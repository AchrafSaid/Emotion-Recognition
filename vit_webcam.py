import argparse
from pathlib import Path
from typing import Sequence, Tuple

import cv2
import numpy as np
import timm
import torch
from PIL import Image
from torchvision import transforms


PROJECT_ROOT = Path(r"C:\Emotion-Recognition")
REGULARIZED_CHECKPOINT = PROJECT_ROOT / "vit_runs" / "vit_fer2013_6class_regularized_best.pth"
ORIGINAL_CHECKPOINT = PROJECT_ROOT / "vit_runs" / "vit_fer2013_6class_best.pth"
DEFAULT_CHECKPOINT = REGULARIZED_CHECKPOINT if REGULARIZED_CHECKPOINT.exists() else ORIGINAL_CHECKPOINT


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run ViT emotion recognition on webcam.")
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--camera", type=int, default=0)
    parser.add_argument("--width", type=int, default=960)
    parser.add_argument("--height", type=int, default=540)
    parser.add_argument("--face-margin", type=float, default=0.08)
    parser.add_argument("--smoothing", type=float, default=0.70)
    parser.add_argument(
        "--preprocess",
        choices=("gray-eq", "gray", "color"),
        default="gray-eq",
        help="gray-eq usually matches FER-style grayscale training images best.",
    )
    parser.add_argument("--no-mirror", action="store_true")
    return parser.parse_args()


def load_checkpoint(path: Path, device: torch.device) -> dict:
    if not path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {path}")
    try:
        return torch.load(path, map_location=device, weights_only=False)
    except TypeError:
        return torch.load(path, map_location=device)


def build_transform(img_size: int) -> transforms.Compose:
    return transforms.Compose(
        [
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )


def load_model(checkpoint_path: Path, device: torch.device):
    checkpoint = load_checkpoint(checkpoint_path, device)
    class_names = checkpoint["class_names"]
    display_class_names = checkpoint.get(
        "display_class_names",
        ["surprised" if name == "suprised" else name for name in class_names],
    )
    img_size = int(checkpoint.get("img_size", 224))
    model_name = checkpoint.get("model_name", "vit_tiny_patch16_224")

    model = timm.create_model(model_name, pretrained=False, num_classes=len(class_names))
    model.load_state_dict(checkpoint["model_state"])
    model.to(device)
    model.eval()

    return model, display_class_names, build_transform(img_size)


def preprocess_face_for_model(face_bgr: np.ndarray, mode: str) -> Image.Image:
    if mode == "color":
        face_rgb = cv2.cvtColor(face_bgr, cv2.COLOR_BGR2RGB)
        return Image.fromarray(face_rgb)

    gray = cv2.cvtColor(face_bgr, cv2.COLOR_BGR2GRAY)
    if mode == "gray-eq":
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        gray = clahe.apply(gray)

    face_rgb = cv2.cvtColor(gray, cv2.COLOR_GRAY2RGB)
    return Image.fromarray(face_rgb)


@torch.no_grad()
def predict_face(
    model: torch.nn.Module,
    transform: transforms.Compose,
    face_bgr: np.ndarray,
    device: torch.device,
    preprocess_mode: str,
) -> np.ndarray:
    image = preprocess_face_for_model(face_bgr, preprocess_mode)
    tensor = transform(image).unsqueeze(0).to(device)
    logits = model(tensor)
    return torch.softmax(logits, dim=1).squeeze(0).detach().cpu().numpy()


def largest_face(faces: Sequence[Tuple[int, int, int, int]]) -> Tuple[int, int, int, int] | None:
    if len(faces) == 0:
        return None
    return max(faces, key=lambda item: item[2] * item[3])


def square_face_crop(
    frame: np.ndarray,
    face_box: Tuple[int, int, int, int],
    margin_ratio: float,
) -> Tuple[np.ndarray, Tuple[int, int, int, int]]:
    x, y, w, h = face_box
    side = int(max(w, h) * (1.0 + 2.0 * margin_ratio))
    cx = x + w // 2
    cy = y + h // 2

    x1 = max(0, cx - side // 2)
    y1 = max(0, cy - side // 2)
    x2 = min(frame.shape[1], x1 + side)
    y2 = min(frame.shape[0], y1 + side)

    x1 = max(0, x2 - side)
    y1 = max(0, y2 - side)

    return frame[y1:y2, x1:x2], (x1, y1, x2, y2)


def draw_percent_panel(
    frame: np.ndarray,
    class_names: Sequence[str],
    probabilities: np.ndarray | None,
    panel_width: int = 320,
) -> np.ndarray:
    height = frame.shape[0]
    panel = np.full((height, panel_width, 3), 245, dtype=np.uint8)

    cv2.putText(panel, "ViT Emotion %", (18, 34), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (25, 25, 25), 2)
    cv2.putText(panel, "press q to quit", (18, 62), cv2.FONT_HERSHEY_SIMPLEX, 0.52, (80, 80, 80), 1)

    if probabilities is None:
        cv2.putText(panel, "No face detected", (18, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (30, 30, 200), 2)
        return np.hstack([frame, panel])

    order = np.argsort(-probabilities)
    y = 112
    bar_x = 18
    bar_w = panel_width - 36
    bar_h = 18

    for idx in order:
        name = class_names[idx]
        pct = float(probabilities[idx] * 100)
        label = f"{name}: {pct:5.1f}%"

        cv2.putText(panel, label, (bar_x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.58, (35, 35, 35), 2)
        y += 10

        cv2.rectangle(panel, (bar_x, y), (bar_x + bar_w, y + bar_h), (220, 220, 220), -1)
        filled = int(bar_w * probabilities[idx])
        color = (30, 150, 80) if idx == int(np.argmax(probabilities)) else (120, 120, 120)
        cv2.rectangle(panel, (bar_x, y), (bar_x + filled, y + bar_h), color, -1)
        cv2.rectangle(panel, (bar_x, y), (bar_x + bar_w, y + bar_h), (180, 180, 180), 1)
        y += 42

    return np.hstack([frame, panel])


def open_camera(camera_index: int, width: int, height: int) -> cv2.VideoCapture:
    capture = cv2.VideoCapture(camera_index, cv2.CAP_DSHOW)
    if not capture.isOpened():
        capture = cv2.VideoCapture(camera_index)

    capture.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    capture.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
    return capture


def main() -> None:
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, class_names, transform = load_model(args.checkpoint, device)

    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
    if face_cascade.empty():
        raise RuntimeError("Could not load OpenCV Haar face detector.")

    camera = open_camera(args.camera, args.width, args.height)
    if not camera.isOpened():
        raise RuntimeError(
            f"Could not open camera {args.camera}. Try --camera 1 if you have another webcam."
        )

    print("Webcam started. Press q to quit.")
    print("Classes:", ", ".join(class_names))
    print("Checkpoint:", args.checkpoint)
    print("Preprocess:", args.preprocess)

    smoothed_probabilities = None

    while True:
        ok, frame = camera.read()
        if not ok:
            print("Could not read frame from camera.")
            break

        if not args.no_mirror:
            frame = cv2.flip(frame, 1)

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.15, minNeighbors=5, minSize=(45, 45))
        probabilities = None

        for x, y, w, h in faces:
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 190, 80), 2)

        main_face = largest_face(faces)
        if main_face is not None:
            face, (x1, y1, x2, y2) = square_face_crop(frame, main_face, args.face_margin)

            raw_probabilities = predict_face(model, transform, face, device, args.preprocess)
            if smoothed_probabilities is None:
                smoothed_probabilities = raw_probabilities
            else:
                alpha = float(np.clip(args.smoothing, 0.0, 0.95))
                smoothed_probabilities = alpha * smoothed_probabilities + (1.0 - alpha) * raw_probabilities
                smoothed_probabilities = smoothed_probabilities / smoothed_probabilities.sum()

            probabilities = smoothed_probabilities
            pred_idx = int(np.argmax(probabilities))
            label = f"{class_names[pred_idx]} {probabilities[pred_idx] * 100:.1f}%"

            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 210, 80), 3)
            cv2.putText(
                frame,
                label,
                (x1, max(28, y1 - 10)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.85,
                (0, 210, 80),
                2,
            )
        else:
            smoothed_probabilities = None

        output = draw_percent_panel(frame, class_names, probabilities)
        cv2.imshow("ViT Emotion Recognition", output)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    camera.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
