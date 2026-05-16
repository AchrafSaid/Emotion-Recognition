import cv2
import numpy as np
import tensorflow as tf
import os

# ── Model ────────────────────────────────────────────────────────────────────
MODEL_PATH = "best_fer2013_cnn_model_train_new.h5"
model = tf.keras.models.load_model(MODEL_PATH)

CLASS_NAMES = ["angry", "fearful", "happy", "neutral", "sad", "surprised"]
EMOJI      = {"angry": "😠", "fearful": "😨", "happy": "😊",
               "neutral": "😐", "sad": "😢", "surprised": "😲"}
COLORS     = {            # BGR
    "angry":    (40,  40, 210),
    "fearful":  (180, 80, 120),
    "happy":    (40, 180,  40),
    "neutral":  (130,130, 130),
    "sad":      (200, 100, 25),
    "surprised":(20, 160, 220),
}

# ── Face detector ─────────────────────────────────────────────────────────────
face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)

# ── Helpers ───────────────────────────────────────────────────────────────────
def preprocess(face_gray):
    face = cv2.resize(face_gray, (48, 48)).astype("float32") / 255.0
    return face[np.newaxis, ..., np.newaxis]          # (1,48,48,1)

def draw_rounded_rect(img, x, y, w, h, r, color, thickness=2):
    """Draw a rounded rectangle."""
    pts = [
        ((x + r, y),           (x + w - r, y)),
        ((x + w, y + r),       (x + w, y + h - r)),
        ((x + w - r, y + h),   (x + r, y + h)),
        ((x, y + h - r),       (x, y + r)),
    ]
    arcs = [
        (x,         y,         (x + 2*r,     y + 2*r),     180, 270),
        (x + w-2*r, y,         (x + w,       y + 2*r),     270, 360),
        (x + w-2*r, y + h-2*r, (x + w,       y + h),       0,    90),
        (x,         y + h-2*r, (x + 2*r,     y + h),       90,  180),
    ]
    for (p1, p2) in pts:
        cv2.line(img, p1, p2, color, thickness)
    for (ax, ay, (bx, by), a0, a1) in arcs:
        cv2.ellipse(img, ((ax+bx)//2, (ay+by)//2),
                    (r, r), 0, a0, a1, color, thickness)

def put_text_bg(img, text, org, font, scale, color, thickness=2, pad=6):
    """Draw text with a filled background pill."""
    (tw, th), base = cv2.getTextSize(text, font, scale, thickness)
    x, y = org
    cv2.rectangle(img,
                  (x - pad, y - th - pad),
                  (x + tw + pad, y + base + pad),
                  color, -1)
    cv2.rectangle(img,
                  (x - pad, y - th - pad),
                  (x + tw + pad, y + base + pad),
                  (255,255,255), 1)
    cv2.putText(img, text, org, font, scale, (255,255,255), thickness, cv2.LINE_AA)

# ── Bar chart drawn on frame ──────────────────────────────────────────────────
def draw_bars(img, scores, top_emotion, x0=10, y0=10, bar_w=120, bar_h=12, gap=4):
    font = cv2.FONT_HERSHEY_SIMPLEX
    for i, name in enumerate(CLASS_NAMES):
        val  = float(scores[i])
        y    = y0 + i * (bar_h + gap)
        fill = int(val * bar_w)
        col  = COLORS[name]
        alpha_overlay = img.copy()
        cv2.rectangle(alpha_overlay, (x0, y), (x0 + bar_w, y + bar_h), (50,50,50), -1)
        cv2.addWeighted(alpha_overlay, 0.5, img, 0.5, 0, img)
        cv2.rectangle(img, (x0, y), (x0 + fill, y + bar_h), col, -1)
        label = f"{name[:3]}  {val*100:4.1f}%"
        cv2.putText(img, label, (x0 + bar_w + 6, y + bar_h - 2),
                    font, 0.38, (220,220,220), 1, cv2.LINE_AA)

# ── Main loop ─────────────────────────────────────────────────────────────────
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    raise RuntimeError("Cannot open camera")

FONT = cv2.FONT_HERSHEY_SIMPLEX

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame  = cv2.flip(frame, 1)          # mirror so it feels natural
    gray   = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces  = face_cascade.detectMultiScale(
        gray, scaleFactor=1.2, minNeighbors=5, minSize=(80, 80)
    )

    for (x, y, w, h) in faces:
        face_crop = preprocess(gray[y:y+h, x:x+w])
        scores    = model.predict(face_crop, verbose=0)[0]
        idx       = int(np.argmax(scores))
        emotion   = CLASS_NAMES[idx]
        conf      = scores[idx] * 100
        color     = COLORS[emotion]

        # rounded face box
        draw_rounded_rect(frame, x, y, w, h, r=14, color=color, thickness=2)

        # emotion label above face
        label = f"{emotion}  {conf:.0f}%"
        lx = x
        ly = y - 12 if y > 30 else y + h + 22
        put_text_bg(frame, label, (lx, ly), FONT, 0.65, color, thickness=2)

        # bar chart (top-left corner)
        draw_bars(frame, scores, emotion, x0=10, y0=10)

    # window title shows current emotion too
    title = "Emotion Recognition  |  q = quit"
    cv2.imshow(title, frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()