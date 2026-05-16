"""
clean_dataset.py

Purpose:
Create a cleaned copy of the dataset using only image cleaning techniques.

Cleaning techniques included:
1. Corrupted image detection
2. Blur detection / low-quality image filtering
3. Deblurring using sharpening filter
4. Pixel value clipping to 0-255

This script creates ONLY TWO cleaned dataset folders inside archive:
- train_clean
- test_clean

Rejected blurry images are not saved separately.
A cleaning log CSV is saved inside archive.
"""

import os
import csv
from pathlib import Path

import numpy as np
from PIL import Image


# =========================
# Paths
# =========================

DATASET_ROOT = Path(r"C:\Users\nadee\Downloads\Emotion-Recognition\archive")

# Input folders
TRAIN_INPUT = DATASET_ROOT / "train"
TEST_INPUT = DATASET_ROOT / "test"

# Output folders inside archive
TRAIN_OUTPUT = DATASET_ROOT / "train_clean"
TEST_OUTPUT = DATASET_ROOT / "test_clean"

# Cleaning log
LOG_PATH = DATASET_ROOT / "cleaning_log.csv"

# Image extensions
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp"}

# Blur threshold
BLUR_THRESHOLD = 20.0


# =========================
# Helper functions
# =========================

def load_grayscale_image(image_path):
    """
    Loads image and converts it to grayscale.
    """
    img = Image.open(image_path).convert("L")
    img_array = np.array(img, dtype=np.float32)
    return img_array


def pixel_value_clipping(img):
    """
    Keeps pixel values inside the valid range 0-255.
    """
    return np.clip(img, 0, 255)


def apply_kernel(img, kernel):
    """
    Manual 2D convolution.
    """
    h, w = img.shape
    pad = kernel.shape[0] // 2

    padded = np.pad(img, pad_width=pad, mode="edge")
    output = np.zeros_like(img, dtype=np.float32)

    for y in range(h):
        for x in range(w):
            region = padded[y:y + kernel.shape[0], x:x + kernel.shape[1]]
            output[y, x] = np.sum(region * kernel)

    return output


def calculate_blur_score(img):
    """
    Calculates blur score using Laplacian variance.
    Low score means the image is blurry.
    """
    laplacian_kernel = np.array([
        [0,  1, 0],
        [1, -4, 1],
        [0,  1, 0]
    ], dtype=np.float32)

    laplacian = apply_kernel(img, laplacian_kernel)
    return float(np.var(laplacian))


def deblur_using_sharpening(img):
    """
    Simple deblurring using sharpening filter.
    """
    sharpening_kernel = np.array([
        [0, -1, 0],
        [-1, 5, -1],
        [0, -1, 0]
    ], dtype=np.float32)

    sharpened = apply_kernel(img, sharpening_kernel)
    sharpened = pixel_value_clipping(sharpened)

    return sharpened


def save_grayscale_image(img, output_path):
    """
    Saves image as uint8 grayscale.
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)
    img_uint8 = pixel_value_clipping(img).astype(np.uint8)
    Image.fromarray(img_uint8, mode="L").save(output_path)


def clean_one_image(input_path, output_path):
    """
    Cleans one image:
    - Detects corrupted images
    - Detects blurry images
    - Applies deblurring/sharpening
    - Clips pixel values
    - Saves cleaned image
    """
    try:
        img = load_grayscale_image(input_path)
    except Exception as e:
        return {
            "status": "corrupted_not_saved",
            "input_path": str(input_path),
            "output_path": "",
            "blur_score": "",
            "reason": f"Could not open image: {e}"
        }

    blur_score = calculate_blur_score(img)

    if blur_score < BLUR_THRESHOLD:
        return {
            "status": "blurry_not_saved",
            "input_path": str(input_path),
            "output_path": "",
            "blur_score": blur_score,
            "reason": "Image blur score below threshold"
        }

    cleaned_img = deblur_using_sharpening(img)
    cleaned_img = pixel_value_clipping(cleaned_img)

    save_grayscale_image(cleaned_img, output_path)

    return {
        "status": "saved_cleaned",
        "input_path": str(input_path),
        "output_path": str(output_path),
        "blur_score": blur_score,
        "reason": "Image cleaned and saved"
    }


def clean_dataset_folder(input_folder, output_folder):
    """
    Cleans all images inside a folder.
    Keeps the same class-folder structure.
    """
    logs = []

    if not input_folder.exists():
        raise FileNotFoundError(f"Input folder not found: {input_folder}")

    for class_name in sorted(os.listdir(input_folder)):
        class_input_path = input_folder / class_name

        if not class_input_path.is_dir():
            continue

        for image_name in os.listdir(class_input_path):
            input_path = class_input_path / image_name

            if input_path.suffix.lower() not in IMAGE_EXTENSIONS:
                continue

            output_path = output_folder / class_name / image_name
            result = clean_one_image(input_path, output_path)
            logs.append(result)

    return logs


def save_log(logs, log_path):
    """
    Saves cleaning results into CSV file.
    """
    with open(log_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["status", "input_path", "output_path", "blur_score", "reason"]
        )
        writer.writeheader()
        writer.writerows(logs)


def print_summary(logs, title):
    """
    Prints summary of cleaning results.
    """
    total = len(logs)
    saved = sum(1 for item in logs if item["status"] == "saved_cleaned")
    blurry = sum(1 for item in logs if item["status"] == "blurry_not_saved")
    corrupted = sum(1 for item in logs if item["status"] == "corrupted_not_saved")

    print()
    print(title)
    print("-" * len(title))
    print("Total images checked:", total)
    print("Saved cleaned images:", saved)
    print("Blurry images not saved:", blurry)
    print("Corrupted images not saved:", corrupted)


# =========================
# Run cleaning
# =========================

if __name__ == "__main__":
    print("Cleaning training dataset...")
    train_logs = clean_dataset_folder(TRAIN_INPUT, TRAIN_OUTPUT)

    print("Cleaning testing dataset...")
    test_logs = clean_dataset_folder(TEST_INPUT, TEST_OUTPUT)

    all_logs = train_logs + test_logs
    save_log(all_logs, LOG_PATH)

    print_summary(train_logs, "Training Cleaning Summary")
    print_summary(test_logs, "Testing Cleaning Summary")

    print()
    print("Cleaned train folder:")
    print(TRAIN_OUTPUT)

    print()
    print("Cleaned test folder:")
    print(TEST_OUTPUT)

    print()
    print("Cleaning log:")
    print(LOG_PATH)
