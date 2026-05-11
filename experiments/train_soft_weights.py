import json
import os
from collections import Counter
from pathlib import Path

import cv2
import glob
import numpy as np
import tensorflow as tf
from sklearn.metrics import accuracy_score, classification_report, f1_score, precision_score, recall_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.utils.class_weight import compute_class_weight
from tensorflow.keras import layers
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.utils import to_categorical


os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

PROJECT_ROOT = Path(r"C:\Emotion Reco\Emotion-Recognition")
TRAIN_PATH = PROJECT_ROOT / "train"
TEST_PATH = PROJECT_ROOT / "test"
OUTPUT_MODEL = PROJECT_ROOT / "best_fer2013_cnn_soft_weights.h5"
OUTPUT_METRICS = PROJECT_ROOT / "soft_weights_metrics.json"

HEIGHT = 48
WIDTH = 48
BATCH_SIZE = 64
EPOCHS = 50
SEED = 42


def load_folder(folder_path):
    images = []
    labels = []
    for class_name in sorted(os.listdir(folder_path)):
        class_path = folder_path / class_name
        if not class_path.is_dir():
            continue
        files = []
        for ext in ("*.png", "*.jpg", "*.jpeg"):
            files.extend(glob.glob(str(class_path / ext)))
        for file_path in files:
            image = cv2.imread(file_path, cv2.IMREAD_GRAYSCALE)
            if image is None:
                continue
            image = cv2.resize(image, (WIDTH, HEIGHT)).astype("float32") / 255.0
            images.append(np.expand_dims(image, axis=-1))
            labels.append(class_name)
    return np.array(images), np.array(labels)


def create_model(num_classes):
    model = tf.keras.Sequential(
        [
            layers.Input(shape=(HEIGHT, WIDTH, 1)),
            layers.Conv2D(32, (3, 3), padding="same", activation="relu", kernel_initializer="he_normal"),
            layers.Conv2D(32, (3, 3), padding="same", activation="relu", kernel_initializer="he_normal"),
            layers.MaxPooling2D((2, 2)),
            layers.Dropout(0.20),
            layers.Conv2D(64, (3, 3), padding="same", activation="relu", kernel_initializer="he_normal"),
            layers.Conv2D(64, (3, 3), padding="same", activation="relu", kernel_initializer="he_normal"),
            layers.MaxPooling2D((2, 2)),
            layers.Dropout(0.25),
            layers.Conv2D(128, (3, 3), padding="same", activation="relu", kernel_initializer="he_normal"),
            layers.Conv2D(128, (3, 3), padding="same", activation="relu", kernel_initializer="he_normal"),
            layers.MaxPooling2D((2, 2)),
            layers.Dropout(0.30),
            layers.Flatten(),
            layers.Dense(256, activation="relu", kernel_initializer="he_normal"),
            layers.Dropout(0.50),
            layers.Dense(num_classes, activation="softmax"),
        ]
    )
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.0005),
        loss="categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model


def main():
    print("TensorFlow:", tf.__version__)
    print("GPU devices:", tf.config.list_physical_devices("GPU"))

    x_train, y_train = load_folder(TRAIN_PATH)
    x_test, y_test = load_folder(TEST_PATH)

    label_encoder = LabelEncoder()
    y_train_int = label_encoder.fit_transform(y_train)
    y_test_int = label_encoder.transform(y_test)
    class_names = list(label_encoder.classes_)
    num_classes = len(class_names)

    y_train_encoded = to_categorical(y_train_int, num_classes=num_classes)
    x_train_split, x_val, y_train_split_encoded, y_val_encoded, y_train_split_int, _ = train_test_split(
        x_train,
        y_train_encoded,
        y_train_int,
        test_size=0.2,
        random_state=SEED,
        stratify=y_train_int,
    )

    balanced_values = compute_class_weight(
        class_weight="balanced",
        classes=np.unique(y_train_split_int),
        y=y_train_split_int,
    )
    soft_class_weights = {
        class_index: min(float(np.sqrt(weight)), 3.0)
        for class_index, weight in dict(enumerate(balanced_values)).items()
    }

    print("Train split:", x_train_split.shape)
    print("Validation split:", x_val.shape)
    print("Test split:", x_test.shape)
    print("Class names:", class_names)
    print("Train class distribution:", Counter(y_train_split_int))
    print("Soft class weights:", {class_names[k]: round(v, 4) for k, v in soft_class_weights.items()})

    train_datagen = ImageDataGenerator(
        rotation_range=8,
        width_shift_range=0.08,
        height_shift_range=0.08,
        zoom_range=0.08,
        horizontal_flip=True,
    )

    model = create_model(num_classes)
    callbacks = [
        EarlyStopping(monitor="val_accuracy", patience=12, restore_best_weights=True),
        ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=4, min_lr=1e-6, verbose=1),
        ModelCheckpoint(str(OUTPUT_MODEL), monitor="val_accuracy", save_best_only=True, verbose=1),
    ]

    history = model.fit(
        train_datagen.flow(x_train_split, y_train_split_encoded, batch_size=BATCH_SIZE, shuffle=True, seed=SEED),
        epochs=EPOCHS,
        validation_data=(x_val, y_val_encoded),
        class_weight=soft_class_weights,
        callbacks=callbacks,
    )

    if OUTPUT_MODEL.exists():
        model = tf.keras.models.load_model(str(OUTPUT_MODEL))

    predictions = model.predict(x_test, batch_size=BATCH_SIZE)
    y_pred = np.argmax(predictions, axis=1)

    metrics = {
        "accuracy": float(accuracy_score(y_test_int, y_pred)),
        "precision_weighted": float(precision_score(y_test_int, y_pred, average="weighted", zero_division=0)),
        "recall_weighted": float(recall_score(y_test_int, y_pred, average="weighted", zero_division=0)),
        "f1_weighted": float(f1_score(y_test_int, y_pred, average="weighted", zero_division=0)),
        "classification_report": classification_report(
            y_test_int,
            y_pred,
            target_names=class_names,
            zero_division=0,
            output_dict=True,
        ),
        "predicted_distribution": {class_names[k]: int(v) for k, v in Counter(y_pred).items()},
        "best_val_accuracy": float(max(history.history["val_accuracy"])),
    }
    OUTPUT_METRICS.write_text(json.dumps(metrics, indent=2), encoding="utf-8")

    print("Best validation accuracy:", metrics["best_val_accuracy"])
    print("Test accuracy:", metrics["accuracy"])
    print("Test weighted F1:", metrics["f1_weighted"])
    print(
        classification_report(
            y_test_int,
            y_pred,
            target_names=class_names,
            zero_division=0,
        )
    )


if __name__ == "__main__":
    main()
