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
from tensorflow.keras import layers
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.utils import to_categorical


os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

PROJECT_ROOT = Path(r"C:\Emotion Reco\Emotion-Recognition")
TRAIN_PATH = PROJECT_ROOT / "train"
TEST_PATH = PROJECT_ROOT / "test"
OUTPUT_MODEL = PROJECT_ROOT / "best_fer2013_mobilenetv2_transfer.h5"
OUTPUT_METRICS = PROJECT_ROOT / "mobilenetv2_transfer_metrics.json"

HEIGHT = 48
WIDTH = 48
TRANSFER_SIZE = 96
BATCH_SIZE = 64
HEAD_EPOCHS = 12
FINE_TUNE_EPOCHS = 28
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


def build_model(num_classes):
    inputs = layers.Input(shape=(HEIGHT, WIDTH, 1))
    x = layers.Resizing(TRANSFER_SIZE, TRANSFER_SIZE)(inputs)
    x = layers.Concatenate(axis=-1)([x, x, x])
    x = layers.Lambda(lambda t: tf.keras.applications.mobilenet_v2.preprocess_input(t * 255.0))(x)

    base_model = tf.keras.applications.MobileNetV2(
        include_top=False,
        weights="imagenet",
        input_shape=(TRANSFER_SIZE, TRANSFER_SIZE, 3),
    )
    base_model.trainable = False

    x = base_model(x, training=False)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dropout(0.35)(x)
    x = layers.Dense(256, activation="relu")(x)
    x = layers.Dropout(0.35)(x)
    outputs = layers.Dense(num_classes, activation="softmax")(x)

    model = tf.keras.Model(inputs, outputs)
    return model, base_model


def compile_model(model, learning_rate):
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
        loss="categorical_crossentropy",
        metrics=["accuracy"],
    )


def evaluate_and_save(model, x_test, y_test_int, class_names, best_val_accuracy):
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
        "best_val_accuracy": float(best_val_accuracy),
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

    x_train_split, x_val, y_train_split_encoded, y_val_encoded, _, _ = train_test_split(
        x_train,
        y_train_encoded,
        y_train_int,
        test_size=0.2,
        random_state=SEED,
        stratify=y_train_int,
    )

    print("Train split:", x_train_split.shape)
    print("Validation split:", x_val.shape)
    print("Test split:", x_test.shape)
    print("Class names:", class_names)

    train_datagen = ImageDataGenerator(
        rotation_range=8,
        width_shift_range=0.08,
        height_shift_range=0.08,
        zoom_range=0.08,
        horizontal_flip=True,
    )

    model, base_model = build_model(num_classes)
    compile_model(model, learning_rate=0.001)
    model.summary()

    callbacks = [
        EarlyStopping(monitor="val_accuracy", patience=8, restore_best_weights=True),
        ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=3, min_lr=1e-6, verbose=1),
        ModelCheckpoint(str(OUTPUT_MODEL), monitor="val_accuracy", save_best_only=True, verbose=1),
    ]

    print("Training classifier head...")
    head_history = model.fit(
        train_datagen.flow(x_train_split, y_train_split_encoded, batch_size=BATCH_SIZE, shuffle=True, seed=SEED),
        epochs=HEAD_EPOCHS,
        validation_data=(x_val, y_val_encoded),
        callbacks=callbacks,
    )

    print("Fine-tuning top MobileNetV2 layers...")
    base_model.trainable = True
    for layer in base_model.layers[:-35]:
        layer.trainable = False
    for layer in base_model.layers:
        if isinstance(layer, layers.BatchNormalization):
            layer.trainable = False

    compile_model(model, learning_rate=1e-5)
    fine_history = model.fit(
        train_datagen.flow(x_train_split, y_train_split_encoded, batch_size=BATCH_SIZE, shuffle=True, seed=SEED),
        epochs=FINE_TUNE_EPOCHS,
        validation_data=(x_val, y_val_encoded),
        callbacks=callbacks,
    )

    best_val_accuracy = max(
        max(head_history.history["val_accuracy"]),
        max(fine_history.history["val_accuracy"]),
    )
    evaluate_and_save(model, x_test, y_test_int, class_names, best_val_accuracy)


if __name__ == "__main__":
    main()
