"""!
@file train_classifier.py
@brief Trains and exports a multitask chess-piece classifier from labeled square datasets.
"""

import json
from pathlib import Path

import cv2
import numpy as np
import tensorflow as tf
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight
from tensorflow import keras
from tensorflow.keras import layers

print("TensorFlow version:", tf.__version__)

# Settings
IMG_SIZE = 160
BATCH_SIZE = 16
EPOCHS = 24
WARMUP_EPOCHS = 8
CENTER_CROP = 0.82
DATASET_DIR = Path("square_dataset_organized")
RANDOM_SEED = 42

PIECE_TYPES = ["P", "N", "B", "R", "Q", "K"]
COLOR_CLASSES = ["black", "white"]

# Explicit directory names avoid Windows case-insensitive collisions.
CLASS_SPECS = [
    ("P", "P", "white"),
    ("N", "N", "white"),
    ("B", "B", "white"),
    ("R", "R", "white"),
    ("Q", "Q", "white"),
    ("K", "K", "white"),
    ("p", "black_p", "black"),
    ("n", "black_n", "black"),
    ("b", "black_b", "black"),
    ("r", "black_r", "black"),
    ("q", "black_q", "black"),
    ("k", "black_k", "black"),
]

np.random.seed(RANDOM_SEED)
tf.random.set_seed(RANDOM_SEED)


def preprocess_square(img):
    """Center-crop and normalize a square for MobileNetV2."""
    if img is None:
        return None

    h, w = img.shape[:2]
    crop_h = max(1, int(h * CENTER_CROP))
    crop_w = max(1, int(w * CENTER_CROP))
    y0 = max(0, (h - crop_h) // 2)
    x0 = max(0, (w - crop_w) // 2)
    cropped = img[y0:y0 + crop_h, x0:x0 + crop_w]

    resized = cv2.resize(cropped, (IMG_SIZE, IMG_SIZE), interpolation=cv2.INTER_AREA)
    rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
    rgb = rgb.astype(np.float32)
    return tf.keras.applications.mobilenet_v2.preprocess_input(rgb)


def load_dataset():
    """Load only occupied-square samples with separate type and color labels."""
    images = []
    type_labels = []
    color_labels = []
    stratify_labels = []

    print("\nLoading occupied-square dataset...")
    for fen_label, folder_name, color_name in CLASS_SPECS:
        class_dir = DATASET_DIR / folder_name
        if not class_dir.exists():
            print(f"Warning: {folder_name} folder not found for class {fen_label}, skipping")
            continue

        image_files = sorted(class_dir.glob("*.jpg"))
        piece_type = fen_label.upper()
        type_index = PIECE_TYPES.index(piece_type)
        color_index = COLOR_CLASSES.index(color_name)

        print(f"  {fen_label:6s}: {len(image_files):4d} images from {folder_name}")
        for img_file in image_files:
            img = cv2.imread(str(img_file))
            processed = preprocess_square(img)
            if processed is None:
                continue

            images.append(processed)
            type_labels.append(type_index)
            color_labels.append(color_index)
            stratify_labels.append(f"{piece_type}_{color_name}")

    images = np.asarray(images, dtype=np.float32)
    type_labels = np.asarray(type_labels, dtype=np.int32)
    color_labels = np.asarray(color_labels, dtype=np.int32)
    stratify_labels = np.asarray(stratify_labels)

    print(f"\nTotal occupied images: {len(images)}")
    print(f"Image shape: {images.shape}")
    return images, type_labels, color_labels, stratify_labels


def create_model():
    """Create a transfer-learning model with separate type and color heads."""
    inputs = keras.Input(shape=(IMG_SIZE, IMG_SIZE, 3), name="image")
    augmented = keras.Sequential(
        [
            layers.RandomFlip("horizontal"),
            layers.RandomRotation(0.03),
            layers.RandomZoom(0.08),
            layers.RandomContrast(0.15),
            layers.GaussianNoise(0.03),
        ],
        name="augmentation",
    )(inputs)

    try:
        backbone = tf.keras.applications.MobileNetV2(
            input_shape=(IMG_SIZE, IMG_SIZE, 3),
            include_top=False,
            weights="imagenet",
        )
        print("Loaded MobileNetV2 ImageNet weights")
    except Exception as exc:
        print(f"Warning: could not load ImageNet weights ({exc})")
        print("Falling back to randomly initialized MobileNetV2")
        backbone = tf.keras.applications.MobileNetV2(
            input_shape=(IMG_SIZE, IMG_SIZE, 3),
            include_top=False,
            weights=None,
        )

    backbone.trainable = False
    features = backbone(augmented, training=False)
    features = layers.GlobalAveragePooling2D()(features)
    features = layers.Dropout(0.25)(features)
    shared = layers.Dense(256, activation="relu")(features)
    shared = layers.Dropout(0.25)(shared)

    piece_type_output = layers.Dense(
        len(PIECE_TYPES), activation="softmax", name="piece_type"
    )(shared)
    piece_color_output = layers.Dense(
        len(COLOR_CLASSES), activation="softmax", name="piece_color"
    )(shared)

    model = keras.Model(
        inputs=inputs,
        outputs={"piece_type": piece_type_output, "piece_color": piece_color_output},
        name="piece_classifier_multitask",
    )
    return model, backbone


def compile_model(model, learning_rate, color_loss_weight=1.35):
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=learning_rate),
        loss={
            "piece_type": keras.losses.SparseCategoricalCrossentropy(),
            "piece_color": keras.losses.SparseCategoricalCrossentropy(),
        },
        loss_weights={
            "piece_type": 1.0,
            "piece_color": color_loss_weight,
        },
        metrics={
            "piece_type": ["accuracy"],
            "piece_color": ["accuracy"],
        },
    )


def make_callbacks():
    return [
        keras.callbacks.ModelCheckpoint(
            "piece_classifier_best.keras",
            monitor="val_piece_type_accuracy",
            mode="max",
            save_best_only=True,
            verbose=1,
        ),
        keras.callbacks.EarlyStopping(
            monitor="val_piece_type_accuracy",
            patience=6,
            mode="max",
            restore_best_weights=True,
        ),
        keras.callbacks.ReduceLROnPlateau(
            monitor="val_piece_type_accuracy",
            factor=0.5,
            patience=3,
            mode="max",
            min_lr=1e-6,
        ),
    ]


def build_sample_weights(y_type, y_color):
    """Compute per-sample weights to counter class imbalance."""
    type_classes = np.unique(y_type)
    color_classes = np.unique(y_color)

    type_class_weights = compute_class_weight(
        class_weight="balanced", classes=type_classes, y=y_type
    )
    color_class_weights = compute_class_weight(
        class_weight="balanced", classes=color_classes, y=y_color
    )

    type_weight_map = {int(c): float(w) for c, w in zip(type_classes, type_class_weights)}
    color_weight_map = {int(c): float(w) for c, w in zip(color_classes, color_class_weights)}

    type_sample_weights = np.array([type_weight_map[int(label)] for label in y_type], dtype=np.float32)
    color_sample_weights = np.array([color_weight_map[int(label)] for label in y_color], dtype=np.float32)

    return (
        {"piece_type": type_sample_weights, "piece_color": color_sample_weights},
        {"piece_type": type_weight_map, "piece_color": color_weight_map},
    )


def print_confusion(name, matrix, labels):
    print(f"\n{name} confusion matrix (rows=true, cols=pred):")
    header = "      " + " ".join(f"{lbl:>6s}" for lbl in labels)
    print(header)
    for idx, row in enumerate(matrix):
        row_values = " ".join(f"{int(v):6d}" for v in row)
        print(f"{labels[idx]:>6s} {row_values}")


def unfreeze_backbone(backbone, trainable_tail=40):
    """Fine-tune the last part of the backbone while keeping batch norms frozen."""
    backbone.trainable = True
    for layer in backbone.layers[:-trainable_tail]:
        layer.trainable = False
    for layer in backbone.layers[-trainable_tail:]:
        if isinstance(layer, layers.BatchNormalization):
            layer.trainable = False


def main():
    if not DATASET_DIR.exists():
        print(f"Error: {DATASET_DIR} not found!")
        print("Run organize_dataset.py first to label your images")
        return

    images, type_labels, color_labels, stratify_labels = load_dataset()
    if len(images) == 0:
        print("No occupied images found! Check your dataset folders.")
        return

    print("\nPiece-type distribution:")
    unique_types, type_counts = np.unique(type_labels, return_counts=True)
    for cls_idx, count in zip(unique_types, type_counts):
        print(f"  {PIECE_TYPES[cls_idx]:6s}: {count:4d} samples")

    print("\nColor distribution:")
    unique_colors, color_counts = np.unique(color_labels, return_counts=True)
    for cls_idx, count in zip(unique_colors, color_counts):
        print(f"  {COLOR_CLASSES[cls_idx]:6s}: {count:4d} samples")

    split = train_test_split(
        images,
        type_labels,
        color_labels,
        stratify_labels,
        test_size=0.2,
        random_state=RANDOM_SEED,
        stratify=stratify_labels,
    )
    X_train, X_val, y_type_train, y_type_val, y_color_train, y_color_val, _, _ = split

    print(f"\nTrain samples: {len(X_train)}")
    print(f"Val samples:   {len(X_val)}")

    model, backbone = create_model()
    print("\nModel summary:")
    model.summary()

    train_targets = {"piece_type": y_type_train, "piece_color": y_color_train}
    val_targets = {"piece_type": y_type_val, "piece_color": y_color_val}
    train_sample_weights, class_weight_maps = build_sample_weights(y_type_train, y_color_train)

    print("\nClass weights:")
    print(f"  Piece-type weights: {class_weight_maps['piece_type']}")
    print(f"  Color weights:      {class_weight_maps['piece_color']}")

    print("\n" + "=" * 60)
    print("Stage 1: training heads")
    print("=" * 60 + "\n")
    compile_model(model, learning_rate=1e-3)
    history_warmup = model.fit(
        X_train,
        train_targets,
        sample_weight=train_sample_weights,
        batch_size=BATCH_SIZE,
        epochs=WARMUP_EPOCHS,
        validation_data=(X_val, val_targets),
        callbacks=make_callbacks(),
        verbose=1,
    )

    print("\n" + "=" * 60)
    print("Stage 2: fine-tuning backbone tail")
    print("=" * 60 + "\n")
    unfreeze_backbone(backbone)
    compile_model(model, learning_rate=1e-5)
    model.fit(
        X_train,
        train_targets,
        sample_weight=train_sample_weights,
        batch_size=BATCH_SIZE,
        initial_epoch=len(history_warmup.history["loss"]),
        epochs=EPOCHS,
        validation_data=(X_val, val_targets),
        callbacks=make_callbacks(),
        verbose=1,
    )

    print("\n" + "=" * 60)
    print("Final evaluation:")
    eval_results = model.evaluate(X_val, val_targets, verbose=0, return_dict=True)
    print(f"Validation total loss:        {eval_results['loss']:.4f}")
    print(f"Validation piece-type acc:   {eval_results['piece_type_accuracy']:.4f}")
    print(f"Validation piece-color acc:  {eval_results['piece_color_accuracy']:.4f}")

    val_pred = model.predict(X_val, verbose=0)
    type_pred = np.argmax(val_pred["piece_type"], axis=1)
    color_pred = np.argmax(val_pred["piece_color"], axis=1)

    type_cm = confusion_matrix(y_type_val, type_pred, labels=np.arange(len(PIECE_TYPES)))
    color_cm = confusion_matrix(y_color_val, color_pred, labels=np.arange(len(COLOR_CLASSES)))
    print_confusion("Piece type", type_cm, PIECE_TYPES)
    print_confusion("Piece color", color_cm, COLOR_CLASSES)

    model.save("piece_classifier.keras")
    print("\n✓ Model saved as: piece_classifier.keras")

    metadata = {
        "model_kind": "piece_type_color_multitask_v1",
        "img_size": IMG_SIZE,
        "center_crop": CENTER_CROP,
        "preprocessing": "mobilenet_v2",
        "piece_types": PIECE_TYPES,
        "color_classes": COLOR_CLASSES,
        "class_specs": CLASS_SPECS,
        "color_confidence_recommended": 0.80,
        "type_confidence_recommended": 0.50,
        "class_weights": class_weight_maps,
    }
    with open("piece_classifier_classes.json", "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)
    print("✓ Model metadata saved as: piece_classifier_classes.json")

    print("\n" + "=" * 60)
    print("Training complete!")
    print("=" * 60)
    print("\nNext step: board_to_fen.py will load the multitask classifier automatically")


if __name__ == "__main__":
    main()
