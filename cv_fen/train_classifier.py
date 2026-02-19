"""
Train a piece classifier on organized square images
"""

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import numpy as np
import cv2
from pathlib import Path
from sklearn.model_selection import train_test_split
import json

print("TensorFlow version:", tf.__version__)

# Settings
IMG_SIZE = 64
BATCH_SIZE = 32
EPOCHS = 50
DATASET_DIR = Path('square_dataset_organized')

# Piece classes
CLASSES = ['empty', 'P', 'N', 'B', 'R', 'Q', 'K', 'p', 'n', 'b', 'r', 'q', 'k']

def load_dataset():
    """Load and preprocess all images"""
    images = []
    labels = []
    
    print("\nLoading dataset...")
    for class_idx, class_name in enumerate(CLASSES):
        class_dir = DATASET_DIR / class_name
        if not class_dir.exists():
            print(f"Warning: {class_name} folder not found, skipping")
            continue
        
        image_files = list(class_dir.glob('*.jpg'))
        print(f"  {class_name:6s}: {len(image_files):4d} images")
        
        for img_file in image_files:
            img = cv2.imread(str(img_file))
            if img is None:
                continue
            
            # Resize to standard size
            img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
            # Normalize to [0, 1]
            img = img.astype(np.float32) / 255.0
            
            images.append(img)
            labels.append(class_idx)
    
    images = np.array(images)
    labels = np.array(labels)
    
    print(f"\nTotal images: {len(images)}")
    print(f"Image shape: {images.shape}")
    
    return images, labels

def create_model(num_classes):
    """Create a lightweight CNN for piece classification"""
    model = keras.Sequential([
        # Input layer
        layers.Input(shape=(IMG_SIZE, IMG_SIZE, 3)),
        
        # Data augmentation
        layers.RandomFlip("horizontal"),
        layers.RandomRotation(0.05),
        layers.RandomBrightness(0.1),
        
        # Conv blocks
        layers.Conv2D(32, 3, activation='relu', padding='same'),
        layers.BatchNormalization(),
        layers.MaxPooling2D(2),
        
        layers.Conv2D(64, 3, activation='relu', padding='same'),
        layers.BatchNormalization(),
        layers.MaxPooling2D(2),
        
        layers.Conv2D(128, 3, activation='relu', padding='same'),
        layers.BatchNormalization(),
        layers.MaxPooling2D(2),
        
        layers.Conv2D(256, 3, activation='relu', padding='same'),
        layers.BatchNormalization(),
        layers.GlobalAveragePooling2D(),
        
        # Dense layers
        layers.Dropout(0.3),
        layers.Dense(128, activation='relu'),
        layers.Dropout(0.3),
        layers.Dense(num_classes, activation='softmax')
    ])
    
    return model

def main():
    # Check if dataset exists
    if not DATASET_DIR.exists():
        print(f"Error: {DATASET_DIR} not found!")
        print("Run organize_dataset.py first to label your images")
        return
    
    # Load data
    images, labels = load_dataset()
    
    if len(images) == 0:
        print("No images found! Run organize_dataset.py first")
        return
    
    # Check class distribution
    unique, counts = np.unique(labels, return_counts=True)
    print("\nClass distribution:")
    for cls_idx, count in zip(unique, counts):
        print(f"  {CLASSES[cls_idx]:6s}: {count:4d} samples")
    
    # Split into train/validation
    X_train, X_val, y_train, y_val = train_test_split(
        images, labels, test_size=0.2, random_state=42, stratify=labels
    )
    
    print(f"\nTrain samples: {len(X_train)}")
    print(f"Val samples:   {len(X_val)}")
    
    # Create model
    model = create_model(len(CLASSES))
    
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=0.001),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    
    print("\nModel summary:")
    model.summary()
    
    # Callbacks
    callbacks = [
        keras.callbacks.EarlyStopping(
            monitor='val_loss',
            patience=10,
            restore_best_weights=True
        ),
        keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=5,
            min_lr=1e-6
        )
    ]
    
    # Train
    print("\n" + "="*60)
    print("Starting training...")
    print("="*60 + "\n")
    
    history = model.fit(
        X_train, y_train,
        batch_size=BATCH_SIZE,
        epochs=EPOCHS,
        validation_data=(X_val, y_val),
        callbacks=callbacks,
        verbose=1
    )
    
    # Evaluate
    print("\n" + "="*60)
    print("Final evaluation:")
    val_loss, val_acc = model.evaluate(X_val, y_val, verbose=0)
    print(f"Validation Loss: {val_loss:.4f}")
    print(f"Validation Accuracy: {val_acc:.4f}")
    
    # Save model
    model.save('piece_classifier.h5')
    print("\n✓ Model saved as: piece_classifier.h5")
    
    # Save class mapping
    class_mapping = {i: cls for i, cls in enumerate(CLASSES)}
    with open('piece_classifier_classes.json', 'w') as f:
        json.dump(class_mapping, f, indent=2)
    print("✓ Class mapping saved as: piece_classifier_classes.json")
    
    print("\n" + "="*60)
    print("Training complete!")
    print("="*60)
    print("\nNext step: The model will be automatically used by board_to_fen.py")

if __name__ == "__main__":
    main()
