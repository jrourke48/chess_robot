"""
Helper script to organize square images into labeled folders for training
Shows each square image and asks you to label it
"""

import cv2
import os
import shutil
from pathlib import Path
import numpy as np

# Define piece classes
CLASSES = ['empty', 'P', 'N', 'B', 'R', 'Q', 'K', 'p', 'n', 'b', 'r', 'q', 'k']
CLASS_NAMES = {
    'empty': 'Empty square',
    'P': 'White Pawn',
    'N': 'White Knight', 
    'B': 'White Bishop',
    'R': 'White Rook',
    'Q': 'White Queen',
    'K': 'White King',
    'p': 'Black Pawn',
    'n': 'Black Knight',
    'b': 'Black Bishop',
    'r': 'Black Rook',
    'q': 'Black Queen',
    'k': 'Black King'
}

# Create organized dataset directory
output_dir = Path('square_dataset_organized')
for cls in CLASSES:
    (output_dir / cls).mkdir(parents=True, exist_ok=True)

# Get all captured positions
dataset_dir = Path('square_dataset')
positions = [d for d in dataset_dir.iterdir() if d.is_dir()]

if not positions:
    print("No data found in square_dataset/")
    print("Run extract_squares.py first to capture board positions")
    exit()

print(f"Found {len(positions)} captured positions")
print("\n=== Square Labeling Tool ===")
print("\nFor each image, enter the piece label:")
for cls in CLASSES:
    print(f"  {cls:6s} = {CLASS_NAMES[cls]}")
print("\nOr press 's' to skip, 'exit' to quit")
print("\n" + "="*60 + "\n")

labeled_count = 0
skipped_count = 0

for position in positions:
    print(f"\nProcessing position: {position.name}")
    
    # Get all square images
    square_files = sorted([f for f in position.glob('*.jpg') if not f.name.startswith('_')])
    
    for square_file in square_files:
        # Load image
        img = cv2.imread(str(square_file))
        if img is None:
            continue
        
        # Get image properties as hints
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray, 50, 150)
        edge_count = cv2.countNonZero(edges)
        brightness = np.mean(gray)
        file_size = square_file.stat().st_size
        
        hint = f"(size: {file_size//1000}KB, edges: {edge_count}, brightness: {brightness:.0f})"
        
        # Get label from user
        while True:
            label = input(f"{square_file.stem} {hint}: ").strip().lower()
            
            if label == 'exit':
                print("\nQuitting...")
                print(f"\nLabeled: {labeled_count}, Skipped: {skipped_count}")

                exit()
            
            if label == 's':
                skipped_count += 1
                break
            
            if label == 'empty':
                dest = output_dir / 'empty' / f"{position.name}_{square_file.name}"
                shutil.copy2(square_file, dest)
                labeled_count += 1
                print(f"  ✓ empty")
                break
            
            # Check if it's a valid piece type (case-insensitive)
            piece_type = label.lower()
            if piece_type in ['p', 'n', 'b', 'r', 'q', 'k']:
                # Ask for color
                color = input(f"    White (w) or Black (b)? ").strip().lower()
                if color == 'w':
                    final_label = piece_type.upper()
                elif color == 'b':
                    final_label = piece_type.lower()
                else:
                    print(f"    Invalid. Use w or b")
                    continue
                
                dest = output_dir / final_label / f"{position.name}_{square_file.name}"
                shutil.copy2(square_file, dest)
                labeled_count += 1
                print(f"  ✓ {final_label}")
                break
            else:
                print(f"  Invalid. Use: p/n/b/r/q/k, empty, s, or 'exit'")
        
print("\n" + "="*60)
print(f"\nLabeling complete!")
print(f"  Labeled: {labeled_count}")
print(f"  Skipped: {skipped_count}")
print(f"\nOrganized dataset saved to: {output_dir}/")
print("\nClass distribution:")
for cls in CLASSES:
    count = len(list((output_dir / cls).glob('*.jpg')))
    if count > 0:
        print(f"  {cls:6s} ({CLASS_NAMES[cls]:15s}): {count:4d} images")

print("\nNext step: python train_classifier.py")
