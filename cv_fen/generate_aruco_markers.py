import cv2
import numpy as np

# Use the same dictionary as in aruco_warp.py
DICT = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)

# Marker IDs for the four corners
MARKER_IDS = [0, 1, 2, 3]
MARKER_LABELS = ["Top-Left", "Top-Right", "Bottom-Right", "Bottom-Left"]

# Marker size in pixels (will be printed)
MARKER_SIZE = 200

print("Generating ArUco markers for chess board corners...")
print(f"Dictionary: DICT_4X4_50")
print(f"Marker size: {MARKER_SIZE}x{MARKER_SIZE} pixels")
print()

for marker_id, label in zip(MARKER_IDS, MARKER_LABELS):
    # Generate the marker
    marker_img = cv2.aruco.generateImageMarker(DICT, marker_id, MARKER_SIZE)
    
    # Add white border for better detection
    border = 20
    marker_with_border = np.ones((MARKER_SIZE + 2*border, MARKER_SIZE + 2*border), dtype=np.uint8) * 255
    marker_with_border[border:border+MARKER_SIZE, border:border+MARKER_SIZE] = marker_img
    
    # Save the marker
    filename = f"aruco_marker_{marker_id}_{label.replace('-', '_').lower()}.png"
    cv2.imwrite(filename, marker_with_border)
    print(f"✓ Created: {filename} (ID={marker_id}, {label})")

print()
print("=" * 60)
print("INSTRUCTIONS:")
print("=" * 60)
print("1. Print all 4 marker images")
print("2. Cut them out and mount on cardboard/foam board")
print("3. Place them OUTSIDE the corners of your chessboard:")
print("   - Marker 0 (ID=0): Top-Left corner")
print("   - Marker 1 (ID=1): Top-Right corner")
print("   - Marker 2 (ID=2): Bottom-Right corner")
print("   - Marker 3 (ID=3): Bottom-Left corner")
print()
print("4. Make sure markers are flat and clearly visible to camera")
print("5. The white border helps with detection - don't cut it off!")
print()
print("TIP: For best results, print them at actual size (not 'fit to page')")
print("     Each marker should be about 2-3 inches square.")
