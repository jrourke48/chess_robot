"""
Extract and save individual square images for training a piece classifier
Creates a dataset folder with labeled squares
"""

import cv2
import numpy as np
from picamera2 import Picamera2
import time
import os

# ---- Settings ----
BOARD_PIX = 800
DICT = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
ID_TL, ID_TR, ID_BR, ID_BL = 0, 1, 2, 3

DST = np.array([
    [0, 0],
    [BOARD_PIX - 1, 0],
    [BOARD_PIX - 1, BOARD_PIX - 1],
    [0, BOARD_PIX - 1],
], dtype=np.float32)

# Initialize camera
picam2 = Picamera2()
full_w, full_h = picam2.camera_properties["PixelArraySize"]
config = picam2.create_preview_configuration(
    main={"size": (1640, 1232), "format": "RGB888"},
    controls={"ScalerCrop": (0, 0, full_w, full_h)}
)
picam2.configure(config)
picam2.start()
time.sleep(2)


def capture_frame():
    frame = picam2.capture_array()
    return cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)


def detect_markers(img_bgr):
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    params = cv2.aruco.DetectorParameters()
    params.adaptiveThreshWinSizeMin = 3
    params.adaptiveThreshWinSizeMax = 23
    params.adaptiveThreshWinSizeStep = 10
    params.minMarkerPerimeterRate = 0.03
    params.maxMarkerPerimeterRate = 4.0
    params.polygonalApproxAccuracyRate = 0.05
    params.minCornerDistanceRate = 0.05
    params.minDistanceToBorder = 3
    params.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_SUBPIX
    detector = cv2.aruco.ArucoDetector(DICT, params)
    corners, ids, _ = detector.detectMarkers(gray)
    
    if ids is None:
        return {}
    
    found = {}
    for c, mid in zip(corners, ids.flatten()):
        pts = c.reshape(4, 2).astype(np.float32)
        found[int(mid)] = pts
    
    return found


def warp_board(img_bgr, found):
    required = [ID_TL, ID_TR, ID_BR, ID_BL]
    if not all(mid in found for mid in required):
        return None
    
    src = np.array([
        found[ID_TL][2],
        found[ID_TR][3],
        found[ID_BR][0],
        found[ID_BL][1],
    ], dtype=np.float32)
    
    H = cv2.getPerspectiveTransform(src, DST)
    warped = cv2.warpPerspective(img_bgr, H, (BOARD_PIX, BOARD_PIX))
    return warped


def normalize_board(board_warp):
    """Normalize lighting"""
    lab = cv2.cvtColor(board_warp, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    l_norm = clahe.apply(l)
    lab_norm = cv2.merge([l_norm, a, b])
    board_norm = cv2.cvtColor(lab_norm, cv2.COLOR_LAB2BGR)
    return cv2.GaussianBlur(board_norm, (3, 3), 0)


def extract_squares(board_warp):
    """Extract 64 individual squares"""
    cell_size = BOARD_PIX // 8
    squares = {}
    
    files = 'abcdefgh'
    ranks = '87654321'
    
    for row in range(8):
        for col in range(8):
            y0 = row * cell_size
            y1 = (row + 1) * cell_size
            x0 = col * cell_size
            x1 = (col + 1) * cell_size
            
            square_img = board_warp[y0:y1, x0:x1]
            square_name = files[col] + ranks[row]
            squares[square_name] = square_img
    
    return squares


def main():
    print("=== Chess Square Dataset Extractor ===\n")
    
    # Create output directory
    output_dir = "square_dataset"
    os.makedirs(output_dir, exist_ok=True)
    
    print("This tool will capture squares from your current board position.")
    print("Use this to build a training dataset for piece classification.\n")
    
    # Get position name from user
    position_name = input("Enter a name for this position (e.g., 'starting_position'): ").strip()
    if not position_name:
        position_name = f"capture_{int(time.time())}"
    
    position_dir = os.path.join(output_dir, position_name)
    os.makedirs(position_dir, exist_ok=True)
    
    print("\nCapturing...")
    
    img = capture_frame()
    found = detect_markers(img)
    
    if len(found) < 4:
        print(f"Error: Only found {len(found)}/4 markers!")
        picam2.stop()
        return
    
    warped = warp_board(img, found)
    if warped is None:
        print("Failed to warp board!")
        picam2.stop()
        return
    
    # Normalize and extract
    warped_norm = normalize_board(warped)
    squares = extract_squares(warped_norm)
    
    # Save all squares
    for square_name, square_img in squares.items():
        filename = os.path.join(position_dir, f"{square_name}.jpg")
        cv2.imwrite(filename, square_img)
    
    # Also save the full warped board
    cv2.imwrite(os.path.join(position_dir, "_full_board.jpg"), warped_norm)
    
    print(f"\n✓ Saved 64 squares to: {position_dir}/")
    print(f"✓ Full board saved as: _full_board.jpg")
    print("\nNext steps:")
    print("  1. Organize squares into folders by piece type (P, N, B, R, Q, K, p, n, b, r, q, k, empty)")
    print("  2. Capture multiple positions to build a diverse dataset")
    print("  3. Train a classifier on the organized dataset")
    
    picam2.stop()


if __name__ == "__main__":
    main()
