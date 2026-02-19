"""
Visualize the board square extraction to verify correct alignment
Saves annotated images showing the 8x8 grid overlay
"""

import cv2
import numpy as np
from picamera2 import Picamera2
import time

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
        found[ID_TL][2],  # TL marker: bottom-right
        found[ID_TR][3],  # TR marker: bottom-left
        found[ID_BR][0],  # BR marker: top-left
        found[ID_BL][1],  # BL marker: top-right
    ], dtype=np.float32)
    
    H = cv2.getPerspectiveTransform(src, DST)
    warped = cv2.warpPerspective(img_bgr, H, (BOARD_PIX, BOARD_PIX))
    return warped


def draw_grid_on_board(warped):
    """Draw 8x8 grid lines and labels on the warped board"""
    annotated = warped.copy()
    cell_size = BOARD_PIX // 8
    
    # Draw grid lines
    for i in range(9):
        pos = i * cell_size
        # Vertical lines
        cv2.line(annotated, (pos, 0), (pos, BOARD_PIX), (0, 255, 0), 2)
        # Horizontal lines
        cv2.line(annotated, (0, pos), (BOARD_PIX, pos), (0, 255, 0), 2)
    
    # Add square labels (a1-h8)
    files = 'abcdefgh'
    ranks = '87654321'  # Top to bottom in image
    
    for row in range(8):
        for col in range(8):
            x = col * cell_size + cell_size // 2 - 15
            y = row * cell_size + cell_size // 2 + 10
            label = files[col] + ranks[row]
            cv2.putText(annotated, label, (x, y), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    
    return annotated


def main():
    print("Capturing frame for grid visualization...")
    
    img = capture_frame()
    found = detect_markers(img)
    
    if len(found) < 4:
        print(f"Error: Only found {len(found)}/4 markers. Need all 4 corners.")
        print(f"Found marker IDs: {list(found.keys())}")
        picam2.stop()
        return
    
    warped = warp_board(img, found)
    if warped is None:
        print("Failed to warp board!")
        picam2.stop()
        return
    
    # Create visualization
    annotated = draw_grid_on_board(warped)
    
    # Save outputs
    cv2.imwrite('board_warped.jpg', warped)
    cv2.imwrite('board_grid_overlay.jpg', annotated)
    
    print("\n✓ Saved images:")
    print("  - board_warped.jpg (clean warped board)")
    print("  - board_grid_overlay.jpg (with 8x8 grid overlay)")
    print("\nCheck these images to verify:")
    print("  1. Board is properly aligned (not rotated)")
    print("  2. Squares line up correctly with actual board")
    print("  3. Labels match your board (a1 should be bottom-left from white's view)")
    
    picam2.stop()


if __name__ == "__main__":
    main()
