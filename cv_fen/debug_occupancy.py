"""
Debug occupancy detection - visualize which squares are detected as occupied
"""

import cv2
import numpy as np
from picamera2 import Picamera2
import time

BOARD_PIX = 800
DICT = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
ID_TL, ID_TR, ID_BR, ID_BL = 0, 1, 2, 3

# Same settings as board_to_fen
EDGE_THRESHOLD = 180
DIFF_THRESHOLD = 20
DIFF_PIXELS = 500
CENTER_CROP = 0.75

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
    lab = cv2.cvtColor(board_warp, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    l_norm = clahe.apply(l)
    lab_norm = cv2.merge([l_norm, a, b])
    board_norm = cv2.cvtColor(lab_norm, cv2.COLOR_LAB2BGR)
    return cv2.GaussianBlur(board_norm, (3, 3), 0)

def extract_squares(board_warp):
    cell_size = BOARD_PIX // 8
    squares = []
    for row in range(8):
        row_squares = []
        for col in range(8):
            y0 = row * cell_size
            y1 = (row + 1) * cell_size
            x0 = col * cell_size
            x1 = (col + 1) * cell_size
            square_img = board_warp[y0:y1, x0:x1]
            row_squares.append(square_img)
        squares.append(row_squares)
    return squares

def detect_occupancy(squares, empty_ref=None):
    occupied = []
    for row in range(8):
        row_occupied = []
        for col in range(8):
            square = squares[row][col]
            
            # Focus on center
            h, w = square.shape[:2]
            ch = int(h * CENTER_CROP / 2)
            cw = int(w * CENTER_CROP / 2)
            cy, cx = h // 2, w // 2
            y0, y1 = cy - ch, cy + ch
            x0, x1 = cx - cw, cx + cw
            square_center = square[y0:y1, x0:x1]
            
            if empty_ref is not None:
                # Use background subtraction if we have empty board reference
                ref_square = empty_ref[row][col]
                ref_center = ref_square[y0:y1, x0:x1]
                diff = cv2.absdiff(square_center, ref_center)
                gray_diff = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY)
                _, thresh = cv2.threshold(gray_diff, DIFF_THRESHOLD, 255, cv2.THRESH_BINARY)
                occupied_pixels = cv2.countNonZero(thresh)
                is_occupied = occupied_pixels > DIFF_PIXELS
            else:
                # Fall back to edge density method
                gray = cv2.cvtColor(square_center, cv2.COLOR_BGR2GRAY)
                edges = cv2.Canny(gray, 50, 150)
                edge_pixels = cv2.countNonZero(edges)
                is_occupied = edge_pixels > EDGE_THRESHOLD
            
            row_occupied.append(is_occupied)
        occupied.append(row_occupied)
    return occupied

def draw_occupancy_overlay(warped, occupied):
    """Draw colored overlays on occupied squares"""
    overlay = warped.copy()
    cell_size = BOARD_PIX // 8
    
    files = 'abcdefgh'
    ranks = '87654321'
    
    for row in range(8):
        for col in range(8):
            y0 = row * cell_size
            y1 = (row + 1) * cell_size
            x0 = col * cell_size
            x1 = (col + 1) * cell_size
            
            if occupied[row][col]:
                # Green overlay for occupied
                cv2.rectangle(overlay, (x0, y0), (x1, y1), (0, 255, 0), -1)
                label = "OCC"
                color = (0, 255, 0)
            else:
                # Red overlay for empty
                cv2.rectangle(overlay, (x0, y0), (x1, y1), (0, 0, 255), -1)
                label = "EMPTY"
                color = (0, 0, 255)
            
            # Draw square label
            text_x = x0 + 5
            text_y = y0 + 20
            cv2.putText(overlay, files[col] + ranks[row], (text_x, text_y),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
    
    # Blend with original
    result = cv2.addWeighted(warped, 0.6, overlay, 0.4, 0)
    
    # Draw grid
    for i in range(9):
        pos = i * cell_size
        cv2.line(result, (pos, 0), (pos, BOARD_PIX), (255, 255, 255), 2)
        cv2.line(result, (0, pos), (BOARD_PIX, pos), (255, 255, 255), 2)
    
    return result

print("Capturing board state...")
img = capture_frame()
found = detect_markers(img)

if len(found) < 4:
    print(f"Error: Only found {len(found)}/4 markers")
    picam2.stop()
    exit()

warped = warp_board(img, found)
if warped is None:
    print("Failed to warp")
    picam2.stop()
    exit()

warped_norm = normalize_board(warped)
squares = extract_squares(warped_norm)

# Try to load empty board reference if it exists
import os
import pickle
empty_ref = None
if os.path.exists('empty_board_ref.pkl'):
    print("Loading empty board reference...")
    with open('empty_board_ref.pkl', 'rb') as f:
        empty_ref = pickle.load(f)
    print("Using background subtraction method")
else:
    print("No empty board reference found - using edge detection only")
    print("Run board_to_fen.py with option 1 to capture empty board")

occupied = detect_occupancy(squares, empty_ref)

# Count occupied squares
occupied_count = sum(sum(row) for row in occupied)
print(f"\nDetected {occupied_count}/64 squares as occupied")

# Show which squares
print("\nOccupied squares:")
files = 'abcdefgh'
ranks = '87654321'
for row in range(8):
    for col in range(8):
        if occupied[row][col]:
            print(f"  {files[col]}{ranks[row]}")

# Create visualization
result = draw_occupancy_overlay(warped, occupied)

cv2.imwrite('occupancy_debug.jpg', result)
print("\n✓ Saved: occupancy_debug.jpg")
print("\nGreen overlay = detected as OCCUPIED")
print("Red overlay = detected as EMPTY")
print("\nCheck the image to see if detection matches reality!")

picam2.stop()
