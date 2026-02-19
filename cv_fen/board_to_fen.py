"""
Full ArUco-based chessboard to FEN pipeline for Raspberry Pi Chess Robot
Pipeline: Capture → ArUco detect → Warp → Normalize → Split 8x8 → Occupancy → Classify → FEN
"""

import time
import cv2
import numpy as np
from picamera2 import Picamera2
from collections import deque
import sys
import os
import json
sys.path.append('/home/jfrourke/chess_robot/chessboard2fen')

# Try to load TensorFlow
try:
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # Suppress TF warnings
    import tensorflow as tf
    tf.get_logger().setLevel('ERROR')
    HAS_TENSORFLOW = True
except ImportError:
    print("Warning: TensorFlow not available. Classifier will not work.")
    HAS_TENSORFLOW = False

# ---- Settings ----
BOARD_PIX = 800  # output warp size (800x800)
DICT = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)

# Occupancy tuning - balanced for pieces vs empty
EDGE_THRESHOLD = 300  # Balanced threshold
DIFF_THRESHOLD = 40   # Moderate sensitivity
DIFF_PIXELS = 1200    # Reasonable pixel count
CENTER_CROP = 0.75    # Good center focus

# ArUco marker IDs for corners (TOP-LEFT, TOP-RIGHT, BOTTOM-RIGHT, BOTTOM-LEFT)
ID_TL, ID_TR, ID_BR, ID_BL = 0, 1, 2, 3

# Destination points in the warped image
DST = np.array([
    [0, 0],                              # TL
    [BOARD_PIX - 1, 0],                  # TR
    [BOARD_PIX - 1, BOARD_PIX - 1],      # BR
    [0, BOARD_PIX - 1],                  # BL
], dtype=np.float32)

# Initialize Pi Camera
picam2 = Picamera2()
full_w, full_h = picam2.camera_properties["PixelArraySize"]
config = picam2.create_preview_configuration(
    main={"size": (1640, 1232), "format": "RGB888"},
    controls={"ScalerCrop": (0, 0, full_w, full_h)}
)
picam2.configure(config)
picam2.start()

# Give camera time to stabilize
time.sleep(2)

# Lock exposure and white balance for stability (optional but recommended)
# picam2.set_controls({"ExposureTime": 10000, "AnalogueGain": 1.0})
# picam2.set_controls({"AwbEnable": 0})  # Lock white balance after it settles


class BoardStateTracker:
    """Temporal smoothing and sanity checking for board states"""
    
    def __init__(self, stability_frames=3):
        self.stability_frames = stability_frames
        self.recent_states = deque(maxlen=stability_frames)
        self.current_stable_state = None
        
    def update(self, new_state):
        """
        Add a new board state and check if it's stable
        Returns: (stable_state, is_new) or (None, False) if not stable yet
        """
        self.recent_states.append(new_state)
        
        if len(self.recent_states) < self.stability_frames:
            return self.current_stable_state, False
            
        # Check if all recent states are identical
        if all(s == new_state for s in self.recent_states):
            if new_state != self.current_stable_state:
                # New stable state
                if self.is_valid_position(new_state):
                    self.current_stable_state = new_state
                    return new_state, True
        
        return self.current_stable_state, False
    
    def is_valid_position(self, board_state):
        """Basic sanity checks for chess position"""
        if board_state is None:
            return False
            
        # Count kings
        white_kings = sum(row.count('K') for row in board_state)
        black_kings = sum(row.count('k') for row in board_state)
        
        if white_kings != 1 or black_kings != 1:
            return False
        
        # Count pawns (max 8 per side)
        white_pawns = sum(row.count('P') for row in board_state)
        black_pawns = sum(row.count('p') for row in board_state)
        
        if white_pawns > 8 or black_pawns > 8:
            return False
            
        return True


# ===== Step 1: Capture Frame =====
def capture_frame():
    """Capture a frame from the Pi camera"""
    frame = picam2.capture_array()
    # Convert from RGB to BGR for OpenCV
    img_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
    return img_bgr


# ===== Step 2: Detect ArUco Markers =====
def detect_markers(img_bgr):
    """Detect ArUco markers and return dictionary of marker_id: corners"""
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
        return {}, corners, ids
    
    ids = ids.flatten()
    found = {}
    for c, mid in zip(corners, ids):
        # c shape: (1,4,2) - corners are [TL, TR, BR, BL] of the marker
        pts = c.reshape(4, 2).astype(np.float32)
        found[int(mid)] = pts
    
    return found, corners, ids


# ===== Step 3: Pick Board Corners from Markers =====
def marker_corner_for_board(marker_pts, which):
    """
    Extract the specific corner of a marker that corresponds to the board corner.
    
    ArUco markers have corners in order: [top-left, top-right, bottom-right, bottom-left]
    If markers are placed OUTSIDE the board, the corner nearest the board is:
      TL marker → its bottom-right corner (index 2)
      TR marker → its bottom-left corner  (index 3)
      BR marker → its top-left corner     (index 0)
      BL marker → its top-right corner    (index 1)
    """
    return marker_pts[which]


# ===== Step 4: Homography / Perspective Warp =====
def warp_board(img_bgr, found):
    """Apply perspective transform to get top-down board view"""
    required = [ID_TL, ID_TR, ID_BR, ID_BL]
    if not all(mid in found for mid in required):
        return None
    
    # Extract the board corners from marker corners
    src = np.array([
        marker_corner_for_board(found[ID_TL], 2),  # TL marker: bottom-right
        marker_corner_for_board(found[ID_TR], 3),  # TR marker: bottom-left
        marker_corner_for_board(found[ID_BR], 0),  # BR marker: top-left
        marker_corner_for_board(found[ID_BL], 1),  # BL marker: top-right
    ], dtype=np.float32)
    
    H = cv2.getPerspectiveTransform(src, DST)
    warped = cv2.warpPerspective(img_bgr, H, (BOARD_PIX, BOARD_PIX))
    return warped


# ===== Step 5: Color Normalization =====
def normalize_board(board_warp):
    """Normalize lighting to reduce variation"""
    # Convert to LAB color space
    lab = cv2.cvtColor(board_warp, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    
    # Apply CLAHE to L channel for brightness normalization
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    l_norm = clahe.apply(l)
    
    # Merge back
    lab_norm = cv2.merge([l_norm, a, b])
    board_norm = cv2.cvtColor(lab_norm, cv2.COLOR_LAB2BGR)
    
    # Mild blur to reduce noise
    board_norm = cv2.GaussianBlur(board_norm, (3, 3), 0)
    
    return board_norm


# ===== Step 6: Split into 8x8 Squares =====
def extract_squares(board_warp):
    """Extract 64 individual square images from the warped board"""
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
    
    return squares  # 8x8 list of images


# ===== Step 7: Occupancy Detection =====
def detect_occupancy(squares, empty_board_ref=None, edge_threshold=EDGE_THRESHOLD):
    """
    Determine which squares are occupied using edge detection
    
    If empty_board_ref is provided, use background subtraction
    Otherwise, use edge/texture heuristic
    """
    occupied = []
    
    for row in range(8):
        row_occupied = []
        for col in range(8):
            square = squares[row][col]

            # Focus on the center of each square to avoid borders/grid lines
            h, w = square.shape[:2]
            ch = int(h * CENTER_CROP / 2)
            cw = int(w * CENTER_CROP / 2)
            cy, cx = h // 2, w // 2
            y0, y1 = cy - ch, cy + ch
            x0, x1 = cx - cw, cx + cw
            square_center = square[y0:y1, x0:x1]
            
            if empty_board_ref is not None:
                # Background subtraction method
                ref_square = empty_board_ref[row][col]
                ref_center = ref_square[y0:y1, x0:x1]
                diff = cv2.absdiff(square_center, ref_center)
                gray_diff = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY)
                _, thresh = cv2.threshold(gray_diff, DIFF_THRESHOLD, 255, cv2.THRESH_BINARY)
                occupied_pixels = cv2.countNonZero(thresh)
                is_occupied = occupied_pixels > DIFF_PIXELS
            else:
                # Edge density method
                gray = cv2.cvtColor(square_center, cv2.COLOR_BGR2GRAY)
                edges = cv2.Canny(gray, 50, 150)
                edge_pixels = cv2.countNonZero(edges)
                is_occupied = edge_pixels > edge_threshold
            
            row_occupied.append(is_occupied)
        
        occupied.append(row_occupied)
    
    return occupied  # 8x8 boolean array


# ===== Step 8: Piece Classification =====
def classify_pieces(squares, occupied, classifier=None):
    """
    Classify pieces on occupied squares
    
    Returns 8x8 grid with piece labels:
    - None for empty squares
    - 'P','N','B','R','Q','K' for white pieces
    - 'p','n','b','r','q','k' for black pieces
    """
    board = []
    
    for row in range(8):
        row_pieces = []
        for col in range(8):
            if not occupied[row][col]:
                row_pieces.append(None)
            else:
                square = squares[row][col]
                
                if classifier is not None:
                    # Use real classifier
                    piece = classify_piece_with_model(square, classifier)
                else:
                    # Placeholder: random piece for testing
                    piece = 'P'  # Replace with actual classification
                
                row_pieces.append(piece)
        
        board.append(row_pieces)
    
    return board


def classify_piece_with_model(square_img, classifier_data):
    """
    Classify a single square using the trained model
    """
    model, class_mapping = classifier_data
    
    # Resize to model input size
    square_resized = cv2.resize(square_img, (64, 64))
    
    # Normalize to [0, 1]
    square_normalized = square_resized.astype(np.float32) / 255.0
    
    # Add batch dimension
    square_batch = np.expand_dims(square_normalized, axis=0)
    
    # Predict
    prediction = model.predict(square_batch, verbose=0)
    piece_idx = np.argmax(prediction)
    
    # Map to FEN character
    piece = class_mapping[str(piece_idx)]
    
    # Return None for empty squares
    if piece == 'empty':
        return None
    
    return piece


# ===== Step 9: Convert Grid to FEN =====
def board_to_fen(board):
    """
    Convert 8x8 board representation to FEN string
    
    board[row][col] where row=0 is rank 8 (top of board from white's perspective)
    """
    fen_parts = []
    
    for row in range(8):
        empty_count = 0
        row_fen = ""
        
        for col in range(8):
            piece = board[row][col]
            
            if piece is None:
                empty_count += 1
            else:
                if empty_count > 0:
                    row_fen += str(empty_count)
                    empty_count = 0
                row_fen += piece
        
        # Add remaining empty squares
        if empty_count > 0:
            row_fen += str(empty_count)
        
        fen_parts.append(row_fen)
    
    fen = '/'.join(fen_parts)
    return fen


# ===== Step 10: Main Pipeline =====
def process_frame(empty_board_ref=None, classifier=None):
    """
    Process one frame through the full pipeline
    Returns: (fen_string, warped_board_image, debug_info)
    """
    # Step 1: Capture
    img = capture_frame()
    if img is None:
        return None, None, "Failed to capture frame"
    
    # Step 2: Detect markers
    found, corners, ids = detect_markers(img)
    if len(found) < 4:
        return None, None, f"Only found {len(found)}/4 markers"
    
    # Step 3 & 4: Warp board
    warped = warp_board(img, found)
    if warped is None:
        return None, None, "Failed to warp board"
    
    # Step 5: Normalize
    warped_norm = normalize_board(warped)
    
    # Step 6: Extract squares
    squares = extract_squares(warped_norm)
    
    # Step 7: Detect occupancy
    occupied = detect_occupancy(squares, empty_board_ref)
    
    # Step 8: Classify pieces
    board = classify_pieces(squares, occupied, classifier)
    
    # Step 9: Generate FEN
    fen = board_to_fen(board)
    
    debug_info = {
        'markers_found': ids.flatten().tolist() if ids is not None else [],
        'occupied_count': sum(sum(row) for row in occupied)
    }
    
    return fen, warped, debug_info


# ===== Capture Empty Board Reference =====
def capture_empty_board_reference():
    """Capture and store empty board for background subtraction"""
    import pickle
    
    print("Capturing empty board reference...")
    print("Make sure the board is EMPTY and press Enter")
    input()
    
    img = capture_frame()
    found, _, _ = detect_markers(img)
    warped = warp_board(img, found)
    
    if warped is None:
        print("Failed to capture empty board!")
        return None
    
    warped_norm = normalize_board(warped)
    empty_squares = extract_squares(warped_norm)
    
    # Save to disk
    with open('empty_board_ref.pkl', 'wb') as f:
        pickle.dump(empty_squares, f)
    
    print("Empty board reference captured and saved!")
    return empty_squares


# ===== Main Loop =====
def main():
    import pickle
    import os
    
    print("=== ArUco Chess Board to FEN Pipeline ===")
    
    # Try to load trained classifier
    classifier = None
    if HAS_TENSORFLOW and os.path.exists('piece_classifier.h5'):
        print("\nLoading trained piece classifier...")
        try:
            model = tf.keras.models.load_model('piece_classifier.h5')
            with open('piece_classifier_classes.json', 'r') as f:
                class_mapping = json.load(f)
            classifier = (model, class_mapping)
            print("✓ Classifier loaded successfully")
        except Exception as e:
            print(f"Warning: Could not load classifier: {e}")
            print("Will use placeholder classification")
    else:
        if not HAS_TENSORFLOW:
            print("\nTensorFlow not available - cannot use classifier")
        else:
            print("\nNo trained classifier found (piece_classifier.h5)")
            print("Run train_classifier.py to train a model first")
        print("Will use placeholder classification (all pieces = 'P')")
    
    print("\nOptions:")
    print("1. Capture empty board reference (recommended)")
    print("2. Run without empty board reference (edge detection only)")
    
    choice = input("\nEnter choice (1 or 2): ").strip()
    
    empty_board_ref = None
    if choice == '1':
        empty_board_ref = capture_empty_board_reference()
        if empty_board_ref is None:
            print("Failed to capture reference. Exiting.")
            return
    elif os.path.exists('empty_board_ref.pkl'):
        print("Loading saved empty board reference...")
        with open('empty_board_ref.pkl', 'rb') as f:
            empty_board_ref = pickle.load(f)
        print("Loaded empty board reference from disk")
    
    # Initialize state tracker
    tracker = BoardStateTracker(stability_frames=3)
    
    print("\n=== Starting FEN Detection ===")
    print("Press Ctrl+C to stop")
    
    frame_count = 0
    try:
        while True:
            fen, warped, debug = process_frame(empty_board_ref)
            
            if fen is not None:
                # Update tracker
                # For now, we'll just use the FEN string as the state
                # In production, you'd use the board array
                stable_fen, is_new = tracker.update(fen)
                
                if is_new:
                    print(f"\n{'='*60}")
                    print(f"NEW STABLE POSITION (frame {frame_count}):")
                    print(f"FEN: {stable_fen}")
                    print(f"Markers: {debug['markers_found']}")
                    print(f"Occupied squares: {debug['occupied_count']}")
                    print(f"{'='*60}\n")
                else:
                    print(f"Frame {frame_count}: {fen} (unstable)", end='\r')
            else:
                print(f"Frame {frame_count}: {debug}", end='\r')
            
            frame_count += 1
            time.sleep(0.1)  # 10 FPS
            
    except KeyboardInterrupt:
        print("\n\nStopping...")
    finally:
        picam2.stop()


if __name__ == "__main__":
    main()
