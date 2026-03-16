"""!
@file board_to_fen.py
@brief End-to-end ArUco vision pipeline that converts camera frames into board FEN strings.
"""

import time
import cv2
import numpy as np
from picamera2 import Picamera2
from collections import deque
import sys
import os
import json
import pickle
sys.path.append('/home/jfrourke/chess_robot/chessboard2fen')

MODULE_DIR = os.path.dirname(os.path.abspath(__file__))

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

# Occupancy tuning - optimized for background subtraction + edge detection fallback
EDGE_THRESHOLD = 1500  # Higher threshold reduces false positives from texture/shadows
DIFF_THRESHOLD = 30   # Sensitive for piece detection (background subtraction is very reliable)
DIFF_PIXELS = 1200    # Increased to eliminate corner shadow false positives
CENTER_CROP = 0.75    # Good center focus

TYPE_CONFIDENCE_THRESHOLD = 0.45
COLOR_CONFIDENCE_THRESHOLD = 0.80
COLOR_MARGIN_THRESHOLD = 0.25
USE_ML_COLOR = False  # Current color head overfits; keep deterministic color by default.

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
    Determine which squares are occupied using best-available method:
    1. Background subtraction (if empty_board_ref provided) - MOST RELIABLE
    2. Edge detection fallback (less reliable but works without reference)
    
    Background subtraction handles same-colored squares (white-on-white, brown-on-brown)
    much better than edge detection.
    """
    occupied = []
    
    for row in range(8):
        row_occupied = []
        for col in range(8):
            square = squares[row][col]

            # Focus on center to avoid borders/grid lines
            h, w = square.shape[:2]
            ch = int(h * CENTER_CROP / 2)
            cw = int(w * CENTER_CROP / 2)
            cy, cx = h // 2, w // 2
            y0, y1 = cy - ch, cy + ch
            x0, x1 = cx - cw, cx + cw
            square_center = square[y0:y1, x0:x1]
            
            if empty_board_ref is not None:
                # PRIMARY METHOD: Background subtraction (recommended)
                # Works reliably for all piece colors including same-colored squares
                ref_square = empty_board_ref[row][col]
                ref_center = ref_square[y0:y1, x0:x1]
                
                # Absolute difference
                diff = cv2.absdiff(square_center, ref_center)
                gray_diff = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY)
                
                # Bilateral filtering - reduces noise better than Gaussian while preserving edges
                gray_diff = cv2.bilateralFilter(gray_diff, 5, 20, 20)
                
                # Threshold to binary - any color change indicates a piece
                _, thresh = cv2.threshold(gray_diff, DIFF_THRESHOLD, 255, cv2.THRESH_BINARY)
                
                # Morphological cleanup - remove small noise (multiple iterations for aggressive cleaning)
                kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
                thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=2)
                
                # Count significant pixels
                occupied_pixels = cv2.countNonZero(thresh)
                is_occupied = occupied_pixels > DIFF_PIXELS
            else:
                # FALLBACK METHOD: Edge detection (no reference available)
                # Less reliable but better than nothing
                gray = cv2.cvtColor(square_center, cv2.COLOR_BGR2GRAY)
                
                # Morphological opening to reduce noise
                kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
                gray_opened = cv2.morphologyEx(gray, cv2.MORPH_OPEN, kernel)
                
                # Canny edge detection with stricter thresholds
                edges = cv2.Canny(gray_opened, 100, 200)
                
                # Dilate edges to connect nearby edge fragments
                kernel_dilate = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2, 2))
                edges_dilated = cv2.dilate(edges, kernel_dilate, iterations=1)
                
                edge_pixels = cv2.countNonZero(edges_dilated)
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
                    piece = _brightness_color_fallback(square)
                
                row_pieces.append(piece)
        
        board.append(row_pieces)
    
    return board


def _prepare_square_for_model(square_img, metadata):
    """Apply the same center crop and preprocessing used in training."""
    crop_fraction = float(metadata.get('center_crop', 1.0))
    img_size = int(metadata.get('img_size', 64))

    h, w = square_img.shape[:2]
    crop_h = max(1, int(h * crop_fraction))
    crop_w = max(1, int(w * crop_fraction))
    y0 = max(0, (h - crop_h) // 2)
    x0 = max(0, (w - crop_w) // 2)
    cropped = square_img[y0:y0 + crop_h, x0:x0 + crop_w]

    resized = cv2.resize(cropped, (img_size, img_size), interpolation=cv2.INTER_AREA)

    preprocessing = metadata.get('preprocessing')
    if preprocessing == 'mobilenet_v2':
        rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB).astype(np.float32)
        processed = tf.keras.applications.mobilenet_v2.preprocess_input(rgb)
    else:
        processed = resized.astype(np.float32) / 255.0

    return np.expand_dims(processed, axis=0)


def _brightness_color_fallback(square_img):
    """Fallback white/black guess based on square-center brightness."""
    gray = cv2.cvtColor(square_img, cv2.COLOR_BGR2GRAY)
    h, w = gray.shape[:2]
    crop_h = max(1, int(h * 0.6))
    crop_w = max(1, int(w * 0.6))
    y0 = max(0, (h - crop_h) // 2)
    x0 = max(0, (w - crop_w) // 2)
    center = gray[y0:y0 + crop_h, x0:x0 + crop_w]

    p90 = np.percentile(center, 90)
    p10 = np.percentile(center, 10)
    mean_val = float(np.mean(center))

    if p90 >= 155:
        return 'P'
    if p10 <= 70:
        return 'p'
    return 'P' if mean_val >= 120 else 'p'


def _extract_multitask_predictions(prediction):
    if isinstance(prediction, dict):
        return prediction['piece_type'], prediction['piece_color']
    if isinstance(prediction, (list, tuple)) and len(prediction) == 2:
        return prediction[0], prediction[1]
    raise ValueError('Unexpected multitask prediction format')


def classify_piece_with_model(square_img, classifier_data):
    """
    Classify a single square using the trained model
    """
    model, metadata = classifier_data

    if metadata.get('model_kind') == 'piece_type_color_multitask_v1':
        square_batch = _prepare_square_for_model(square_img, metadata)
        prediction = model.predict(square_batch, verbose=0)
        type_probs, color_probs = _extract_multitask_predictions(prediction)

        type_probs = type_probs[0]
        color_probs = color_probs[0]
        type_idx = int(np.argmax(type_probs))
        color_idx = int(np.argmax(color_probs))
        type_conf = float(type_probs[type_idx])
        color_conf = float(color_probs[color_idx])
        sorted_color = np.sort(color_probs)
        color_margin = float(sorted_color[-1] - sorted_color[-2]) if len(sorted_color) > 1 else 1.0

        piece_type = metadata['piece_types'][type_idx]
        if type_conf < TYPE_CONFIDENCE_THRESHOLD:
            # Keep top-1 type prediction but demand higher confidence for color.
            piece_type = metadata['piece_types'][type_idx]

        heuristic_is_white = _brightness_color_fallback(square_img).isupper()

        if (
            USE_ML_COLOR
            and color_conf >= COLOR_CONFIDENCE_THRESHOLD
            and color_margin >= COLOR_MARGIN_THRESHOLD
        ):
            color_name = metadata['color_classes'][color_idx]
            is_white = color_name == 'white'
        else:
            is_white = heuristic_is_white

        return piece_type if is_white else piece_type.lower()

    square_batch = _prepare_square_for_model(square_img, metadata)
    prediction = model.predict(square_batch, verbose=0)
    piece_idx = int(np.argmax(prediction))
    piece = metadata[str(piece_idx)]
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


def load_trained_classifier():
    """Load classifier + metadata for external callers (returns None if unavailable)."""
    if not HAS_TENSORFLOW:
        print("TensorFlow not available - cannot use classifier")
        return None

    model_path = None
    keras_path = os.path.join(MODULE_DIR, 'piece_classifier.keras')
    h5_path = os.path.join(MODULE_DIR, 'piece_classifier.h5')
    metadata_path = os.path.join(MODULE_DIR, 'piece_classifier_classes.json')

    if os.path.exists(keras_path):
        model_path = keras_path
    elif os.path.exists(h5_path):
        model_path = h5_path
    else:
        print("No trained classifier found (piece_classifier.keras or piece_classifier.h5)")
        return None

    if not os.path.exists(metadata_path):
        print("Classifier metadata missing: piece_classifier_classes.json")
        return None

    try:
        model = tf.keras.models.load_model(model_path)
        with open(metadata_path, 'r', encoding='utf-8') as f:
            metadata = json.load(f)
        print(f"Classifier loaded from {os.path.basename(model_path)}")
        return (model, metadata)
    except Exception as e:
        print(f"Warning: Could not load classifier: {e}")
        return None


def load_empty_board_reference(path=None):
    """Load empty-board reference if available, else return None."""
    ref_path = path or os.path.join(MODULE_DIR, 'empty_board_ref.pkl')
    if not os.path.exists(ref_path):
        return None

    try:
        with open(ref_path, 'rb') as f:
            return pickle.load(f)
    except Exception as e:
        print(f"Warning: Could not load empty board reference: {e}")
        return None


def detect_fen_once(empty_board_ref=None, classifier=None):
    """One-shot callable API for external modules."""
    return process_frame(empty_board_ref=empty_board_ref, classifier=classifier)


# ===== Capture Empty Board Reference =====
def capture_empty_board_reference():
    """Capture and store empty board for background subtraction"""
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
    ref_path = os.path.join(MODULE_DIR, 'empty_board_ref.pkl')
    with open(ref_path, 'wb') as f:
        pickle.dump(empty_squares, f)
    
    print("Empty board reference captured and saved!")
    return empty_squares


# ===== Main Loop =====
def main():
    import os
    
    print("=== ArUco Chess Board to FEN Pipeline ===")
    
    # Try to load trained classifier
    print("\nLoading trained piece classifier...")
    classifier = load_trained_classifier()
    if classifier is None:
        print("Will use brightness fallback classification")
    
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
    elif os.path.exists(os.path.join(MODULE_DIR, 'empty_board_ref.pkl')):
        print("Loading saved empty board reference...")
        empty_board_ref = load_empty_board_reference()
        print("Loaded empty board reference from disk")
    
    # Initialize state tracker
    tracker = BoardStateTracker(stability_frames=3)
    
    print("\n=== Starting FEN Detection ===")
    print("Press Ctrl+C to stop")
    
    frame_count = 0
    try:
        while True:
            fen, warped, debug = process_frame(empty_board_ref, classifier)
            
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
