# Computer Vision System - Board Detection & Piece Classification

The CV subsystem transforms raw camera frames into chess board FEN strings. It combines ArUco marker detection for board localization, piece occupancy detection via background subtraction, and neural network classification for piece identification.

## System Overview

```
Raw Camera Frame
       ↓
[1] ArUco Marker Detection
       ↓ (find board corners)
[2] Perspective Warp (to top-down view)
       ↓
[3] Lighting Normalization (CLAHE)
       ↓
[4] Square Extraction (8×8 grid)
       ↓
[5] Occupancy Detection (background subtraction)
       ↓
[6] Piece Classification (MobileNetV2 CNN)
       ↓
[7] FEN String Generation
       ↓
Updated Board State
```

## Core Pipeline (`board_to_fen.py`)

### Function Reference

#### `detect_markers(img_bgr)`
Detects ArUco markers in the image and returns their corner positions.

```python
found, corners, ids = detect_markers(img_bgr)
# found: {marker_id: [[corner_pts]]} 
# ids: [0, 1, 2, 3] (TL, TR, BR, BL expected)
```

**Parameters:**
- `img_bgr`: BGR image from camera

**Returns:**
- `found`: Dictionary mapping marker IDs to corner coordinates
- `corners`: Raw OpenCV corner data
- `ids`: Array of detected marker IDs

**Configuration (tuned for chess board):**
```python
params.adaptiveThreshWinSizeMin = 3
params.adaptiveThreshWinSizeMax = 23
params.minMarkerPerimeterRate = 0.03
params.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_SUBPIX
```

---

#### `warp_board(img_bgr, found)`
Applies perspective transform to convert camera view to top-down board view.

```python
warped = warp_board(img_bgr, found)  # 800×800 image
```

**Marker Placement:**
- **TL marker**: bottom-right corner used as board TL
- **TR marker**: bottom-left corner used as board TR
- **BR marker**: top-left corner used as board BR
- **BL marker**: top-right corner used as board BL

**Output:** 800×800 gray-corrected image

---

#### `normalize_board(board_warp)`
Reduces lighting variation using CLAHE (Contrast Limited Adaptive Histogram Equalization).

```python
board_norm = normalize_board(board_warp)
```

**Steps:**
1. Convert BGR → LAB color space
2. Apply CLAHE to L (brightness) channel
3. Gaussian blur for noise reduction
4. Convert back to BGR

---

#### `extract_squares(board_warp)`
Splits the warped board into 64 individual square images.

```python
squares = extract_squares(board_warp)  # 8×8 list of 100×100 images
```

**Returns:** `squares[row][col]` where row/col ∈ [0,7]

---

#### `detect_occupancy(squares, empty_board_ref, edge_threshold)`
Determines which squares contain chess pieces using automatic method selection.

```python
occupied = detect_occupancy(squares, empty_board_ref)  # 8×8 boolean array
```

**Method 1: Background Subtraction (Preferred)**
- Requires: Empty board reference image captured beforehand
- Process: Absolute difference, bilateral filtering, morphological cleanup
- Advantages: Handles same-colored pieces (white-on-white, etc.)
- Sensitivity: `DIFF_THRESHOLD=30`, `DIFF_PIXELS=1200`

```python
# Occupancy math:
diff = |current_square - empty_square|
occupied_pixels = morphology_open(diff > DIFF_THRESHOLD)
is_occupied = count_nonzero(occupied_pixels) > DIFF_PIXELS
```

**Method 2: Edge Detection (Fallback)**
- Used when empty reference unavailable
- Process: Canny edge detection + dilation + pixel counting
- Sensitivity: `EDGE_THRESHOLD=1500`
- Less reliable (misses featureless pieces)

**Tuning Parameters:**
```python
DIFF_THRESHOLD = 30        # Color change sensitivity
DIFF_PIXELS = 1200         # Min pixels for "occupied"
EDGE_THRESHOLD = 1500      # Edge pixel count threshold
CENTER_CROP = 0.75         # Focus on square center (avoids borders)
```

---

#### `classify_pieces(squares, occupied, classifier)`
Classifies pieces on occupied squares.

```python
board = classify_pieces(squares, occupied, classifier)
# board[row][col] = 'P', 'N', 'B', 'R', 'Q', 'K', 'p', 'n', ... or None
```

**Classification Method:**
1. **Model-Based** (if `classifier` provided):
   - Input: Normalized square image
   - Model: `piece_classifier.keras` (MobileNetv2-based)
   - Output: Piece type + color

2. **Fallback**: Brightness-based color detection
   - White piece if median brightness > 120
   - Black piece otherwise

---

#### `board_to_fen(board)`
Converts 8×8 board representation to FEN string.

```python
fen = board_to_fen(board)
# fen = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR"
```

**FEN Format:**
- Rows encoded rank-by-rank (rank 8 → rank 1)
- Empty squares as digits (e.g., "8" = 8 consecutive empty)
- Pieces as letters (uppercase = white, lowercase = black)

---

#### `capture_empty_board_reference()`
Interactive function to capture and save empty board image for background subtraction.

```python
empty_board_ref = capture_empty_board_reference()
# Saves to: empty_board_ref.pkl
```

**Workflow:**
1. Prompts user to clear board
2. Waits for Enter key
3. Captures image through full pipeline
4. Persists to disk for future sessions

---

#### `process_frame(empty_board_ref, classifier)`
Complete single-frame pipeline from capture to FEN.

```python
fen, warped, debug = process_frame(empty_board_ref, classifier)
```

**Returns:**
- `fen`: FEN string (or None on error)
- `warped`: Top-down board image
- `debug`: Dict with markers_found, occupied_count

---

### Board State Tracking (`BoardStateTracker`)

Temporal smoothing for robust move detection:

```python
tracker = BoardStateTracker(stability_frames=3)
stable_fen, is_new = tracker.update(new_fen)
```

**Logic:**
- Buffers last 3 FEN observations
- Returns "stable" only when 3 consecutive identical states
- Validates position (1 white king, 1 black king, etc.)
- Prevents jitter from momentary misdetections

**Validation Rules:**
- Exactly 1 white king (K) and 1 black king (k)
- Max 8 white pawns (P) and 8 black pawns (p)

## Piece Classifier Model

### Training Data

**Dataset Location:** `square_dataset_organized/`

**Structure:**
```
square_dataset_organized/
├── white_P/    # White pawns
├── white_N/    # White knights
├── ...
├── black_p/    # Black pawns
├── empty/      # Empty squares (for balance)
└── ...
```

**Piece Labels:**
- `P` (white pawn), `N` (knight), `B` (bishop), `R` (rook), `Q` (queen), `K` (king)
- `p`, `n`, `b`, `r`, `q`, `k` (black equivalents)
- `empty` (no piece)

### Model Architecture

**Framework:** TensorFlow/Keras  
**Base Model:** MobileNetV2 (lightweight, accurate)  
**Input:** 64×64 RGB image  
**Output:** 13-class softmax (12 pieces + empty)  
**Inference Time:** ~5ms per square (GPU), ~20ms (CPU)

### Model Files

| File | Purpose | Size |
|------|---------|------|
| `piece_classifier.keras` | Latest trained model | ~26 MB |
| `piece_classifier_best.keras` | Best validation performance | ~26 MB (recommended) |
| `piece_classifier_classes.json` | Class→label mapping + metadata | ~7 KB |

### Loading the Model

```python
import tensorflow as tf
import json

# Load model
model = tf.keras.models.load_model('piece_classifier.keras')

# Load metadata
with open('piece_classifier_classes.json', 'r') as f:
    metadata = json.load(f)

classifier = (model, metadata)
```

### Training New Model

```bash
python train_classifier.py
```

**Script Options:**
```python
# Edit these in train_classifier.py:
epochs = 50
batch_size = 32
learning_rate = 0.001
validation_split = 0.2
```

**Expected Performance:**
- Training accuracy: 98%+
- Validation accuracy: 92-96%
- Training time: 10-30 minutes (GPU)

## Configuration & Tuning

### Camera Settings

```python
# board_to_fen.py (top of file)
BOARD_PIX = 800              # Output warp size
DICT = cv2.aruco.DICT_4X4_50 # ArUco marker dictionary

# Camera capture (Raspberry Pi)
full_w, full_h = picam2.camera_properties["PixelArraySize"]
config = picam2.create_preview_configuration(
    main={"size": (1640, 1232), "format": "RGB888"},  # 4:3 resolution
    controls={"ScalerCrop": (0, 0, full_w, full_h)}
)
```

**Recommended Exposure Settings:**
```python
# Optional: Lock for stability
picam2.set_controls({
    "ExposureTime": 10000,    # 10ms
    "AnalogueGain": 1.0       # Unity gain
})
picam2.set_controls({"AwbEnable": 0})  # Lock white balance
```

### Occupancy Sensitivity

```python
EDGE_THRESHOLD = 1500        # Lower = more sensitive (more false positives)
DIFF_THRESHOLD = 30          # Lower = more sensitive
DIFF_PIXELS = 1200           # Lower = smaller pieces detected
CENTER_CROP = 0.75           # Higher = focus more on center
```

**Tuning Workflow:**
1. Run `debug_occupancy.py` with sample image
2. Adjust thresholds and observe changes in real-time
3. Save optimal values back to `board_to_fen.py`

### Classification Confidence

```python
TYPE_CONFIDENCE_THRESHOLD = 0.45  # Min confidence for piece type
COLOR_CONFIDENCE_THRESHOLD = 0.80 # Min confidence for color
COLOR_MARGIN_THRESHOLD = 0.25     # Margin between top-2 colors
USE_ML_COLOR = False              # Use model color or brightness fallback
```

**Strategy:**
- Keep `USE_ML_COLOR = False` (brightness-based more reliable)
- Adjust `TYPE_CONFIDENCE_THRESHOLD` if misclassifications occur
- Lower values = more permissive, higher = stricter

## Testing & Validation

### Quick Tests

```bash
# Camera functioning
python test_camera.py
# → Display live camera feed

# ArUco marker detection
python debug_corners.py
# → Show detected marker corners

# Board warp alignment
python test_grid_alignment.py
# → Show warped board with grid overlay

# Occupancy detection
python debug_occupancy.py
# → Display occupancy map with parameters
```

### End-to-End Testing

```bash
python board_to_fen.py
# → Interactive: capture empty board, then run detection
```

**Expected Output:**
```
=== ArUco Chess Board to FEN Pipeline ===
Loading trained piece classifier...
✓ Classifier loaded successfully from piece_classifier_best.keras

Options:
1. Capture empty board reference (recommended)
2. Run without empty board reference (edge detection only)

Enter choice (1 or 2): 1
Capturing empty board reference...
Make sure the board is EMPTY and press Enter
[Camera displays, user presses Enter]
Empty board reference captured and saved!

=== Starting FEN Detection ===
Press Ctrl+C to stop

Frame 0: rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR (unstable)
Frame 1: rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR (unstable)
Frame 2: rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR (unstable)

============================================================
NEW STABLE POSITION (frame 3):
FEN: rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR
Markers: [0, 1, 2, 3]
Occupied squares: 16
============================================================
```

## Performance Metrics

| Metric | Value | Notes |
|--------|-------|-------|
| **FPS** | 10 Hz | Configurable via sleep(0.1) |
| **Detection Accuracy** | 95%+ | With training data |
| **Classification Latency** | 5-20 ms/frame | GPU vs CPU |
| **Marker Detection Success** | 98%+ | Under good lighting |
| **False Positive Rate** | <2% | With occupancy parameters tuned |

## Troubleshooting

| Problem | Cause | Solution |
|---------|-------|----------|
| **Markers not detected** | Poor image quality, wrong dictionary | Check lighting, verify DICT_4X4_50 |
| **Warped board distorted** | Marker placement issue | Ensure markers at board corners, not offset |
| **Pieces misclassified** | Lighting variation, ink fade | Capture new training data, retrain |
| **False occupancy** | Shadows or texture | Adjust `DIFF_PIXELS` threshold |
| **Model inference slow** | CPU processing | Use GPU acceleration or mobile model |
| **"Unreachable" FEN** | Invalid piece count | Check dataset for corrupted labels |

## Dependencies

- `opencv-python >= 4.5`
- `tensorflow >= 2.8`
- `numpy >= 1.20`
- `picamera2` (Raspberry Pi only)

## References

- OpenCV ArUco Module: https://docs.opencv.org/4.5.2/d5/dae/tutorial_aruco_detection.html
- MobileNetV2 Paper: https://arxiv.org/abs/1801.04381
- TensorFlow/Keras: https://tensorflow.org/tutorials/computer_vision
