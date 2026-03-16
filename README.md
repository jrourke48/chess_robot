# Chess Robot - Autonomous Chess-Playing Robotic Arm

A sophisticated robotic system that plays chess autonomously using computer vision, inverse kinematics, and trajectory planning. The robot detects board moves via camera, validates game state, plans motion trajectories, and executes moves with servo-controlled robotic arm.

## Project Overview

This is an integrated mechanical engineering senior capstone project (Cal Poly ME 423) combining:
- **Computer Vision**: ArUco marker-based board detection and neural network piece classification
- **Robotics**: 4-DOF robotic arm with servo motors and electromagnetic gripper
- **Control Systems**: Real-time servo control and trajectory planning (5th-order spline)
- **Game Logic**: Chess state management and move validation via python-chess
- **Web UI**: Real-time control, visualization, and calibration dashboard

### Key Features
✅ Autonomous move detection via camera or manual input  
✅ Real-time trajectory visualization (2D board + 3D workspace)  
✅ Two-stage servo calibration with offset adjustment  
✅ WebSocket-based UI with game state tracking  
✅ Support for checkmate, stalemate, and pawn promotion detection  
✅ Current board representation in FEN notation  

## Quick Start

### 1) Environment Setup

```powershell
# Create and activate virtual environment
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install --upgrade pip
pip install -r requirements.txt
```

For Raspberry Pi with camera:
```bash
pip install -r requirements-rpi-camera.txt
```

### 2) Verify Dependencies

```powershell
python scripts/verify_environment.py
```

Checks for:
- Python modules (opencv, tensorflow, chess, fastapi, etc.)
- Stockfish binary availability
- Serial drivers on Windows (ch341ser)

### 3) Run the System

```powershell
python chessrobotclasses/main.py
```

Open browser to: **http://localhost:8000**

## System Architecture

### Task-Based Async Model

All tasks run concurrently using Python `asyncio`:

```
┌─────────────────────────────────────────────────────────────┐
│                   Main Async Loop                           │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌──────────┐   │
│  │  CV Task │  │Servo Ctrl│  │EndGame   │  │   UI     │   │
│  │          │  │   Task   │  │   FSM    │  │  Task    │   │
│  │- Detect  │  │- Motion  │  │- Moves   │  │- Dashboard│  │
│  │- Classify│  │- Trajproj│  │- Validate│  │- WebSocket│  │
│  └──────────┘  └──────────┘  └──────────┘  └──────────┘   │
│                                                             │
│  ┌──────────────────────────────────────┐                  │
│  │    Calibration Task + Sensor Ctrl    │                  │
│  │    - Servo positioning               │                  │
│  │    - Offset tuning                   │                  │
│  │    - Electromagnet control           │                  │
│  └──────────────────────────────────────┘                  │
│                                                             │
├─────────────────────────────────────────────────────────────┤
│         Shared Events & Queues (thread-safe)              │
└─────────────────────────────────────────────────────────────┘
```

### Communication Layer

| Type | Purpose | Example |
|------|---------|---------|
| **Events** | Async synchronization flags | `begin_game`, `ready2move`, `calibrate_servos` |
| **Queues** | Data passing between tasks | FEN strings, joint angles, UI updates |
| **WebSocket** | Real-time UI communication | Game state, plots, log messages |

## Directory Structure

```
chess_robot/
├── README.md                               # Main documentation (this file)
│
├── cv_fen/                                 # Computer Vision System
│   ├── README.md                          # 📖 CV documentation
│   ├── board_to_fen.py                    # Main vision pipeline
│   ├── piece_classifier.keras             # Trained neural network
│   ├── piece_classifier_best.keras        # Best model version
│   ├── piece_classifier_classes.json      # Class mapping
│   ├── aruco_warp.py                      # ArUco detection utilities
│   ├── train_classifier.py                # Model training script
│   └── square_dataset_organized/          # Training data
│
├── chessrobotclasses/                     # Robot Control System
│   ├── README.md                          # 📖 Robot classes documentation
│   ├── main.py                            # Async task orchestration
│   ├── UITask.py                          # Web UI (FastAPI + WebSocket)
│   ├── EndGameEffectorTask.py             # Move execution FSM
│   ├── ServoController.py                 # Servo command interface
│   ├── RobotMotionPlanner.py             # Chess→workspace conversion
│   ├── InverseKinematics_TrajectoryPlanner.py
│   ├── ChessStateValidatorMoveParser.py  # Chess game logic
│   ├── Electromagnet.py                  # Gripper control
│   ├── Stockfish.py                      # Chess engine wrapper
│   ├── test_classes.py                   # Unit tests & visualization
│   ├── Servos/                           # Servo hardware layer
│   │   ├── README.md                     # 📖 Servo documentation
│   │   ├── servo_ping.py                 # Servo discovery
│   │   ├── servo_move.py                 # Joint control utilities
│   │   └── servo_sdk/                    # Protocol drivers
│   │       ├── BusServo.py               # Abstract servo interface
│   │       ├── Hiwonder55Servo.py        # 55-series protocol (115200 bps)
│   │       ├── HiwonderFFServo.py        # FF-series protocol (1000000 bps)
│   │       └── Servo_models.py           # Servo configuration
│   └── ui_plots/                         # Visualization outputs
│
├── scripts/
│   └── verify_environment.py              # Dependency checker
│
├── requirements.txt                       # Python dependencies
├── requirements-rpi-camera.txt           # RPi-specific dependencies
└── ch341ser.exe, ServoStudio_v0.1.5.exe  # Windows hardware tools
```

## Component Guides

### 📖 Computer Vision (`cv_fen/`)
- ArUco marker board detection and perspective warp
- Background subtraction for occupancy detection
- MobileNetV2-based piece classification
- FEN string generation

**[Read cv_fen/README.md →](cv_fen/README.md)**

### 📖 Servo Control (`chessrobotclasses/Servos/`)
- Multi-protocol serial communication (Hiwonder 55-series, FF-series)
- Position/velocity control with limits
- Calibration and offset management
- Servo health checking

**[Read chessrobotclasses/Servos/README.md →](chessrobotclasses/Servos/README.md)**

### 📖 Robot Classes (`chessrobotclasses/`)
- 4-DOF arm kinematics and inverse kinematics
- Motion planning and trajectory generation
- Chess state validation and move parsing
- Async task architecture

**[Read chessrobotclasses/README.md →](chessrobotclasses/README.md)**

## Wiring & Hardware

### Servo Buses

| Bus | Protocol | Baud Rate | Servos | Purpose |
|-----|----------|-----------|---------|---------|
| **Bus 1** | Hiwonder FF | 1,000,000 | 4 (Wrist), 2 (Tilt Shoulder) | Precision upper-arm |
| **Bus 2** | Hiwonder 55 | 115,200 | 3 (Elbow), 1 (Pan Shoulder) | Lower-arm control |

### Servo Configuration

```python
# chessrobotclasses/ServoController.py
WRISTLIMITS_HX_10HM = ServoLimits(
    min_angle=1200, max_angle=3600,
    offset=0  # Tune during calibration
)
ELBOWLIMITS_HX_35HM = ServoLimits(
    pos_max=1000, min_angle=500, max_angle=1000,
    offset=0
)
SHOULDERLIMITS_HX_65HM = ServoLimits(
    min_angle=0, max_angle=4095, offset=0
)
```

### Peripherals

| Device | GPIO/Port | Function |
|--------|-----------|----------|
| **Electromagnet** | GPIO 17 | Piece pickup/release |
| **Pi Camera V2** | Camera Port | Board image capture |
| **Servo Bus 1** | USB Serial | Wrist + tilt control |
| **Servo Bus 2** | USB Serial | Elbow + pan control |

## Game Workflow

```
┌─────────────────────────────────────────────────────────────┐
│  1. BEGIN GAME                                              │
│     - Open game, set initial FEN                           │
│     - UI ready, awaiting opponent move                     │
└────────────────────┬────────────────────────────────────────┘
                     │
┌────────────────────▼────────────────────────────────────────┐
│  2. DETECT MOVE                                             │
│     - CV captures board image (if enabled) OR               │
│     - User enters UCI move manually                         │
└────────────────────┬────────────────────────────────────────┘
                     │
┌────────────────────▼────────────────────────────────────────┐
│  3. VALIDATE MOVE                                           │
│     - Parse UCI or detect FEN change                        │
│     - Check legal move in current position                 │
│     - Update chess board state                             │
│     - Check for checkmate/stalemate/promotion              │
└────────────────────┬────────────────────────────────────────┘
                     │
┌────────────────────▼────────────────────────────────────────┐
│  4. GENERATE ROBOT PATH                                     │
│     - Chess coords (board squares) → 3D workspace coords    │
│     - Inverse kinematics → joint angles                     │
│     - Trajectory planning (5th-order spline)               │
└────────────────────┬────────────────────────────────────────┘
                     │
┌────────────────────▼────────────────────────────────────────┐
│  5. EXECUTE MOVE                                            │
│     - Send servo joint commands in trajectory segments      │
│     - Electromagnet on for piece pickup at source          │
│     - Electromagnet off for piece release at destination    │
└────────────────────┬────────────────────────────────────────┘
                     │
┌────────────────────▼────────────────────────────────────────┐
│  6. UPDATE UI                                               │
│     - Send board FEN, waypoints, plots to dashboard        │
│     - Return to step 2                                     │
└─────────────────────────────────────────────────────────────┘
```

## Calibration Guide

### Servo Offset Calibration

**Via Web UI (Recommended):**

1. Do NOT start a game
2. Click **"Stage 1: Init Calibration"**
   - Servos move to neutral (512 position)
   - Allow 2 seconds for settling
   
3. Manually adjust servos if needed (use ServoStudio.exe)

4. Click **"Stage 2: Confirm Offsets"**
   - Current positions are read and confirmed
   - Offsets saved for session

**Via Python API:**

```python
# Set absolute offset values
servo_controller.set_servo_offsets([10, 20, 30, 40])

# Or adjust incrementally
servo_controller.adjust_servo_offsets([-5, 5, 0, 10])
```

### Camera Calibration

Run before vision tasks:
```bash
python cv_fen/test_camera.py      # Check camera feed
python cv_fen/debug_corners.py    # Verify ArUco detection
python cv_fen/test_grid_alignment.py  # Check board warping
```

## Performance Specifications

| Metric | Value |
|--------|-------|
| **Vision FPS** | 10 Hz |
| **Servo Response Time** | 50-1000 ms (configurable) |
| **End-effector Accuracy** | ±1 cm |
| **Piece Detection Rate** | 95%+ |
| **Calibration Time** | ~5 minutes |
| **Move Execution Time** | 10-30 seconds |

## Dependencies

### Core Python Packages
- `opencv-python >= 4.5` — Computer vision
- `tensorflow >= 2.8` — Neural network inference
- `numpy >= 1.20` — Numerical computing
- `chess >= 1.6` — Chess move validation
- `fastapi >= 0.95` — Web API framework
- `uvicorn >= 0.21` — ASGI server
- `pyserial >= 3.5` — Serial communication

### Optional (Raspberry Pi)
- `picamera2` — Pi Camera integration
- `gpiozero >= 1.6` — GPIO control (electromagnet)

See [requirements.txt](requirements.txt) for complete list with versions.

## Troubleshooting

### Vision Problems
| Issue | Solution |
|-------|----------|
| Poor piece detection | Check lighting, capture new empty board reference |
| Marker not detected | Ensure markers visible and properly spaced |
| Misclassification | Retrain model with more example images |

### Servo Problems
| Issue | Solution |
|-------|----------|
| No servo response | Check USB connections, verify `lsusb` / Device Manager |
| Jerky movement | Adjust `default_move_time_ms` in `Servo_models.py` |
| Position drift | Recalibrate via Web UI |

### Robot Problems
| Issue | Solution |
|-------|----------|
| "Unreachable" position | Check IK solver bounds, verify position is in workspace |
| Gripper won't release | Check electromagnet GPIO, test with `Electromagnet.py` |
| Collision detected | Reduce trajectory smoothing, check obstacle boundaries |

## Development

### Running Tests

```powershell
python chessrobotclasses/run_test_classes.py
```

Tests cover:
- Chess game logic
- IK solver accuracy
- Servo communication
- Vision processing

### Key Constants

Edit these to tune system behavior:

```python
# cv_fen/board_to_fen.py
EDGE_THRESHOLD = 1500        # Piece occupancy sensitivity
DIFF_THRESHOLD = 30          # Background subtraction threshold
DIFF_PIXELS = 1200           # Min pixels for occupied square

# chessrobotclasses/Servo_models.py
default_move_time_ms = 50    # Servo movement speed
max_move_time_ms = 1000      # Max servo speed limit

# chessrobotclasses/InverseKinematics_TrajectoryPlanner.py
TRAJECTORY_TYPE = 'fifth_order'  # 'cubic_spline' or 'fifth_order'
```

## References & Documentation

- **Hiwonder Servo Drivers**: `chessrobotclasses/Servos/servo_sdk/`
- **Chess.py Library**: https://python-chess.readthedocs.io/
- **OpenCV ArUco**: https://docs.opencv.org/master/tag_aruco.html
- **FastAPI**: https://fastapi.tiangolo.com/
- **TensorFlow/Keras**: https://tensorflow.org/

## License

Educational/Research Use - Cal Poly ME 423 Senior Capstone (Winter 2026)

## Authors & Contributors

Capstone Team, Mechanical Engineering Department  
California Polytechnic State University, San Luis Obispo
