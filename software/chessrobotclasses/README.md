# Chess Robot Classes - Core Control Architecture

High-level robot control, motion planning, and finite state machine orchestration. This module coordinates vision, servo control, end-game effector, and UI into a coherent chess-playing system.

## System Architecture

```
                    main.py (Async Orchestrator)
                          │
            ┌─────────────┼─────────────┐
            │             │             │
        [Tasks]      [Events]      [Queues]
            │             │             │
      ┌─────┴─────┐   ┌───┴───┐    ┌───┴───┐
      │           │   │       │    │       │
   cv_task    servo_   begin_   board_    piece_
   (Vision)  task      game     update   states
            (Control) (Event)   (Queue)  (Queue)
            
            ↓         ↓         ↓       ↓         ↓
        
      ┌──────────────────────────────┐
      │     State Machine (FSM)      │
      │  in UITask.py                │
      │                              │
      │  WAIT4GAME                   │
      │    ├─→ CALIBRATE_SERVOS      │
      │    ├─→ CALIBRATE_SERVOS2     │
      │    ├─→ GAME_IN_PROGRESS      │
      │    └─→ GAME_OVER             │
      │                              │
      │  Motion Planning Pipeline:   │
      │  [Chess Move]                │
      │    → IK Solver               │
      │    → Trajectory Gen          │
      │    → Servo Command           │
      │    → Feedback (Vision)       │
      │                              │
      └──────────────────────────────┘
```

## Module Overview

### `main.py` - Task Orchestrator

Async event loop managing 5 concurrent tasks:

```python
async def run_all_tasks():
    """Main async loop - runs all robot subsystems"""
    
    # Task 1: Computer Vision
    cv_task = asyncio.create_task(
        cv_inference_loop(cv_queue, board_update_queue)
    )
    
    # Task 2: Servo Controller
    servo_task = asyncio.create_task(
        servo_controller_task(servo_events)
    )
    
    # Task 3: End-game Effector (Gripper)
    endgame_task = asyncio.create_task(
        endgameeffector_cpu_task(servo_controller, endgame_queue)
    )
    
    # Task 4: UI Server + FSM
    ui_task = asyncio.create_task(
        UITask(events=ui_events, queues=queues).run()
    )
    
    # Task 5: Sensor Control
    sensor_task = asyncio.create_task(
        lowlevel_sensorcontrol_task(events=ui_events)
    )
    
    # All tasks run concurrently until Ctrl+C
    await asyncio.gather(
        cv_task, servo_task, endgame_task, ui_task, sensor_task
    )
```

**Event Dictionary:**
```python
ui_events = {
    'begin_game': asyncio.Event(),        # Start game sequence
    'calibrate_servos': asyncio.Event(),  # Stage 1 calibration
    'calibrate_servos2': asyncio.Event(), # Stage 2 calibration
    'move_gripper': asyncio.Event(),      # Activate gripper
    'stop': asyncio.Event(),              # Emergency stop
}
```

**Queue Dictionary:**
```python
queues = {
    'board_update': asyncio.Queue(),      # Vision → UI (board FEN)
    'piece_states': asyncio.Queue(),      # Piece position updates
    'endgame_commands': asyncio.Queue(),  # UI → EndgameMover
}
```

### `UITask.py` - Web Interface & State Machine

FastAPI web server + FEN state management + robot FSM.

```
HTTP GET /
    ↓ (serves HTML dashboard)
    
HTML UI (http://localhost:8000)
    ├─ [Begin Game] button
    ├─ [Calibrate Stage 1] button
    ├─ [Calibrate Stage 2] button
    ├─ Board visualization
    ├─ FEN display
    └─ Control panel
    
    ↓ (WebSocket messages)
    
FastAPI WebSocket Handler
    ├─ /ws message processor
    ├─ Event dispatcher
    └─ Board state manager
    
    ↓ (asyncio events)
    
FSM State Machine (UITask.run())
    ├─ state_wait4game()
    ├─ state_game_in_progress()
    ├─ state_calibrate_servos()
    ├─ state_calibrate_servos2()
    └─ state_game_over()
```

#### FSM State Diagram

```
             ┌─────────────────────────────────┐
             │      WAIT4GAME (Initial)        │
             │  Waiting for begin_game event   │
             └────┬────────┬────────┬──────────┘
                  │        │        │
        (calibrate(calibrate│      (begin_game)
         _servos)  _servos2)│       │
              │        │     │       ↓
              ↓        ↓     └─→ GAME_IN_PROGRESS
         CALIBRATE  CALIBRATE │   (Move sequence)
         _SERVOS    _SERVOS2  │   ├─ getMove()
              │        │      │   ├─ movePiece()
              └────┬───┘      │   └─ updateBoard()
                   │          │
                 (cleared)     ↓
                   │    ╔═ (game_over)
                   ↓    ║
              WAIT4GAME←╨─ GAME_OVER
                        (Display result)
                        (Return to WAIT4GAME)
```

#### State Handlers

```python
async def state_wait4game(self) -> State:
    """Idle state - wait for game start or calibration request
    
    Priority:
    1. If calibrate_servos event set → return CALIBRATE_SERVOS
    2. Else if begin_game event set → return GAME_IN_PROGRESS
    3. Else → loop
    """
    
async def state_calibrate_servos(self) -> State:
    """Stage 1: Move all servos to neutral positions
    
    Actions:
    1. Move all servos to neutral (500, 512, 650, 2400)
    2. Wait for calibrate_servos2 event
    3. Return upper/lower limits to user
    """
    
async def state_calibrate_servos2(self) -> State:
    """Stage 2: Confirm offsets and return to waiting
    
    Actions:
    1. Await calibrate_servos2 event
    2. Read current servo angles
    3. Calculate and set offsets
    4. Clear both calibration events
    5. Return to WAIT4GAME
    """
    
async def state_game_in_progress(self) -> State:
    """Main game loop: detect moves, execute, update board
    
    Loop:
    ├─ Read board state from vision
    ├─ Validate move (compare to previous FEN)
    ├─ Plan trajectory (IK solver)
    ├─ Execute servo motion
    ├─ Update UI with new board state
    └─ Repeat until game_over event
    """
    
async def state_game_over(self) -> State:
    """End state: display result, cleanup
    
    Actions:
    1. Display checkmate/draw result
    2. Disable all servo commands
    3. Return to WAIT4GAME after user acknowledges
    """
```

### `ServoController.py` - High-Level Motion Control

Wrapper combining 55S and FF servo buses with robot kinematics awareness.

```python
class ServoController:
    def __init__(self, com55s='COM3', com_ff='COM4', timeout=1.0):
        """Initialize servo buses and limits"""
        
    async def move_to_position(self, position_name: str, duration_ms: int = 1000) -> bool:
        """Move to named position (e.g., 'home', 'capture', 'neutral')"""
        
    def move_servo(self, servo_id: int, angle: int, duration: int = 500) -> bool:
        """Single servo direct command"""
        
    async def trajectory_move(self, trajectory: list, total_duration: int) -> bool:
        """Execute trajectory (multiple waypoints over time)"""
        
    def set_servo_offsets(self, offsets: list[int]) -> None:
        """Set calibration offsets for all 4 servos"""
        
    def get_servo_offsets(self) -> list[int]:
        """Read current offsets"""
```

### `RobotMotionPlanner.py` - Trajectory Planning

Convert chess moves (algebraic notation) into robot trajectories.

```python
class RobotMotionPlanner:
    def __init__(self, servo_controller: ServoController):
        """Initialize with servo interface"""
        
    def plan_capture(self, from_square: str, to_square: str) -> Trajectory:
        """Plan move + capture sequence
        
        Args:
            from_square: Starting square (e.g., 'e2')
            to_square: Target square (e.g., 'e4')
            
        Returns:
            Trajectory object with waypoints
        """
        
    def plan_move(self, from_square: str, to_square: str) -> Trajectory:
        """Plan piece movement (non-capture)"""
        
    def chess_space_to_robot_space(self, square: str) -> CartesianPoint:
        """Convert chess board coordinates to robot XYZ"""
```

### `InverseKinematics_TrajectoryPlanner.py` - IK Solver

Convert 3D Cartesian coordinates to servo angles.

```python
class InverseKinematics:
    def __init__(self, robot_params):
        """Initialize with DH parameters"""
        
    def solve(self, target_xyz: tuple) -> list[int] | None:
        """Solve IK for target position
        
        Args:
            target_xyz: (x, y, z) in robot frame (mm)
            
        Returns:
            [servo1_angle, servo2_angle, servo3_angle, servo4_angle] or None
            
        Raises:
            ValueError: If position unreachable
        """
        
    def forward_kinematics(self, angles: list[int]) -> tuple:
        """Compute end-effector position from servo angles
        
        Returns: (x, y, z)
        """
```

### `EndGameEffectorTask.py` - Gripper Control

Control electromagnetic gripper (controlled via GPIO).

```python
class EndGameEffectorTask:
    def __init__(self, servo_controller: ServoController):
        """Initialize gripper FSM"""
        
    async def task(self, queue: asyncio.Queue) -> None:
        """Main task loop - listen for gripper commands"""
        
    async def grip(self, duration_ms: int = 500) -> bool:
        """Activate electromagnet + close gripper"""
        
    async def release(self) -> bool:
        """Deactivate electromagnet + open gripper"""
```

### `Electromagnet.py` - GPIO Control

Low-level GPIO control for electromagnet solenoid.

```python
class Electromagnet:
    def __init__(self, gpio_pin: int = 17):
        """Initialize GPIO pin (BCM numbering on Raspberry Pi)"""
        
    def activate(self) -> bool:
        """Enable electromagnet (set pin HIGH)"""
        
    def deactivate(self) -> bool:
        """Disable electromagnet (set pin LOW)"""
        
    def is_active(self) -> bool:
        """Check current state"""
        
    def cleanup(self) -> None:
        """Release GPIO resource"""
```

### `ChessStateValidatorMoveParser.py` - Move Validation

Parse and validate chess moves against board state.

```python
class ChessStateValidator:
    def __init__(self, fen_string: str):
        """Initialize with starting FEN"""
        
    def detect_player_move(self, old_fen: str, new_fen: str) -> Move | None:
        """Detect move from FEN change
        
        Args:
            old_fen: Previous board state
            new_fen: Current board state
            
        Returns:
            Move object (from_square, to_square) or None if invalid
        """
        
    def detect_opponent_response(self, new_fen: str) -> Move | None:
        """Detect opponent's move from new FEN"""
        
    def is_checkmate(self) -> bool:
        """Test if position is checkmate"""
        
    def get_legal_moves(self) -> list[Move]:
        """List valid moves from current position"""
```

### `Stockfish.py` - Chess Engine Integration

Optional computer-vs-computer play analysis.

```python
class StockfishEngine:
    def __init__(self, fen: str, depth: int = 20):
        """Initialize Stockfish subprocess
        
        Args:
            fen: Starting position (FEN)
            depth: Search depth (higher = stronger, slower)
        """
        
    def get_best_move(self) -> Move:
        """Compute best move for current position
        
        Returns: Move object with coordinates
        """
        
    def evaluate_position(self) -> float:
        """Get position evaluation (centipawns)
        
        Returns:
            +100 = White winning 1 pawn
            -100 = Black winning 1 pawn
            0 = Equal
        """
```

### `ServoController.py` - (Servo Hardware Layer)

See [Servos/README.md](Servos/README.md) for detailed servo control documentation.

## Control Flow Example

### Starting a Game

```
User clicks "Begin Game" button
    ↓ (WebSocket message)
UI event handler sets begin_game event
    ↓ (asyncio event)
state_wait4game() detects event
    ↓ (FSM transition)
FSM → GAME_IN_PROGRESS
    ↓
state_game_in_progress() starts loop:
    1. Read board from CV (via board_update_queue)
    2. Compare to previous state → detect move
    3. Plan trajectory (IK solver)
    4. Execute servo motion
    5. Update UI with new board state
    6. Repeat until game_over
    ↓
User clicks "End Game" button
    ↓
state_game_in_progress() exits
    ↓
FSM → GAME_OVER
    ↓
state_game_over() displays result
    ↓
FSM → WAIT4GAME (loops back)
```

### Servo Calibration Flow

```
User clicks "Stage 1: Init Calibration"
    ↓ (WebSocket)
calibrate_stage1 handler sets calibrate_servos event
    ↓
state_wait4game() detects calibration priority
    ↓
FSM → CALIBRATE_SERVOS
    ↓
state_calibrate_servos():
    - Move all servos to neutral angles
    - Display: "Manually adjust servos, then go to Stage 2"
    - Wait for calibrate_servos2 event
    ↓
User manually adjusts servo positions with hands
    ↓
User clicks "Stage 2: Confirm Offsets"
    ↓
calibrate_stage2 handler sets calibrate_servos2 event
    ↓
state_calibrate_servos2():
    - Read current servo angles
    - Calculate offset = neutral_angle - current_angle
    - Set offsets in ServoLimits
    - Move servos again to verify
    - Clear calibration events
    ↓
FSM → WAIT4GAME (back to idle)
```

## Testing & Validation

### Unit Tests

```bash
# Run test suite
python run_test_classes.py
```

**Core tests:**
- `test_servo_limits()`: Offset calculations
- `test_ik_solver()`: Forward/inverse kinematics
- `test_move_validator()`: Legal move detection
- `test_fsm_transitions()`: State machine paths

### Integration Tests

```bash
# Start server
python server.py

# In browser: http://localhost:8000
# Manual tests:
1. Click "Begin Game" → Verify calibration state
2. Click "Calibrate Stage 1" → Servos move to neutral
3. Click "Calibrate Stage 2" → Offsets set
4. Click "Begin Game" → Vision loop starts
5. Move piece on board → Detect movement
6. Verify servo execution
```

## Performance Metrics

| Metric | Value | Notes |
|--------|-------|-------|
| **Vision FPS** | 10 Hz | Board state update rate |
| **Servo Response** | 50-200 ms | Per-servo command latency |
| **Move Execution** | 2-5 seconds | Full move sequence (grip, move, release) |
| **IK Solver** | <10 ms | Per-move planning |
| **FSM Cycle** | 100 ms | State check frequency |
| **Memory (Resting)** | ~150 MB | Python + TensorFlow + OpenCV |

## Troubleshooting

| Problem | Symptom | Check |
|---------|---------|-------|
| **Move not detected** | Vision reads but no change recognized | Check board state validator, FEN comparison |
| **Servo stops mid-move** | Partial trajectory execution | Verify servo not hitting limits, check offset |
| **Gripper won't release** | Piece stuck after capture | Check GPIO pin value, verify electromagnet power |
| **FSM stuck in state** | UI unresponsive | Check event set/clear in handlers |
| **Memory leak** | Gradual slowdown over time | Check queue growth, verify CV frame cleanup |

## Dependencies

- `python >= 3.8`
- `asyncio` (stdlib)
- `FastAPI >= 0.95`
- `python-chess >= 1.6`
- `OpenCV >= 4.5`
- `TensorFlow >= 2.8`
- `numpy >= 1.20`
- `RPi.GPIO` (Raspberry Pi only)

## Architecture References

- **Async Design**: 5 concurrent tasks coordinated via events/queues
- **FSM Pattern**: State machine with priority event checking
- **Kinematics**: DH parameter-based forward/inverse solvers
- **Computer Vision**: Real-time FEN detection (10 Hz)
- **Web Interface**: FastAPI + WebSocket for low-latency communication

## Related Modules

- **Computer Vision**: [cv_fen/README.md](../cv_fen/README.md)
- **Servo Hardware**: [Servos/README.md](./Servos/README.md)
- **Vision Tests**: [cv_fen/](../cv_fen/) directory scripts
- **Servo Utilities**: [Servos/](./Servos/) directory scripts
