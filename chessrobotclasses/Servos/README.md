# Servo Control System - Hardware Layer

Low-level servo hardware control for the chess robot. Supports dual-protocol multi-bus servo architecture: 55-series (115200 bps) and FF-series (1 Mbps) Hiwonder servos on separate hardware buses.

## System Architecture

```
ServoController (High-Level Interface)
    ↓ (command: servo_id, angle, duration)
    ├─ BusServo_55S (115200 bps bus)
    │  ├─ Servo 1 (Pan, ID 1)
    │  └─ Servo 3 (Elbow, ID 3)
    │
    └─ BusServo_FF (1 Mbps bus)
       ├─ Servo 2 (Tilt, ID 2)
       └─ Servo 4 (Wrist, ID 4)
```

## Hardware Configuration

### Physical Setup

| Servo | Function | Bus | Baud Rate | Limits | Idle Angle | Speed |
|-------|----------|-----|-----------|--------|------------|-------|
| **1** | Pan (Base) | 55S | 115200 | 120-880 | 500 | 240°/s |
| **2** | Tilt (Shoulder) | FF | 1000000 | 0-4095 | 512 | N/A |
| **3** | Elbow | 55S | 115200 | 500-1000 | 650 | 240°/s |
| **4** | Wrist (Grip) | FF | 1000000 | 1200-3600 | 2400 | N/A |

### Serial Connections

**Raspberry Pi:**
- UART 0 (`/dev/ttyAMA0`): 55S bus (Servos 1, 3)
- USB adapter (`/dev/ttyUSB0`): FF bus (Servos 2, 4)

**Development Machine (Windows):**
- COM3: 55S bus
- COM4: FF bus

## Core Components

### ServoLimits Dataclass

Configuration container for individual servo parameters.

```python
@dataclass
class ServoLimits:
    min_angle: int           # Minimum angle value (hardware limit)
    max_angle: int           # Maximum angle value (hardware limit)
    neutral_angle: int       # Home/safe position
    current_offset: int = 0  # Calibration offset (-500 to +500)
    
    def set_offset(self, new_offset: int) -> None:
        """Set absolute calibration offset"""
        self.current_offset = max(-500, min(500, new_offset))
    
    def adjust_offset(self, delta: int) -> None:
        """Fine-tune offset by delta value"""
        self.set_offset(self.current_offset + delta)
    
    def apply_offset(self, target_angle: int) -> int:
        """Return angle with offset applied"""
        clipped = max(self.min_angle, min(self.max_angle, target_angle))
        return clipped + self.current_offset
```

**Usage:**
```python
# Access current limits
servo = ServoController()
servo1_limits = servo.servo_limits[0]
print(f"Servo 1 range: {servo1_limits.min_angle}-{servo1_limits.max_angle}")
print(f"Current offset: {servo1_limits.current_offset}")

# Set calibration offset
servo.servo_limits[0].set_offset(15)

# Fine-tune
servo.servo_limits[0].adjust_offset(5)  # +5 to offset
```

### BusServo Base Protocol

Abstract servo communication protocol. All bus implementations inherit from `BusServo`.

```python
class BusServo:
    def write_angle(self, servo_id: int, angle: int, duration: int) -> bool:
        """Command servo to angle over duration (ms)"""
        
    def read_angle(self, servo_id: int) -> int | None:
        """Query current servo angle (None if failed)"""
        
    def set_id(self, old_id: int, new_id: int) -> bool:
        """Change servo ID (requires power cycle after)"""
        
    def ping(self, servo_id: int) -> bool:
        """Check if servo responding"""
        
    def enable(self, servo_id: int) -> bool:
        """Enable servo torque"""
        
    def disable(self, servo_id: int) -> bool:
        """Disable servo torque (freewheel)"""
```

### BusServo_55S (115200 Protocol)

Low-speed 55-series servo protocol. 115200 bps, command-response pattern.

```python
# Packet structure (8 bytes):
# [0xFF] [0xFF] [ID] [LEN] [CMD] [ADDR] [DATA...] [CHECKSUM]

# Example: Move servo 1 to angle 500 in 1000ms
bus_55s.write_angle(servo_id=1, angle=500, duration=1000)
```

**Supported Operations:**
- `get_angle()`: Query current angle
- `move_with_calibration()`: Move with offset applied
- `read_angle()`: Current servo state
- `ping()`: Alive check

**Parameters:**
```python
BAUD_RATE = 115200
BUS_WRITE_DELAY = 0.01 s  # Spacing between commands
MOVEMENT_TIME_MIN = 100 ms
MOVEMENT_TIME_MAX = 30000 ms
```

### BusServo_FF (1 Mbps Protocol)

High-speed FF-series servo protocol. 1 Mbps, continuous polling.

```python
# Packet structure:
# [0xFF] [0xFF] [ID] [LEN] [CMD] [ADDR] [DATA16(LE)...] [CHECKSUM]

# Example: Move servo 2 to angle 2048
bus_ff.write_angle(servo_id=2, angle=2048, duration=500)
```

**Supported Operations:**
- `get_angle()`: Using angle status register (0x39)
- `move_with_calibration()`: 16-bit angle format
- `read_angle()`: High-resolution position feedback

**Differences from 55S:**
- 16-bit angle format (0-4095 range)
- 1 Mbps communication (10× faster)
- Continuous polling vs command-response
- Per-servo response status bytes

## ServoController (High-Level API)

High-level interface for robot arm control, combining both servo buses.

```python
class ServoController:
    def __init__(self, com55s='COM3', com_ff='COM4', timeout=1.0):
        """Initialize both servo buses
        
        Args:
            com55s: Serial port for 55S bus (e.g. '/dev/ttyAMA0' on RPi)
            com_ff: Serial port for FF bus (e.g. '/dev/ttyUSB0' on RPi)
            timeout: Serial read timeout (seconds)
        """
    
    def move_servo(self, servo_id: int, angle: int, duration: int = 500) -> bool:
        """Move single servo to angle
        
        Args:
            servo_id: 1-4 (router selects correct bus)
            angle: Target angle (within servo limits)
            duration: Movement time (ms)
            
        Returns:
            True if command successful
            
        Raises:
            ValueError: If servo_id invalid or angle out of limits
        """
        
    def move_servos(self, commands: list[tuple]) -> list[bool]:
        """Move multiple servos simultaneously
        
        Args:
            commands: [(servo_id, angle, duration), ...]
            
        Returns: Success result for each command
        """
        
    def move_group(self, group_name: str, angle: int, duration: int = 500) -> bool:
        """Move servo group by name
        
        Args:
            group_name: 'all', 'pan_elbow', 'tilt_wrist', etc.
            
        Returns: All servos in group move to angle
        """
```

### Offset Management API

```python
def set_servo_offsets(self, offsets: list[int]) -> None:
    """Set absolute calibration offsets for all 4 servos
    
    Args:
        offsets: [servo1_offset, servo2_offset, servo3_offset, servo4_offset]
                 Each value: -500 to +500
    
    Example:
        controller.set_servo_offsets([10, 0, -15, 5])
    """
    
def adjust_servo_offsets(self, offset_deltas: list[int]) -> None:
    """Adjust all servo offsets by delta values (fine-tuning)
    
    Args:
        offset_deltas: [delta1, delta2, delta3, delta4]
                       Applied to current offsets
    
    Example:
        controller.adjust_servo_offsets([2, 0, -3, 0])  # Fine-tune
    """
    
def get_servo_offsets(self) -> list[int]:
    """Get current offsets of all 4 servos"""
    
def read_servo_angle(self, servo_id: int) -> int | None:
    """Query current angle of servo (with offset applied)"""
    
def ping_servo(self, servo_id: int) -> bool:
    """Check if servo responding"""
    
def enable_servos(self, servo_ids: list[int] = None) -> list[bool]:
    """Enable servo torque (None = all servos)"""
    
def disable_servos(self, servo_ids: list[int] = None) -> list[bool]:
    """Disable servo torque / freewheel (None = all servos)"""
```

### Servo Groups

```python
# Predefined groups for motion planning:

SERVO_GROUPS = {
    'all': [1, 2, 3, 4],
    'pan_elbow': [1, 3],      # Left column (55S bus)
    'tilt_wrist': [2, 4],     # Right column (FF bus)
    'vertical': [1, 2],        # Pan & Tilt (shoulders)
    'gripper': [3, 4],         # Elbow & Wrist (end effector)
}

# Usage:
servo.move_group('vertical', angle=400, duration=1000)
servo.move_group('all', angle=500, duration=2000)
```

## Calibration Workflow

### Stage 1: Initialize Position

Move all servos to neutral angles for manual adjustment.

```python
from chessrobotclasses import ServoController

servo = ServoController()

# Move to neutral
servo.move_servos([
    (1, 500, 1000),  # Pan to neutral
    (2, 512, 1000),  # Tilt to neutral
    (3, 650, 1000),  # Elbow to neutral
    (4, 2400, 1000), # Wrist to neutral
])

print("Servos at neutral. Manually adjust as needed...")
input("Press Enter when done adjusting")
```

### Stage 2: Measure & Set Offsets

After manual adjustment, read current angles and set offsets.

```python
# Read actual positions
angles = [
    servo.read_servo_angle(i)
    for i in range(1, 5)
]

print(f"Current angles: {angles}")

# Calculate offsets (target_angle - actual_angle)
offsets = [
    500 - angles[0],    # Servo 1
    512 - angles[1],    # Servo 2
    650 - angles[2],    # Servo 3
    2400 - angles[3],   # Servo 4
]

# Set offsets
servo.set_servo_offsets(offsets)
print(f"Offsets set: {offsets}")

# Verify by moving to neutral again
servo.move_servos([
    (1, 500, 500),
    (2, 512, 500),
    (3, 650, 500),
    (4, 2400, 500),
])
```

### UI Integration (Two-Stage)

Accessible through web UI at `http://localhost:8000`:

**Stage 1: Blue Button - "Init Calibration"**
```
→ Moves all servos to neutral
→ User manually adjusts for alignment
→ Waits for Stage 2 trigger
```

**Stage 2: Purple Button - "Confirm Offsets"**
```
→ Reads current angles
→ Calculates offsets
→ Moves to neutral again to verify
```

## Servo Health & Maintenance

### Health Checking

```python
def health_check(servo_id: int) -> dict:
    """Run diagnostic on servo"""
    return {
        'responsive': servo.ping(servo_id),
        'current_angle': servo.read_servo_angle(servo_id),
        'temperature': servo.read_temp(servo_id),
        'voltage': servo.read_voltage(servo_id),
    }

# Check all servos
for i in range(1, 5):
    status = health_check(i)
    print(f"Servo {i}: {status}")
```

### Common Issues

| Issue | Symptom | Debug | Fix |
|-------|---------|-------|-----|
| **Not responsive** | `ping()` returns False | Wrong port/baud | Check serial connections |
| **Servo jitter** | Oscillation at target | Offset miscalibrated | Re-run calibration |
| **Movement too slow** | Commands lag | Duration too long | Reduce duration param |
| **Servo won't move** | No response to `move_servo()` | ID mismatch | Run servo_id_probe.py |
| **Sudden jump** | Angle spike | Offset overflow | Clamp offset to ±500 |

## Testing & Verification

### Quick Tests

```bash
# Probe servo IDs on networks
python servo_id_broadcast_probe.py
# → Output: Connected IDs for each bus

# Move specific servo
python servo_move.py --id 1 --angle 500 --duration 1000

# Set servo ID
python servo_id_set.py --old_id 1 --new_id 5
# → Requires power cycle after

# Ping servo
python servo_ping.py --id 2
# → Returns [OK] or [FAIL]
```

### Integration Test

```bash
python -c "
from chessrobotclasses import ServoController

servo = ServoController()

# Test each servo
for i in range(1, 5):
    print(f'Testing Servo {i}...')
    if servo.ping(servo_id=i):
        print(f'  ✓ Responding')
        servo.move_servo(i, servo.servo_limits[i-1].neutral_angle, 500)
        print(f'  ✓ Moved to neutral')
    else:
        print(f'  ✗ Not responding')
"
```

## Performance Metrics

| Metric | 55S Bus | FF Bus |
|--------|---------|--------|
| **Baud Rate** | 115200 | 1000000 |
| **Command Latency** | 50-100 ms | 5-10 ms |
| **Response Time** | 20-50 ms | <5 ms |
| **Max Rotation** | 300° | 300° |
| **Torque (nominal)** | 30 kg-cm | 30 kg-cm |
| **Operating Voltage** | 4.8-7.2V | 4.8-7.2V |

## Troubleshooting Guide

### Communication Issues

**Problem**: "Connection refused" on serial port
```
→ Check UDEV rules on Linux:
  ls -la /dev/ttyUSB* /dev/ttyAMA*
→ Verify baud rates in ServoController init
→ On Windows, ensure COM ports exist in Device Manager
```

**Problem**: Servos not responding after ID change
```
→ Power cycle robot (servos persist new ID on restart)
→ Use broadcast probe to verify new ID: servo_id_broadcast_probe.py
→ If stuck, contact servo manufacturer for factory reset
```

### Motion Issues

**Problem**: Servo moves to wrong angle
```
→ Verify offset: servo.get_servo_offsets()
→ Check angle limits: servo.servo_limits[i].min_angle
→ Re-run calibration for drift correction
```

**Problem**: Movement jerky / discontinuous
```
→ Increase duration (slower = smoother)
→ Check for offset overflow (|offset| > 500)
→ Verify servo isn't mechanically stuck
```

## Dependencies

- `pyserial >= 3.5`
- Python 3.8+
- Hardware: Raspberry Pi or compatible Linux single-board computer
- Recommended: Oscilloscope for timing verification

## References

- Hiwonder 55S Documentation: [Manufacturer docs]
- Hiwonder FF Documentation: [Manufacturer docs]
- Serial Protocol Analysis: See `BusServo.py` for raw packet structures
