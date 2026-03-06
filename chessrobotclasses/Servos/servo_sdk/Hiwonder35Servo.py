"""
hx_servo.py - Hiwonder Bus Servo Driver (0x55 protocol)

Robust, thread-safe implementation of the Hiwonder/LX/HX "0x55 0x55" bus-servo protocol.

Key reliability features added:
- Thread lock around send/receive (prevents interleaving from multiple threads)
- Resync frame parser (searches for 0x55 0x55 anywhere in stream)
- Read window collector for commands that can return multiple frames (broadcast ID read)
- Convenience "discover_ids" for scanning bus safely

Packet format:
[0x55][0x55][ID][Length][Cmd][Params...][Checksum]
Checksum = ~(ID + Length + Cmd + sum(Params)) & 0xFF
"""

from __future__ import annotations

import time
import threading
from typing import List, Tuple, Optional, Set

# =============================================================================
# PROTOCOL CONSTANTS
# =============================================================================
SERVO_HEADER = 0x55
SERVO_BROADCAST_ID = 254  # 0xFE

# =============================================================================
# COMMANDS (common Hiwonder bus-servo set)
# =============================================================================
# Write
SERVO_MOVE_TIME_WRITE = 1
SERVO_MOVE_TIME_WAIT_WRITE = 7
SERVO_MOVE_START = 11
SERVO_MOVE_STOP = 12

SERVO_ID_WRITE = 13

SERVO_ANGLE_OFFSET_ADJUST = 17
SERVO_ANGLE_OFFSET_WRITE = 18

SERVO_ANGLE_LIMIT_WRITE = 20
SERVO_VIN_LIMIT_WRITE = 22
SERVO_TEMP_MAX_LIMIT_WRITE = 24

SERVO_OR_MOTOR_MODE_WRITE = 29
SERVO_LOAD_OR_UNLOAD_WRITE = 31

SERVO_LED_CTRL_WRITE = 33
SERVO_LED_ERROR_WRITE = 35

# Read
SERVO_MOVE_TIME_READ = 2
SERVO_MOVE_TIME_WAIT_READ = 8
SERVO_ID_READ = 14
SERVO_ANGLE_OFFSET_READ = 19
SERVO_ANGLE_LIMIT_READ = 21
SERVO_VIN_LIMIT_READ = 23
SERVO_TEMP_MAX_LIMIT_READ = 25
SERVO_TEMP_READ = 26
SERVO_VIN_READ = 27
SERVO_POS_READ = 28
SERVO_OR_MOTOR_MODE_READ = 30
SERVO_LOAD_OR_UNLOAD_READ = 32
SERVO_LED_CTRL_READ = 34
SERVO_LED_ERROR_READ = 36

# =============================================================================
# PACKET / PARSING
# =============================================================================


# Compute protocol checksum from ID, LEN, CMD, and parameter bytes.
# Checksum = bitwise NOT of the low 8 bits of the total sum.
def _checksum(sid: int, length: int, cmd: int, params: List[int]) -> int:
    total = (sid + length + cmd + sum(params)) & 0xFF
    return (~total) & 0xFF


# Build one outbound Hiwonder frame:
# [0x55][0x55][ID][LEN][CMD][PARAMS...][CHK]
def _build_packet(sid: int, cmd: int, params: Optional[List[int]] = None) -> bytes:
    if params is None:
        params = []
    length = len(params) + 3  # LEN = CMD + PARAMS + CHK
    chk = _checksum(sid, length, cmd, params)
    pkt = [SERVO_HEADER, SERVO_HEADER, sid & 0xFF, length & 0xFF, cmd & 0xFF, *params, chk & 0xFF]
    return bytes(pkt)


# Scan a raw byte buffer and extract all complete protocol frames.
# Uses header resync by searching for 0x55 0x55.
def _extract_frames(buf: bytes) -> List[bytes]:
    """
    Returns list of raw frames:
      55 55 ID LEN CMD ... CHK
    """
    frames: List[bytes] = []
    i = 0
    n = len(buf)
    while i + 5 < n:
        if buf[i] == SERVO_HEADER and buf[i + 1] == SERVO_HEADER:
            if i + 4 >= n:
                break
            length = buf[i + 3]
            # Hiwonder 0x55 protocol stores LEN as: LEN + CMD + PARAMS + CHK.
            # So total frame bytes = header(2) + ID(1) + LEN bytes.
            frame_len = 3 + length
            if i + frame_len <= n:
                frames.append(buf[i:i + frame_len])
                i += frame_len
                continue
        i += 1
    return frames


# Parse a single frame into (sid, cmd, params) and verify checksum validity.
def _parse_frame(frame: bytes) -> Tuple[Optional[int], Optional[int], List[int], bool]:
    """
    Returns (sid, cmd, params, ok)
    """
    if len(frame) < 7:
        return None, None, [], False
    sid = frame[2]
    length = frame[3]
    cmd = frame[4]
    params = list(frame[5:-1])
    chk = frame[-1]
    ok = (chk == _checksum(sid, length, cmd, params))
    return sid, cmd, params, ok


# Return the low byte of an integer.
def _lo(x: int) -> int:
    return x & 0xFF


# Return the high byte of an integer.
def _hi(x: int) -> int:
    return (x >> 8) & 0xFF


# Combine low and high bytes into a 16-bit unsigned value.
def _word(lo_b: int, hi_b: int) -> int:
    return (hi_b << 8) | lo_b


# Convert unsigned 16-bit value to signed int16.
def _to_signed_short(u16: int) -> int:
    return u16 - 65536 if u16 > 32767 else u16


# Convert signed byte-style value to unsigned byte (0..255).
def _to_unsigned_byte(s8: int) -> int:
    return (s8 + 256) & 0xFF if s8 < 0 else s8 & 0xFF


# Convert unsigned byte (0..255) to signed int8 (-128..127).
def _to_signed_byte(u8: int) -> int:
    return u8 - 256 if u8 > 127 else u8


# =============================================================================
# MAIN HANDLER
# =============================================================================

class Hiwonder55Servo:
    """
    Owns a serial port and provides robust, thread-safe packet IO.
    Expects a pyserial Serial object configured for correct COM port + baud.
    """

    def __init__(self, serial_port):
        self.serial = serial_port
        self._lock = threading.Lock()

    # -------------------------
    # Low-level IO
    # -------------------------

    def _send(self, sid: int, cmd: int, params: Optional[List[int]] = None) -> int:
        pkt = _build_packet(sid, cmd, params)
        return self.serial.write(pkt)

    def _read_available(self, max_bytes: int = 1024) -> bytes:
        n = getattr(self.serial, "in_waiting", 0)
        if not n:
            return b""
        return self.serial.read(min(n, max_bytes))

    def read_frames_for(self, duration_s: float = 0.06) -> List[Tuple[Optional[int], Optional[int], List[int], bool]]:
        """
        Collect bytes for a short window and parse all frames found.
        Useful for broadcast responses (multiple frames).
        """
        t0 = time.time()
        buf = bytearray()
        while (time.time() - t0) < duration_s:
            buf.extend(self._read_available())
            time.sleep(0.001)
        frames = _extract_frames(bytes(buf))
        return [_parse_frame(f) for f in frames]

    def send_and_get_one(self, sid: int, cmd: int, params: Optional[List[int]] = None, timeout_s: float = 0.08):
        """
        Send a command and attempt to read exactly one valid response frame for that servo ID.
        Returns (sid, cmd, params, ok). If nothing: (None, None, [], False)
        """
        t0 = time.time()
        buf = bytearray()
        while (time.time() - t0) < timeout_s:
            buf.extend(self._read_available())
            time.sleep(0.001)

        frames = _extract_frames(bytes(buf))
        for f in frames:
            psid, pcmd, pparams, ok = _parse_frame(f)
            if ok and psid == sid and pcmd == cmd:
                return psid, pcmd, pparams, True

        return None, None, [], False

    # =============================================================================
    # High-level protocol commands
    # =============================================================================
    def Motor_on(self, sid: int, enable: bool) -> int:
        # load/unload: 1=load (torque on), 0=unload
        state = 1 if enable else 0
        with self._lock:
            return self._send(sid, SERVO_LOAD_OR_UNLOAD_WRITE, [state])
    
    #immediately will stop servo
    def move_stop(self, sid: int) -> int:
        with self._lock:
            return self._send(sid, SERVO_MOVE_STOP)

    #move the servo with a given id a certain position in a certain time 
    def move_time(self, sid: int, position: int, time_ms: int) -> int:
        position = max(0, min(1000, int(position)))
        time_ms = max(0, min(30000, int(time_ms)))
        params = [_lo(position), _hi(position), _lo(time_ms), _hi(time_ms)]
        with self._lock:
            return self._send(sid, SERVO_MOVE_TIME_WRITE, params)

    #write to the Servo with x Id a new Servo ID
    def id_write(self, sid: int, new_id: int) -> int:
        new_id = max(0, min(253, int(new_id)))
        with self._lock:
            return self._send(sid, SERVO_ID_WRITE, [new_id])

    #Read Servo ID
    def id_read_one(self, sid: int) -> Tuple[int, bool]:
        """
        Read ID from a specific servo (safer than broadcast).
        """
        with self._lock:
            try:
                self.serial.reset_input_buffer()
            except Exception:
                pass

            self._send(sid, SERVO_ID_READ)
            rsid, rcmd, params, ok = self.send_and_get_one(sid, SERVO_ID_READ)

        if ok and params:
            return params[0], True
        return 0, False

    #read motor position
    def pos_read(self, sid: int) -> Tuple[int, bool]:
        with self._lock:
            try:
                self.serial.reset_input_buffer()
            except Exception:
                pass

            self._send(sid, SERVO_POS_READ)
            rsid, rcmd, params, ok = self.send_and_get_one(sid, SERVO_POS_READ)

        if ok and len(params) >= 2:
            pos = _to_signed_short(_word(params[0], params[1]))
            return pos, True
        return 0, False

    #read motor Temperature
    def temp_read(self, sid: int) -> Tuple[int, bool]:
        with self._lock:
            try:
                self.serial.reset_input_buffer()
            except Exception:
                pass
            self._send(sid, SERVO_TEMP_READ)
            rsid, rcmd, params, ok = self.send_and_get_one(sid, SERVO_TEMP_READ)

        if ok and len(params) >= 1:
            return params[0], True
        return 0, False

    #read motor Voltage
    def vin_read(self, sid: int) -> Tuple[int, bool]:
        with self._lock:
            try:
                self.serial.reset_input_buffer()
            except Exception:
                pass
            self._send(sid, SERVO_VIN_READ)
            rsid, rcmd, params, ok = self.send_and_get_one(sid, SERVO_VIN_READ)

        if ok and len(params) >= 2:
            mv = _word(params[0], params[1])
            return mv, True
        return 0, False
        
    #set the motor mode to servo(position mode) or motor (speed) mode
    def motor_mode(self, sid: int, enable: bool, speed: int = 0) -> int:
        """
        mode=0 position, mode=1 motor
        speed is signed (-1000..1000) when in motor mode
        """
        mode = 1 if enable else 0
        speed = max(-1000, min(1000, int(speed)))
        if speed < 0:
            speed_u = speed + 65536
        else:
            speed_u = speed

        params = [mode, 0, _lo(speed_u), _hi(speed_u)]
        with self._lock:
            return self._send(sid, SERVO_OR_MOTOR_MODE_WRITE, params)
        
    #write the offset angle for the motor must be with in -30 to 30 degrees or -125 to 125 bits in the 10 bit number 
    def angle_offset_write(self, sid: int, offset: int) -> int:
        offset = max(-125, min(125, int(offset)))
        with self._lock:
            return self._send(sid, SERVO_ANGLE_OFFSET_WRITE, [_to_unsigned_byte(offset)])
        
    #read the angular offset
    def angle_offset_read(self, sid: int) -> Tuple[int, bool]:
        with self._lock:
            try:
                self.serial.reset_input_buffer()
            except Exception:
                pass
            self._send(sid, SERVO_ANGLE_OFFSET_READ)
            sid, rcmd, params, ok = self.send_and_get_one(sid, SERVO_ANGLE_OFFSET_READ)

        if ok and len(params) >= 1:
            return _to_signed_byte(params[0]), True
        return 0, False
    
    #write servo angle limits
    def servo_angle_limit_write(self, sid: int, min_angle: int, max_angle: int) -> int:
        min_angle = max(0, min(1000, int(min_angle)))
        max_angle = max(0, min(1000, int(max_angle)))
        with self._lock:
            return self._send(sid, SERVO_ANGLE_LIMIT_WRITE, [_to_unsigned_byte(min_angle), _to_unsigned_byte(max_angle)])

    #read the servo angle limits 
    def servo_angle_limit_read(self, sid: int) -> Tuple[int, int, bool]:
        with self._lock:
            try:
                self.serial.reset_input_buffer()
            except Exception:
                pass
            self._send(sid, SERVO_ANGLE_LIMIT_READ)
            sid, rcmd, params, ok = self.send_and_get_one(sid, SERVO_ANGLE_LIMIT_READ)

        if ok and len(params) >= 2:
            return _to_signed_byte(params[0]), _to_signed_byte(params[1]), True
        return 0, 0, False
