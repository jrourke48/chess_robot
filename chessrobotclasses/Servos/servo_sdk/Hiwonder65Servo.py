# chessrobotclasses/Servos/ff_bus.py
from __future__ import annotations
from dataclasses import dataclass
from typing import Optional, Iterable, Dict, List, Tuple
import time
import serial
import threading

# ---------------------------
# FF protocol constants
# ---------------------------
FF_H0 = 0xFF
FF_H1 = 0xFF

ID_BROADCAST = 0xFE  # 254

INST_PING      = 0x01
INST_READ      = 0x02
INST_WRITE     = 0x03
INST_REG_WRITE = 0x04
INST_ACTION    = 0x05
INST_SYNC_READ = 0x82
INST_SYNC_WRITE= 0x83
INST_RESET     = 0x06

# The goal block at 0x2A is 6 bytes: pos(2), time(2), speed(2)
GOAL_BLOCK_LEN = 6

#baudrate define
SERVO_1M = 0
SERVO_0_5M = 1
SERVO_250K = 2
SERVO_128K = 3
SERVO_115200 = 4
SERVO_76800 = 5
SERVO_57600 = 6
SERVO_38400 = 7

#Memory table

#NVS
SERVO_MAIN_VERSION = 3
SERVO_SEC_VERSION = 4
SERVO_ID = 5
SERVO_BAUD_RATE = 6
SERVO_CW_DEAD = 26
SERVO_CCW_DEAD = 27
SERVO_POS_OFFSET_L = 31
SERVO_POS_OFFSET_H = 32
SERVO_MODE = 33

#SRAM

#Write read
SERVO_TORQUE_ENABLE = 40
SERVO_ACC = 41
SERVO_GOAL_POSITION_L = 42
SERVO_GOAL_POSITION_H = 43
SERVO_PWM_SPEED_L = 44
SERVO_PWM_SPEED_H = 45
SERVO_GOAL_SPEED_L = 46
SERVO_GOAL_SPEED_H = 47
SERVO_MAX_TORQUE_L = 48
SERVO_MAX_TORQUE_H = 49

#Only read
SERVO_PRESENT_POSITION_L = 56
SERVO_PRESENT_POSITION_H = 57
SERVO_PRESENT_SPEED_L = 58
SERVO_PRESENT_SPEED_H = 59
SERVO_PRESENT_LOAD_L = 60
SERVO_PRESENT_LOAD_H = 61
SERVO_PRESENT_VOLTAGE = 62
SERVO_PRESENT_TEMPERATURE = 63
SERVO_MOVING_STATUS = 66
SERVO_PRESENT_CURRENT_L = 69
SERVO_PRESENT_CURRENT_H = 70


# =============================================================================
# PACKET / PARSING
# =============================================================================

# Compute protocol checksum from ID, length, instruction, and parameter bytes.
# Checksum = bitwise NOT of the low 8 bits of the total sum.
def _checksum(sid: int, length: int, inst: int, params: List[int]) -> int:
    total = (sid + length + inst + sum(params)) & 0xFF
    return (~total) & 0xFF


# Build one outbound FF frame:
# [0xFF][0xFF][ID][Length][Inst][PARAMS...][CHK]
def _build_packet(sid: int, inst: int, params: Optional[List[int]] = None) -> bytes:
    if params is None:
        params = []
    # Length = Inst + Params + Checksum
    length = len(params) + 2
    chk = _checksum(sid, length, inst, params)
    pkt = [FF_H0, FF_H1, sid & 0xFF, length & 0xFF, inst & 0xFF, *params, chk & 0xFF]
    return bytes(pkt)


# Scan a raw byte buffer and extract all complete FF protocol frames.
# Uses header resync by searching for 0xFF 0xFF.
def _extract_frames(buf: bytes) -> List[bytes]:
    """
    Returns list of raw frames:
      FF FF ID LEN INST ... CHK
    """
    frames: List[bytes] = []
    i = 0
    n = len(buf)
    while i + 4 < n:
        if buf[i] == FF_H0 and buf[i + 1] == FF_H1:
            if i + 3 >= n:
                break
            length = buf[i + 3]
            # Total frame size = header(2) + ID(1) + LEN(1) + LEN payload bytes
            frame_len = 2 + 1 + 1 + length
            if i + frame_len <= n:
                frames.append(buf[i:i + frame_len])
                i += frame_len
                continue
        i += 1
    return frames


# Parse a single frame into (sid, status, params) and verify checksum validity.
def _parse_frame(frame: bytes) -> Tuple[Optional[int], Optional[int], List[int], bool]:
    """
    Returns (sid, status, params, ok)
    """
    if len(frame) < 6:
        return None, None, [], False
    sid = frame[2]
    length = frame[3]
    status = frame[4]
    params = list(frame[5:-1])
    chk = frame[-1]
    ok = (chk == _checksum(sid, length, status, params))
    return sid, status, params, ok


# Return the low byte of a 16-bit value.
def _lo(x: int) -> int:
    return x & 0xFF


# Return the high byte of a 16-bit value.
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

class HiwonderFFServo:
    """
    Owns a serial port and provides robust, thread-safe packet IO for FF protocol.
    Expects a pyserial Serial object configured for correct COM port + baud.
    """

    def __init__(self, serial_port):
        self.serial = serial_port
        self._lock = threading.Lock()

    # -------------------------
    # Low-level IO
    # -------------------------

    def _send(self, sid: int, inst: int, params: Optional[List[int]] = None) -> int:
        pkt = _build_packet(sid, inst, params)
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

    def send_and_get_one(self, sid: int, inst: int, params: Optional[List[int]] = None, timeout_s: float = 0.08):
        """
        Send a command and attempt to read exactly one valid response frame for that servo ID.
        Returns (sid, status, params, ok). If nothing: (None, None, [], False)
        """
        t0 = time.time()
        buf = bytearray()
        while (time.time() - t0) < timeout_s:
            buf.extend(self._read_available())
            time.sleep(0.001)

        frames = _extract_frames(bytes(buf))
        for f in frames:
            psid, pstatus, pparams, ok = _parse_frame(f)
            if ok and psid == sid:
                return psid, pstatus, pparams, True

        return None, None, [], False


    # =============================================================================
    # High-level protocol commands
    # =============================================================================

    # Enable or disable motor torque output.
    def Motor_on(self, sid: int, enable: bool) -> int:
        state = 1 if enable else 0
        with self._lock:
            return self._send(sid, INST_WRITE, [SERVO_TORQUE_ENABLE, state])
    def move_stop(self, sid: int) -> int:
        """
        Stop motor movement.
        """
        with self._lock:
            return self._send(sid, INST_WRITE, [SERVO_GOAL_SPEED_L, 0, 0, 0, 0, 0, 0])

    # Read ID from a specific servo.
    def id_read_one(self, sid: int) -> Tuple[int, bool]:
        """
        Read ID from a specific servo.
        """
        with self._lock:
            try:
                self.serial.reset_input_buffer()
            except Exception:
                pass

            self._send(sid, INST_READ, [SERVO_ID, 1])
            rsid, status, params, ok = self.send_and_get_one(sid, INST_READ)

        if ok and status == 0 and len(params) >= 1:
            return params[0], True
        return 0, False

    # Write a new ID to a servo.
    def id_write(self, sid: int, new_id: int) -> int:
        new_id = max(0, min(253, int(new_id)))
        with self._lock:
            return self._send(sid, INST_WRITE, [SERVO_ID, new_id])

    # Read present position (SRAM 56/57).
    def pos_read(self, sid: int) -> Tuple[int, bool]:
        with self._lock:
            try:
                self.serial.reset_input_buffer()
            except Exception:
                pass

            self._send(sid, INST_READ, [SERVO_PRESENT_POSITION_L, 2])
            rsid, status, params, ok = self.send_and_get_one(sid, INST_READ)

        if ok and status == 0 and len(params) >= 2:
            pos = _word(params[0], params[1])
            return pos, True
        return 0, False

    # Write goal position (SRAM 42/43).
    def move_time(self, sid: int, position: int, time_ms: int, speed: int = 1000) -> int:
        position = max(0, min(4095, int(position)))
        time_ms = max(0, min(30000, int(time_ms)))
        speed = max(0, min(3400, int(speed)))
        # 0x2A goal block expects 6 bytes: pos_l, pos_h, time_l, time_h, speed_l, speed_h
        params = [
            SERVO_GOAL_POSITION_L,
            _lo(position), _hi(position),
            _lo(time_ms), _hi(time_ms),
            _lo(speed), _hi(speed),
        ]
        with self._lock:
            return self._send(sid, INST_WRITE, params)

    # Read present temperature (SRAM 63).
    def temp_read(self, sid: int) -> Tuple[int, bool]:
        with self._lock:
            try:
                self.serial.reset_input_buffer()
            except Exception:
                pass

            self._send(sid, INST_READ, [SERVO_PRESENT_TEMPERATURE, 1])
            rsid, status, params, ok = self.send_and_get_one(sid, INST_READ)

        if ok and status == 0 and len(params) >= 1:
            return params[0], True
        return 0, False

    # Read supply voltage (SRAM 62).
    def vin_read(self, sid: int) -> Tuple[int, bool]:
        with self._lock:
            try:
                self.serial.reset_input_buffer()
            except Exception:
                pass

            self._send(sid, INST_READ, [SERVO_PRESENT_VOLTAGE, 1])
            rsid, status, params, ok = self.send_and_get_one(sid, INST_READ)

        if ok and status == 0 and len(params) >= 1:
            return params[0], True
        return 0, False

    # Set motor acceleration (SRAM 41).
    def set_acc(self, sid: int, acc: int) -> int:
        acc = max(0, min(255, int(acc)))
        with self._lock:
            return self._send(sid, INST_WRITE, [SERVO_ACC, acc])

    # Set goal speed (SRAM 46/47).
    def set_goal_speed(self, sid: int, speed: int) -> int:
        speed = max(0, min(3400, int(speed)))
        with self._lock:
            return self._send(sid, INST_WRITE, [SERVO_GOAL_SPEED_L, _lo(speed), _hi(speed)])

    # Read present speed (SRAM 58/59).
    def speed_read(self, sid: int) -> Tuple[int, bool]:
        with self._lock:
            try:
                self.serial.reset_input_buffer()
            except Exception:
                pass

            self._send(sid, INST_READ, [SERVO_PRESENT_SPEED_L, 2])
            rsid, status, params, ok = self.send_and_get_one(sid, INST_READ)

        if ok and status == 0 and len(params) >= 2:
            spd = _word(params[0], params[1])
            return spd, True
        return 0, False




