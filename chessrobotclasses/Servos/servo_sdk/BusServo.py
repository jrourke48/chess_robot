"""
BusServo.py - Single bus servo wrapper class.

Provides a clean object-oriented interface for controlling one servo
via either 0x55 or 0xFF protocol.
"""

from __future__ import annotations
from typing import Optional, Tuple
from .Servo_models import ServoLimits


class BusServo:
    """
    Represents a single bus servo on a bus.
    Works with any protocol (0x55 or 0xFF).
    """

    def __init__(self, protocol_driver, servo_id, limits: ServoLimits, name: str = None):
        """
        Initialize a single servo.
        
        Args:
            protocol_driver: Hiwonder55BusServo or HiwonderFFServo instance
            servo_id: ID of this servo on the bus
            limits: ServoLimits instance containing offset, min_angle, max_angle, pos_min, and pos_max
            name: Optional friendly name (e.g., "Arm_Joint_1")

            driver: communication protocol
            id: servo id
            name: specific servo name
            min_angle: Minimum allowed position set by user (raw units)
            max_angle: Maximum allowed position set by user (raw units)
            max_pos: Maximum allowed position for the servo (raw units)
            min_pos: Minimum allowed position for the servo (raw units)

        """

        self.driver = protocol_driver
        self.id = servo_id
        self.name = name or f"Servo_{servo_id}"
        self.offset = limits.offset
        self.min_angle = limits.min_angle
        self.max_angle = limits.max_angle
        self.max_pos = limits.pos_max
        self.min_pos = limits.pos_min
        self.range  = limits.range

    def __repr__(self) -> str:
        return f"BusServo(id={self.id}, name={self.name})"

    # =========================================================================
    # Motor Control
    # =========================================================================
    # =========================================================================
    # Helpers
    # =========================================================================
    #
    def radian2raw_number(self, radian: float) -> int:
        """
        Convert radians to raw servo units.
        
        Args:
            radian: Angle in radians
            
        Returns:
            Raw servo units
        """
        # Example conversion, adjust based on servo type
        return int(radian * (self.max_pos - self.min_pos) / (2 * 3.141592 * self.range / 360) + self.offset)

    def raw_number2radian(self, raw: int) -> float:
        """
        Convert raw servo units to radians.
        
        Args:
            raw: Raw servo units
            
        Returns:
            Angle in radians
        """
        return (raw - self.offset) * (2 * 3.141592 * self.range / 360) / (self.max_pos - self.min_pos)
    def motor_on(self) -> int:
        """Enable motor torque output."""
        return self.driver.Motor_on(self.id, True)

    def motor_off(self) -> int:
        """Disable motor torque output."""
        return self.driver.Motor_on(self.id, False)

    def motor_stop(self) -> int:
        """Stop motor movement."""
        return self.driver.move_stop(self.id)

    def move_rad(self, position: float, time_ms: int) -> int:
        """
        Move servo to position in specified time.
        
        Args:
            position: Target position in radians
            time_ms: Time to move in milliseconds
            
        Returns:
            Bytes written to serial port
        """
        # Convert radians to raw servo units
        raw_position = self.radian2raw_number(position)
        # Clamp positions to the servo's limits
        if raw_position > self.max_angle:
            raw_position = self.max_angle
        elif raw_position < self.min_angle:
            raw_position = self.min_angle
        print(f"{self.name}: Moving to {raw_position} (raw units)")
        return self.driver.move_time(self.id, raw_position, time_ms)
    def move(self, position: int, time_ms: int) -> int:
        """
        Move servo to position in specified time.
        
        Args:
            position: Target position in raw units
            time_ms: Time to move in milliseconds
            
        Returns:
            Bytes written to serial port
        """
        # Clamp positions to the servo's limits
        if position > self.max_angle:
            position = self.max_angle
        elif position < self.min_angle:
            position = self.min_angle
        print(f"{self.name}: Moving to {position} (raw units)")
        return self.driver.move_time(self.id, position, time_ms)
    
    # =========================================================================
    # Angle Offset and limits
    # =========================================================================

    def set_offset(self, offset: int) -> None:
        """
        Set angle offset for this servo instance.
        Offset is applied to all move commands and read positions.
        
        Args:
            offset: Offset in raw units (stored locally, not in servo NVS)
        """
        self.offset = offset
        print(f"{self.name}: Offset set to {offset}")
    
    def set_limits(self, min_angle: int, max_angle: int) -> None:
        """
        Set position limits for this servo.
        Move commands will clamp to [min_angle, max_angle].
        
        Args:
            min_angle: Minimum allowed position (raw units)
            max_angle: Maximum allowed position (raw units)
        """
        if min_angle >= max_angle:
            raise ValueError(f"min_angle ({min_angle}) must be < max_angle ({max_angle})")
        self.min_angle = min_angle
        self.max_angle = max_angle
        print(f"{self.name}: Limits set to [{min_angle}, {max_angle}]")

    # =========================================================================
    # Reading Sensor Data
    # =========================================================================

    def read_rad_position(self) -> float:
        """
        Read current position in radians.
        
        Returns:
            (position, ok) - position value and success flag
        """
        raw_position, ok = self.driver.pos_read(self.id)
        if not ok:
            return None
        return self.raw_number2radian(raw_position)
    
    def read_position(self) -> int:
        """
        Read current position in raw units.
        
        Returns:
            (position, ok) - position value and success flag
        """
        pos, ok = self.driver.pos_read(self.id)
        if not ok:
            return None
        return pos 
    
    def read_temperature(self) -> Tuple[int, bool]:
        """
        Read motor temperature in °C.
        
        Returns:
            (temperature, ok) - temperature value and success flag
        """
        return self.driver.temp_read(self.id)

    def read_voltage(self) -> Tuple[int, bool]:
        """
        Read supply voltage in mV.
        
        Returns:
            (voltage_mv, ok) - voltage value and success flag
        """
        return self.driver.vin_read(self.id)

    # =========================================================================
    # ID Management
    # =========================================================================

    def read_id(self) -> Tuple[int, bool]:
        """
        Read servo ID from NVS.
        
        Returns:
            (id, ok) - ID value and success flag
        """
        return self.driver.id_read_one(self.id)

    def write_id(self, new_id: int) -> int:
        """
        Write new ID to servo (NVS - survives power cycle).
        
        WARNING: This will change the servo's ID permanently.
        You must recreate the BusServo with the new ID after this.
        
        Args:
            new_id: New ID (0-253)
            
        Returns:
            Bytes written to serial port
        """
        return self.driver.id_write(self.id, new_id)

    # =========================================================================
    # Status Helpers
    # =========================================================================

    def status(self) -> dict:
        """
        Read all sensor data at once.
        
        Returns:
            Dictionary with position, temperature, voltage, and success flags
        """
        pos = self.read_position()
        temp = self.read_temperature()
        vin = self.read_voltage()
        
        return {
            "id": self.id,
            "name": self.name,
            "position": pos,
            "temperature": temp,
            "voltage_mv": vin,
        }

    def print_status(self):
        """Print servo status to console."""
        s = self.status()
        print(f"\n{self.name} (ID={self.id})")
        print(f"  Position:    {s['position']}")
        print(f"  Temperature: {s['temperature']}°C")
        print(f"  Voltage:     {s['voltage_mv']}mV")