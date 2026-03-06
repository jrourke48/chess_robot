from __future__ import annotations
from dataclasses import dataclass

@dataclass
class ServoLimits:
    # position units are 0..1000 by default (typical Hiwonder mapping)
    pos_min: int = 0
    pos_max: int = 4095
    #
    range: int = 240
    #servo limits
    min_angle: int = 0
    max_angle: int = 1200
    #angle_offset
    offset: int = 0
    # safety / tuning knobs
    default_move_time_ms: int = 50
    max_move_time_ms: int = 1000

    # motor mode speed limits (-1000..1000)
    motor_speed_min: int = -1000
    motor_speed_max: int = 1000

# Conservative presets (tune as needed)
WRISTLIMITS_HX_10HM = ServoLimits(min_angle=1200, max_angle=3600)
ELBOWLIMITS_HX_35HM = ServoLimits(pos_max=1000, min_angle=500, max_angle=1000)
SHOULDERLIMITS_HX_65HM = ServoLimits()
SHOULDERLIMITS_HX_35HM = ServoLimits(pos_max=1000, min_angle=120, max_angle=880)