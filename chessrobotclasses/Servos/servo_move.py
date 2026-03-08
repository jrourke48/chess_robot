"""
servo_move.py - Test suite for Hiwonder servo drivers.
"""

from __future__ import annotations
import serial
import time
from servo_sdk.Hiwonder35Servo import Hiwonder55Servo
from servo_sdk.Hiwonder65Servo import HiwonderFFServo
from servo_sdk.BusServo import BusServo
from servo_sdk.Servo_models import ServoLimits
# Conservative presets (tune as needed)
WRISTLIMITS_HX_10HM = ServoLimits(min_angle=1200, max_angle=3600)
ELBOWLIMITS_HX_35HM = ServoLimits(pos_max=1000, min_angle=500, max_angle=1000)
SHOULDERLIMITS_HX_65HM = ServoLimits(min_angle=0, max_angle=4095)
SHOULDERLIMITS_HX_35HM = ServoLimits(pos_max=1000, min_angle=120, max_angle=880)


def test_35_series(port: str = "COM4", baudrate: int = 115200):
    """Test 35-series (0x55 protocol) servos."""
    print("\n" + "="*60)
    print("Testing 35-Series Servos (0x55 Protocol)")
    print("="*60)
    
    try:
        ser = serial.Serial(port, baudrate, timeout=0.1)
        driver = Hiwonder55Servo(ser)
        
        # Wrap in BusServo objects
        servos = [BusServo(driver, 15, SHOULDERLIMITS_HX_35HM, f"35_Servo_15")]
        
        # Test each servo
        for servo in servos:
            print(f"\n--- Testing {servo} ---")
            
            servo.print_status()
            
            # Enable motor
            print(f"\nEnabling {servo.name}...")
            servo.motor_on()
            time.sleep(0.1)
            
            # Move servo
            print(f"Moving {servo.name} to position 0 in 50ms...")
            servo.move(000, 50)
            time.sleep(1.5)
            
            # Read final position
            servo.print_status()
            
            # Disable motor
            print(f"\nDisabling {servo.name}...")
            servo.motor_off()
        
        ser.close()
        print("\n✓ 35-Series test complete")
        
    except Exception as e:
        print(f"✗ Error in 35-series test: {e}")
        import traceback
        traceback.print_exc()


def test_65_series(port: str = "COM4", baudrate: int = 1000000):
    """Test 65-series (0xFF protocol) servos."""
    print("\n" + "="*60)
    print("Testing 65-Series Servos (0xFF Protocol)")
    print("="*60)
    
    try:
        ser = serial.Serial(port, baudrate, timeout=0.1)
        driver = HiwonderFFServo(ser)
        
        # Wrap in BusServo objects
        servos = [BusServo(driver, 1, SHOULDERLIMITS_HX_65HM, f"65_Servo_23")]
        
        # Test each servo
        for servo in servos:
            print(f"\n--- Testing {servo} ---")
            
            servo.print_status()
            
            # Enable motor
            print(f"\nEnabling {servo.name}...")
            servo.motor_on()
            time.sleep(0.1)
            
            # Move servo
            print(f"Moving {servo.name} to position 4000 in 100ms...")
            #servo.move(0, 0)
            time.sleep(1.5)
            
            # Read final position
            servo.print_status()
            
            # Disable motor
            print(f"\nDisabling {servo.name}...")
            servo.motor_off()
        
        ser.close()
        print("\n✓ 65-Series test complete")
        
    except Exception as e:
        print(f"✗ Error in 65-series test: {e}")
        import traceback
        traceback.print_exc()


def test_both_series():
    """Run both test suites."""
    test_35_series("COM3", 115200)
    test_65_series("COM4", 115200)


if __name__ == "__main__":
    print("Hiwonder Servo SDK Test Suite")
    print("=" * 60)
    arg = input("input which series to test (35/65/both): ")
    
    if arg == "35":
        test_35_series()
    elif arg == "65":
        test_65_series()
    elif arg == "both":
        test_both_series()
    else:
        test_both_series()
