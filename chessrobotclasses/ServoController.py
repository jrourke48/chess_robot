from __future__ import annotations
import serial
import time
from Servos.servo_sdk.Hiwonder35Servo import Hiwonder55Servo
from Servos.servo_sdk.Hiwonder65Servo import HiwonderFFServo
from Servos.servo_sdk.BusServo import BusServo
from Servos.servo_sdk.Servo_models import ServoLimits
# Conservative presets (tune as needed)
WRISTLIMITS_HX_10HM = ServoLimits(min_angle=1200, max_angle=3600)
ELBOWLIMITS_HX_35HM = ServoLimits(pos_max=1000, min_angle=500, max_angle=1000)
SHOULDERLIMITS_HX_65HM = ServoLimits(min_angle=500, max_angle=2200)
SHOULDERLIMITS_HX_35HM = ServoLimits(pos_max=1000, min_angle=120, max_angle=880)

class ServoController:
    # This class is responsible for controlling the servo motors that move the chess pieces.
    # It provides methods to move the servo to a specific position and to stop the servo.
    #they are bus servos that can be controlled via a serial interface, 
    #so the implementation depends on the hiwonder hardware being used.
    #this will control all four servos for the robot arm, 
    #so the position input will be a list of four values representing the desired position for each servo.
    #sub class for single servo
    def __init__(self, port55, portff):
        #serial protocols for the 55-series (orange) and FF-series (black) servos
        ser55 = serial.Serial(port55, 115200, timeout=0.1)
        serff = serial.Serial(portff, 1_000_000, timeout=0.1)
        #create drivers for each protocol
        driver55 = Hiwonder55Servo(ser55)
        driverff = HiwonderFFServo(serff)
        # Wrap in BusServo objects
        self.Wrist_servo = BusServo(driverff, 4, WRISTLIMITS_HX_10HM, f"Wrist_Servo_4")
        self.Elbow_servo = BusServo(driver55, 3, ELBOWLIMITS_HX_35HM, f"Elbow_Servo_3")
        self.TiltShoulder_servo = BusServo(driverff, 2, SHOULDERLIMITS_HX_65HM, f"TiltShoulder_Servo_2")
        self.PanShoulder_servo = BusServo(driver55, 1, SHOULDERLIMITS_HX_35HM, f"PanShoulder_Servo_1")
        # Store all servos in a list for easy iteration
        self.Servo_Motors = [self.Wrist_servo, self.Elbow_servo, self.TiltShoulder_servo, self.PanShoulder_servo]

    #update all the servo positions at once, with a list of four values representing the desired position for each servo.
    def update_servo_positions(self, position_vector: list):
        for i in range(4):
            self.Servo_Motors[i].move(position_vector[i], 50)
    #turn on all the servos
    def turn_on_all(self):
        for servo in self.Servo_Motors:
            servo.motor_on()
    #turn off all the servos
    def turn_off_all(self):
        for servo in self.Servo_Motors:
            servo.motor_off()
    #stop all the servos
    def stop_all(self):
        for servo in self.Servo_Motors:
            servo.motor_off()
    #print the status of all the servos
    def print_servo_statuses(self): 
        for servo in self.Servo_Motors:
            servo.print_status()
    #print the position of all the servos
    def print_servo_positions(self):
        for servo in self.Servo_Motors:
            print(f"{servo.name} position: {servo.read_position()}")
    #get the position of all the servos as a list of four values representing the current position of each servo.
    def get_servo_positions(self):
        positions = []
        for servo in self.Servo_Motors:
            positions.append(servo.read_position())
        return positions
    #set the servo offsets during calibration, with a list of four values representing the desired offset for each servo.
    def set_servo_offsets(self):
        """Set absolute offset values for each servo during calibration"""
        for servo in self.Servo_Motors:
            servo.limits.set_offset(servo.read_position())
    #==================================
    #individual servo control methods
    #==================================
    #print the status of a single servo by index
    def print_servo_status(self, servo_index):
        if 0 <= servo_index < len(self.Servo_Motors):
            self.Servo_Motors[servo_index].print_status()
        else:
            print(f"Invalid servo index: {servo_index}")
    #print the position of a single servo by index
    def print_servo_position(self, servo_index):
        if 0 <= servo_index < len(self.Servo_Motors):
            position = self.Servo_Motors[servo_index].read_position()
            print(f"{self.Servo_Motors[servo_index].name} position: {position}")
        else:
            print(f"Invalid servo index: {servo_index}")
    #move a single servo to a specific position by index
    def move_servo_to_position(self, servo_index, position, duration=1000):
        if 0 <= servo_index < len(self.Servo_Motors):
            self.Servo_Motors[servo_index].move(position, duration)
        else:
            print(f"Invalid servo index: {servo_index}")
    #turn on a single servo by index
    def on_servo(self, servo_index):
        if 0 <= servo_index < len(self.Servo_Motors):
            self.Servo_Motors[servo_index].motor_on()
        else:
            print(f"Invalid servo index: {servo_index}")
    #turn off a single servo by index
    def off_servo(self, servo_index):
        if 0 <= servo_index < len(self.Servo_Motors):
            self.Servo_Motors[servo_index].motor_off()
        else:
            print(f"Invalid servo index: {servo_index}")
    #stop a single servo by index
    def stop_servo(self, servo_index):
        if 0 <= servo_index < len(self.Servo_Motors):
            self.Servo_Motors[servo_index].motor_off()
        else:
            print(f"Invalid servo index: {servo_index}")

    