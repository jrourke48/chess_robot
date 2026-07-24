from gpiozero import OutputDevice
from time import sleep

class Electromagnet:
    def __init__(self, pin):
        self.device = OutputDevice(pin)
    
    def on(self):
        self.device.on()
    
    def off(self):
        self.device.off()