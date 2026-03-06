import serial
import time

PORT = "COM3"      # change to your port
BAUD = 1000000     # try 1000000 first

ser = serial.Serial(PORT, BAUD, timeout=0.1)

# Broadcast ping packet (Hiwonder / Lobot protocol)
packet = bytes([0x55, 0x55, 0xFE, 0x02, 0x01, 0xFC])

ser.write(packet)

time.sleep(0.1)

data = ser.read(100)

print("Response:", data)

ser.close()