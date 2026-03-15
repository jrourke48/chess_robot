import serial
import time

PORT = "COM3"       # <-- change to your BusLinker COM port
BAUD = 115200

OLD_ID = 128
NEW_ID = 4         # <-- your requested ID

CMD_ID_WRITE = 0x0D  # SERVO_ID_WRITE
HEADER = b"\x55\x55"

def checksum(id_byte, length_byte, cmd_byte, params):
    s = (id_byte + length_byte + cmd_byte + sum(params)) & 0xFF
    return (~s) & 0xFF

def make_packet(id_byte, cmd_byte, params):
    # length includes: LEN, CMD, params..., CHK
    length = 3 + len(params)
    chk = checksum(id_byte, length, cmd_byte, params)
    return bytes([0x55, 0x55, id_byte, length, cmd_byte, *params, chk])

def read_id_broadcast(ser):
    # SERVO_ID_READ (0x0E) sent to broadcast ID 0xFE
    pkt = make_packet(0xFE, 0x0E, [])
    ser.reset_input_buffer()
    ser.write(pkt)
    ser.flush()
    time.sleep(0.05)
    return ser.read(64)

def main():
    with serial.Serial(PORT, BAUD, timeout=0.2) as ser:
        print("Broadcast ID read (before):")
        rx = read_id_broadcast(ser)
        print("RX:", rx.hex(" "))

        print(f"\nWriting ID: {OLD_ID} -> {NEW_ID}")
        pkt = make_packet(OLD_ID, CMD_ID_WRITE, [NEW_ID])
        ser.write(pkt)
        ser.flush()
        time.sleep(0.1)

        print("\nBroadcast ID read (after):")
        rx2 = read_id_broadcast(ser)
        print("RX:", rx2.hex(" "))

        print("\nDone. If the 'after' packet shows ... 0F ... then ID=15 is set.")

if __name__ == "__main__":
    main()