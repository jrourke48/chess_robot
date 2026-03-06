import serial
import time

# ===== USER SETTINGS =====
PORT = "COM3"  # <-- change this
BAUDS_TO_TRY = [115200, 1000000, 500000, 38400, 9600]
# =========================

def checksum_lobot(id_byte, length_byte, cmd_byte, params=b""):
    s = (id_byte + length_byte + cmd_byte + sum(params)) & 0xFF
    return (~s) & 0xFF

def make_id_read_broadcast_packet():
    # Protocol frame: 0x55 0x55 ID LEN CMD [params...] CHK
    # SERVO_ID_READ: CMD = 14 (0x0E)
    # Read command has no params -> LEN = 3 (LEN, CMD, CHK)
    ID = 0xFE               # broadcast ID
    LEN = 0x03
    CMD = 0x0E              # SERVO_ID_READ
    CHK = checksum_lobot(ID, LEN, CMD)
    return bytes([0x55, 0x55, ID, LEN, CMD, CHK])

def try_baud(baud):
    pkt = make_id_read_broadcast_packet()
    with serial.Serial(PORT, baudrate=baud, timeout=0.2) as ser:
        ser.reset_input_buffer()
        ser.reset_output_buffer()

        # send a few times in case timing is touchy
        for _ in range(3):
            ser.write(pkt)
            ser.flush()
            time.sleep(0.05)

        data = ser.read(64)
        return data, pkt

def pretty_hex(b: bytes):
    return " ".join(f"{x:02X}" for x in b)

if __name__ == "__main__":
    pkt = make_id_read_broadcast_packet()
    print("TX packet:", pretty_hex(pkt))

    for baud in BAUDS_TO_TRY:
        try:
            rx, _ = try_baud(baud)
            print(f"\nBAUD {baud}: RX {len(rx)} bytes")
            print(pretty_hex(rx))

            # Heuristic: a valid response usually starts with 55 55 and includes cmd 0E
            if len(rx) >= 6 and rx[0] == 0x55 and rx[1] == 0x55 and 0x0E in rx:
                print("✅ Looks like a valid response at this baud.")
                break
        except serial.SerialException as e:
            print(f"\nBAUD {baud}: serial error: {e}")