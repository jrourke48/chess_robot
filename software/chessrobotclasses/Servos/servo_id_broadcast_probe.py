import serial
import time

# ===== USER SETTINGS =====
PORT = "COM3"
BAUDS_TO_TRY = [115200, 1000000, 500000, 38400, 9600]
TARGET_IDS = [1, 2, 3, 4, 128]
# =========================


def pretty_hex(data: bytes) -> str:
    return " ".join(f"{byte:02X}" for byte in data) if data else "(empty)"


def checksum_55(servo_id: int, length: int, command: int, params=b"") -> int:
    total = (servo_id + length + command + sum(params)) & 0xFF
    return (~total) & 0xFF


def checksum_ff(servo_id: int, length: int, instruction: int, params=b"") -> int:
    total = (servo_id + length + instruction + sum(params)) & 0xFF
    return (~total) & 0xFF


def make_55_broadcast_id_read() -> bytes:
    servo_id = 0xFE
    length = 0x03
    command = 0x0E
    checksum = checksum_55(servo_id, length, command)
    return bytes([0x55, 0x55, servo_id, length, command, checksum])


def make_55_targeted_id_read(target_id: int) -> bytes:
    servo_id = target_id & 0xFF
    length = 0x03
    command = 0x0E
    checksum = checksum_55(servo_id, length, command)
    return bytes([0x55, 0x55, servo_id, length, command, checksum])


def make_ff_ping(target_id: int) -> bytes:
    servo_id = target_id & 0xFF
    length = 0x02
    instruction = 0x01
    checksum = checksum_ff(servo_id, length, instruction)
    return bytes([0xFF, 0xFF, servo_id, length, instruction, checksum])


def make_ff_targeted_id_read(target_id: int) -> bytes:
    servo_id = target_id & 0xFF
    instruction = 0x02
    params = bytes([0x05, 0x01])
    length = len(params) + 0x02
    checksum = checksum_ff(servo_id, length, instruction, params)
    return bytes([0xFF, 0xFF, servo_id, length, instruction, *params, checksum])


def run_once(packet: bytes, baud: int, timeout: float = 0.15) -> bytes:
    with serial.Serial(PORT, baudrate=baud, timeout=timeout) as ser:
        ser.reset_input_buffer()
        ser.reset_output_buffer()
        ser.write(packet)
        ser.flush()
        time.sleep(0.08)
        return ser.read(64)


def run_broadcast_55(baud: int) -> tuple[bytes, bytes]:
    packet = make_55_broadcast_id_read()
    with serial.Serial(PORT, baudrate=baud, timeout=0.2) as ser:
        ser.reset_input_buffer()
        ser.reset_output_buffer()
        for _ in range(3):
            ser.write(packet)
            ser.flush()
            time.sleep(0.05)
        rx = ser.read(64)
    return packet, rx


def targeted_check_55(baud: int, target_ids: list[int]) -> None:
    print("\n55 protocol targeted ID-read check:")
    for servo_id in target_ids:
        packet = make_55_targeted_id_read(servo_id)
        rx = run_once(packet, baud)
        valid = False
        for index in range(len(rx) - 6):
            if rx[index] == 0x55 and rx[index + 1] == 0x55 and rx[index + 2] == servo_id and rx[index + 4] == 0x0E:
                reported_id = rx[index + 5]
                print(f"  ID {servo_id}: RESPONDED (reports ID={reported_id}) raw={pretty_hex(rx)}")
                valid = True
                break
        if not valid:
            print(f"  ID {servo_id}: no valid 55 response raw={pretty_hex(rx)}")


def targeted_check_ff(baud: int, target_ids: list[int]) -> None:
    print("\nFF protocol targeted ping + ID-read check:")
    for servo_id in target_ids:
        ping_rx = run_once(make_ff_ping(servo_id), baud)
        read_rx = run_once(make_ff_targeted_id_read(servo_id), baud)

        ping_valid = False
        for index in range(len(ping_rx) - 5):
            if ping_rx[index] == 0xFF and ping_rx[index + 1] == 0xFF and ping_rx[index + 2] == servo_id:
                ping_valid = True
                break

        read_valid = False
        for index in range(len(read_rx) - 6):
            if read_rx[index] == 0xFF and read_rx[index + 1] == 0xFF and read_rx[index + 2] == servo_id:
                read_valid = True
                break

        if ping_valid or read_valid:
            print(f"  ID {servo_id}: POSSIBLE FF RESPONSE ping={pretty_hex(ping_rx)} read={pretty_hex(read_rx)}")
        else:
            print(f"  ID {servo_id}: no valid FF response ping={pretty_hex(ping_rx)} read={pretty_hex(read_rx)}")


if __name__ == "__main__":
    print("55 protocol broadcast ID-read packet:", pretty_hex(make_55_broadcast_id_read()))

    for baud in BAUDS_TO_TRY:
        try:
            packet, rx = run_broadcast_55(baud)
            print(f"\nBAUD {baud}: TX={pretty_hex(packet)}")
            print(f"BAUD {baud}: RX {len(rx)} bytes")
            print(pretty_hex(rx))
        except serial.SerialException as error:
            print(f"\nBAUD {baud}: serial error: {error}")

    print("\n" + "=" * 60)
    print("Targeted checks at 115200")
    print("=" * 60)
    targeted_check_55(115200, TARGET_IDS)
    targeted_check_ff(115200, TARGET_IDS)