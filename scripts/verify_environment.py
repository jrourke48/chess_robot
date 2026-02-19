import importlib
import os
import shutil
import sys
from pathlib import Path

REQUIRED_MODULES = [
    "cv2",
    "numpy",
    "tensorflow",
    "sklearn",
    "chess",
    "serial",  # pyserial
]

OPTIONAL_MODULES = [
    "picamera2",   # only needed on Raspberry Pi camera setups
    "hiwonder",    # if using vendor python package
    "lx16a",       # common LX-16A servo package name
]

WINDOWS_REQUIRED_TOOLS = [
    "ch341ser.exe",
    "ServoStudio_v0.1.5.exe",
]


def check_module(module_name):
    try:
        importlib.import_module(module_name)
        return True, None
    except Exception as exc:
        return False, str(exc)


def check_stockfish_binary():
    in_path = shutil.which("stockfish") is not None
    common_paths = [
        Path("/usr/games/stockfish"),
        Path("C:/Program Files/Stockfish/stockfish.exe"),
        Path("C:/stockfish/stockfish.exe"),
    ]
    explicit = any(path.exists() for path in common_paths)
    return in_path or explicit


def check_windows_tools():
    missing = []
    for tool_name in WINDOWS_REQUIRED_TOOLS:
        if Path(tool_name).exists():
            print(f"[OK] Windows hardware tool found: {tool_name}")
        else:
            print(f"[MISSING] Windows hardware tool not found in repo root: {tool_name}")
            missing.append(tool_name)
    return missing


def main():
    print("=== Chess Robot Environment Check ===")

    missing_required = []
    for module_name in REQUIRED_MODULES:
        ok, err = check_module(module_name)
        if ok:
            print(f"[OK] Required module: {module_name}")
        else:
            print(f"[MISSING] Required module: {module_name} ({err})")
            missing_required.append(module_name)

    for module_name in OPTIONAL_MODULES:
        ok, err = check_module(module_name)
        if ok:
            print(f"[OK] Optional module: {module_name}")
        else:
            print(f"[INFO] Optional module not found: {module_name}")

    if check_stockfish_binary():
        print("[OK] Stockfish binary found")
    else:
        print("[MISSING] Stockfish binary not found in PATH/common install locations")
        missing_required.append("stockfish-binary")

    if os.name == "nt":
        missing_required.extend(check_windows_tools())

    if missing_required:
        print("\nEnvironment check FAILED.")
        print("Missing required dependencies:", ", ".join(missing_required))
        sys.exit(1)

    print("\nEnvironment check PASSED.")
    sys.exit(0)


if __name__ == "__main__":
    main()
