# Chess Robot Project

This repo contains the chess state validation, Stockfish integration, CV pipeline, and robot motion tooling.

## 1) Python setup

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install --upgrade pip
pip install -r requirements.txt
```

If you are running camera scripts on Raspberry Pi:

```bash
pip install -r requirements-rpi-camera.txt
```

## 2) System dependencies

### Stockfish engine binary (required)
Install Stockfish and make sure the executable is accessible.
- Windows: add `stockfish.exe` to PATH or pass its absolute path to `ChessBoard(engine_path=...)`
- Linux/RPi: often available at `/usr/games/stockfish`

### Hiwonder servo interface (required for low-level hardware control)
The exact package depends on your controller/SDK.
- Install the official Hiwonder/LX-16A Python package for your hardware
- If your interface is serial-based, `pyserial` is already included in `requirements.txt`
- Keep vendor SDK docs in this repo under a `docs/` folder if needed

### Windows hardware tools (included in this repo)
These two files are committed in the repo root for Windows setup:
- `ch341ser.exe` (CH341 USB-serial driver installer)
- `ServoStudio_v0.1.5.exe` (servo configuration/calibration tool)

For a new machine, run both once as Administrator before first hardware bring-up.

## 3) Verify environment

Run:

```powershell
python scripts/verify_environment.py
```

This checks required Python modules and whether a Stockfish binary is discoverable.
On Windows, it also verifies that `ch341ser.exe` and `ServoStudio_v0.1.5.exe` are present.

## 4) Create/push GitHub repo

From project root:

```powershell
git add .
git commit -m "Initial commit"
gh repo create chess_robot --private --source . --remote origin --push
```

If you want public instead, replace `--private` with `--public`.

## Notes
- `venv/` is ignored via `.gitignore`
- Large datasets can make GitHub pushes very heavy. If needed, move big assets to Git LFS later.
