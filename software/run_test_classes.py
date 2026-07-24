from runpy import run_path
from pathlib import Path
import sys

root = Path(__file__).resolve().parent
classes_dir = root / "chessrobotclasses"
sys.path.insert(0, str(classes_dir))

run_path(classes_dir / "test_classes.py", run_name="__main__")
