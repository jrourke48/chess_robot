import importlib.util
from pathlib import Path

module_path = Path(__file__).with_name("ChessStateValidator&MoveParser.py")
spec = importlib.util.spec_from_file_location("chess_state_validator_move_parser_impl", module_path)
module = importlib.util.module_from_spec(spec)

if spec is None or spec.loader is None:
    raise ImportError(f"Could not load module from {module_path}")

spec.loader.exec_module(module)

ChessBoard = module.ChessBoard
