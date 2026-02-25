"""!
@file ChessStateValidatorMoveParser.py
@brief Compatibility loader for the implementation file with special characters in its name.
"""

import importlib.util
import sys
from pathlib import Path

module_path = Path(__file__).with_name("ChessStateValidator&MoveParser.py")
spec = importlib.util.spec_from_file_location(
    "chessrobotclasses.chess_state_validator_move_parser_impl", 
    module_path,
    submodule_search_locations=[]
)
module = importlib.util.module_from_spec(spec)

# Register the module in sys.modules with proper package context
sys.modules["chessrobotclasses.chess_state_validator_move_parser_impl"] = module
module.__package__ = "chessrobotclasses"

if spec is None or spec.loader is None:
    raise ImportError(f"Could not load module from {module_path}")

spec.loader.exec_module(module)

ChessBoard = module.ChessBoard
