#!/usr/bin/env python3
"""
End-to-end pipeline tests:
previous/current FEN -> infer move -> validate move -> generate next move ->
parse to chess waypoints -> convert to robot coordinates.
"""

import chess
from ChessStateValidatorMoveParser import ChessBoard
from RobotMotionPlanner import RobotMotionPlanner


def _patch_chessboard_contract_for_robot_planner():
    if not hasattr(ChessBoard, "square_size"):
        ChessBoard.square_size = 10 / 8

    def _height_lookup(piece_or_type):
        if piece_or_type is None:
            return 0.0

        if isinstance(piece_or_type, chess.Piece):
            piece_type = piece_or_type.piece_type
        else:
            piece_type = piece_or_type

        piece_to_height = {
            chess.PAWN: 1.23,
            chess.KNIGHT: 1.575,
            chess.BISHOP: 1.97,
            chess.ROOK: 1.39,
            chess.QUEEN: 2.3,
            chess.KING: 2.15,
        }
        return piece_to_height.get(piece_type, 0.0)

    ChessBoard.get_piece_height = staticmethod(_height_lookup)
    ChessBoard.get_chess_piece_height = staticmethod(_height_lookup)


def _select_safe_next_move(board):
    for move in board.legal_moves:
        if board.is_castling(move):
            continue
        if move.promotion is not None:
            continue
        if board.is_capture(move):
            continue
        return move

    for move in board.legal_moves:
        return move

    return None


def _get_engine_next_move_or_fallback(curr_fen):
    try:
        board_obj = ChessBoard()
        next_move = board_obj.engine.get_best_move(curr_fen, depth=10)
        board_obj.engine.quit()
        if next_move:
            return next_move, "stockfish"
    except Exception:
        pass

    board = chess.Board(curr_fen)
    fallback_move = _select_safe_next_move(board)
    if fallback_move is None:
        return None, "none"
    return fallback_move.uci(), "fallback-legal-move"


def run_pipeline_for_transition(previous_fen, current_fen):
    previous_board = chess.Board(previous_fen)
    current_board = chess.Board(current_fen)

    checker = ChessBoard.__new__(ChessBoard)
    checker.board = previous_board.copy(stack=False)
    checker.waypoints = []

    inferred_move = checker.getsinglemove(previous_board, current_board)
    if not isinstance(inferred_move, chess.Move):
        return False, f"Infer failed: {inferred_move}"

    if not previous_board.is_legal(inferred_move):
        return False, f"Inferred move illegal: {inferred_move.uci()}"

    candidate = previous_board.copy(stack=False)
    candidate.push(inferred_move)
    if not checker._positions_match(candidate, current_board):
        return False, f"Inferred move does not reach target: {inferred_move.uci()}"

    next_move_uci, source = _get_engine_next_move_or_fallback(current_fen)
    if next_move_uci is None:
        return False, "No legal next move available"

    checker.board = current_board.copy(stack=False)
    parse_ok, parse_result = checker.parsemove(next_move_uci)
    if not parse_ok:
        return False, f"Parse failed for next move {next_move_uci}: {parse_result}"

    _patch_chessboard_contract_for_robot_planner()
    planner = RobotMotionPlanner()

    try:
        robot_waypoints = planner.parse_chesswaypoints(checker.waypoints)
    except Exception as exc:
        return False, f"RobotMotionPlanner conversion failed: {exc}"

    if not robot_waypoints or not all(len(waypoint) == 3 for waypoint in robot_waypoints):
        return False, "Robot waypoints invalid shape"

    return True, (
        f"OK | inferred={inferred_move.uci()} | next={next_move_uci} ({source}) | "
        f"chess_waypoints={len(checker.waypoints)} | robot_waypoints={len(robot_waypoints)}"
    )


def run_full_pipeline_sequence_test():
    print("=" * 90)
    print("Full Pipeline Test: prev/current FEN -> validate -> next move -> chess wp -> robot wp")
    print("=" * 90)

    moves = ["e2e4", "e7e5", "g1f3", "b8c6", "f1b5"]
    board = chess.Board()
    fens = [board.fen()]
    for move in moves:
        board.push(chess.Move.from_uci(move))
        fens.append(board.fen())

    pass_count = 0
    fail_count = 0

    for i in range(1, len(fens)):
        prev_fen = fens[i - 1]
        curr_fen = fens[i]
        ok, message = run_pipeline_for_transition(prev_fen, curr_fen)
        if ok:
            print(f"✓ Transition {i}: {message}")
            pass_count += 1
        else:
            print(f"✗ Transition {i}: {message}")
            fail_count += 1

    print("-" * 90)
    print(f"Summary: PASS={pass_count}, FAIL={fail_count}")
    print("=" * 90)


if __name__ == "__main__":
    run_full_pipeline_sequence_test()
