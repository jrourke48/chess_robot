#!/usr/bin/env python3
"""!
@file test_classes.py
@brief End-to-end tests for FEN parsing, move validation, and motion planning integration.
"""

import chess
import math
import numpy as np
from pathlib import Path
from ChessStateValidatorMoveParser import ChessBoard
from RobotMotionPlanner import RobotMotionPlanner
from InverseKinematics_TrajectoryPlanner import (
    chess_robot_inversekinematics,
    fullvector_fifth_order_spline,
    fullvector_cubic_spline
)


# Utility function to plot robot waypoints in 3D space with optional chess board
def plot_robot_waypoints(robot_waypoints, title="Robot Waypoints", save_path=None, show_plot=False, chess_board=None):
    try:
        import matplotlib.pyplot as plt
        import matplotlib.patches as patches
    except ImportError:
        print("Plot skipped: matplotlib is not installed.")
        return

    if chess_board is not None:
        fig = plt.figure(figsize=(14, 6))
        ax_board = fig.add_subplot(121)
        _draw_chessboard(ax_board, chess_board, robot_waypoints)
        ax_3d = fig.add_subplot(122, projection="3d")
    else:
        fig = plt.figure(figsize=(8, 6))
        ax_3d = fig.add_subplot(111, projection="3d")

    xs = [point[0] for point in robot_waypoints]
    ys = [point[1] for point in robot_waypoints]
    zs = [point[2] for point in robot_waypoints]

    ax_3d.plot(xs, ys, zs, marker="o", linewidth=1.5)

    for idx, (x_coord, y_coord, z_coord) in enumerate(robot_waypoints):
        ax_3d.text(x_coord, y_coord, z_coord, str(idx), fontsize=8)

    ax_3d.set_title("3D Waypoints")
    ax_3d.set_xlabel("X (in)")
    ax_3d.set_ylabel("Y (in)")
    ax_3d.set_zlabel("Z (in)")
    plt.tight_layout()

    if save_path is not None:
        output_path = Path(save_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_path, dpi=150)
        print(f"Saved waypoint plot: {output_path}")

    if show_plot:
        plt.show()
    else:
        plt.close(fig)


def _draw_chessboard(ax, chess_board, robot_waypoints=None):
    import matplotlib.patches as patches
    import chess

    ax.set_xlim(-0.5, 9.5)
    ax.set_ylim(-0.5, 9.5)
    ax.set_aspect("equal")
    ax.set_title("Chess Board with Waypoints")
    ax.set_xlabel("File (a-h)")
    ax.set_ylabel("Rank (1-8)")

    file_labels = ["a", "b", "c", "d", "e", "f", "g", "h"]
    rank_labels = ["8", "7", "6", "5", "4", "3", "2", "1"]
    ax.set_xticks(range(8))
    ax.set_xticklabels(file_labels)
    ax.set_yticks(range(8))
    ax.set_yticklabels(rank_labels)

    colors = ["#F0D9B5", "#B58863"]
    for rank in range(8):
        for file in range(8):
            color = colors[(rank + file) % 2]
            rect = patches.Rectangle((file - 0.5, rank - 0.5), 1, 1, linewidth=0, facecolor=color)
            ax.add_patch(rect)

    piece_symbols = {
        chess.PAWN: "♟",
        chess.KNIGHT: "♞",
        chess.BISHOP: "♝",
        chess.ROOK: "♜",
        chess.QUEEN: "♛",
        chess.KING: "♚",
    }

    for square in chess.SQUARES:
        piece = chess_board.piece_at(square)
        if piece:
            file = chess.square_file(square)
            rank = 7 - chess.square_rank(square)
            symbol = piece_symbols.get(piece.piece_type, "?")
            color = "white" if piece.color == chess.WHITE else "black"
            ax.text(file, rank, symbol, ha="center", va="center", fontsize=20, color=color)

    if robot_waypoints:
        ax.plot([7 - w[1] / 1.25 for w in robot_waypoints], [w[0] / 1.25 for w in robot_waypoints], "r-o", linewidth=2, markersize=4, label="Robot Path")
        ax.legend()


#Full chess robot test sequence:
#1. Initialize the chess board and motion planner
#2. Prompt the user to input a move in UCI format (e.g., e2e4)
#3. Validate the move against the current board state and generate the detected FEN after the move
#4. Use the chess board's checkstate_thenrun() function to validate the detected FEN and get the engine reply move
def run_full_pipeline_sequence_test():
    """!
    @brief Runs a full sequence test from FEN parsing to robot waypoint generation.
    """
    chess_board = ChessBoard()

    motion_planner = RobotMotionPlanner()
    if input("Run full pipeline sequence test? (y/n): ").lower() != "y":
        print("Test skipped.")
        return
    show_plot = input("Show plot window each move? (y/n): ").lower() == "y"

    while True:
        chess_board.display_board()
        user_move = input("Enter a move in UCI format (e.g., e2e4) or 'exit' to quit: ")
        if user_move.lower() == "exit":
             print("Exiting test.")
             break
        current_fen = chess_board.current_state
        try:
            #create a temporary chess board to validate the move and generate the detected FEN after the move
            temp_board = chess.Board(current_fen)
            move_obj = chess.Move.from_uci(user_move)
            if move_obj not in temp_board.legal_moves:
                print(f"Illegal move for current position: {user_move}")
                continue
            temp_board.push(move_obj)
        except ValueError:
            print(f"Invalid move: {user_move}")
            continue
        detected_fen = temp_board.fen()  # Simulate the detected FEN after the move
        #validate the detected FEN and update the chess board state
        #checkstate_thenrun() returns the move outputted by stockfish if the detected FEN is valid
        robot_move = chess_board.checkstate_thenrun(detected_fen)
        if isinstance(robot_move, tuple) and len(robot_move) == 2 and robot_move[0] is False:
            print(f"State validation/engine failed: {robot_move[1]}")
            continue

        #create chessspace waypoints based on the move
        parse_ok, parse_result = chess_board.parsemove(robot_move)
        if not parse_ok:
            print(f"Could not parse engine move {robot_move}: {parse_result}")
            continue
        chess_waypoints = chess_board.waypoints
        for chess_wp in chess_waypoints:
            print(f"Chess waypoint: {chess_wp.position} with piece {chess_wp.piece}") 
        #convert the chessspace waypoints to robot manipulator space waypoints 
        robot_waypoints = motion_planner.parse_chesswaypoints(chess_waypoints)
        for waypoint in robot_waypoints:
            print(f"Robot waypoint: {waypoint}")
        print(motion_planner.emag_on)
    
        #convert the robot waypoints to joint space waypoints using the inverse kinematics function
        jointspace_waypoints = []
        for waypoint in robot_waypoints:
            inverse_kinematics_result = chess_robot_inversekinematics(waypoint[0], waypoint[1], waypoint[2])
            jointspace_waypoints.append(inverse_kinematics_result)
        #now we need to use the trajectory planner to generate a trajectory for the robot to move from its
        #  current waypoint to the next waypoint target using a fifth-order spline trajectory
        for i in range(1, len(jointspace_waypoints)):
            cur_thetas = jointspace_waypoints[i - 1]
            next_thetas = jointspace_waypoints[i]
            trajectory_coeffs = fullvector_cubic_spline(0, 5, cur_thetas, next_thetas)
            #print(f"Trajectory coefficients for move from waypoint {i-1} to {i}:"
              #    f"\n{trajectory_coeffs}\n")
        chess_board.move_completed()
        
        print("Move complete. Enter the next move or 'exit'.")

        plot_robot_waypoints(
            robot_waypoints,
            title=f"Robot Waypoints for {robot_move} Move",
            save_path=f"test_outputs/{robot_move}_robot_waypoints.png",
            show_plot=show_plot,
            chess_board=chess_board.board,
        )
        
       
        

if __name__ == "__main__":
    run_full_pipeline_sequence_test()
