import chess
from ChessBoard import ChessBoard
import numpy as np

class RobotMotionPlanner:
    def __init__(self):
        self.chessboard = ChessBoard(engine_path="/usr/games/stockfish")
        self.robothomeframe = np.array([ # apprixomate matrix representing robot end effector home frame
                            [1, 0, 0, 5],
                            [0, 1, 0, 0],
                            [0, 0, 1, 12],
                            [0, 0, 0, 1]
                        ])
        self.robot_current_frame = self.robothomeframe # Initialize to home frame
        self.curmove_vector = np.array([0, 0, 0]) # Initialize move vector

        #dictionary mapping chess files to indices
        self.file_to_index = {
            'a': 0,
            'b': 1,
            'c': 2,
            'd': 3,
            'e': 4,
            'f': 5,
            'g': 6,
            'h': 7
        }

    def get_frame_positions(self, frame):
        return frame[:3, 3]

    def plan_move(self, start_pos, end_pos):
        # Placeholder for motion planning logic
        print(f"Planning move from {start_pos} to {end_pos}")
        return [start_pos, end_pos]  # Simple direct path for illustration
        
    def chessmove_to_coordinates(self):
        square_size = self.chessboard.square_size
        board_origin = np.array([0, 0, 0])  # Assuming bottom-left corner of the board is at (0,0)
        move = self.chessboard.best_move
        if move is None:
            raise ValueError("No best move available from chess engine.")
        # Parse move like 'e2e4'
        start_square = move[:2]
        end_square = move[2:4]
        # Convert squares to coordinates in inches
        start_pos = board_origin + square_size*np.array([self.file_to_index[start_square[0]], (int(start_square[1]) - 1), 0])
        end_pos = board_origin + square_size*np.array([self.file_to_index[end_square[0]], (int(end_square[1]) - 1), 0])
        self.curmove_vector = end_pos - start_pos
        
        return start_pos, end_pos #x, y coordinates in inches


        