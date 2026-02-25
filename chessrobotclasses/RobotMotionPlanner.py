"""!
@file RobotMotionPlanner.py
@brief Generates robot waypoints for legal chess moves on the physical board.
"""

import chess
from ChessStateValidatorMoveParser import ChessBoard
import numpy as np

class RobotMotionPlanner:
    """!
    @brief Converts chess moves into spatial waypoints for robot end-effector motion.
    """

    def __init__(self):
        # Z offset for the robot to lift pieces off the board in inches, calculated as twice the
        #height of the tallest piece (queen) plus a clearance of 1 inch
        self.intermediate_z_offset = 2*ChessBoard.get_chess_piece_height(chess.QUEEN)+1
        self.move_waypoints = [] # waypoints for the current move
        self.frameEhome = np.array([ # apprixomate matrix representing robot end effector home frame
                            [1, 0, 0, 2.5],
                            [0, 1, 0, 4.375],
                            [0, 0, 1, 17.5],
                            [0, 0, 0, 1]
                        ])
        self.HomePos = self.get_frame_positions(self.frameEhome) # home position of the robot end effector in inches
        self.robot_current_frame = self.frameEhome # Initialize to home frame
        self.curmove_vector = np.array([0, 0, 0]) # Initialize move vector
        self.off = np.array([5, -5, -1])  # A position off the board for captured pieces to be moved to in inches
        #dictionary mapping chess files to indices
        self.file_to_index = {
            'a': 7,
            'b': 6,
            'c': 5,
            'd': 4,
            'e': 3,
            'f': 2,
            'g': 1,
            'h': 0
        }


    #function to get only the position from a transformation matrix
    def get_frame_positions(self, frame):
        return frame[:3, 3]
    

    #function to get only the rotation matrix from a transformation matrix
    def get_rotation_matrix(self, frame):
        return frame[:3, :3]
    

    #function to convert an array of waypoints in chesssquare format: postion (e.g., e2) 
    # and piece type (e.g. PAWN) to robot coordinates for the start and end positions of the move
    def parse_chesswaypoints(self, chesswaypoints):
        robot_waypoints = []  # Start with the home position as the first waypoint
        # Convert all chess square waypoints to robot coordinates
        for square in chesswaypoints:
            robot_waypoint = self.chessposition_to_coordinates(square)
            robot_waypoints.append(robot_waypoint)
        #add intermediate waypoints to lift the piece off the board 
        robot_waypoints = self.add_intermediate_waypoints(robot_waypoints)
        robot_waypoints.insert(0, self.HomePos)  # Start with the home position as the first waypoint
        robot_waypoints.append(self.HomePos)  # End with the home position as the last waypoint
        self.move_waypoints = robot_waypoints
        return robot_waypoints
        
    
    #utility function to convert chess square waypoints to robot coordinates
    def chessposition_to_coordinates(self, square: ChessBoard.ChessSquare):
        chess_pos = square.position
        print(chess_pos)
        if chess_pos == "off":
            return self.off
        square_size = ChessBoard.square_size
        # front right corner of the board is at (0,0)
        #this is a8 in chess coordinates, which corresponds to (0,0) in robot coordinates
        file_index = self.file_to_index[chess_pos[0]]  # Get file index from 'a' to 'h'
        rank_index = 8 - int(chess_pos[1])  # Get rank index inverted: rank 1->7, rank 8->0
        x = float(rank_index * square_size)
        y = float(file_index * square_size)
        z_offset = ChessBoard.get_piece_height(square.piece)  # Get height of the piece for z-coordinate
        print(f"x: {x}, y: {y}, z_offset: {z_offset}")
        position = np.array([x, y, z_offset])
        return position #x, y, z coordinates in inches
    

    #add intermediate waypoints to the move waypoints to lift the piece off the board
    def add_intermediate_waypoints(self, waypoints):
        new_waypoints = []#initialize new waypoints list
        #for all waypoints in the original list, add an intermediate waypoint with the same 
        # x and y coordinates but with a z offset to lift the piece off the board before 
        # moving to the next waypoint
        for waypoint in waypoints:
            intermediate_waypoint = [waypoint[0], waypoint[1], self.intermediate_z_offset]  # Same position in x and y, but with z offset to lift the piece
            new_waypoints.append(intermediate_waypoint)
            new_waypoints.append(waypoint)
            new_waypoints.append(intermediate_waypoint)  # Add the intermediate waypoint again to return to the lifted position after placing the piece down
        return new_waypoints