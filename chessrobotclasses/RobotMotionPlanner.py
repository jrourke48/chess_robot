import chess
from ChessStateValidatorMoveParser import ChessBoard
import numpy as np

class RobotMotionPlanner:
    def __init__(self):
        # Z offset for the robot to lift pieces off the board in inches, calculated as twice the
        #height of the tallest piece (queen) plus a clearance of 1 inch
        self.intermediate_z_offset = 2*ChessBoard.get_chess_piece_height(chess.QUEEN)+1
        self.move_waypoints = [] # waypoints for the current move
        self.frameEhome = np.array([ # apprixomate matrix representing robot end effector home frame
                            [1, 0, 0, 3],
                            [0, 1, 0, 4.375],
                            [0, 0, 1, 16],
                            [0, 0, 0, 1]
                        ])
        self.HomePos = self.get_frame_positions(self.frameEhome) # home position of the robot end effector in inches
        self.robot_current_frame = self.frameEhome # Initialize to home frame
        self.curmove_vector = np.array([0, 0, 0]) # Initialize move vector
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
        robot_waypoints = [self.HomePos]  # Start with the home position as the first waypoint
        # Convert all chess square waypoints to robot coordinates
        for square in chesswaypoints:
            robot_waypoint = self.chessposition_to_coordinates(square)
            robot_waypoints.append(robot_waypoint)
        #add intermediate waypoints to lift the piece off the board 
        robot_waypoints = self.add_intermediate_waypoints(robot_waypoints)
        robot_waypoints.append(self.HomePos)  # End with the home position as the last waypoint
        return robot_waypoints
        
    
    #utility function to convert chess square waypoints to robot coordinates
    def chessposition_to_coordinates(self, square: ChessBoard.ChessSquare):
        square_size = ChessBoard.square_size
        board_origin = np.array([0, 0, 0])  # bottom right corner of the board is at (0,0)
        file_index = self.file_to_index[square.position[0]]  # Get file index from 'a' to 'h'
        rank_index = int(square.position[1]) - 1  # Get rank index from '1' to '8'
        z_offset = ChessBoard.get_piece_height(square.piece)  # Get height of the piece for z-coordinate
        position = board_origin + square_size * np.array([file_index, rank_index, 0])
        position[2] += z_offset  # Add the z-offset to the position
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
        return new_waypoints