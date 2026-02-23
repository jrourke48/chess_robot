"""!
@file main.py
@brief Entry point for integrating chess state parsing, planning, and robot motion execution.
"""

import asyncio
import chess
import numpy as np
from ChessStateValidatorMoveParser import ChessBoard
from RobotMotionPlanner import RobotMotionPlanner
from InverseKinematics_TrajectoryPlanner import (
    chess_robot_inversekinematics,
    cubic_spline,
    fifth_order_spline,
)
def main():
    """!
    @brief Application entry point for the chess robot control loop.
    """
    #flowchart of the main function:
    #1. Initialize the chess board and robot motion planner
    #3 wait for a move to be made on the chess board
    #run the computer vision system to detect the move 
    #put the detected move in the chessboard object to validate it and update the chess board state
    #4. If the move is valid, use parsemove() to parse the move into chessspace waypoints 
    #use the robot motion planner to generate manipulator space waypoints for the move using
    #5.parse_chesswaypoints() inputting the chessspace waypoints
    #6. Use the inverse kinematics to change the list of waypoints in manipulator space 
    # to joint space waypoints for the robot to execute the move
    #7. Use the trajectory planner to generate a trajectory for the robot to move from its
    # current waypoint to the next waypoint target using a fifth-order spline trajectory
    pass
if __name__ == "__main__":
    main()