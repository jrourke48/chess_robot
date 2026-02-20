# InverseKinematics_TrajectoryPlanner.py
#inverse kinematics function to convert from desired chessboard coordinates to robot joint angles
#inputs: x, y, z coordinates of the target position in inches
#outputs: (theta1, theta2, theta3, theta4) joint angles for the robot to move to the target position
def chess_robot_inversekinematics(x, y, z):
    pass
#trajectory planner function to generate a trajectory for the robot to move from its 
# current position to the target position using a cubic spline trajectory
#inputs: initial time, final time
# and two joint angle vectors each 4X1:(theta1, theta2, theta3, theta4)
#representing the initial and final joint angles for the robot to move between
#outputs a matrix of coefficents for a cubic spline trajectory for each joint angle
def cubic_spline(t0, tf, theta0, thetaf):
    pass
#trajectory planner function to generate a trajectory for the robot to move from its current
#position to the target position using a fifth-order spline trajectory
#inputs: initial time, final time
# and two joint angle vectors each 4X1:(theta1, theta2, theta3, theta4)
#representing the initial and final joint angles for the robot to move between
#outputs a matrix of coefficents for a fifth-order spline trajectory for each joint angle
def fifth_order_spline(t0, tf, theta0, thetaf):
    pass