import numpy as np
import math
"""!
@file InverseKinematics_TrajectoryPlanner.py
@brief Defines inverse kinematics and spline trajectory helper functions for robot arm motion.
"""


def _as_joint_vector(theta, name):
    values = np.asarray(theta, dtype=float).reshape(-1)
    if values.size != 4:
        raise ValueError(f"{name} must contain exactly 4 joint values")
    return values


def _clamp_unit(value):
    return max(-1.0, min(1.0, float(value)))

#inverse kinematics function to convert from desired chessboard coordinates to robot joint angles
#inputs: x, y, z coordinates of the target position in inches
#outputs: (theta1, theta2, theta3, theta4) joint angles for the robot to move to the target position
def chess_robot_inversekinematics(x, y, z):
    """!
    @brief Computes a feasible 4-DOF arm pose for a desired Cartesian target.
    @param x Target x-position in inches.
    @param y Target y-position in inches.
    @param z Target z-position in inches.
    @return Tuple of 4 joint angles in radians (theta1, theta2, theta3, theta4).
    @throws ValueError If the target is unreachable for the assumed arm geometry.
    """
    lux = 3.5  # base-to-shoulder length [in]
    luy = 4.375 # base-to-shoulder length [in]
    luz = 3.5  # base-to-shoulder length  [in]
    l1 = 8.0   # shoulder-to-elbow length [in]
    l2 = 7.0   # elbow-wrist length  [in]
    l3 = 6.0   # manhattan x offset  [in]
    l4 = 13.0   # manhattan z offset [in]
    lem = 1  # end-effector frame electromagnet z offset [in]
    #intermediate variables for geometric calculations
    int1 = l1**2 + l2**2 
    int2 = l3**2 + l4**2
    lam1_arg = (int2 - int1) / (-2.0 * l1 * l2)
    lam1 = math.acos(_clamp_unit(lam1_arg))
    lam2_arg = l2 * math.sin(lam1) / math.sqrt(int2)
    lam2 = math.asin(_clamp_unit(lam2_arg))
    lam3 = math.atan2(l4, l3)
    #calculate the geometric parameters for the inverse kinematics calculations
    phi = lam2+lam3
    beta = math.pi - lam1
    #calculate the frame Tnoa based on our desired position vector geometric parameters and the target position
    px = x+lux
    py = y-luy
    pz = z-luz-lem
    if px == 0.0 and py == 0.0:
        raise ValueError("Target x and y cannot both be zero for orientation calculation")
    try:
        nx = 1/ math.sqrt(1 + (px/py)**2)
        ny = - (px/py) * nx
    except ZeroDivisionError:
        nx = 1
        ny = 0
    ox = ny
    oy = nx

    T0Edes = np.array([
        [nx, ox, 0, px],
        [-ny, -oy, 0, py],
        [0, 0, -1, pz],
        [0, 0, 0, 1]
    ])
    print(f"Desired end-effector frame T0Edes:\n{T0Edes}")
    #extract the position and orientation components from the desired end-effector frame
    nz = T0Edes[2, 0]
    ox = T0Edes[0, 1]
    oy = T0Edes[1, 1]
    az = T0Edes[2, 2]
    px = T0Edes[0, 3]
    py = T0Edes[1, 3]
    pz = T0Edes[2, 3]
    #the orientation of the end-effector frame constrains the sum of theta2-theta4, so we can calculate theta1 first using the position of the target and the orientation of the end-effector frame
    theta1 = math.atan2(py, px)
    #next get the sum of theta2-theta4
    THETA234 = math.atan2(-nz, -az)
    print(f"Sum of theta2-theta4 (THETA234): {THETA234}")
    #get theta 3 next using the geometric parameters and the desired position
    #intermediate variables for the inverse kinematics calculations
    radius = math.hypot(float(px), float(py))
    theta3_arg = (radius**2 + pz**2 - l1**2 - l2**2) / (2.0 * l1 * l2)
    print(f"Intermediate variables: radius={radius}, pz={pz}, theta3_arg={theta3_arg}")
    print(f"theta3_arg: {theta3_arg}")  # Debug print to check the value before acos
    if theta3_arg < -1.0 or theta3_arg > 1.0:
        raise ValueError("Target position is outside reachable workspace")
    theta3 = beta - math.acos(_clamp_unit(theta3_arg))
    #finally get theta2 and theta4 using the geometric parameters and the desired position
    delta = theta3 - beta
    X1 = l1 + l2 * math.cos(delta)
    X2 = l2 * math.sin(delta)
    denominator = X1**2 + X2**2
    if denominator == 0.0:
        raise ValueError("Degenerate IK geometry encountered")
    theta2_arg = (X1 * pz - X2 * radius) / denominator
    theta2 = math.asin(_clamp_unit(theta2_arg)) - phi
    theta4 = THETA234 - theta2 - theta3
    print(f"Inverse Kinematics Solution: theta1={math.degrees(theta1):.3f}, theta2={math.degrees(theta2):.3f}, theta3={math.degrees(theta3):.3f}, theta4={math.degrees(theta4):.3f}")
    return (theta1, theta2, theta3, theta4)
#trajectory planner function to generate a trajectory for the robot to move from its 
# current position to the target position using a cubic spline trajectory
#inputs: initial time, final time
# and two joint angle vectors each 4X1:(theta1, theta2, theta3, theta4)
#representing the initial and final joint angles for the robot to move between
#outputs a matrix of coefficents for a cubic spline trajectory for each joint angle
def cubic_spline(t0, tf, theta0, thetaf):
    """!
    @brief Generates cubic polynomial coefficients with zero endpoint velocity.
    @param t0 Initial time.
    @param tf Final time.
    @param theta0 Initial 4-joint vector.
    @param thetaf Final 4-joint vector.
    @return 4x4 matrix where each row contains [a0, a1, a2, a3] for one joint.
    @throws ValueError If timing is invalid or joint vectors are not length 4.
    """
    t0 = float(t0)
    tf = float(tf)
    if tf <= t0:
        raise ValueError("tf must be greater than t0")

    theta0_vec = _as_joint_vector(theta0, "theta0")
    thetaf_vec = _as_joint_vector(thetaf, "thetaf")

    a_matrix = np.array([
        [1.0, t0, t0**2, t0**3],
        [0.0, 1.0, 2.0 * t0, 3.0 * t0**2],
        [1.0, tf, tf**2, tf**3],
        [0.0, 1.0, 2.0 * tf, 3.0 * tf**2],
    ], dtype=float)

    coeffs = np.zeros((4, 4), dtype=float)
    for joint_index in range(4):
        b_vector = np.array([theta0_vec[joint_index], 0.0, thetaf_vec[joint_index], 0.0], dtype=float)
        coeffs[joint_index, :] = np.linalg.solve(a_matrix, b_vector)

    return coeffs
#trajectory planner function to generate a trajectory for the robot to move from its current
#position to the target position using a fifth-order spline trajectory
#inputs: initial time, final time
# and two joint angle vectors each 4X1:(theta1, theta2, theta3, theta4)
#representing the initial and final joint angles for the robot to move between
#outputs a matrix of coefficents for a fifth-order spline trajectory for each joint angle
def fifth_order_spline(t0, tf, theta0, thetaf):
    """!
    @brief Generates quintic polynomial coefficients with zero endpoint velocity and acceleration.
    @param t0 Initial time.
    @param tf Final time.
    @param theta0 Initial 4-joint vector.
    @param thetaf Final 4-joint vector.
    @return 4x6 matrix where each row contains [a0..a5] for one joint.
    @throws ValueError If timing is invalid or joint vectors are not length 4.
    """
    t0 = float(t0)
    tf = float(tf)
    if tf <= t0:
        raise ValueError("tf must be greater than t0")

    theta0_vec = _as_joint_vector(theta0, "theta0")
    thetaf_vec = _as_joint_vector(thetaf, "thetaf")

    a_matrix = np.array([
        [1.0, t0, t0**2, t0**3, t0**4, t0**5],
        [0.0, 1.0, 2.0 * t0, 3.0 * t0**2, 4.0 * t0**3, 5.0 * t0**4],
        [0.0, 0.0, 2.0, 6.0 * t0, 12.0 * t0**2, 20.0 * t0**3],
        [1.0, tf, tf**2, tf**3, tf**4, tf**5],
        [0.0, 1.0, 2.0 * tf, 3.0 * tf**2, 4.0 * tf**3, 5.0 * tf**4],
        [0.0, 0.0, 2.0, 6.0 * tf, 12.0 * tf**2, 20.0 * tf**3],
    ], dtype=float)

    coeffs = np.zeros((4, 6), dtype=float)
    for joint_index in range(4):
        b_vector = np.array([theta0_vec[joint_index], 0.0, 0.0, thetaf_vec[joint_index], 0.0, 0.0], dtype=float)
        coeffs[joint_index, :] = np.linalg.solve(a_matrix, b_vector)

    return coeffs

def main():
    # Example usage of the inverse kinematics and trajectory planner functions
    target_x = 8-3.5  # inches
    target_y = 2+4.375  # inches
    target_z = 8+4.5  # inches

    try:
        joint_angles = chess_robot_inversekinematics(target_x, target_y, target_z)
        print(f"Calculated joint angles: {joint_angles}")
    except ValueError as e:
        print(f"Error: {e}")
if __name__ == "__main__":
    main()