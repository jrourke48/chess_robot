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
    #get theta 3 next using the geometric parameters and the desired position
    #intermediate variables for the inverse kinematics calculations
    radius = math.hypot(float(px), float(py))
    theta3_arg = (radius**2 + pz**2 - l1**2 - l2**2) / (2.0 * l1 * l2)
    #check if the target position is within the reachable workspace of the robot based 
    # on the argument to the arccos function for theta3
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
    return np.array([theta1, theta2, theta3, theta4])
#trajectory planner function to generate a trajectory for the robot to move from its 
# current position to the target position using a cubic spline trajectory
#inputs: initial time, final time
# and two joint angle vectors each 4X1:(theta1, theta2, theta3, theta4)
#representing the initial and final joint angles for the robot to move between
#outputs a matrix of coefficents for a cubic spline trajectory for each joint angle
def cubic_spline(t0, tf, theta0, thetaf, theta_dot0=None, theta_dotf=None):
    """!
    @brief Generates cubic polynomial coefficients with zero endpoint velocity.
    @param t0 Initial time.
    @param tf Final time.
    @param theta0 Initial n-joint vector.
    @param thetaf Final n-joint vector.
    @return 1x4 matrix where each row contains [a0, a1, a2, a3] for one joint.
    @throws ValueError If timing is invalid or joint vectors are not length n.
    """
    t0 = float(t0)
    tf = float(tf)
    if tf <= t0:
        raise ValueError("tf must be greater than t0")
    T = tf - t0
    #create the theta vectors and the a_matrix for solving the cubic spline coefficients
    #where the velocity at the endpoints is zero
    if theta_dot0 is None:
        theta_dot0 = 0
    if theta_dotf is None:
        theta_dotf = 0
    theta_vec = np.array([theta0, theta_dot0, thetaf, theta_dotf], dtype=float)
    #the a_matrix is based on the cubic spline equations for the boundary conditions of the trajectory
    a_matrix = np.array([
        [1.0, 0, 0, 0],
        [0.0, 1.0, 0, 0],
        [1.0, T, T**2, T**3],
        [0.0, 1.0, 2.0 * T, 3.0 * T**2],
    ], dtype=float)
    #want to solve for the a_coefficients for the joint angle trajectory
    coeffs = np.linalg.solve(a_matrix, theta_vec)
    #return the coefficients for the cubic spline trajectory for each joint angle
    return coeffs
#the full vector version of the cubic spline function where the input theta vectors are 4X1
# vectors representing the joint angles for each of the 4 joints and the output is a 4X4 matrix 
# where each row contains the coefficients for the cubic spline trajectory for each joint angle
def fullvector_cubic_spline(t0, tf, theta0, thetaf, dtheta0=None, dthetaf=None):
    if dtheta0 is None:
        dtheta0 = np.zeros(4)
    if dthetaf is None:
        dthetaf = np.zeros(4)
    all_coeffs = None  # Initialize to None to build 4x4 matrix
    for i in range(4):
        coeffs = cubic_spline(t0, tf, theta0[i], thetaf[i], dtheta0[i], dthetaf[i])
        if all_coeffs is None:
            all_coeffs = coeffs.reshape(1, -1)  # Reshape first row to 1x4
        else:
            all_coeffs = np.vstack((all_coeffs, coeffs))  # Stack rows to build 4x4 matrix
    return all_coeffs

#function to evaluate the cubic spline at a given time t to get the current joint angles for
# the robot to move to at time t
def evaluate_cubic_spline(coeffs, t):
    """!
    @brief Evaluates the cubic spline at a given time t.
    @param coeffs 4x4 matrix of coefficients for each joint's cubic spline.
    @param t Time at which to evaluate the spline.
    @return 4-joint vector of angles at time t.
    """
    a0 = coeffs[:, 0]
    a1 = coeffs[:, 1]
    a2 = coeffs[:, 2]
    a3 = coeffs[:, 3]
    return a0 + a1 * t + a2 * t**2 + a3 * t**3
#trajectory planner function to generate a trajectory for the robot to move from its current
#position to the target position using a fifth-order spline trajectory
#inputs: initial time, final time
# and two joint angle vectors each 4X1:(theta1, theta2, theta3, theta4)
#representing the initial and final joint angles for the robot to move between
#outputs a matrix of coefficents for a fifth-order spline trajectory for each joint angle
def fifth_order_spline(t0, tf, theta0, thetaf, theta_dot0=None, theta_dotf=None, theta_ddot0=None, theta_ddotf=None):
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
    T = tf - t0
    #create the theta vectors and the a_matrix for solving the cubic spline coefficients
    #where the velocity at the endpoints is zero
    if theta_dot0 is None:
        theta_dot0 = 0
    if theta_dotf is None:
        theta_dotf = 0
    if theta_ddot0 is None:
        theta_ddot0 = 0
    if theta_ddotf is None:
        theta_ddotf = 0
    theta_vec = np.array([theta0, theta_dot0, theta_ddot0, thetaf, theta_dotf, theta_ddotf], dtype=float)
    #the a_matrix is based on the cubic spline equations for the boundary conditions of the trajectory
    a_matrix = np.array([
        [1.0, 0, 0, 0, 0, 0],
        [0.0, 1.0, 0, 0, 0, 0],
        [0.0, 0.0, 2.0, 0, 0, 0],
        [1.0, T, T**2, T**3, T**4, T**5],
        [0.0, 1.0, 2.0 * T, 3.0 * T**2, 4.0 * T**3, 5.0 * T**4],
        [0.0, 0.0, 2.0, 6.0 * T, 12.0 * T**2, 20.0 * T**3],
    ], dtype=float)
    #want to solve for the a_coefficients for the joint angle trajectory
    coeffs = np.linalg.solve(a_matrix, theta_vec)
    #return the coefficients for the cubic spline trajectory for each joint angle
    return coeffs
    

#the full vector version of the fifth order spline function where the input theta vectors are 4X1 vectors representing the joint angles for each of the 
# 4 joints and the output is a 4X6 matrix where each row contains the coefficients for 
# the fifth order spline trajectory for each joint angle  
def fullvector_fifth_order_spline(t0, tf, theta0, thetaf, dtheta0=None, dthetaf=None, ddtheta0=None, ddthetaf=None):
    if dtheta0 is None:
        dtheta0 = np.zeros(4)
    if dthetaf is None:
        dthetaf = np.zeros(4)
    if ddtheta0 is None:
        ddtheta0 = np.zeros(4)  
    if ddthetaf is None:
        ddthetaf = np.zeros(4)
    all_coeffs = None  # Initialize to None to build 4x6 matrix
    for i in range(4):
        coeffs = fifth_order_spline(t0, tf, theta0[i], thetaf[i], dtheta0[i], dthetaf[i], ddtheta0[i], ddthetaf[i])
        if all_coeffs is None:
            all_coeffs = coeffs.reshape(1, -1)  # Reshape first row to 1x6
        else:
            all_coeffs = np.vstack((all_coeffs, coeffs))  # Stack rows to build 4x6 matrix
    return all_coeffs

def evaluate_fifth_order_spline(coeffs, t):
    """!
    @brief Evaluates the quintic spline at a given time t.
    @param coeffs 4x6 matrix of coefficients for each joint's quintic spline.
    @param t Time at which to evaluate the spline.
    @return 4-joint vector of angles at time t.
    """
    a0 = coeffs[:, 0]
    a1 = coeffs[:, 1]
    a2 = coeffs[:, 2]
    a3 = coeffs[:, 3]
    a4 = coeffs[:, 4]
    a5 = coeffs[:, 5]
    return a0 + a1 * t + a2 * t**2 + a3 * t**3 + a4 * t**4 + a5 * t**5


def main():
    # Example usage of the inverse kinematics and trajectory planner functions
    target_x = 10-3.5  # inches
    target_y = 5+4.375  # inches
    target_z = 8+4.5  # inches

    try:
        joint_angles = chess_robot_inversekinematics(target_x, target_y, target_z)
        print(f"Calculated joint angles: {joint_angles}")
    except ValueError as e:
        print(f"Error: {e}")
    
    # Test trajectory planning
    print("\n--- Trajectory Planning Tests ---")
    
    # Define two waypoints (initial and final joint angles)
    theta0 = np.array([0.0, 0.5, -0.3, 0.2])  # Initial joint angles (radians)
    thetaf = np.array([0.5, 0.8, 0.1, -0.1])  # Final joint angles (radians)
    t0, tf = 0.0, 2.0  # Trajectory from t=0 to t=2 seconds
    
    # Test single joint cubic spline
    print("\n1. Single joint cubic spline (joint 0):")
    coeffs_cubic = cubic_spline(t0, tf, theta0[0], thetaf[0])
    print(f"   Coefficients [a0, a1, a2, a3]: {coeffs_cubic}")
    
    # Test single joint fifth-order spline
    print("\n2. Single joint fifth-order spline (joint 0):")
    coeffs_fifth = fifth_order_spline(t0, tf, theta0[0], thetaf[0])
    print(f"   Coefficients [a0..a5]: {coeffs_fifth}")
    
    # Test full vector cubic spline (4 joints)
    print("\n3. Full vector cubic spline (4 joints):")
    all_coeffs_cubic = fullvector_cubic_spline(t0, tf, theta0, thetaf)
    print(f"   Coefficient matrix shape: {all_coeffs_cubic.shape}")
    print(f"   Coefficients:\n{all_coeffs_cubic}")
    
    # Evaluate trajectory at different times
    print("\n4. Evaluating trajectory at key times:")
    for t in [0.0, 0.5, 1.0, 1.5, 2.0]:
        angles = evaluate_cubic_spline(all_coeffs_cubic, t)
        print(f"   t={t:.1f}s: {angles}")
    
    # Verify boundary conditions
    print("\n5. Verify boundary conditions:")
    angles_t0 = evaluate_cubic_spline(all_coeffs_cubic, t0)
    angles_tf = evaluate_cubic_spline(all_coeffs_cubic, tf)
    print(f"   At t0: expected {theta0}, got {angles_t0}")
    print(f"   At tf: expected {thetaf}, got {angles_tf}")
    print(f"   Initial match: {np.allclose(angles_t0, theta0)}")
    print(f"   Final match: {np.allclose(angles_tf, thetaf)}")
    
    # Test fifth order full vector
    print("\n6. Full vector fifth-order spline (4 joints):")
    all_coeffs_fifth = fullvector_fifth_order_spline(t0, tf, theta0, thetaf)
    print(f"   Coefficient matrix shape: {all_coeffs_fifth.shape}")
    angles_fifth_tf = evaluate_fifth_order_spline(all_coeffs_fifth, tf)
    print(f"   At tf: expected {thetaf}, got {angles_fifth_tf}")
    print(f"   Final match: {np.allclose(angles_fifth_tf, thetaf)}")
if __name__ == "__main__":
    main()