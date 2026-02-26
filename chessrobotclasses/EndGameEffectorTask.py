import asyncio
import time
import chess
import RobotMotionPlanner, ServoController, InverseKinematics_TrajectoryPlanner, InverseKinematics_TrajectoryPlanner

class EndgameEffectorFSM:
    """State machine for CPU task"""
    
    def __init__(self, queues, events, chess_board, motion_planner):
        self.opponent_move = queues['opponent_move']
        self.detected_fen = queues['detected_fen']
        self.theta_vector = queues['theta_vector']
        self.begin_game = events['begin_game']
        self.servo_mode = events['servo_mode']
        self.ready2move = events['ready2move']
        self.use_cv = events['use_cv']
        self.emag_on = events['emag_on']
        self.move_completed = events['move_completed']
        #data storage for the queues
        self.current_detected_fen = None
        self.current_theta_vector = None
        #data storage for the state machine
        self.joint_space_waypoints = None
        self.cur_jwaypoint = None
        self.next_jwaypoint = None
        self.trajectory_coeffs = None
        self.cur_trajectory_time = 0.0
        #store the chess board and motion planner objects for use in the state machine
        self.chess_board = chess_board
        self.motion_planner = motion_planner
        
        # State definitions
        self.WAIT4GAME = 0
        self.WAIT4MOVE = 1
        self.WAIT4CV_ALGO = 2
        self.VALIDATIONANDMOVERPARSER = 3
        self.CHESS2JOINTSPACE = 4
        self.EXECUTE_MOVE = 5
        self.EXECUTE_CURRENT_WAYPOINT = 6
        
        
        self.state = self.WAIT4GAME
    

    #state function to wait for the game to begin, which is signaled by the begin_game event being set
    async def state_wait4game(self):
        #wait for the game to begin
        await self.begin_game.wait()
        print("Game started!")
        return self.WAIT4MOVE
    

    #state function to wait for the opponent's move, which is signaled by the ready2move event being set
    async def state_wait4move(self):
        await self.ready2move.wait()  # Wait for opponent's move to be ready
        self.ready2move.clear()  # Reset the event
        return self.WAIT4CV_ALGO
    

    #state function to wait for the computer vision algorithm to provide the detected FEN string after a move is made
    async def statewait4cv_algo(self):
        #check if we are using CV to detect moves
        if self.use_cv.is_set():
            try:
                self.current_detected_fen = await asyncio.wait_for(
                    self.detected_fen.get(), 
                    timeout=10.0  # 10 second timeout for CV
                )
                return self.VALIDATIONANDMOVERPARSER
            except asyncio.TimeoutError:
                print("CV timeout - no FEN detected")
                return self.WAIT4CV_ALGO
        else:
            return self.VALIDATIONANDMOVERPARSER
       
    
    #state function to validate the detected FEN and parse the move into chess waypoints
    async def state_validate_parse_move(self):
        #validate the detected FEN and parse the move
        if self.use_cv.is_set():
            #validate the detected FEN and update the chess board state
            robot_move = self.chess_board.checkstate_thenrun(self.current_detected_fen)
            if isinstance(robot_move, tuple) and len(robot_move) == 2 and robot_move[0] is False:
                print(f"State validation/engine failed: {robot_move[1]}")
                return self.WAIT4CV_ALGO  # Go back to generate a new fen with CV
            else:
                #create chessspace waypoints based on the move
                parse_ok, parse_result = self.chess_board.parsemove(robot_move)
                if not parse_ok:
                    print(f"Could not parse engine move {robot_move}: {parse_result}")
                    return self.WAIT4MOVE  # Go back to waiting for the next move
                else:
                    self.chess_waypoints = self.chess_board.waypoints
                    return self.CHESS2JOINTSPACE
        else:
            #if not using CV, just read the opponent move from the queue
            try:
                user_move = self.opponent_move.get_nowait()
                current_fen = self.chess_board.current_state
                try:
                    #create a temporary chess board to validate the move and generate the detected FEN after the move
                    temp_board = chess.Board(current_fen)
                    move_obj = chess.Move.from_uci(user_move)
                    if move_obj not in temp_board.legal_moves:
                        print(f"Illegal move for current position: {user_move}")
                        return self.WAIT4MOVE
                    temp_board.push(move_obj)
                except ValueError:
                    print(f"Invalid move: {user_move}")
                    return self.WAIT4MOVE
                detected_fen = temp_board.fen()  # Simulate the detected FEN after the move
                #validate the detected FEN and update the chess board state
                #checkstate_thenrun() returns the move outputted by stockfish if the detected FEN is valid
                robot_move = self.chess_board.checkstate_thenrun(detected_fen)
                #create chessspace waypoints based on the move
                parse_ok, parse_result = self.chess_board.parsemove(robot_move)
                if not parse_ok:
                    print(f"Could not parse engine move {robot_move}: {parse_result}")
                    return self.WAIT4MOVE
                #store the chess waypoints for use in the next state
                self.chess_waypoints = self.chess_board.waypoints
                return self.CHESS2JOINTSPACE
            except asyncio.QueueEmpty:
                print("No opponent move provided in queue.")
                # Stay in this state until opponent move is provided
                return self.WAIT4MOVE
    

    #state function to convert the chessspace waypoints to joint space waypoints for the robot motion planner
    async def state_chess2jointspace(self):
        #convert the chessspace waypoints to robot manipulator space waypoints
        robot_waypoints = self.motion_planner.parse_chesswaypoints(self.chess_waypoints)
        #convert the robot waypoints to joint space waypoints using the inverse kinematics function
        jointspace_waypoints = []
        for waypoint in robot_waypoints:
            inverse_kinematics_result = InverseKinematics_TrajectoryPlanner.chess_robot_inversekinematics(waypoint[0], waypoint[1], waypoint[2])
            jointspace_waypoints.append(inverse_kinematics_result)
        #store the joint space waypoints for use in the next state
        self.joint_space_waypoints = jointspace_waypoints
        self.servo_mode.set()  # Signal the servo controller task to execute the move
        return self.EXECUTE_MOVE
    

    #once we have the waypoints in the joint space we can begin to plan trajectories and execute the move
    async def state_execute_move(self):
        #get the current joint angles and the next joint space waypoint target
        #to generate the trajectory coefficients for a cubic-order spline trajectory
        if self.cur_jwaypoint is None:
            self.cur_jwaypoint = self.joint_space_waypoints[0]  # Start at the first waypoint
            self.next_jwaypoint = self.joint_space_waypoints[1]  # Set the next waypoint
        #check if we have reached the last waypoint, if so we can end the move execution and go back to waiting for the next move
        if self.joint_space_waypoints.index(self.cur_jwaypoint) + 1 >= len(self.joint_space_waypoints):
            print("Move execution complete.")
            self.chess_board.move_completed()  # Update the chess board state after move execution
            self.move_completed.set()  # Signal that the move has been completed
            self.servo_mode.clear() #clear servo mode now that move is over
            return self.WAIT4MOVE  # Go back to waiting for the next move
        #calculate the euclidean distance in joint space to the next waypoint to determine
        #the trajectory time based on the move distance and time calculated in the motion planner
        self.next_jwaypoint = self.joint_space_waypoints[self.joint_space_waypoints.index(self.cur_jwaypoint) + 1]  # Get the next waypoint
        path_distance = self.motion_planner.get_single_path_distance(self.cur_jwaypoint, self.next_jwaypoint)
        self.cur_trajectory_time = (path_distance/self.motion_planner.move_distance)*self.motion_planner.move_time  # Total time for the move trajectory
        #generate the trajectory coefficients for a cubic-order spline trajectory from the current waypoint to the next waypoint
        self.trajectory_coeffs = InverseKinematics_TrajectoryPlanner.fullvector_cubic_spline(
            0, self.cur_trajectory_time, self.cur_jwaypoint, self.next_jwaypoint)
        return self.EXECUTE_CURRENT_WAYPOINT

    #execute the move by sending the joint angle commands to the servo controller task
    async def state_execute_current_waypoint(self):
        #query the trajectory planner for the next trajectory segment to execute and send the joint angle commands to the robot
        start_time = time.time()
        while time.time() - start_time < self.cur_trajectory_time:
            t = time.time() - start_time
            current_thetas = InverseKinematics_TrajectoryPlanner.evaluate_cubic_spline(self.trajectory_coeffs, t)
            await self.theta_vector.put(current_thetas)  # Send the current joint angles to the servo controller task
            await asyncio.sleep(0.01)  # Control loop delay (10 ms)
        #after executing the trajectory for the current waypoint, update the current waypoint to the next waypoint and go back to executing the move
        self.cur_jwaypoint = self.next_jwaypoint
        return self.EXECUTE_MOVE
    
    
    async def run(self):
        """Main state machine loop"""
        while True:
            if self.state == self.WAIT4GAME:
                self.state = await self.state_wait4game()
            elif self.state == self.WAIT4MOVE:
                self.state = await self.state_wait4move()
            elif self.state == self.WAIT4CV_ALGO:
                self.state = await self.statewait4cv_algo()
            elif self.state == self.VALIDATIONANDMOVERPARSER:
                self.state = await self.state_validate_parse_move()
            elif self.state == self.CHESS2JOINTSPACE:
                self.state = await self.state_chess2jointspace()
            elif self.state == self.EXECUTE_MOVE:
                self.state = await self.state_execute_move()
            elif self.state == self.EXECUTE_CURRENT_WAYPOINT:
                self.state = await self.state_execute_current_waypoint()
            else:
                print(f"Unknown state: {self.state}, returning to WAIT state")
                self.state = self.WAIT4GAME
