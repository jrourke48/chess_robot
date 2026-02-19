import chess
from Stockfish import Stockfish

class ChessBoard:
    class ChessSquare:
        def __init__(self, position, piece):
            self.position = position  # file and rank 'a' through 'h' and 1 through 8 or off the board for captured pieces
            self.piece = piece  # chess.Piece object or None for empty squares
    def __init__(self, engine_path=None):
        self.engine = Stockfish(engine_path)
        self.dimension = 10 # size of chessboard in inches
        self.square_size = self.dimension / 8 # size of each square in inches  
        self.board = chess.Board()
        self.moves_made = [] # list of moves made in the game, in UCI format
        self.current_state = self.board.fen()
        self.previous_state = None
        self.best_move = None
        self.detected_move = None
        self.waypoints = [] #waypoints to execute the move

    def get_piece_height(self, piece):
        self.piece_to_height = { #these are the actual piece heights in inches that we will use 
        # for the robot waypoints.
            chess.PAWN: 1.23,
            chess.KNIGHT: 1.575,
            chess.BISHOP: 1.97,
            chess.ROOK: 1.39,
            chess.QUEEN: 2.3,
            chess.KING: 2.15
        }
        return self.piece_to_height.get(piece.piece_type, 0)  # Return 0 if piece is None or not found

    #utility function to display the board in the console for debugging
    def display_board(self):
        print(self.board)

    #utility function to compare two chess.Board objects and determine 
    #if they represent the same position (ignores move history)
    def _positions_match(self, board_a, board_b):
        return board_a.board_fen() == board_b.board_fen() and board_a.turn == board_b.turn

    #given two chess.Board objects, determine if they differ by exactly one legal move, and if so return that move. 
    #If they differ by more than one move, or if the second board is not reachable from the first by one legal move, return False
    def getsinglemove(self, previous_board, current_board):
        if self._positions_match(previous_board, current_board):
            return False, "No move detected between previous and current state"
        #check each square to determine which piece moved where, and then check if that move is legal
        diff_squares = []
        for i in range(0, 64):
            prev_piece = previous_board.piece_at(i)
            curr_piece = current_board.piece_at(i)
            if prev_piece != curr_piece:
                #found a square that changed, add it to the list of different squares along
                #with the piece that was on that square before and after the move
                diff_squares.append((i, prev_piece, curr_piece))
        if len(diff_squares) == 0:
            return False, "No move detected between previous and current state"
         #two squares changed, so it's a normal move or a capture
        elif len(diff_squares) == 2:
            #one square should have a piece and one is empty, or both 
            # squares should have pieces but are different if it's a capture
            #the start square is always the one that had a piece and now is empty
            if (diff_squares[0][2] is None):
                start_square = diff_squares[0][0]
                end_square = diff_squares[1][0]
            elif (diff_squares[1][2] is None):
                start_square = diff_squares[1][0]
                end_square = diff_squares[0][0]
            else:
                return False, "Invalid move detected (ambiguous squares)"
            return chess.Move(from_square=start_square, to_square=end_square)
        #three squares changed, so it's an en passant move.
        if len(diff_squares) == 3:
            #one square should have a piece and two should be empty, and the piece that 
            #moved should be a pawn and the piece that was captured should be a pawn as well,
            #and the captured piece should not be on the target square
            #the end square was empty before and now has a piece,
            #the start square had a piece and is now empty
            end_square = None
            start_square = None
            colorturn = None
            for square in diff_squares:
                if square[1] is None and square[2] is not None and square[2].piece_type == chess.PAWN:
                    end_square = square[0]
                    colorturn = square[2].color
                elif square[1] is not None and square[2] is None:
                    if colorturn is not None and square[1].color == colorturn and square[1].piece_type == chess.PAWN:
                        start_square = square[0]
            if start_square is not None and end_square is not None:
                return chess.Move(from_square=start_square, to_square=end_square)
            return False, "Invalid move detected (ambiguous squares)"
        #four squares changed, so it's a castling move
        elif len(diff_squares) == 4:
            #two squares should have pieces and two should be empty, and the pieces that 
            # moved should be a king and a rook.
            #the start square of the king is the one that had a king and is now empty, and the end square of the king is the one that is empty before and now has a king
            #the start square of the rook is the one that had a rook and is now empty, and the end square of the rook is the one that is empty before and now has a rook
            king_start_square = None
            king_end_square = None
            for square in diff_squares:
                if square[1] is not None and square[2] is None and square[1].piece_type == chess.KING:
                    king_start_square = square[0]
                elif square[1] is None and square[2] is not None and square[2].piece_type == chess.KING:
                    king_end_square = square[0]
            if king_start_square is not None and king_end_square is not None:
                return chess.Move(from_square=king_start_square, to_square=king_end_square)
            return False, "Invalid move detected (ambiguous squares)"

    #uses the utility function above to check if the transition from the current board state 
    #to the detected board state is valid (i.e., differs by exactly one legal move), 
    #and if so updates the board state and detected move accordingly.
    #  Returns a tuple of (is_valid_transition, message)
    def _validate_transition_and_update(self, detected_fen):
        try:
            detected_board = chess.Board(detected_fen)
            #check if the detected board is a fen string
        except ValueError:
            return False, "Invalid FEN string"
            #check if it is a valid chess state
        if not detected_board.is_valid():
            return False, "Illegal position"
            #check if the detected board is the same as the current board
        if self._positions_match(self.board, detected_board):
            return False, "No move detected between previous and current state"
        #check if the detected board can be reached from the current board by one legal move
        detected_move = self.getsinglemove(self.board, detected_board)
        if detected_move is False:
            return False, "Detected FEN is not reachable from previous state by one legal move"
        if not self.board.is_legal(detected_move):  # check if the move is legal in the current position
            return False, "Detected move is not legal in the current position"
        else:
            #only one legal move matches the detected board, so we can update the state
            self.previous_state = self.board.fen()
            self.board = self.board.push(detected_move)
            self.current_state = self.board.fen()
            self.detected_move = detected_move.uci()
            self.moves_made.append(self.detected_move)
            return True, f"Detected valid move: {self.detected_move}"
    
     #once we have executed the move, we can clear the waypoints, update the moves made
    #and the states of the board
    def move_completed(self):
        self.waypoints = []  # Clear waypoints after move is completed
        self.previous_state = self.board.fen()
        self.board.push(chess.Move.from_uci(self.best_move))  # Update board state with the move that was executed
        self.current_state = self.board.fen()
        self.moves_made.append(self.best_move)  # Add the move to the moves made list

    # Once robot completes move, update states
    def update_state(self, move):
        """Update state by applying a move (in UCI like 'e2e4')"""
        try:
            # Parse and apply the move
            move_obj = chess.Move.from_uci(move)  # e.g., "e2e4"
            if move_obj not in self.board.legal_moves:
                return False, f"Illegal move: {move}"
            
            # Apply the move
            self.previous_state = self.board.fen()
            self.board.push(move_obj)
            self.current_state = self.board.fen()
            self.detected_move = move
            self.moves_made.append(self.detected_move)
            return True, "State updated successfully"
        except ValueError:
            return False, "Invalid move format"


    #inputs the detected fen string and validates the transition, 
    #then gets the next move from engine
    def checkState_thenRun(self, detected_fen):
        #first validate the transition from the current board state to the detected board state, 
        # and update the board state if valid
        is_valid_transition, message = self._validate_transition_and_update(detected_fen)
        if not is_valid_transition:
            return False, message
        #if the transition is valid, get the next move from the engine
        try:
            best_move = self.engine.get_best_move(self.current_state, depth=10)
            if best_move is None:
                return False, "Engine could not evaluate position"
        except Exception as e:
            return False, f"Engine error: {str(e)}"
        #if we got a valid move from the engine, we can store it 
        # and return it in the message
        self.best_move = best_move
        return True, f"{message}. Engine move: {best_move}"
    

    #parse move from stockfish into waypoints for the robot
    def parsemove(self, move):
        try:
            move_obj = chess.Move.from_uci(move)
            if move_obj not in self.board.legal_moves:
                return False, f"Illegal move: {move}"

            move_uci = move_obj.uci()
            if len(move_uci) not in (4, 5):
                return False, "Move must be in UCI format (e.g., 'e2e4' or 'e7e8q')"
            #5 possible move formats and how there are parsed into waypoints: 
            # promotion move (e.g., e7e8q): IDK how we will handle the promotion with the robot yet
            # en passant (e.g., e5d6): [start, end, captured, off]
            # castling(e.g., e1g1): [king start, king end, rook start, rook end]
            # capture move (e.g., e4d5): [captured, off, start, end]
            # no capture move (e.g., e2e4): [start, end]
            #get the start and end squares and pieces for the move
            start_square = move_uci[:2]
            end_square = move_uci[2:4]
            start_piece = self.board.piece_at(chess.parse_square(start_square))
            end_piece = self.board.piece_at(chess.parse_square(end_square))
            off = self.ChessSquare("off", None)  # represents pieces that are captured and off the board
            #create waypoints for the robot to execute the move
            waypoints = [self.ChessSquare(start_square, start_piece), self.ChessSquare(end_square, end_piece)]
            #check edge cases: promotion move, en passant, castling
            #first promotion move: must be 5 characters, last character must be q, r, b, or n, and starting piece must be a pawn
            if len(move_uci) == 5:
                promotion_piece = move_uci[-1]
                if promotion_piece not in ['q', 'r', 'b', 'n']:
                    return False, "Invalid promotion piece. Use 'q', 'r', 'b', or 'n'."
                if start_piece is None or start_piece.piece_type != chess.PAWN:
                    return False, "Promotion move must start from a pawn."
                #not sure how we will handle the actual promotion with the robot,
                #but for now we can just treat it as a normal move and then handle 
                #the promotion after the move is executed by placing the new piece on the board
            if start_piece is not None:
                #next handle en passant: 
                # if the move is done by a pawn and it moves diagonally to an empty square,
                # then the move must be anan en passant move meeaning we need to capture the
                #pawn on the square behind the target square instead of the target square
                if end_piece is None and start_square[0] != end_square[0] and start_piece.piece_type == chess.PAWN:
                    #now we can add the waypoints for the en passant move
                    captured_square = end_square[0] + start_square[1]  # the square behind the target square
                    waypoints.append(self.ChessSquare(captured_square, chess.PAWN), self.ChessSquare(off, None))
                #next handle castling:    
                #if the move is a king move that moves two squares, it's a castling move. The piece on the target square is not relevant for castling
                if start_piece.piece_type == chess.KING and abs(chess.parse_square(end_square) - chess.parse_square(start_square)) == 2:
                    #add waypoints to move the rook as well
                    if end_square[0] == 'g':  # kingside castling
                        rook_start = 'h' + start_square[1]
                        rook_end = 'f' + start_square[1]
                    else:  # queenside castling
                        rook_start = 'a' + start_square[1]
                        rook_end = 'd' + start_square[1]
                    waypoints.append(self.ChessSquare(rook_start, chess.ROOK), self.ChessSquare(rook_end, chess.ROOK))
                #if piece is captured
                if end_piece is not None:
                     #capturing move, we want to add waypoints to remove the captured piece before moving the new piece
                    waypoints.insert(0, self.ChessSquare(end_square, end_piece))  # add a waypoint to capture the piece before moving the new piece
                    waypoints.insert(1, self.ChessSquare(off, None))  # add a waypoint to move the captured piece off the board
                self.waypoints = waypoints
                return True, move_obj
            else:
                return False, "Invalid move"
        except ValueError:
            return False, "Invalid move format"
        

   