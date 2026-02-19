import chess
from Stockfish import Stockfish

class ChessBoard:
    class ChessSquare:
        def __init__(self, position, piece):
            self.position = position  # file and rank 'a' through 'h' and 1 through 8
            self.piece = piece  # chess.Piece object or None for empty squares
    def __init__(self, engine_path=None):
        self.engine = Stockfish(engine_path)
        self.dimension = 16 # size of chessboard in inches
        self.square_size = self.dimension / 8 # size of each square in inches   
        self.board = chess.Board()
        self.current_state = self.board.fen()
        self.previous_state = None
        self.best_move = None
        self.detected_move = None
        self.waypoints = [] #waypoints to execute the move

    def _positions_match(self, board_a, board_b):
        return board_a.board_fen() == board_b.board_fen() and board_a.turn == board_b.turn

    def _validate_transition_and_update(self, detected_fen):
        try:
            detected_board = chess.Board(detected_fen)
        except ValueError:
            return False, "Invalid FEN string"

        if not detected_board.is_valid():
            return False, "Illegal position"

        if self._positions_match(self.board, detected_board):
            return False, "No move detected between previous and current state"

        matching_moves = []
        for legal_move in self.board.legal_moves:
            candidate = self.board.copy(stack=False)
            candidate.push(legal_move)
            if self._positions_match(candidate, detected_board):
                matching_moves.append((legal_move, candidate))

        if len(matching_moves) == 0:
            return False, "Detected FEN is not reachable from previous state by one legal move"

        if len(matching_moves) > 1:
            return False, "Detected FEN maps to multiple legal moves (ambiguous)"

        detected_move, updated_board = matching_moves[0]
        self.previous_state = self.board.fen()
        self.board = updated_board
        self.current_state = self.board.fen()
        self.detected_move = detected_move.uci()
        return True, f"Detected valid move: {self.detected_move}"

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
            return True, "State updated successfully"
        except ValueError:
            return False, "Invalid move format"

    #get next move from engine
    def checkState_thenRun(self, fen):
        is_valid_transition, message = self._validate_transition_and_update(fen)
        if not is_valid_transition:
            return False, message

        try:
            best_move = self.engine.get_best_move(self.current_state, depth=15)
            if best_move is None:
                return False, "Engine could not evaluate position"
        except Exception as e:
            return False, f"Engine error: {str(e)}"

        self.best_move = best_move
        return True, f"{message}. Engine move: {best_move}"

    def parsemove(self, move):
        try:
            move_obj = chess.Move.from_uci(move)
            if move_obj not in self.board.legal_moves:
                return False, f"Illegal move: {move}"

            move_uci = move_obj.uci()
            if len(move_uci) not in (4, 5):
                return False, "Move must be in UCI format (e.g., 'e2e4' or 'e7e8q')"

            start_square = move_uci[:2]
            end_square = move_uci[2:4]
            start_piece = self.board.piece_at(chess.parse_square(start_square))
            end_piece = self.board.piece_at(chess.parse_square(end_square))
            #check edge cases: promotion move, en passant, castling
            #first promotion move: must be 5 characters, last character must be q, r, b, or n, and starting piece must be a pawn
            if len(move_uci) == 5:
                promotion_piece = move_uci[-1]
                if promotion_piece not in ['q', 'r', 'b', 'n']:
                    return False, "Invalid promotion piece. Use 'q', 'r', 'b', or 'n'."
                if start_piece is None or start_piece.piece_type != chess.PAWN:
                    return False, "Promotion move must start from a pawn."
            #en passant: if the move is a pawn move that captures an opponent's pawn on the 5th rank, and the captured piece is not on the target square, then it's an en passant move
            if start_piece is not None and start_piece.piece_type == chess.PAWN:
                if end_piece is None and chess.parse_square(end_square) in self.board.attacks(chess.parse_square(start_square)):
                    if self.board.is_en_passant(move_obj):
                        end_piece = chess.Piece(chess.PAWN, not self.board.turn)  # The captured pawn
                        #need to finish this logic: the captured pawn is actually on the square behind the target square, so we need to calculate that square and check if the captured piece is there
            #castling: if the move is a king move that moves two squares, it's a castling move. The piece on the target square is not relevant for castling
            if start_piece is not None and start_piece.piece_type == chess.KING:
                if abs(chess.parse_square(end_square) - chess.parse_square(start_square)) == 2:
                    end_piece = None  # Castling move, ignore piece on target square
                    #also need to finsih this logic: we should check if the move is actually a legal castling move, but since we already checked if the move is legal earlier, we can assume it's valid if it reaches this point
            #if piece is captured
            #need to do 
            #standard move: piece moves to an empty square or captures an opponent's piece
            #need to do
            start = self.ChessSquare(start_square, start_piece)
            end = self.ChessSquare(end_square, end_piece)
            self.waypoints = [start, end]

            return True, move_obj
        except ValueError:
            return False, "Invalid move format"
       
    

    def display_board(self):
        print(self.board)