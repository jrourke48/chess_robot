import asyncio
import chess


class UITaskFSM:
    """State machine for the UI task."""

    def __init__(self, queues, events, chess_board, motion_planner):
        self.detected_fen = queues['detected_fen']
        self.winner = queues['winner']
        self.update_ui_boardstate = queues['update_ui_boardstate']
        self.update_ui_robot_waypoints = queues['update_ui_robot_waypoints']
        self.update_ui_move_list = queues['update_ui_move_list']

        self.begin_game = events['begin_game']
        self.ready2move = events['ready2move']
        self.move_completed = events['move_completed']
        self.use_cv = events['use_cv']

        self.chess_board = chess_board
        self.motion_planner = motion_planner
        self.current_detected_fen = None

        self.WAIT4GAME = 0
        self.WAIT4ROBOTTURN = 1
        self.CVALGO = 2
        self.ROBOTTURN = 3

        self.state = self.WAIT4GAME

    async def state_wait4game(self):
        await self.begin_game.wait()
        print("UI task started.")
        return self.WAIT4ROBOTTURN

    async def state_wait4roboturn(self):
        await self.ready2move.wait()
        return self.CVALGO

    async def state_cvalgo(self):
        if not self.use_cv.is_set():
            self.current_detected_fen = None
            return self.ROBOTTURN

        try:
            self.current_detected_fen = await asyncio.wait_for(
                self.detected_fen.get(),
                timeout=10.0,
            )
            return self.ROBOTTURN
        except asyncio.TimeoutError:
            print("UI CV timeout - no FEN detected")
            return self.CVALGO

    def _board_payload(self):
        board = chess.Board(self.chess_board.current_state)
        payload = {
            "fen": board.fen(),
            "ascii": str(board),
            "turn": "white" if board.turn == chess.WHITE else "black",
        }
        if self.current_detected_fen is not None:
            payload["detected_fen"] = self.current_detected_fen
        return payload

    def _move_list_payload(self):
        board = chess.Board(self.chess_board.current_state)
        return [move.uci() for move in board.move_stack]

    def _winner_payload(self):
        board = chess.Board(self.chess_board.current_state)
        if not board.is_game_over(claim_draw=True):
            return None

        outcome = board.outcome(claim_draw=True)
        if outcome is None or outcome.winner is None:
            return "draw"
        return "white" if outcome.winner == chess.WHITE else "black"

    async def state_robotturn(self):
        await self.update_ui_boardstate.put(self._board_payload())

        if self.motion_planner.move_waypoints:
            await self.update_ui_robot_waypoints.put(self.motion_planner.move_waypoints)

        await self.update_ui_move_list.put(self._move_list_payload())

        winner = self._winner_payload()
        if winner is not None:
            await self.winner.put(winner)

        await self.move_completed.wait()
        self.move_completed.clear()
        return self.WAIT4ROBOTTURN

    async def run(self):
        """Main state machine loop."""
        while True:
            if self.state == self.WAIT4GAME:
                self.state = await self.state_wait4game()
            elif self.state == self.WAIT4ROBOTTURN:
                self.state = await self.state_wait4roboturn()
            elif self.state == self.CVALGO:
                self.state = await self.state_cvalgo()
            elif self.state == self.ROBOTTURN:
                self.state = await self.state_robotturn()
            else:
                print(f"Unknown UI state: {self.state}, returning to WAIT4GAME")
                self.state = self.WAIT4GAME
