"""!
@file UITask.py
@brief UI task FSM + FastAPI websocket bridge for inter-task queues/events.
"""
#!/usr/bin/env python3
import asyncio
import time
from pathlib import Path

import chess
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles

from ChessStateValidatorMoveParser import ChessBoard
from RobotMotionPlanner import RobotMotionPlanner
from test_classes import _draw_chessboard


class UITaskFSM:
    """State machine for the UI task."""

    def __init__(self, queues, events, chess_board, motion_planner):
        self.detected_fen = queues["detected_fen"]
        self.winner = queues["winner"]
        self.update_ui_boardstate = queues["update_ui_boardstate"]
        self.update_ui_robot_waypoints = queues["update_ui_robot_waypoints"]
        self.update_ui_move_list = queues["update_ui_move_list"]
        self.opponent_move = queues["opponent_move"]

        self.begin_game = events["begin_game"]
        self.ready2move = events["ready2move"]
        self.move_completed = events["move_completed"]
        self.use_cv = events["use_cv"]
        self.calibrate_servos = events["calibrate_servos"]
        self.calibrate_servos2 = events["calibrate_servos2"]

        self.chess_board = chess_board
        self.motion_planner = motion_planner
        self.current_detected_fen = None

        self.WAIT4GAME = 0
        self.WAIT4ROBOTTURN = 1
        self.CVALGO = 2
        self.ROBOTTURN = 3
        self.CALIBRATE_SERVOS = 4
        self.CALIBRATE_SERVOS2 = 5
        self.state = self.WAIT4GAME

    async def state_wait4game(self):
        # Check if calibration was requested (has priority over game start)
        if self.calibrate_servos.is_set():
            return self.CALIBRATE_SERVOS
        
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
            self.current_detected_fen = await asyncio.wait_for(self.detected_fen.get(), timeout=10.0)
            return self.ROBOTTURN
        except asyncio.TimeoutError:
            print("UI CV timeout - no FEN detected")
            return self.CVALGO

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
        # Per requirement: queue carries raw FEN board state for UI session updates.
        await self.update_ui_boardstate.put(self.chess_board.current_state)

        if self.motion_planner.move_waypoints:
            await self.update_ui_robot_waypoints.put(self.motion_planner.move_waypoints)

        await self.update_ui_move_list.put(self._move_list_payload())

        winner = self._winner_payload()
        if winner is not None:
            await self.winner.put(winner)

        await self.move_completed.wait()
        self.move_completed.clear()
        return self.WAIT4ROBOTTURN
    async def state_calibrate_servos(self):
        """Stage 1: Initiate calibration - move servos to known position"""
        await self.calibrate_servos.wait()
        # Signal that calibration stage 1 was initiated
        await self.update_ui_boardstate.put("CALIBRATION_STAGE_1_INITIATED")
        print("Calibration Stage 1: Servos moved to calibration position. Waiting for stage 2...")
        return self.CALIBRATE_SERVOS2
    
    async def state_calibrate_servos2(self):
        """Stage 2: Confirm calibration position and set offsets"""
        await self.calibrate_servos2.wait()
        # Signal that calibration stage 2 was completed
        await self.update_ui_boardstate.put("CALIBRATION_STAGE_2_COMPLETE")
        print("Calibration Stage 2: Offsets confirmed. Returning to game.")
        self.calibrate_servos.clear()
        self.calibrate_servos2.clear()
        return self.WAIT4GAME
    
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
            elif self.state == self.CALIBRATE_SERVOS:
                self.state = await self.state_calibrate_servos()
            elif self.state == self.CALIBRATE_SERVOS2:
                self.state = await self.state_calibrate_servos2()
            else:
                print(f"Unknown UI state: {self.state}, returning to WAIT4GAME")
                self.state = self.WAIT4GAME


# -------------------- Web UI bridge runtime --------------------

_UI_RUNTIME = {
    "queues": None,
    "events": None,
    "chess_board": None,
    "motion_planner": None,
}


def init_ui_runtime(queues, events, chess_board, motion_planner):
    """Called by main() so websocket handlers can access shared task resources."""
    _UI_RUNTIME["queues"] = queues
    _UI_RUNTIME["events"] = events
    _UI_RUNTIME["chess_board"] = chess_board
    _UI_RUNTIME["motion_planner"] = motion_planner


def _get_ui_runtime_or_none():
    if _UI_RUNTIME["queues"] is None or _UI_RUNTIME["events"] is None:
        return None
    return _UI_RUNTIME


app = FastAPI()

PLOT_DIR = Path(__file__).resolve().parent / "ui_plots"
PLOT_DIR.mkdir(exist_ok=True)
app.mount("/ui_plots", StaticFiles(directory=str(PLOT_DIR)), name="ui_plots")


class RobotSession:
    """UI-side session model. Board + motion planner are updated from task queues."""

    def __init__(self):
        self.chess_board = ChessBoard()
        self.motion_planner = RobotMotionPlanner()
        self.lock = asyncio.Lock()
        self.game_started = False
        self.use_cv = False


async def ws_send(ws: WebSocket, payload: dict):
    await ws.send_json(payload)


async def ws_log(ws: WebSocket, msg: str, level: str = "info"):
    await ws_send(ws, {"type": "log", "level": level, "msg": msg})


async def publish_board(ws: WebSocket, session: RobotSession):
    board = chess.Board(session.chess_board.current_state)
    await ws_send(
        ws,
        {
            "type": "board",
            "fen": board.fen(),
            "ascii": str(board),
            "turn": "white" if board.turn == chess.WHITE else "black",
        },
    )


def _save_visualizations(robot_waypoints, fen):
    """Reuses the test_classes plotting style as web images."""
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception:
        return None, None, "matplotlib is not installed"

    board_img = PLOT_DIR / "board_waypoints.png"
    traj_img = PLOT_DIR / "trajectory_3d.png"

    fig2d = plt.figure(figsize=(7, 7))
    ax2d = fig2d.add_subplot(111)
    _draw_chessboard(ax2d, chess.Board(fen), robot_waypoints)
    plt.tight_layout()
    fig2d.savefig(board_img, dpi=150)
    plt.close(fig2d)

    fig3d = plt.figure(figsize=(8, 6))
    ax3d = fig3d.add_subplot(111, projection="3d")
    xs = [point[0] for point in robot_waypoints]
    ys = [point[1] for point in robot_waypoints]
    zs = [point[2] for point in robot_waypoints]
    ax3d.plot(xs, ys, zs, marker="o", linewidth=1.5)
    for idx, (x_coord, y_coord, z_coord) in enumerate(robot_waypoints):
        ax3d.text(x_coord, y_coord, z_coord, str(idx), fontsize=8)

    ax3d.set_title("3D Waypoints")
    ax3d.set_xlabel("X (in)")
    ax3d.set_ylabel("Y (in)")
    ax3d.set_zlabel("Z (in)")
    plt.tight_layout()
    fig3d.savefig(traj_img, dpi=150)
    plt.close(fig3d)

    stamp = int(time.time() * 1000)
    return f"/ui_plots/{board_img.name}?t={stamp}", f"/ui_plots/{traj_img.name}?t={stamp}", None


async def _pump_task_updates_to_ui(ws: WebSocket, session: RobotSession, runtime):
    queues = runtime["queues"]
    while True:
        did_work = False

        while not queues["update_ui_boardstate"].empty():
            did_work = True
            new_fen = await queues["update_ui_boardstate"].get()
            if isinstance(new_fen, str):
                session.chess_board.current_state = new_fen
                await publish_board(ws, session)

        while not queues["update_ui_robot_waypoints"].empty():
            did_work = True
            robot_waypoints = await queues["update_ui_robot_waypoints"].get()
            session.motion_planner.move_waypoints = robot_waypoints
            board_url, traj_url, plot_err = await asyncio.to_thread(
                _save_visualizations,
                robot_waypoints,
                session.chess_board.current_state,
            )
            if plot_err is None:
                await ws_send(ws, {"type": "plots", "board_plot": board_url, "traj_plot": traj_url})
            else:
                await ws_log(ws, f"Plotting unavailable: {plot_err}", "warn")

        while not queues["update_ui_move_list"].empty():
            did_work = True
            moves = await queues["update_ui_move_list"].get()
            await ws_send(ws, {"type": "move_list", "moves": moves})

        while not queues["winner"].empty():
            did_work = True
            winner = await queues["winner"].get()
            await ws_send(ws, {"type": "winner", "winner": winner})

        if not did_work:
            await asyncio.sleep(0.05)


@app.websocket("/ws")
async def ws_endpoint(ws: WebSocket):
    await ws.accept()
    session = RobotSession()
    runtime = _get_ui_runtime_or_none()

    if runtime is None:
        await ws_log(ws, "UI runtime not initialized. Start through main.py.", "error")
        await ws.close()
        return

    queues = runtime["queues"]
    events = runtime["events"]

    session.game_started = events["begin_game"].is_set()
    session.use_cv = events["use_cv"].is_set()
    session.chess_board.current_state = runtime["chess_board"].current_state

    await ws_log(ws, "Connected.")
    await ws_send(ws, {"type": "session", "game_started": session.game_started, "use_cv": session.use_cv})
    await publish_board(ws, session)

    pump_task = asyncio.create_task(_pump_task_updates_to_ui(ws, session, runtime))

    try:
        while True:
            msg = await ws.receive_json()
            mtype = msg.get("type")

            if mtype == "begin_game":
                events["begin_game"].set()
                session.game_started = True
                await ws_log(ws, "Game started.", "success")
                await ws_send(ws, {"type": "session", "game_started": True, "use_cv": session.use_cv})
                await ws_send(ws, {"type": "status", "state": "wait4roboturn"})

            elif mtype == "set_use_cv":
                enabled = bool(msg.get("enabled", False))
                if enabled:
                    events["use_cv"].set()
                else:
                    events["use_cv"].clear()
                session.use_cv = enabled
                await ws_log(ws, f"Use CV set to {enabled}")
                await ws_send(ws, {"type": "session", "game_started": session.game_started, "use_cv": session.use_cv})

            elif mtype in {"move_over", "human_move"}:
                if not session.game_started:
                    await ws_log(ws, "Click Begin Game first.", "error")
                    continue

                if session.use_cv:
                    fen = (msg.get("fen") or "").strip()
                    if fen:
                        await queues["detected_fen"].put(fen)
                    events["ready2move"].set()
                    await ws_send(ws, {"type": "status", "state": "cvalgo"})
                    await ws_log(ws, "Signaled CV/CPU tasks that opponent move is complete.")
                    continue

                uci = (msg.get("uci") or "").strip().lower()
                if len(uci) < 4:
                    await ws_log(ws, "Enter a valid UCI move like e2e4", "error")
                    continue

                await queues["opponent_move"].put(uci)
                events["ready2move"].set()
                await ws_send(ws, {"type": "status", "state": "robotturn", "uci": uci})
                await ws_log(ws, f"Queued opponent move for CPU task: {uci}")

            elif mtype == "cv_fen":
                fen = (msg.get("fen") or "").strip()
                if not fen:
                    await ws_log(ws, "cv_fen missing fen payload", "error")
                    continue
                await queues["detected_fen"].put(fen)
                await ws_log(ws, "Queued detected FEN for CPU task")

            elif mtype == "get_board":
                await publish_board(ws, session)

            elif mtype == "resign":
                if not session.game_started:
                    await ws_log(ws, "No game in progress to resign.", "error")
                    continue
                # Human resigns, robot (Black) wins
                await queues["winner"].put("black")
                events["ready2move"].set()  # Trigger the CPU task to check for winner
                await ws_log(ws, "You resigned. Robot wins!", "success")
                await ws_send(ws, {"type": "winner", "winner": "black", "reason": "resignation"})
            
            elif mtype == "calibrate_stage1":
                if session.game_started:
                    await ws_log(ws, "Cannot calibrate during an active game.", "error")
                    continue
                events["calibrate_servos"].set()
                await ws_log(ws, "Calibration Stage 1: Moving servos to calibration position...", "success")
                await ws_send(ws, {"type": "calibration", "stage": 1, "status": "initiated"})
            
            elif mtype == "calibrate_stage2":
                if session.game_started:
                    await ws_log(ws, "Cannot calibrate during an active game.", "error")
                    continue
                events["calibrate_servos2"].set()
                await ws_log(ws, "Calibration Stage 2: Confirming offset positions...", "success")
                await ws_send(ws, {"type": "calibration", "stage": 2, "status": "completed"})

            else:
                await ws_log(ws, f"Unknown message type: {mtype}", level="warn")

    except WebSocketDisconnect:
        return
    finally:
        pump_task.cancel()


@app.get("/")
def index():
    return HTMLResponse(
        """
<!doctype html>
<html>
<head>
    <meta charset="utf-8" />
    <meta name="viewport" content="width=device-width, initial-scale=1.0" />
    <title>Chess Robot UI</title>
    <style>
        * { box-sizing: border-box; }
        body {
            margin: 0;
            font-family: "Segoe UI", Tahoma, sans-serif;
            background: #f3f4f6;
            color: #111827;
        }
        .wrap {
            max-width: 1320px;
            margin: 0 auto;
            padding: 18px;
            display: grid;
            grid-template-columns: 360px 1fr;
            gap: 16px;
        }
        .card {
            background: #ffffff;
            border: 1px solid #e5e7eb;
            border-radius: 10px;
            padding: 14px;
        }
        h2 { margin: 0 0 12px 0; font-size: 1.1rem; }
        h3 { margin: 8px 0; font-size: 0.95rem; }
        .controls { display: grid; gap: 10px; }
        label { font-size: 0.9rem; }
        input[type="text"] {
            width: 100%;
            padding: 8px;
            border: 1px solid #d1d5db;
            border-radius: 8px;
        }
        button {
            width: 100%;
            padding: 9px 10px;
            border: 1px solid #111827;
            background: #111827;
            color: #fff;
            border-radius: 8px;
            cursor: pointer;
        }
        button:disabled {
            opacity: 0.5;
            cursor: not-allowed;
        }
        button.secondary {
            background: #fff;
            color: #111827;
        }
        .pill {
            display: inline-block;
            font-size: 0.85rem;
            padding: 4px 9px;
            border-radius: 999px;
            background: #e5e7eb;
            margin-right: 6px;
        }
        .viz-grid {
            display: grid;
            grid-template-columns: 1fr 1fr;
            gap: 14px;
        }
        .viz img {
            width: 100%;
            height: auto;
            border: 1px solid #d1d5db;
            border-radius: 8px;
            background: #fff;
        }
        .logs {
            margin-top: 10px;
            height: 260px;
            overflow: auto;
            border: 1px solid #e5e7eb;
            border-radius: 8px;
            padding: 8px;
            font-family: Consolas, monospace;
            font-size: 0.82rem;
            background: #f9fafb;
        }
        .entry { padding: 5px; border-left: 3px solid #9ca3af; margin: 4px 0; background: #fff; }
        .entry.error { border-left-color: #dc2626; }
        .entry.success { border-left-color: #16a34a; }
        .entry.warn { border-left-color: #d97706; }
        .board-ascii {
            margin-top: 10px;
            white-space: pre;
            font-family: Consolas, monospace;
            font-size: 0.85rem;
            border: 1px solid #e5e7eb;
            border-radius: 8px;
            padding: 8px;
            background: #fff;
        }
        @media (max-width: 1024px) {
            .wrap { grid-template-columns: 1fr; }
            .viz-grid { grid-template-columns: 1fr; }
        }
    </style>
</head>
<body>
    <div class="wrap">
        <div class="card">
            <h2>Controls</h2>
            <div class="controls">
                <button onclick="beginGame()">Begin Game</button>

                <label>
                    <input type="checkbox" id="useCv" onchange="setCvMode()" />
                    Use CV for move detection
                </label>

                <div id="manualMoveSection">
                    <label for="moveInput">Manual Opponent Move (UCI)</label>
                    <input type="text" id="moveInput" placeholder="e2e4" maxlength="5" />
                </div>

                <button class="secondary" onclick="moveOverClicked()">My Move Is Over</button>
                <button class="secondary" style="background: #dc2626; border-color: #dc2626; color: #fff;" onclick="resignGame()">Resign</button>
                
                <hr style="margin: 12px 0; border: none; border-top: 1px solid #e5e7eb;">
                <h3 style="margin-top: 0;">Servo Calibration</h3>
                <button id="calibrateBtn1" class="secondary" style="background: #3b82f6; border-color: #3b82f6; color: #fff;" onclick="calibrateStage1()">Stage 1: Init Calibration</button>
                <button id="calibrateBtn2" class="secondary" style="background: #8b5cf6; border-color: #8b5cf6; color: #fff;" onclick="calibrateStage2()">Stage 2: Confirm Offsets</button>
            </div>

            <h3>Session</h3>
            <div>
                <span id="gamePill" class="pill">game: not started</span>
                <span id="cvPill" class="pill">cv: off</span>
                <span id="statusPill" class="pill">state: idle</span>
                <span id="turnPill" class="pill">turn: ?</span>
            </div>

            <div id="asciiBoard" class="board-ascii">(waiting for board)</div>
            <div id="logs" class="logs"></div>
        </div>

        <div class="card">
            <h2>Visualizations</h2>
            <div class="viz-grid">
                <div class="viz">
                    <h3>2D Board + Waypoints</h3>
                    <img id="boardPlot" alt="2D board with waypoint projection" />
                </div>
                <div class="viz">
                    <h3>3D End-Effector Trajectory</h3>
                    <img id="trajPlot" alt="3D trajectory" />
                </div>
            </div>
        </div>
    </div>

    <script>
        const ws = new WebSocket(`ws://${location.host}/ws`);

        function addLog(msg, level = "info") {
            const logsEl = document.getElementById("logs");
            const entry = document.createElement("div");
            entry.className = `entry ${level === "warning" ? "warn" : level}`;
            entry.textContent = `[${new Date().toLocaleTimeString()}] ${msg}`;
            logsEl.appendChild(entry);
            logsEl.scrollTop = logsEl.scrollHeight;
        }

        function setPill(id, text) {
            document.getElementById(id).textContent = text;
        }

        function beginGame() {
            ws.send(JSON.stringify({ type: "begin_game" }));
        }

        function setCvMode() {
            const enabled = document.getElementById("useCv").checked;
            ws.send(JSON.stringify({ type: "set_use_cv", enabled }));
            document.getElementById("manualMoveSection").style.display = enabled ? "none" : "block";
        }

        function moveOverClicked() {
            const useCv = document.getElementById("useCv").checked;
            if (useCv) {
                ws.send(JSON.stringify({ type: "move_over" }));
                return;
            }

            const uci = document.getElementById("moveInput").value.trim().toLowerCase();
            if (uci.length < 4) {
                addLog("Enter a valid UCI move like e2e4", "error");
                return;
            }

            ws.send(JSON.stringify({ type: "move_over", uci }));
            document.getElementById("moveInput").value = "";
        }

        function resignGame() {
            if (confirm("Are you sure you want to resign?")) {
                ws.send(JSON.stringify({ type: "resign" }));
            }
        }

        function calibrateStage1() {
            ws.send(JSON.stringify({ type: "calibrate_stage1" }));
        }

        function calibrateStage2() {
            ws.send(JSON.stringify({ type: "calibrate_stage2" }));
        }

        function updateCalibrationButtonState(gameStarted) {
            document.getElementById("calibrateBtn1").disabled = gameStarted;
            document.getElementById("calibrateBtn2").disabled = gameStarted;
        }

        ws.onopen = () => {
            addLog("Connected", "success");
            updateCalibrationButtonState(false);  // Enable calibration buttons on connect
            ws.send(JSON.stringify({ type: "get_board" }));
        };

        ws.onmessage = (event) => {
            const msg = JSON.parse(event.data);

            if (msg.type === "log") {
                addLog(msg.msg, msg.level || "info");
                return;
            }

            if (msg.type === "session") {
                setPill("gamePill", `game: ${msg.game_started ? "started" : "not started"}`);
                setPill("cvPill", `cv: ${msg.use_cv ? "on" : "off"}`);
                document.getElementById("useCv").checked = !!msg.use_cv;
                document.getElementById("manualMoveSection").style.display = msg.use_cv ? "none" : "block";
                updateCalibrationButtonState(msg.game_started);
                return;
            }

            if (msg.type === "status") {
                setPill("statusPill", `state: ${msg.state}`);
                return;
            }

            if (msg.type === "board") {
                setPill("turnPill", `turn: ${msg.turn}`);
                document.getElementById("asciiBoard").textContent = msg.ascii;
                return;
            }

            if (msg.type === "move_list") {
                addLog(`Moves: ${msg.moves.join(" ") || "(none)"}`);
                return;
            }

            if (msg.type === "winner") {
                let winnerText;
                const w = msg.winner.toLowerCase();
                if (w === "white" || w === "black") {
                    winnerText = `Checkmate! ${msg.winner.charAt(0).toUpperCase() + msg.winner.slice(1)} wins!`;
                } else if (w === "draw") {
                    winnerText = "Game Over - Draw!";
                } else if (w === "stalemate") {
                    winnerText = "Game Over - Stalemate!";
                } else {
                    winnerText = `Game Over: ${msg.winner}`;
                }
                if (msg.reason === "resignation") {
                    winnerText = `Resignation! ${msg.winner.charAt(0).toUpperCase() + msg.winner.slice(1)} wins!`;
                }
                addLog(winnerText, "success");
                setPill("statusPill", "state: game over");
                alert(winnerText);
                return;
            }

            if (msg.type === "plots") {
                document.getElementById("boardPlot").src = msg.board_plot;
                document.getElementById("trajPlot").src = msg.traj_plot;
                addLog("Updated plots", "success");
                return;
            }

            if (msg.type === "calibration") {
                const stage = msg.stage;
                const status = msg.status;
                const statusText = status === "initiated" ? "initiated - check robot position" : "completed - offsets confirmed";
                addLog(`Calibration Stage ${stage}: ${statusText}`, "success");
                setPill("statusPill", `state: calibration stage ${stage}`);
                return;
            }
        };

        ws.onerror = () => addLog("WebSocket error", "error");
        ws.onclose = () => addLog("Disconnected", "error");
    </script>
</body>
</html>
"""
    )


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="127.0.0.1", port=8000)
