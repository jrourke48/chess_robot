"""!
@file ChessRobotUI.py
@brief Web UI orchestration for chess state validation, engine move generation, robot waypoint planning, and plotting.
"""
#!/usr/bin/env python3
import asyncio
import math
import time
from pathlib import Path

import chess

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles

from ChessStateValidatorMoveParser import ChessBoard
from RobotMotionPlanner import RobotMotionPlanner
from InverseKinematics_TrajectoryPlanner import chess_robot_inversekinematics, cubic_spline
from test_classes import _draw_chessboard


app = FastAPI()

PLOT_DIR = Path(__file__).resolve().parent / "ui_plots"
PLOT_DIR.mkdir(exist_ok=True)
app.mount("/ui_plots", StaticFiles(directory=str(PLOT_DIR)), name="ui_plots")


def _patch_chessboard_contract_for_robot_planner(chess_board):
    ChessBoard.square_size = chess_board.square_size

    def _height_lookup(piece_or_type):
        if piece_or_type is None:
            return 0.0
        if isinstance(piece_or_type, chess.Piece):
            piece_type = piece_or_type.piece_type
        else:
            piece_type = piece_or_type

        piece_to_height = {
            chess.PAWN: 1.23,
            chess.KNIGHT: 1.575,
            chess.BISHOP: 1.97,
            chess.ROOK: 1.39,
            chess.QUEEN: 2.3,
            chess.KING: 2.15,
        }
        return piece_to_height.get(piece_type, 0.0)

    ChessBoard.get_piece_height = staticmethod(_height_lookup)
    ChessBoard.get_chess_piece_height = staticmethod(_height_lookup)


class RobotSession:
    """Holds state across moves for one websocket session."""

    def __init__(self):
        self.chess_board = ChessBoard()
        _patch_chessboard_contract_for_robot_planner(self.chess_board)
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


def _save_visualizations(robot_waypoints, chess_board_obj):
    """Reuses the test_classes plotting style as web images."""
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception:
        return None, None, "matplotlib is not installed"

    board_img = PLOT_DIR / "board_waypoints.png"
    traj_img = PLOT_DIR / "trajectory_3d.png"

    # 2D chess board with projected robot path (same style helper from test_classes)
    fig2d = plt.figure(figsize=(7, 7))
    ax2d = fig2d.add_subplot(111)
    _draw_chessboard(ax2d, chess_board_obj, robot_waypoints)
    plt.tight_layout()
    fig2d.savefig(board_img, dpi=150)
    plt.close(fig2d)

    # 3D end-effector trajectory (same point/line style as test_classes)
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


def compute_full_pipeline_for_one_human_move(session: RobotSession, user_move_uci: str):
    """
    Runs the chess validation/engine/planner pipeline for one manual move.
    """
    chess_board = session.chess_board
    motion_planner = session.motion_planner

    out = {
        "ok": False,
        "error": None,
        "robot_move": None,
        "robot_waypoints": None,
        "jointspace_waypoints": None,
        "trajectory_debug": [],
    }

    current_fen = chess_board.current_state

    # 1) validate human move
    try:
        temp_board = chess.Board(current_fen)
        move_obj = chess.Move.from_uci(user_move_uci)
        if move_obj not in temp_board.legal_moves:
            out["error"] = f"Illegal move for current position: {user_move_uci}"
            return out
        temp_board.push(move_obj)
    except ValueError:
        out["error"] = f"Invalid move: {user_move_uci}"
        return out

    detected_fen = temp_board.fen()

    # 2) state validator + engine
    robot_move = chess_board.checkstate_thenrun(detected_fen)
    if isinstance(robot_move, tuple) and len(robot_move) == 2 and robot_move[0] is False:
        out["error"] = f"State validation/engine failed: {robot_move[1]}"
        return out

    out["robot_move"] = robot_move

    # 3) parse engine move -> chess waypoints -> robot waypoints
    parse_ok, parse_result = chess_board.parsemove(robot_move)
    if not parse_ok:
        out["error"] = f"Could not parse engine move {robot_move}: {parse_result}"
        return out

    chess_waypoints = chess_board.waypoints
    robot_waypoints = motion_planner.parse_chesswaypoints(chess_waypoints)
    out["robot_waypoints"] = robot_waypoints

    # 4) IK to joint waypoints
    jointspace_waypoints = []
    for waypoint in robot_waypoints:
        thetas = chess_robot_inversekinematics(waypoint[0], waypoint[1], waypoint[2])
        jointspace_waypoints.append(thetas)
    out["jointspace_waypoints"] = jointspace_waypoints

    # 5) trajectory coefficients debug
    for i in range(1, len(jointspace_waypoints)):
        cur_thetas = jointspace_waypoints[i - 1]
        next_thetas = jointspace_waypoints[i]
        for j in range(4):
            cur_theta = cur_thetas[j]
            next_theta = next_thetas[j]
            coeffs = cubic_spline(0, 2, cur_theta, next_theta)
            out["trajectory_debug"].append(
                {
                    "joint": j + 1,
                    "from_deg": math.degrees(cur_theta),
                    "to_deg": math.degrees(next_theta),
                    "coeffs": [float(c) for c in coeffs],
                    "T": 2.0,
                }
            )

    chess_board.move_completed()
    out["ok"] = True
    return out


@app.websocket("/ws")
async def ws_endpoint(ws: WebSocket):
    await ws.accept()
    session = RobotSession()

    await ws_log(ws, "Connected.")
    await ws_send(ws, {"type": "session", "game_started": False, "use_cv": False})
    await publish_board(ws, session)

    try:
        while True:
            msg = await ws.receive_json()
            mtype = msg.get("type")

            if mtype == "begin_game":
                session.game_started = True
                await ws_log(ws, "Game started.", "success")
                await ws_send(ws, {"type": "session", "game_started": True, "use_cv": session.use_cv})
                await ws_send(ws, {"type": "status", "state": "wait4roboturn"})

            elif mtype == "set_use_cv":
                session.use_cv = bool(msg.get("enabled", False))
                await ws_log(ws, f"Use CV set to {session.use_cv}")
                await ws_send(ws, {"type": "session", "game_started": session.game_started, "use_cv": session.use_cv})

            elif mtype in {"move_over", "human_move"}:
                if not session.game_started:
                    await ws_log(ws, "Click Begin Game first.", "error")
                    continue

                if session.use_cv:
                    await ws_send(ws, {"type": "status", "state": "cvalgo"})
                    await ws_log(
                        ws,
                        "CV mode selected. CV-to-FEN input is not wired in this UI yet, so manual move processing is disabled.",
                        "warn",
                    )
                    continue

                uci = (msg.get("uci") or "").strip().lower()
                async with session.lock:
                    await ws_send(ws, {"type": "status", "state": "robotturn", "uci": uci})
                    await ws_log(ws, f"Processing move-over with manual move: {uci}")

                    result = await asyncio.to_thread(compute_full_pipeline_for_one_human_move, session, uci)

                    if not result["ok"]:
                        await ws_send(ws, {"type": "status", "state": "error"})
                        await ws_log(ws, result["error"], level="error")
                        await publish_board(ws, session)
                        continue

                    board_url, traj_url, plot_err = await asyncio.to_thread(
                        _save_visualizations,
                        result["robot_waypoints"],
                        session.chess_board.board,
                    )

                    await ws_send(ws, {"type": "robot_move", "uci": result["robot_move"]})
                    await ws_log(ws, f"Engine move: {result['robot_move']}", "success")

                    if plot_err is None:
                        await ws_send(ws, {"type": "plots", "board_plot": board_url, "traj_plot": traj_url})
                    else:
                        await ws_log(ws, f"Plotting unavailable: {plot_err}", "warn")

                    await ws_send(ws, {"type": "status", "state": "wait4roboturn"})
                    await publish_board(ws, session)

            elif mtype == "get_board":
                await publish_board(ws, session)

            else:
                await ws_log(ws, f"Unknown message type: {mtype}", level="warn")

    except WebSocketDisconnect:
        return


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

        ws.onopen = () => {
            addLog("Connected", "success");
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

            if (msg.type === "robot_move") {
                addLog(`Engine move: ${msg.uci}`, "success");
                return;
            }

            if (msg.type === "plots") {
                document.getElementById("boardPlot").src = msg.board_plot;
                document.getElementById("trajPlot").src = msg.traj_plot;
                addLog("Updated plots", "success");
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
