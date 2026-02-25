"""!
@file ChessRobotUI.py
@brief Simple UI orchestration class for chess state validation, engine move generation, and robot waypoint planning.
"""
#!/usr/bin/env python3
import asyncio
import math
import chess

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse

from ChessStateValidatorMoveParser import ChessBoard
from RobotMotionPlanner import RobotMotionPlanner
from InverseKinematics_TrajectoryPlanner import chess_robot_inversekinematics, cubic_spline

app = FastAPI()


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
    """Holds state across moves (like your CLI loop)."""
    def __init__(self):
        self.chess_board = ChessBoard()
        _patch_chessboard_contract_for_robot_planner(self.chess_board)
        self.motion_planner = RobotMotionPlanner()
        self.lock = asyncio.Lock()  # prevent concurrent moves


async def ws_send(ws: WebSocket, payload: dict):
    await ws.send_json(payload)


async def ws_log(ws: WebSocket, msg: str, level: str = "info"):
    await ws_send(ws, {"type": "log", "level": level, "msg": msg})


async def publish_board(ws: WebSocket, session: RobotSession):
    # Send FEN + an ASCII representation for quick UI
    board = chess.Board(session.chess_board.current_state)
    await ws_send(ws, {
        "type": "board",
        "fen": board.fen(),
        "ascii": str(board),
        "turn": "white" if board.turn == chess.WHITE else "black"
    })


def compute_full_pipeline_for_one_human_move(session: RobotSession, user_move_uci: str):
    """
    Runs your existing pipeline for ONE input move.
    Returns a dict of results and debug info to send back.
    This is synchronous; we'll run it in a thread to avoid blocking the event loop.
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

    # 3) parse engine move → chess waypoints → robot waypoints
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

    # 5) trajectory coefficients debug (same as your code)
    for i in range(1, len(jointspace_waypoints)):
        cur_thetas = jointspace_waypoints[i - 1]
        next_thetas = jointspace_waypoints[i]
        for j in range(4):
            cur_theta = cur_thetas[j]
            next_theta = next_thetas[j]
            coeffs = cubic_spline(0, 2, cur_theta, next_theta)
            out["trajectory_debug"].append({
                "joint": j + 1,
                "from_deg": math.degrees(cur_theta),
                "to_deg": math.degrees(next_theta),
                "coeffs": [float(c) for c in coeffs],
                "T": 2.0
            })

    # Mark the robot's move as completed in the board state
    chess_board.move_completed()
    out["ok"] = True
    return out


@app.websocket("/ws")
async def ws_endpoint(ws: WebSocket):
    await ws.accept()
    session = RobotSession()

    await ws_log(ws, "Connected. Ready for moves.")
    await publish_board(ws, session)

    try:
        while True:
            msg = await ws.receive_json()
            mtype = msg.get("type")

            if mtype == "human_move":
                uci = (msg.get("uci") or "").strip()

                async with session.lock:
                    await ws_send(ws, {"type": "status", "state": "planning", "uci": uci})
                    await ws_log(ws, f"Received human move: {uci}")

                    # Run your heavy pipeline without blocking WS:
                    result = await asyncio.to_thread(compute_full_pipeline_for_one_human_move, session, uci)

                    if not result["ok"]:
                        await ws_send(ws, {"type": "status", "state": "error"})
                        await ws_log(ws, result["error"], level="error")
                        await publish_board(ws, session)
                        continue

                    await ws_send(ws, {"type": "robot_move", "uci": result["robot_move"]})
                    await ws_log(ws, f"Engine move: {result['robot_move']}")

                    # Stream debug trajectory info
                    for item in result["trajectory_debug"]:
                        await ws_send(ws, {"type": "traj_coeffs", **item})

                    # TODO: Here is where you would actually execute motion:
                    # await ws_send(ws, {"type":"status","state":"executing"})
                    # await run_trajectory_and_send_servo_commands(...)

                    await ws_send(ws, {"type": "status", "state": "idle"})
                    await publish_board(ws, session)

            elif mtype == "get_board":
                await publish_board(ws, session)

            else:
                await ws_log(ws, f"Unknown message type: {mtype}", level="warn")

    except WebSocketDisconnect:
        return


# Minimal test page (optional)
@app.get("/")
def index():
    return HTMLResponse("""
<!doctype html>
<html>
<head>
    <title>Chess Robot UI</title>
    <meta charset="utf-8" />
    <meta name="viewport" content="width=device-width, initial-scale=1.0" />
    <script src="https://cdnjs.cloudflare.com/ajax/libs/three.js/r128/three.min.js"></script>
    <style>
        * {
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }
        body {
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            min-height: 100vh;
            padding: 20px;
        }
        .container {
            max-width: 1400px;
            margin: 0 auto;
            display: grid;
            grid-template-columns: 1fr 1fr;
            gap: 20px;
            height: 90vh;
        }
        .panel {
            background: white;
            border-radius: 12px;
            box-shadow: 0 10px 40px rgba(0,0,0,0.2);
            padding: 20px;
            display: flex;
            flex-direction: column;
        }
        .panel h2 {
            color: #333;
            margin-bottom: 15px;
            font-size: 1.5em;
            border-bottom: 3px solid #667eea;
            padding-bottom: 10px;
        }
        
        /* Chess Board */
        .chessboard {
            display: grid;
            grid-template-columns: repeat(8, 1fr);
            gap: 0;
            aspect-ratio: 1;
            background: #8B7355;
            padding: 2px;
            border-radius: 8px;
            overflow: hidden;
        }
        .square {
            aspect-ratio: 1;
            display: flex;
            align-items: center;
            justify-content: center;
            font-size: 2.5em;
            cursor: pointer;
            transition: all 0.2s ease;
            user-select: none;
        }
        .square.light {
            background: #F0D9B5;
        }
        .square.dark {
            background: #B58863;
        }
        .square:hover {
            filter: brightness(1.1);
        }
        .square.from {
            background: #BCE6FE !important;
            box-shadow: inset 0 0 10px rgba(0,150,255,0.5);
        }
        .square.to {
            background: #BACA44 !important;
            box-shadow: inset 0 0 10px rgba(186,202,68,0.5);
        }
        
        /* Controls */
        .controls {
            display: flex;
            gap: 10px;
            margin-top: 15px;
            flex-wrap: wrap;
        }
        input[type="text"] {
            flex: 1;
            padding: 10px 15px;
            border: 2px solid #e0e0e0;
            border-radius: 6px;
            font-size: 1em;
            transition: border-color 0.3s;
        }
        input[type="text"]:focus {
            outline: none;
            border-color: #667eea;
        }
        button {
            padding: 10px 20px;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            border: none;
            border-radius: 6px;
            cursor: pointer;
            font-weight: bold;
            transition: transform 0.2s, box-shadow 0.2s;
        }
        button:hover {
            transform: translateY(-2px);
            box-shadow: 0 5px 15px rgba(102,126,234,0.4);
        }
        button:active {
            transform: translateY(0);
        }
        
        /* Logs & Info */
        .logs {
            flex: 1;
            overflow-y: auto;
            background: #f9f9f9;
            border: 1px solid #e0e0e0;
            border-radius: 6px;
            padding: 15px;
            margin-top: 15px;
            font-family: 'Courier New', monospace;
            font-size: 0.9em;
        }
        .log-entry {
            padding: 8px;
            margin: 5px 0;
            border-left: 4px solid #667eea;
            background: white;
            border-radius: 4px;
        }
        .log-entry.info {
            border-left-color: #667eea;
        }
        .log-entry.success {
            border-left-color: #4caf50;
            background: #f1f8f4;
        }
        .log-entry.error {
            border-left-color: #f44336;
            background: #fef5f4;
        }
        .log-entry.warning {
            border-left-color: #ff9800;
            background: #fff8f3;
        }
        
        /* 3D Visualization */
        #canvas {
            width: 100%;
            height: 100%;
            border-radius: 8px;
            background: linear-gradient(135deg, #e0e0e0 0%, #f5f5f5 100%);
        }
        
        .status {
            display: flex;
            gap: 15px;
            margin-top: 15px;
            flex-wrap: wrap;
        }
        .status-item {
            padding: 10px 15px;
            background: #f0f0f0;
            border-radius: 6px;
            font-size: 0.9em;
            border-left: 4px solid #667eea;
        }
        .status-item strong {
            color: #333;
        }
        
        .pill {
            display: inline-block;
            background: #e0e0e0;
            padding: 4px 10px;
            border-radius: 20px;
            font-size: 0.85em;
            margin: 2px;
        }
        .pill.success {
            background: #c8e6c9;
            color: #2e7d32;
        }
        .pill.error {
            background: #ffcdd2;
            color: #c62828;
        }
        .pill.info {
            background: #bbdefb;
            color: #1565c0;
        }
    </style>
</head>
<body>
    <div class="container">
        <!-- Left Panel: Chess Board -->
        <div class="panel">
            <h2>♟ Chess Board</h2>
            <div class="chessboard" id="board"></div>
            <div class="controls">
                <input type="text" id="moveInput" placeholder="Enter move (e.g., e2e4)" maxlength="4" />
                <button onclick="sendMove()">Send Move</button>
            </div>
            <div class="status">
                <div class="status-item">Turn: <span id="turn" class="pill info">White</span></div>
                <div class="status-item">Status: <span id="status" class="pill">Idle</span></div>
            </div>
            <div class="logs" id="logs"></div>
        </div>
        
        <!-- Right Panel: 3D Visualization -->
        <div class="panel">
            <h2>🤖 Robot Waypoints 3D</h2>
            <canvas id="canvas"></canvas>
        </div>
    </div>

    <script>
        const PIECES = {
            'P': '♟', 'N': '♞', 'B': '♝', 'R': '♜', 'Q': '♛', 'K': '♚',
            'p': '♙', 'n': '♘', 'b': '♗', 'r': '♖', 'q': '♕', 'k': '♔'
        };
        
        let currentBoard = null;
        let selectedFrom = null;
        let scene, camera, renderer;
        const waypointData = [];
        
        // Initialize 3D scene
        function initThreeJS() {
            const canvas = document.getElementById('canvas');
            scene = new THREE.Scene();
            scene.background = new THREE.Color(0xf0f0f0);
            
            camera = new THREE.PerspectiveCamera(75, canvas.clientWidth / canvas.clientHeight, 0.1, 1000);
            camera.position.set(5, 5, 5);
            camera.lookAt(0, 0, 0);
            
            renderer = new THREE.WebGLRenderer({ canvas, antialias: true });
            renderer.setSize(canvas.clientWidth, canvas.clientHeight);
            
            // Lighting
            const ambientLight = new THREE.AmbientLight(0xffffff, 0.7);
            scene.add(ambientLight);
            const directionalLight = new THREE.DirectionalLight(0xffffff, 0.5);
            directionalLight.position.set(10, 10, 10);
            scene.add(directionalLight);
            
            // Add axes helper
            const axesHelper = new THREE.AxesHelper(5);
            scene.add(axesHelper);
            
            animate();
        }
        
        function animate() {
            requestAnimationFrame(animate);
            renderer.render(scene, camera);
        }
        
        function drawWaypoints(waypoints) {
            // Remove previous waypoint visualization
            scene.children = scene.children.filter(child => !(child instanceof THREE.Line) && !(child instanceof THREE.Points));
            
            if (!waypoints || waypoints.length === 0) return;
            
            // Draw line through waypoints
            const points = waypoints.map(w => new THREE.Vector3(w[0], w[1], w[2]));
            const geometry = new THREE.BufferGeometry().setFromPoints(points);
            const material = new THREE.LineBasicMaterial({ color: 0xff0000, linewidth: 3 });
            const line = new THREE.Line(geometry, material);
            scene.add(line);
            
            // Draw spheres at waypoints
            points.forEach((point, idx) => {
                const sphere = new THREE.Mesh(
                    new THREE.SphereGeometry(0.2, 8, 8),
                    new THREE.MeshPhongMaterial({ color: idx === 0 ? 0x00ff00 : idx === points.length - 1 ? 0xff0000 : 0x0088ff })
                );
                sphere.position.copy(point);
                scene.add(sphere);
                
                // Add text labels
                const canvas = new THREE.CanvasTexture(createTextTexture(String(idx)));
                const label = new THREE.Mesh(
                    new THREE.PlaneGeometry(0.5, 0.5),
                    new THREE.MeshBasicMaterial({ map: canvas })
                );
                label.position.copy(point).addScaledVector(new THREE.Vector3(1, 1, 0), 0.3);
                scene.add(label);
            });
        }
        
        function createTextTexture(text) {
            const canvas = document.createElement('canvas');
            canvas.width = 64;
            canvas.height = 64;
            const ctx = canvas.getContext('2d');
            ctx.fillStyle = '#ffffff';
            ctx.fillRect(0, 0, 64, 64);
            ctx.fillStyle = '#000000';
            ctx.font = 'bold 40px Arial';
            ctx.textAlign = 'center';
            ctx.textBaseline = 'middle';
            ctx.fillText(text, 32, 32);
            return canvas;
        }
        
        // Render chess board
        function renderBoard(fen) {
            const boardEl = document.getElementById('board');
            boardEl.innerHTML = '';
            currentBoard = fenToBoard(fen);
            
            for (let rank = 7; rank >= 0; rank--) {
                for (let file = 0; file < 8; file++) {
                    const square = document.createElement('div');
                    const isLight = (rank + file) % 2 === 0;
                    const squareIdx = rank * 8 + file;
                    square.className = 'square ' + (isLight ? 'light' : 'dark');
                    const piece = currentBoard[squareIdx];
                    if (piece) {
                        square.textContent = PIECES[piece] || piece;
                    }
                    square.dataset.square = String.fromCharCode(97 + file) + (rank + 1);
                    square.onclick = () => selectSquare(square, String.fromCharCode(97 + file) + (rank + 1));
                    boardEl.appendChild(square);
                }
            }
        }
        
        function fenToBoard(fen) {
            const parts = fen.split(' ');
            const board = new Array(64).fill(null);
            const rows = parts[0].split('/');
            let idx = 0;
            for (const row of rows) {
                for (const char of row) {
                    if (isNaN(char)) {
                        board[idx] = char;
                        idx++;
                    } else {
                        idx += parseInt(char);
                    }
                }
            }
            return board;
        }
        
        function selectSquare(el, square) {
            document.querySelectorAll('.square.from, .square.to').forEach(s => {
                s.classList.remove('from', 'to');
            });
            if (!selectedFrom) {
                selectedFrom = square;
                el.classList.add('from');
            } else {
                el.classList.add('to');
                const move = selectedFrom + square;
                document.getElementById('moveInput').value = move;
                selectedFrom = null;
                sendMove();
            }
        }
        
        const ws = new WebSocket(`ws://${location.host}/ws`);
        
        ws.onopen = () => {
            addLog('Connected to Chess Robot', 'success');
        };
        
        ws.onmessage = (event) => {
            const msg = JSON.parse(event.data);
            handleMessage(msg);
        };
        
        ws.onerror = () => {
            addLog('WebSocket error', 'error');
        };
        
        ws.onclose = () => {
            addLog('Disconnected', 'error');
        };
        
        function handleMessage(msg) {
            switch(msg.type) {
                case 'board':
                    renderBoard(msg.fen);
                    document.getElementById('turn').textContent = msg.turn.toUpperCase();
                    document.getElementById('turn').className = 'pill ' + (msg.turn === 'white' ? 'info' : '');
                    break;
                case 'status':
                    const statusEl = document.getElementById('status');
                    statusEl.textContent = msg.state.toUpperCase();
                    statusEl.className = 'pill ' + (msg.state === 'error' ? 'error' : msg.state === 'idle' ? 'info' : '');
                    break;
                case 'log':
                    addLog(msg.msg, msg.level);
                    break;
                case 'robot_move':
                    addLog(`🤖 Engine Move: ${msg.uci}`, 'success');
                    break;
                case 'traj_coeffs':
                    // Could visualize trajectory data here
                    break;
                case 'error':
                    addLog(msg.error, 'error');
                    break;
            }
        }
        
        function sendMove() {
            const uci = document.getElementById('moveInput').value.trim().toLowerCase();
            if (uci.length !== 4) {
                addLog('Invalid move format (e.g., e2e4)', 'error');
                return;
            }
            addLog(`Sending move: ${uci}`, 'info');
            ws.send(JSON.stringify({ type: 'human_move', uci }));
            document.getElementById('moveInput').value = '';
        }
        
        function addLog(msg, level = 'info') {
            const logsEl = document.getElementById('logs');
            const entry = document.createElement('div');
            entry.className = 'log-entry ' + level;
            entry.textContent = `[${new Date().toLocaleTimeString()}] ${msg}`;
            logsEl.appendChild(entry);
            logsEl.scrollTop = logsEl.scrollHeight;
        }
        
        // Initialize
        window.addEventListener('load', () => {
            initThreeJS();
            ws.send(JSON.stringify({ type: 'get_board' }));
        });
        
        // Allow Enter key to send move
        document.addEventListener('keypress', (e) => {
            if (e.key === 'Enter' && document.getElementById('moveInput') === document.activeElement) {
                sendMove();
            }
        });
    </script>
</body>
</html>
""")
