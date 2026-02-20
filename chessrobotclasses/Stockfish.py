import subprocess
import os
import shutil
from pathlib import Path

class Stockfish:
    def __init__(self, path=None):
        resolved_path = self._resolve_path(path)
        self.process = subprocess.Popen(
            [resolved_path],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            bufsize=1
        )
        self.path = resolved_path
        self.send("uci")
        # Wait for uciok
        while True:
            line = self.process.stdout.readline()
            if "uciok" in line:
                break

    def _resolve_path(self, path):
        if path:
            return path

        candidates = []
        repo_root = Path(__file__).resolve().parent.parent

        env_path = os.getenv("STOCKFISH_PATH")
        if env_path:
            candidates.append(env_path)

        candidates.extend([
            repo_root / "stockfishwindow" / "stockfish-windows-x86-64-avx2.exe",
            repo_root / "stockfishlinux" / "stockfish-ubuntu-x86-64-avx2",
        ])

        in_path = shutil.which("stockfish")
        if in_path:
            candidates.append(in_path)

        candidates.extend([
            "/usr/games/stockfish",
            "/usr/local/bin/stockfish",
            "/opt/homebrew/bin/stockfish",
            "C:/Program Files/Stockfish/stockfish.exe",
            "C:/stockfish/stockfish.exe",
        ])

        for candidate in candidates:
            if candidate and Path(candidate).exists():
                return str(candidate)

        raise FileNotFoundError(
            "Stockfish executable not found. Install Stockfish and either add it to PATH or set STOCKFISH_PATH."
        )
    
    def send(self, command):
        self.process.stdin.write(command + "\n")
        self.process.stdin.flush()
    
    def get_best_move(self, fen, depth=20):
        self.send(f"position fen {fen}")
        self.send(f"go depth {depth}")
        
        best_move = None
        while True:
            line = self.process.stdout.readline()
            if line.startswith("bestmove"):
                best_move = line.split()[1]
                break
        return best_move
    
    def quit(self):
        self.send("quit")
        self.process.wait()