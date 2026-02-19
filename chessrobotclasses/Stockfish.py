import subprocess

class Stockfish:
    def __init__(self, path="/usr/games/stockfish"):
        self.process = subprocess.Popen(
            [path],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            bufsize=1
        )
        self.send("uci")
        # Wait for uciok
        while True:
            line = self.process.stdout.readline()
            if "uciok" in line:
                break
    
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