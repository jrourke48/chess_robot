"""!
@file server.py
@brief FastAPI web server for the Chess Robot UI
"""

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from chessrobotclasses.ChessRobotUI import app

# Import the app that's already configured in ChessRobotUI module

# Add CORS middleware to allow requests from any origin
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)
