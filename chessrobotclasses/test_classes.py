#!/usr/bin/env python3
"""
Test script for ChessBoard, Stockfish, and RobotMotionPlanner classes.
Tests the complete workflow: FEN validation -> Stockfish move -> Coordinate output
"""

from ChessBoard import ChessBoard
from RobotMotionPlanner import RobotMotionPlanner
from Stockfish import Stockfish

def test_full_workflow():
    """Test the complete workflow from FEN validation to coordinate output"""
    
    print("=" * 70)
    print("Testing Chess Robot Classes Workflow")
    print("=" * 70)
    
    # Test FENs
    test_cases = [
        {
            "name": "Starting Position",
            "fen": "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"
        },
        {
            "name": "Position after 1.e4",
            "fen": "rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq e3 0 1"
        },
        {
            "name": "Mid-game Position",
            "fen": "r1bqkbnr/pppp1ppp/2n5/4p3/4P3/5N2/PPPP1PPP/RNBQKB1R w KQkq - 2 3"
        },
        {
            "name": "Invalid FEN (should fail)",
            "fen": "invalid_fen_string"
        }
    ]
    
    # Initialize RobotMotionPlanner (which initializes ChessBoard and Stockfish)
    print("\n[1] Initializing RobotMotionPlanner...")
    try:
        robot_planner = RobotMotionPlanner()
        print("✓ RobotMotionPlanner initialized successfully")
    except Exception as e:
        print(f"✗ Failed to initialize RobotMotionPlanner: {e}")
        return
    
    # Test each FEN
    for i, test_case in enumerate(test_cases, 1):
        print("\n" + "-" * 70)
        print(f"Test Case {i}: {test_case['name']}")
        print("-" * 70)
        print(f"FEN: {test_case['fen']}")
        
        # Step 1: Validate FEN and get best move from Stockfish
        print("\n[2] Validating FEN and getting best move from Stockfish...")
        is_valid, message = robot_planner.chessboard.checkState_thenRun(test_case['fen'])
        
        if not is_valid:
            print(f"✗ Validation failed: {message}")
            continue
        
        print(f"✓ {message}")
        
        # Step 2: Get the best move
        best_move = robot_planner.chessboard.best_move
        print(f"\n[3] Best move from Stockfish: {best_move}")
        
        if best_move is None:
            print("✗ No move available")
            continue
        
        # Step 3: Convert move to coordinates
        print("\n[4] Converting move to robot coordinates...")
        try:
            start_pos, end_pos = robot_planner.chessmove_to_coordinates()
            print(f"✓ Start position (inches): ({start_pos[0]:.2f}, {start_pos[1]:.2f})")
            print(f"✓ End position (inches): ({end_pos[0]:.2f}, {end_pos[1]:.2f})")
            
            # Calculate distance
            distance = ((end_pos[0] - start_pos[0])**2 + (end_pos[1] - start_pos[1])**2)**0.5
            print(f"✓ Move distance: {distance:.2f} inches")
            
        except Exception as e:
            print(f"✗ Failed to convert to coordinates: {e}")
            continue
        
        # Step 4: Display the current board state
        print("\n[5] Current board state:")
        robot_planner.chessboard.display_board()
    
    # Clean up
    print("\n" + "=" * 70)
    print("Cleaning up...")
    print("=" * 70)
    robot_planner.chessboard.engine.quit()
    print("✓ Stockfish engine closed")
    print("\nAll tests completed!")


def test_individual_classes():
    """Test each class individually"""
    
    print("\n" + "=" * 70)
    print("Testing Individual Classes")
    print("=" * 70)
    
    # Test 1: Stockfish class
    print("\n[Test 1] Stockfish Class")
    print("-" * 70)
    try:
        engine = Stockfish()
        fen = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"
        move = engine.get_best_move(fen, depth=10)
        print(f"✓ Stockfish initialized and returned move: {move}")
        engine.quit()
    except Exception as e:
        print(f"✗ Stockfish test failed: {e}")
    
    # Test 2: ChessBoard class
    print("\n[Test 2] ChessBoard Class")
    print("-" * 70)
    try:
        chessboard = ChessBoard(engine_path="/usr/games/stockfish")
        
        # Test FEN validation
        fen = "rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq e3 0 1"
        is_valid, msg = chessboard.checkState_thenRun(fen)
        print(f"✓ FEN validation: {is_valid}, {msg}")
        print(f"✓ Best move: {chessboard.best_move}")
        
        # Test state update
        success, msg = chessboard.update_state("e7e5")
        print(f"✓ State update: {success}, {msg}")
        
        chessboard.engine.quit()
    except Exception as e:
        print(f"✗ ChessBoard test failed: {e}")
    
    # Test 3: RobotMotionPlanner class
    print("\n[Test 3] RobotMotionPlanner Class")
    print("-" * 70)
    try:
        planner = RobotMotionPlanner()
        
        # Set a known move
        planner.chessboard.best_move = "e2e4"
        start, end = planner.chessmove_to_coordinates()
        print(f"✓ Move e2e4 coordinates:")
        print(f"  Start: ({start[0]:.2f}, {start[1]:.2f})")
        print(f"  End: ({end[0]:.2f}, {end[1]:.2f})")
        
        planner.chessboard.engine.quit()
    except Exception as e:
        print(f"✗ RobotMotionPlanner test failed: {e}")


if __name__ == "__main__":
    # Run full workflow test
    test_full_workflow()
    
    # Uncomment to run individual class tests
    # test_individual_classes()
