import sys
import os
import chess

PROJECT_ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
if PROJECT_ROOT_DIR not in sys.path:
    sys.path.insert(0, PROJECT_ROOT_DIR)

from openai import OpenAI
from methods.two_agent_debate_method import TwoAgentDebateMethod
from evaluation.evaluator import Evaluator # To ensure it can be initialized
from config import OPENAI_API_KEY, STOCKFISH_PATH # For initialization

def run_test():
    print("--- Testing TwoAgentDebateMethod ---")

    # --- 1. Initialize OpenAI Client ---
    openai_client = None
    if OPENAI_API_KEY and OPENAI_API_KEY != "YOUR_API_KEY_HERE":
        try:
            openai_client = OpenAI()
            print("OpenAI client initialized successfully.")
        except Exception as e:
            print(f"Error: Failed to initialize OpenAI client: {e}. Check API key.")
            return
    else:
        print("Warning: OPENAI_API_KEY not configured. Will try environment variable.")
        try:
            openai_client = OpenAI()
            print("OpenAI client initialized (likely from environment variable).")
        except Exception as e:
            print(f"Failed to initialize OpenAI client from environment: {e}")
            return
            
    if not openai_client:
        print("Critical Error: OpenAI client could not be initialized. Exiting test.")
        return

    evaluator = None
    if STOCKFISH_PATH:
        try:
            evaluator = Evaluator(stockfish_path=STOCKFISH_PATH)
            print("Stockfish Evaluator initialized successfully (for path check).")
        except Exception as e:
            print(f"Warning: Failed to initialize Stockfish Evaluator: {e}. This test can proceed but main experiment might fail.")
            # We can continue this specific test if Stockfish isn't the focus here
    else:
        print("Warning: STOCKFISH_PATH not configured. Main experiment would fail.")

    try:
        debate_method = TwoAgentDebateMethod(openai_client=openai_client, model_name="gpt-4o")
        print(f"TwoAgentDebateMethod initialized with model '{debate_method.model_name}'.")
    except Exception as e:
        print(f"Error initializing TwoAgentDebateMethod: {e}")
        return

    test_fens = {
        "Standard Start": "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1",
        "FEN 1 (Prev Invalid)": "4R3/8/8/2Pkp3/N7/4rnKB/1nb5/b1r5"#, # Will be completed by python-chess
        #"FEN 2 (Prev Invalid)": "b1B3Q1/5K2/5NP1/n7/2p2k1P/3pN2R/1B1P4/4qn2", # Will be completed
        #"FEN 3 (Prev Had Issues)": "1k6/1P5Q/8/7B/8/5K2/8/8", # Will be completed
    }

    for name, fen_str_original in test_fens.items():
        print(f"\\n--- Test Case: {name} ---")
        print(f"Original FEN: {fen_str_original}")

        try:
            board = chess.Board(fen_str_original)
            current_fen = board.fen()
            if fen_str_original != current_fen:
                 print(f"Standardized FEN: {current_fen}")
        except ValueError:
            print(f"  ERROR: Could not parse original FEN string with python-chess: {fen_str_original}. Skipping this test case.")
            continue
        
        print(f"Running debate for FEN: {current_fen}")
        final_move_uci, debate_transcript = debate_method.run_debate(fen_position=current_fen)

        print("\\n--- Debate Result ---")
        if final_move_uci:
            print(f"Extracted and Validated Final Move: {final_move_uci}")
            if evaluator:
                eval_dict = evaluator.get_evaluation_dict_after_move(current_fen, final_move_uci)
                if eval_dict:
                    print(f"  Stockfish evaluation of position after '{final_move_uci}': {eval_dict}")
                else:
                    print(f"  Could not get Stockfish evaluation for position after '{final_move_uci}'. (Move might be illegal despite internal check - this would be a discrepancy)")
            else:
                print("  (Skipping Stockfish evaluation as evaluator was not initialized)")
        else:
            print("No valid final move was extracted or validated from the debate.")
            print("Showing last 5 entries of transcript for context:")
            for entry in debate_transcript[-5:]: # Print last few entries
                 print(f"  Round {entry['round']} - {entry['speaker']}: {entry['text'][:200]}...") # Truncate long text

    print("\\n--- Test Finished ---")

if __name__ == "__main__":
    run_test() 