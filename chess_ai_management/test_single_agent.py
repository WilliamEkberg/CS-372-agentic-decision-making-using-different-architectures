import sys
import os
import chess


PROJECT_ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
if PROJECT_ROOT_DIR not in sys.path:
    sys.path.insert(0, PROJECT_ROOT_DIR)

from openai import OpenAI
from methods.single_agent_method import SingleAgentMethod
from evaluation.evaluator import Evaluator # To check evaluation of valid moves
from config import OPENAI_API_KEY, STOCKFISH_PATH # For initialization

def run_test():
    print("--- Testing SingleAgentMethod ---")

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

    # --- 2. Initialize Evaluator (optional, for checking evaluation of valid moves) ---
    evaluator = None
    if STOCKFISH_PATH:
        try:
            evaluator = Evaluator(stockfish_path=STOCKFISH_PATH)
            print("Stockfish Evaluator initialized successfully (for path check).")
        except Exception as e:
            print(f"Warning: Failed to initialize Stockfish Evaluator: {e}.")
    else:
        print("Warning: STOCKFISH_PATH not configured.")

    # --- 3. Initialize SingleAgentMethod ---
    try:
        # You can change the agent_model_name if needed
        single_agent_method = SingleAgentMethod(openai_client=openai_client, agent_model_name="gpt-4o")
        print(f"SingleAgentMethod initialized with agent model '{single_agent_method.agent.model_name if single_agent_method.agent else 'None'}'.")
    except Exception as e:
        print(f"Error initializing SingleAgentMethod: {e}")
        return

    if not single_agent_method.agent:
        print("Critical Error: SingleAgentMethod could not initialize its internal MoveProposingAgent. Exiting test.")
        return

    # --- 4. Define Test FENs ---
    test_fens = {
        "Standard Start": "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1",
        "FEN 1 (Problematic Previously)": "4R3/8/8/2Pkp3/N7/4rnKB/1nb5/b1r5", 
        "FEN 2 (Problematic Previously)": "b1B3Q1/5K2/5NP1/n7/2p2k1P/3pN2R/1B1P4/4qn2",
        "White to play, clear move": "r1bqkbnr/pp1ppppp/2n5/2p5/4P3/5N2/PPPP1PPP/RNBQKB1R w KQkq - 2 3" # e.g. Bc4 or Bb5
    }

    for name, fen_str_original in test_fens.items():
        print(f"\n--- Test Case: {name} ---")
        print(f"Original FEN: {fen_str_original}")

        try:
            board = chess.Board(fen_str_original)
            current_fen = board.fen() # Standardize FEN
            if fen_str_original != current_fen:
                 print(f"Standardized FEN: {current_fen}")
        except ValueError:
            print(f"  ERROR: Could not parse original FEN string with python-chess: {fen_str_original}. Skipping test case.")
            continue

        print(f"Requesting move from SingleAgentMethod for FEN: {current_fen}")
        proposed_move_uci = single_agent_method.decide_move(current_fen)

        print(f"  Agent proposed: {proposed_move_uci}")

        if proposed_move_uci and isinstance(proposed_move_uci, str) and "Error:" not in proposed_move_uci:
            try:
                board_check = chess.Board(current_fen)
                move_obj = chess.Move.from_uci(proposed_move_uci)
                if move_obj in board_check.legal_moves:
                    print(f"  VALIDATION: Move '{proposed_move_uci}' is LEGAL for FEN '{current_fen}'.")
                    if evaluator:
                        eval_dict = evaluator.get_evaluation_dict_after_move(current_fen, proposed_move_uci)
                        if eval_dict:
                            print(f"    Stockfish evaluation of position after '{proposed_move_uci}': {eval_dict}")
                        else:
                            print(f"    Could not get Stockfish evaluation for position after '{proposed_move_uci}'.")
                    else:
                        print("    (Skipping Stockfish evaluation as evaluator was not initialized)")
                else:
                    print(f"  VALIDATION ERROR: Move '{proposed_move_uci}' is ILLEGAL for FEN '{current_fen}'. Legal moves: {[m.uci() for m in board_check.legal_moves]}")
            except ValueError:
                print(f"  VALIDATION ERROR: Proposed move '{proposed_move_uci}' is not in valid UCI format.")
            except Exception as e:
                 print(f"  VALIDATION ERROR: Unexpected error during local validation for '{proposed_move_uci}': {e}")
        elif proposed_move_uci and "Error:" in proposed_move_uci:
            print(f"  AGENT ERROR: SingleAgentMethod returned an error message: {proposed_move_uci}")
        else:
            print("  AGENT BEHAVIOR: SingleAgentMethod did not return a typical move string or error.")

    print("\n--- Test Finished ---")

if __name__ == "__main__":
    run_test() 