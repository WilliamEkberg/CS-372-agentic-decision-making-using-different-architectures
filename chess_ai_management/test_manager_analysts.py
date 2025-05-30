import sys
import os

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
# Get the parent of that (e.g. .../Final project)
PROJECT_ROOT_PARENT = os.path.dirname(SCRIPT_DIR) 
if PROJECT_ROOT_PARENT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT_PARENT)


from openai import OpenAI
from chess_ai_management.methods.Manager_analysts_method import ManagerAnalystsMethod # Adjusted import
from chess_ai_management.config import OPENAI_API_KEY # Adjusted import


def run_manager_analysts_test():
    print("--- Testing ManagerAnalystsMethod (Agent-Only Validation Flow) ---")

    # --- 1. Initialize OpenAI Client ---
    openai_client = None
    if OPENAI_API_KEY and OPENAI_API_KEY != "YOUR_API_KEY_HERE":
        try:
            openai_client = OpenAI() # Relies on OPENAI_API_KEY env var or direct setting
            print("OpenAI client initialized successfully.")
        except Exception as e:
            print(f"Error: Failed to initialize OpenAI client: {e}. Check API key.")
            return
    else:
        print("Warning: OPENAI_API_KEY not configured in config.py or is placeholder. Will try environment variable.")
        try:
            openai_client = OpenAI() 
            print("OpenAI client initialized (likely from environment variable).")
        except Exception as e:
            print(f"Failed to initialize OpenAI client from environment: {e}")
            return
            
    if not openai_client:
        print("Critical Error: OpenAI client could not be initialized. Exiting test.")
        return

    # --- 2. Initialize ManagerAnalystsMethod ---
    try:
        manager_method = ManagerAnalystsMethod(
            openai_client=openai_client,
            manager_model="gpt-4o", 
            analyst_model="gpt-4o-mini" 
        )
        print(f"ManagerAnalystsMethod initialized with Manager: '{manager_method.manager_model_name}', Analysts/PA Service: '{manager_method.analyst_model_name}'.")
    except Exception as e:
        print(f"Error initializing ManagerAnalystsMethod: {e}")
        return

    # --- 3. Define Test FEN ---
    # A standard opening position
    test_fen = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"
    # A slightly more complex FEN
    # test_fen = "r1bqkbnr/pp1ppppp/2n5/2p5/4P3/5N2/PPPP1PPP/RNBQKB1R w KQkq - 2 3"
    # FEN for promotion
    # test_fen = "1k6/1P5Q/8/7B/8/5K2/8/8 w - - 0 1" 
    # FEN where previous methods struggled with legality
    # test_fen = "4R3/8/8/2Pkp3/N7/4rnKB/1nb5/b1r5 w - - 0 1" # Will be standardized by python-chess if that lib was used
                                                              # For this pure agent test, we pass the FEN as is.
                                                              # The PA LLM needs to handle it.
                                                              # The ManagerAnalystsMethod currently doesn't standardize FENs internally.
                                                              # Let's use a full FEN for the PA.
    if "w" not in test_fen and "b" not in test_fen.split(" ")[1]: # Quick check if FEN is incomplete
        try:
            import chess
            board = chess.Board(test_fen)
            completed_fen = board.fen()
            print(f"Original FEN '{test_fen}' appears incomplete, completed to '{completed_fen}' for PA.")
            test_fen = completed_fen
        except ImportError:
            print(f"Warning: chess library not found, passing FEN '{test_fen}' as is. PA might struggle if incomplete.")
        except ValueError:
            print(f"Warning: FEN '{test_fen}' could not be parsed by chess library. Passing as is.")


    print(f"\n--- Test Case FEN: {test_fen} ---")
    
    # The ManagerAnalystsMethod already has internal print statements for flow.
    final_decision_uci = manager_method.decide_move(test_fen)

    print("\n--- ManagerAnalystsMethod Test Script Final Result ---")
    if "Error:" in final_decision_uci:
        print(f"Method failed with error: {final_decision_uci}")
    else:
        print(f"Method decided on move: {final_decision_uci}")
    
        print("\n--- External Verification (Optional) ---")
        try:
            import chess
            board_verify = chess.Board(test_fen)
            move_obj_verify = chess.Move.from_uci(final_decision_uci)
            if move_obj_verify in board_verify.legal_moves:
                print(f"  Verification (python-chess): Move '{final_decision_uci}' IS legal for FEN '{test_fen}'. PA LLM was correct.")
            else:
                print(f"  VERIFICATION ERROR (python-chess): Move '{final_decision_uci}' IS ILLEGAL for FEN '{test_fen}'. PA LLM was incorrect.")
        except ImportError:
            print("  (Skipping python-chess verification as 'chess' library is not available for this external check)")
        except ValueError: # Handles if final_decision_uci is not valid UCI format
            print(f"  VERIFICATION ERROR (python-chess): Final decision '{final_decision_uci}' is not valid UCI format.")
        except Exception as e_ver:
            print(f"  Verification error: {e_ver}")


    print("\n--- Test Finished ---")

if __name__ == "__main__":
    run_manager_analysts_test()