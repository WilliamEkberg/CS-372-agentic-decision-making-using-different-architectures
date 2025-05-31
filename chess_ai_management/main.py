from openai import OpenAI
import os
import sys
import chess
import datetime # For timestamping results file entries

from config import OPENAI_API_KEY, STOCKFISH_PATH 

from data.fen_loader import load_fens_from_file
from evaluation.evaluator import Evaluator
from methods.single_agent_method import SingleAgentMethod
from methods.two_agent_debate_method import TwoAgentDebateMethod
from methods.Manager_analysts_method import ManagerAnalystsMethod

FINAL_RESULTS_FILENAME = "final_summary_results.txt"

def run_experiment(max_fens_to_test: int = None):
    """
    Runs the full chess AI experiment.

    Args:
        max_fens_to_test (int, optional): Maximum number of FENs to test. 
                                         If None, tests all loaded FENs.
    """
    print("--- Starting Chess AI Experiment --- ")


    fens = load_fens_from_file() # Uses default "100-chess-to-solve.txt"
    if not fens:
        print("Error: No FENs loaded from data/100-chess-to-solve.txt. Exiting experiment.")
        return

    if max_fens_to_test is not None and max_fens_to_test > 0:
        fens = fens[:max_fens_to_test]
        print(f"Loaded {len(fens)} FENs for the experiment (limited to first {max_fens_to_test}).")
    else:
        print(f"Loaded {len(fens)} FENs for the experiment.")

    # --- 2. Initialize Clients and Evaluator ---
    openai_client = None
    if OPENAI_API_KEY and OPENAI_API_KEY != "YOUR_API_KEY_HERE":
        try:
            openai_client = OpenAI() 
            print("OpenAI client initialized successfully.")
        except Exception as e:
            print(f"Error: Failed to initialize OpenAI client: {e}. \nCheck API key and openai library.")
            openai_client = None
    else:
        print("Warning: OPENAI_API_KEY not configured in config.py or is placeholder. \nOpenAI dependent methods may fail if key not in environment.")
        try:
            openai_client = OpenAI()
            print("OpenAI client initialized (likely from environment variable).")
        except Exception as e:
            print(f"Failed to initialize OpenAI client from environment either: {e}")
            openai_client = None

    evaluator = None
    if STOCKFISH_PATH:
        try:
            evaluator = Evaluator(stockfish_path=STOCKFISH_PATH)
            print("Stockfish Evaluator initialized successfully.")
        except Exception as e:
            print(f"Error: Failed to initialize Stockfish Evaluator: {e}. \nPlease check STOCKFISH_PATH in config.py and Stockfish installation.")
            evaluator = None
    else:
        print("Error: STOCKFISH_PATH not configured in config.py. Evaluation cannot proceed.")

    if not openai_client:
        print("Critical Error: OpenAI client could not be initialized. Cannot run AI methods. Exiting.")
        return
    if not evaluator:
        print("Critical Error: Stockfish Evaluator could not be initialized. Cannot score moves. Exiting.")
        return

    # 3. Initialize AI Methods
    methods_to_test = []
    The_model_name = "o4-mini"
    #The_model_name = "gpt-4.1"
    try:
        single_agent = SingleAgentMethod(openai_client=openai_client, agent_model_name=The_model_name)
        methods_to_test.append({"name": "SingleAgent", "method_obj": single_agent, "total_score": 0, "fens_processed_count": 0, "illegal_move_count": 0})
    except Exception as e:
        print(f"Error initializing SingleAgentMethod: {e}")

    try:
        debate_method = TwoAgentDebateMethod(openai_client=openai_client, model_name=The_model_name)
        methods_to_test.append({"name": f"TwoAgentDebate-{The_model_name}", "method_obj": debate_method, "total_score": 0, "fens_processed_count": 0, "illegal_move_count": 0})
    except Exception as e:
        print(f"Error initializing TwoAgentDebateMethod: {e}")

    try:
        manager_analyst_method = ManagerAnalystsMethod(openai_client=openai_client, 
                                                     manager_model=The_model_name, 
                                                     analyst_model=The_model_name)
        methods_to_test.append({
            "name": f"ManagerAnalysts-{The_model_name}_mgr-{The_model_name}_ana", 
            "method_obj": manager_analyst_method, 
            "total_score": 0, 
            "fens_processed_count": 0,
            "illegal_move_count": 0
        })
    except Exception as e:
        print(f"Error initializing ManagerAnalystsMethod: {e}")
    
    if not methods_to_test:
        print("No AI methods were initialized successfully. Exiting experiment.")
        return
    
    print(f"\nInitialized {len(methods_to_test)} AI method(s) for testing.")

    # --- 4. Experiment Loop ---
    for i, fen_from_file in enumerate(fens):
        print(f"\n--- Processing FEN {i+1}/{len(fens)}: {fen_from_file} ---")

        initial_board = None
        fen = None
        is_white_turn_for_fen = None
        try:
    
            initial_board = chess.Board(fen_from_file) 
            fen = initial_board.fen()
            is_white_turn_for_fen = (initial_board.turn == chess.WHITE)
            if fen_from_file != fen:
                print(f"  Info: Original FEN '{fen_from_file}' standardized to '{fen}' by python-chess.")
        except ValueError:
            print(f"  Warning: Invalid FEN format (could not be parsed by python-chess): {fen_from_file}. Skipping this FEN.")
            continue
        
        if not evaluator.stockfish.is_fen_valid(fen): 
             print(f"  Warning: Stockfish considers FEN invalid (even after python-chess completion): {fen}. Original from file: '{fen_from_file}'. Skipping.")
             continue

        current_fen_agent_moves = {} 
        current_fen_agent_eval_scores = {}
        any_method_produced_evaluable_move = False

        for agent_info in methods_to_test:
            method_name = agent_info["name"]
            method_obj = agent_info["method_obj"]
            agent_move_uci = None
            print(f"  Running method: {method_name}...")

            try:
                if hasattr(method_obj, 'decide_move'):
                    agent_move_uci = method_obj.decide_move(fen)
                elif hasattr(method_obj, 'run_debate'):
                    agent_move_uci, _ = method_obj.run_debate(fen)
                else:
                    print(f"    Error: Method {method_name} does not have a recognized execution function.")
                    continue
                
                agent_info["fens_processed_count"] += 1

                if agent_move_uci and isinstance(agent_move_uci, str) and "Error:" not in agent_move_uci.lower():
                    print(f"    {method_name} proposed: {agent_move_uci}")
                    current_fen_agent_moves[method_name] = agent_move_uci
                    
                    eval_dict = evaluator.get_evaluation_dict_after_move(fen, agent_move_uci)
                    
                    if eval_dict:
                        any_method_produced_evaluable_move = True
                        score = 0.0 # This will be White's POV score
                        MATE_VALUE_SCALE = 100000.0 
                        
                        # eval_dict is from perspective of player whose turn it is in new_fen.
                        # If White made the original move (`is_white_turn_for_fen == True`), then in `new_fen` it's Black's turn.
                        # If Black made the original move (`is_white_turn_for_fen == False`), then in `new_fen` it's White's turn.

                        if eval_dict["type"] == "mate":
                            mate_value_for_new_player = eval_dict["value"] # This value is for the player whose turn it is in new_fen

                            if is_white_turn_for_fen: 
                                # White made the original move; Black is to play in new_fen.
                                # mate_value_for_new_player is for Black.
                                if mate_value_for_new_player == 0: # Black is checkmated now (by White's last move)
                                    score = MATE_VALUE_SCALE    # Max positive for White
                                elif mate_value_for_new_player > 0: # Black (in new_fen) can mate White in X moves
                                    score = -MATE_VALUE_SCALE + mate_value_for_new_player # Very bad for White
                                else: # mate_value_for_new_player < 0 => Black (in new_fen) IS MATED by White in X moves
                                    score = MATE_VALUE_SCALE - abs(mate_value_for_new_player) # Very good for White
                            else: 
                                # Black made the original move; White is to play in new_fen.
                                # mate_value_for_new_player is for White.
                                if mate_value_for_new_player == 0: # White is checkmated now (by Black's last move)
                                    score = -MATE_VALUE_SCALE   # Max negative for White
                                elif mate_value_for_new_player > 0: # White (in new_fen) can mate Black in X moves
                                    score = MATE_VALUE_SCALE - mate_value_for_new_player # Very good for White
                                else: # mate_value_for_new_player < 0 => White (in new_fen) IS MATED by Black in X moves
                                    score = -MATE_VALUE_SCALE + abs(mate_value_for_new_player) # Very bad for White
                        
                        elif eval_dict["type"] == "cp":
                            cp_value_from_stockfish = float(eval_dict["value"])
                            # cp_value_from_stockfish is from the POV of whose turn it is in new_fen.
                            
                            if is_white_turn_for_fen: 
                                # White made the original move. In new_fen, it's Black's turn.
                                # So, cp_value_from_stockfish is Black's advantage.
                                # To get White's POV, we negate it.
                                score = -cp_value_from_stockfish 
                            else: 
                                # Black made the original move. In new_fen, it's White's turn.
                                # So, cp_value_from_stockfish is already White's advantage.
                                score = cp_value_from_stockfish
                        
                        current_fen_agent_eval_scores[method_name] = score
                        print(f"    {method_name} move {agent_move_uci} -> eval score (White's POV): {score:.2f} (raw: {eval_dict})")
                    else:
                        print(f"    Warning: Could not evaluate move '{agent_move_uci}' by {method_name}. Considered worst.")
                        current_fen_agent_eval_scores[method_name] = -float('inf') if is_white_turn_for_fen else float('inf')
                else:
                    print(f"    {method_name} did not produce a valid move string for evaluation (Proposed: {agent_move_uci}). Considered worst.")
                    current_fen_agent_eval_scores[method_name] = -float('inf') if is_white_turn_for_fen else float('inf')

            except Exception as e:
                print(f"    Error running method {method_name} for FEN {fen}: {e}")
                current_fen_agent_eval_scores[method_name] = -float('inf') if is_white_turn_for_fen else float('inf')

        if any_method_produced_evaluable_move and current_fen_agent_eval_scores:
            best_eval_score_for_this_fen = None
            if is_white_turn_for_fen:
                best_eval_score_for_this_fen = max(current_fen_agent_eval_scores.values())
            else:
                best_eval_score_for_this_fen = min(current_fen_agent_eval_scores.values())
            
            if best_eval_score_for_this_fen == -float('inf') or best_eval_score_for_this_fen == float('inf'):
                 print("  No method produced a practically evaluable move. No competitive points awarded.")
            else:
                methods_achieving_best_eval = []
                for method_name, eval_score in current_fen_agent_eval_scores.items():
                    if abs(eval_score - best_eval_score_for_this_fen) < 1e-9: # Tolerance for float comparison
                        methods_achieving_best_eval.append(method_name)
                
                if methods_achieving_best_eval:
                    print(f"  Best relative evaluation score for this FEN ({best_eval_score_for_this_fen:.2f}) achieved by: {', '.join(methods_achieving_best_eval)}")
                    for agent_info in methods_to_test:
                        if agent_info["name"] in methods_achieving_best_eval:
                            agent_info["total_score"] += 1
        elif current_fen_agent_eval_scores:
             print(f"  No method produced a practically evaluable move for FEN {fen}. No competitive points awarded.")
        else:
            print(f"  No method produced moves for evaluation for FEN {fen}. No competitive points awarded.")


    # 5. Aggregate and Collect Results for File and Console 
    print("\n--- Experiment Finished ---") # Console only
    print("\n--- Final Results (Scored by best move among competitors) ---") # Console only

    final_results_lines = [] # List to hold lines for the results file
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    final_results_lines.append(f"\n=== Experiment Run: {timestamp} ===")
    final_results_lines.append("--- Final Results (Scored by best move among competitors) ---")

    if not any(agent_info["fens_processed_count"] > 0 for agent_info in methods_to_test):
        no_fens_message = "No FENs were processed by any method for scoring."
        print(no_fens_message) # Console
        final_results_lines.append(no_fens_message) # File
    else:
        for agent_info in methods_to_test:
            method_name = agent_info["name"]
            score = agent_info["total_score"]
            fens_processed_for_method = agent_info["fens_processed_count"]
            illegal_moves = agent_info["illegal_move_count"]
            
            # Lines for console
            print(f"  Method: {method_name}")
            if fens_processed_for_method > 0:
                percentage_best_moves = (score / fens_processed_for_method) * 100
                print(f"    Total Points (achieved best relative eval): {score} out of {fens_processed_for_method} FENs where method produced output")
                print(f"    Percentage of Best Relative Moves: {percentage_best_moves:.2f}%")
                print(f"    Illegal/Malformed Moves Proposed: {illegal_moves}")
            else:
                print(f"    No FENs were processed/attempted by this method.")

            # Lines for file
            final_results_lines.append(f"  Method: {method_name}")
            if fens_processed_for_method > 0:
                percentage_best_moves = (score / fens_processed_for_method) * 100
                final_results_lines.append(f"    Total Points (achieved best relative eval): {score} out of {fens_processed_for_method} FENs where method produced output")
                final_results_lines.append(f"    Percentage of Best Relative Moves: {percentage_best_moves:.2f}%")
                final_results_lines.append(f"    Illegal/Malformed Moves Proposed: {illegal_moves}")
            else:
                final_results_lines.append(f"    No FENs were processed/attempted by this method.")
    
    final_results_lines.append(f"--- End of Experiment Run: {timestamp} ---\n")

    # Write collected results to the file
    try:
        with open(FINAL_RESULTS_FILENAME, 'a', encoding='utf-8') as f_out:
            for line in final_results_lines:
                f_out.write(line + "\n")
        print(f"\nFinal summary results also written to {FINAL_RESULTS_FILENAME}")
    except Exception as e:
        print(f"\nError writing final summary results to file: {e}")

if __name__ == "__main__":
    # Basic pre-checks for config, actual error handling is in initializations
    if not OPENAI_API_KEY or OPENAI_API_KEY == "YOUR_API_KEY_HERE":
        print("INFO: OPENAI_API_KEY in config.py is not set or is placeholder. Will try environment variable.")
    
    if not STOCKFISH_PATH:
        print("CRITICAL INFO: STOCKFISH_PATH is not set in config.py. Evaluation will fail if not found elsewhere.")

    #run_experiment(max_fens_to_test=1) 
    run_experiment() 