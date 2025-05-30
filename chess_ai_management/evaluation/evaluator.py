from stockfish import Stockfish
import chess
import sys
import os

current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
if project_root not in sys.path:
    sys.path.insert(0, project_root)

try:
    from config import STOCKFISH_PATH
except ImportError:
    print("Error: Could not import STOCKFISH_PATH from config.py.")
    print("Please ensure config.py exists in the chess_ai_management root and STOCKFISH_PATH is set.")
    STOCKFISH_PATH = None 

class Evaluator:
    """
    Handles interaction with the Stockfish chess engine to provide an evaluation
    for a position resulting from a given move, using python-chess for board operations.
    """
    DEFAULT_STOCKFISH_PARAMS = {
        "Debug Log File": "", "Contempt": 0, "Min Split Depth": 0,
        "Threads": 1, "Ponder": "false", "Hash": 128, "MultiPV": 1,
        "Skill Level": 20, "Move Overhead": 10, "Minimum Thinking Time": 20, # Lowered for faster static evals
        "Slow Mover": 100, "UCI_Chess960": "false", "UCI_LimitStrength": "false",
    }

    def __init__(self, stockfish_path: str = None, stockfish_parameters: dict = None):
        resolved_path = stockfish_path if stockfish_path else STOCKFISH_PATH
        if not resolved_path:
            raise ValueError("Stockfish path must be provided either as an argument or in config.py.")
        
        try:
            self.stockfish = Stockfish(path=resolved_path)
        except Exception as e:
            if "No such file or directory" in str(e) or "OSError" in str(e):
                raise FileNotFoundError(
                    f"Stockfish executable not found or could not be run at path: '{resolved_path}'. "
                    f"Please ensure Stockfish is installed and the path is correct. Original error: {e}"
                ) from e
            raise RuntimeError(f"Failed to initialize Stockfish engine: {e}") from e

        params_to_set = stockfish_parameters if stockfish_parameters else self.DEFAULT_STOCKFISH_PARAMS
        try:
            # Set a reasonable depth for static evaluations
            self.stockfish.set_depth(15) 
            self.stockfish.update_engine_parameters(params_to_set)
            print(f"Stockfish initialized successfully from path: {resolved_path}")
        except Exception as e:
            print(f"Warning: Could not set all Stockfish parameters: {e}.")

    def _get_static_evaluation(self, fen_position: str) -> dict | None:
        """
        Gets Stockfish's static evaluation for a given FEN position.
        """
        if not self.stockfish.is_fen_valid(fen_position): # Stockfish's own FEN check
            print(f"Error (_get_static_evaluation): Invalid FEN provided to Stockfish: {fen_position}")
            return None
        try:
            self.stockfish.set_fen_position(fen_position)
            evaluation = self.stockfish.get_evaluation()
            return evaluation
        except Exception as e:
            print(f"Error (_get_static_evaluation) for FEN '{fen_position}': {e}")
            return None

    def get_evaluation_dict_after_move(self, start_fen: str, move_uci: str) -> dict | None:

        try:
            board = chess.Board(start_fen)
        except ValueError as e: 
            print(f"Error (get_evaluation_dict_after_move): Invalid starting FEN \'{start_fen}\': {e}")
            return None

        try:
            move = chess.Move.from_uci(move_uci)
            if move not in board.legal_moves:
                print(f"Error (get_evaluation_dict_after_move): Illegal move \'{move_uci}\' for FEN \'{start_fen}\'.")
                return None
            
            board.push(move)
            new_fen = board.fen()
            
            return self._get_static_evaluation(new_fen)

        except ValueError as e: 
            print(f"Error (get_evaluation_dict_after_move): Could not parse UCI move \'{move_uci}\': {e}")
            return None
        except Exception as e: 
            print(f"Error (get_evaluation_dict_after_move) processing move \'{move_uci}\' on FEN \'{start_fen}\': {e}")
            return None

    def get_stockfish_best_move(self, fen_position: str, thinking_time_ms: int = 1000) -> str | None:
        """ Gets Stockfish's best move for a given FEN. (Utility method) """
        if not self.stockfish.is_fen_valid(fen_position):
            print(f"Error (get_stockfish_best_move): Invalid FEN: {fen_position}")
            return None
        try:
            self.stockfish.set_fen_position(fen_position) 
            best_move = self.stockfish.get_best_move_time(thinking_time_ms)
            return best_move
        except Exception as e:
            print(f"Error getting best move from Stockfish for FEN '{fen_position}': {e}")
            return None

if __name__ == '__main__':
    print("Testing Simplified Evaluator (with python-chess) & Stockfish...")
    if not STOCKFISH_PATH:
        print("Skipping test: STOCKFISH_PATH not configured.")
    else:
        try:
            evaluator = Evaluator() 
            print("Evaluator initialized.")

            start_fen = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"
            
            print(f"\n--- Testing get_evaluation_dict_after_move for various moves from FEN: {start_fen} ---")
            
            moves_to_test = {
                "e2e4": "(Strong opening)",
                "d2d4": "(Strong opening)",
                "g1f3": "(Good development)",
                "a2a3": "(Passive)",
                "e7e5": "(Illegal - Black's piece/turn, White to play)", # python-chess should catch as illegal
                "e1g1": "(Illegal - Castling through occupied squares from start)" # python-chess should catch
            }
            
            for move, desc in moves_to_test.items():
                eval_dict = evaluator.get_evaluation_dict_after_move(start_fen, move)
                if eval_dict and 'type' in eval_dict and 'value' in eval_dict:
                    print(f"  Evaluation dict after '{move}' {desc}: {{type: '{eval_dict['type']}', value: {eval_dict['value']}}}")
                else:
                    print(f"  Could not get evaluation dict for move '{move}' {desc}. Result: {eval_dict}")
        
            # Test a position with a mate
            # Classic back-rank mate setup for White
            back_rank_mate_fen = "6k1/5ppp/8/8/8/8/5PPP/R4RK1 w - - 0 1" # White to play Ra8#
            print(f"\n--- Testing mate position (Ra8#): {back_rank_mate_fen} ---")
            eval_dict_mate = evaluator.get_evaluation_dict_after_move(back_rank_mate_fen, "a1a8") 
            if eval_dict_mate:
                print(f"  Evaluation dict after 'a1a8#': {{type: '{eval_dict_mate['type']}', value: {eval_dict_mate['value']}}}")
            else:
                print("  Could not get evaluation dict for a1a8#.")

            # Test a position where White is about to be mated by Black
            white_is_about_to_be_mated_fen = "rnbqkbnr/pppp1ppp/8/4p3/6P1/5P2/PPPPP2P/RNBQKBNR b KQkq g3 0 2"
            print(f"\n--- Testing position where White is about to be mated by Black (FEN: {white_is_about_to_be_mated_fen}) ---")
            eval_dict_white_mated = evaluator._get_static_evaluation(white_is_about_to_be_mated_fen)
            if eval_dict_white_mated:
                print(f"  Static evaluation of this position: {{type: '{eval_dict_white_mated['type']}', value: {eval_dict_white_mated['value']}}}")
            else:
                print(f"  Could not get static evaluation for the impending mate position for White.")

        except FileNotFoundError as e:
            print(f"TEST FAILED (FileNotFoundError): {e}")
        except ValueError as e: # Can be raised by chess.Board(invalid_fen)
             print(f"TEST FAILED (ValueError): {e}")
        except RuntimeError as e: # Can be raised by Stockfish init
            print(f"TEST FAILED (RuntimeError): {e}")
        except Exception as e:
            print(f"An unexpected error occurred during testing: {e}") 