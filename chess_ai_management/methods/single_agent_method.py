from openai import OpenAI
import sys
import os

current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from agents.move_proposing_agent import MoveProposingAgent

class SingleAgentMethod:
    """
    A simple management method that uses a single MoveProposingAgent
    to decide on a chess move.
    """
    def __init__(self, openai_client: OpenAI = None, agent_model_name: str = "gpt-4o"):
        """
        Initializes the SingleAgentMethod.

        Args:
            openai_client (OpenAI, optional): An initialized OpenAI client instance.
                If None, an attempt will be made to initialize one (requires OPENAI_API_KEY env var).
            agent_model_name (str, optional): The model name to be used by the internal MoveProposingAgent.
                                            Defaults to "gpt-4o".
        """
        if openai_client is None:
            try:
                self.openai_client = OpenAI()
                print("SingleAgentMethod: OpenAI client initialized successfully (likely from environment variable).")
            except Exception as e:
                print(f"SingleAgentMethod: Failed to auto-initialize OpenAI client: {e}")
                print("SingleAgentMethod: Please provide an initialized client or ensure OPENAI_API_KEY is set.")
                self.openai_client = None # Ensure it's None if initialization failed
        else:
            self.openai_client = openai_client

        if self.openai_client:
            self.agent = MoveProposingAgent(openai_client=self.openai_client, model_name=agent_model_name)
            print(f"SingleAgentMethod: Initialized with MoveProposingAgent using model '{agent_model_name}'.")
        else:
            self.agent = None
            print("SingleAgentMethod: Could not initialize MoveProposingAgent due to missing OpenAI client.")

    def decide_move(self, fen: str) -> str:
        """
        Uses the internal MoveProposingAgent to get a move for the given FEN.

        Args:
            fen (str): The FEN string of the chess position.

        Returns:
            str: The proposed move in UCI format, or an error message if the agent isn't available or fails.
        """
        if not self.agent:
            return "Error: MoveProposingAgent not available in SingleAgentMethod."
        
        move = self.agent.propose_move(fen)
        return move

if __name__ == '__main__':
    print("Testing SingleAgentMethod...")

    try:
        method = SingleAgentMethod(agent_model_name="gpt-4o") 
        print("\nAttempting to initialize SingleAgentMethod (relies on env var for OpenAI client)...")
        method = SingleAgentMethod(agent_model_name="gpt-4o") 

        if method.agent:
            test_fen = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"
            print(f"\nTest FEN: {test_fen}")

            print("\n--- Deciding Move --- ")
            proposed_move = method.decide_move(test_fen)
            print(f"Move decided by SingleAgentMethod: {proposed_move}")
        else:
            print("SingleAgentMethod could not be fully initialized for testing (agent missing).")

    except Exception as e:
        print(f"An error occurred during SingleAgentMethod test: {e}")
        print("Ensure your OPENAI_API_KEY environment variable is set correctly.") 