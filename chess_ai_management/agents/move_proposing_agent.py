from openai import OpenAI # For type hinting, client is passed in
import json # Needed to parse the tool call arguments



class MoveProposingAgent:
    """
    A chess agent that uses the OpenAI API with a simple JSON schema (tool calling)
    to reliably propose a single best move for a given chess position (FEN).
    """
    def __init__(self, 
                 openai_client: OpenAI = None, 
                 model_name: str = "gpt-3.5-turbo-0125", # Model supporting tool calling is recommended
                 agent_name: str = "Move Proposing Agent", 
                 agent_role: str = "an AI chess assistant focused on proposing the best move via a structured tool call."):
        self.agent_name = agent_name
        self.agent_role = agent_role
        self.openai_client = openai_client
        self.model_name = model_name

        if self.openai_client is None:
            print(f"Warning: {self.agent_name} initialized without an OpenAI client. API calls will fail.")

        self.move_tool_schema = {
            "type": "function",
            "function": {
                "name": "propose_chess_move",
                "description": "Provides the best chess move in UCI format.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "move": {
                            "type": "string",
                            "description": "The chess move in UCI format (e.g., e2e4)."
                        }
                    },
                    "required": ["move"]
                }
            }
        }

    def _create_prompt_for_tool_call(self, fen: str) -> list[dict[str, str]]:
    
        system_message = (
            f"You are {self.agent_name}, an AI chess assistant and a chess grandmaster. Your role is: {self.agent_role}.\n"
            f"You are analyzing the chess position represented by the FEN: {fen}.\n"
            f"You MUST propose the single best chess move by calling the 'propose_chess_move' tool.\n"
            f"The move you propose via the tool call MUST be a valid, legal chess move in UCI format for the provided FEN: {fen}.\n"
            f"Ensure the 'move' argument in your tool call is ONLY the UCI string (e.g., 'e2e4') and nothing else.\n"
            f"Do not provide any other commentary or text outside of the tool call itself."
        )
        user_content = f"Given the FEN: {fen}, determine the single best legal move and provide it using the 'propose_chess_move' tool. Start by analyzing the board so you understand it." 
        
        messages = [
            {"role": "system", "content": system_message},
            {"role": "user", "content": user_content}
        ]
        return messages

    def propose_move(self, fen: str) -> str:
        
        if not self.openai_client:
            return "Error: OpenAI client not provided."

        prompt_messages = self._create_prompt_for_tool_call(fen)

        try:
            response = self.openai_client.chat.completions.create(
                model=self.model_name,
                messages=prompt_messages,
                tools=[self.move_tool_schema],
                tool_choice={"type": "function", "function": {"name": "propose_chess_move"}}, # Force this tool
                #temperature=0.1 # Low temperature for deterministic move choice
            )

            message = response.choices[0].message

            # 3. Extract the move from the tool call in the response.
            if message.tool_calls:
                tool_call = message.tool_calls[0]
                if tool_call.function.name == "propose_chess_move":
                    try:
                        arguments = json.loads(tool_call.function.arguments)
                        move = arguments.get("move")
                        if move and isinstance(move, str):
                            if 4 <= len(move.strip()) <= 5 and move.strip().isalnum():
                                return move.strip()
                            else:
                                return f"Error: LLM proposed an invalidly formatted move: '{move}'"
                        else:
                            return "Error: 'move' argument not found or invalid in tool call."
                    except json.JSONDecodeError:
                        return "Error: Could not parse JSON arguments from tool call."
                else:
                    return f"Error: LLM called an unexpected tool: {tool_call.function.name}"
            else:
                return "Error: LLM did not make the expected tool call. Response: " + (message.content if message.content else "No content")
        
        except Exception as e:
            return f"Error during OpenAI API call: {str(e)}"

if __name__ == '__main__':
    print("MoveProposingAgent class defined (Simplified JSON Schema / Tool Calling).")
    print("To use it, initialize with an OpenAI client instance and call propose_move(fen).")
    print("\nExample (conceptual, requires OPENAI_API_KEY):")

    try:
        import sys
        import os
        # current_dir = os.path.dirname(os.path.abspath(__file__))
        # project_root = os.path.dirname(current_dir)
        # if project_root not in sys.path:
        #     sys.path.insert(0, project_root)
        
        # from config import OPENAI_API_KEY # We will let OpenAI client find it from env

        # if OPENAI_API_KEY and OPENAI_API_KEY != "YOUR_API_KEY_HERE": # Condition removed
        print("\nAttempting to initialize MoveProposingAgent (will try to use OPENAI_API_KEY from environment)...")
        try:
            client = OpenAI() # Initialize without explicit api_key to use environment variable
            agent = MoveProposingAgent(openai_client=client, model_name="gpt-4o")
            print(f"{agent.agent_name} initialized with model {agent.model_name}.")
            
            test_fen = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1" 
            print(f"\nTesting with FEN: {test_fen}")
            
            print("\n--- Proposing Move (Example Call) ---")
            move = agent.propose_move(test_fen)
            print(f"Proposed Move: {move}")

        except Exception as e:
            print(f"Could not initialize or test agent. Error: {e}")
            print("Ensure your OPENAI_API_KEY environment variable is set correctly.")
        # else:
            # print("\nSkipping live agent initialization: OPENAI_API_KEY not set in config.py or is placeholder.")
            
    # except ImportError:
        # print("\nCould not import config.py. Ensure it exists and OPENAI_API_KEY is set.")
    except Exception as e:
        print(f"An error occurred in the __main__ example block: {e}")