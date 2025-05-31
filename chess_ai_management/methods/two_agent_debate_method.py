from openai import OpenAI
import re
import chess

class TwoAgentDebateMethod:
    """
    Manages a 1v1 debate between two LLM agents over a specific topic (e.g., best chess move for a FEN).
    The debate progresses through 4 rounds, aiming for disagreement then compromise.
    """

    # --- Constants for Personas and Round Logic ---
    SYSTEM_PROMPT_ALPHA = ("You are Chess Debater Alpha. You are a chess grandmaster. You are analytical, daring, and prefer aggressive, tactical solutions. "
                           "Your primary goal is to argue for the objectively strongest chess move for the provided FEN string. "
                           "Emphasize tactical advantages, direct threats, and forcing sequences. "
                           "When proposing a chess move, you MUST state it clearly in UCI format (e.g., e2e4) within your reasoning. "
                           "Any move you propose or discuss MUST be a strictly legal chess move for the given FEN position."
                           "You must be very careful to ensure that the move you propose is legal for the given FEN position."
                           "Be on point and concise. Do not be verbose. Do not be redundant. Do not be repetitive. Do not be lazy.")

    SYSTEM_PROMPT_BETA = ("You are Chess Debater Beta. You are a chess grandmaster. You are cautious, strategic, and prefer solid, positional solutions. "
                          "Your primary goal is to argue for the objectively strongest chess move for the provided FEN string. "
                          "Emphasize long-term positional soundness, king safety, pawn structure, and piece coordination. "
                          "When proposing a chess move, you MUST state it clearly in UCI format (e.g., e2e4) within your reasoning. "
                          "Any move you propose or discuss MUST be a strictly legal chess move for the given FEN position."
                          "You must be very careful to ensure that the move you propose is legal for the given FEN position."
                          "Be on point and concise. Do not be verbose. Do not be redundant. Do not be repetitive. Do not be lazy.")

    ROUND_INSTRUCTIONS = {
        1: "Present your initial, strong, independent analysis of the FEN position and propose the single best move from your unique perspective. Justify your choice thoroughly based on your defined persona. The move MUST be legal for the FEN.",
        2: "Review your opponent's Round 1 argument. Provide a strong, direct counter-argument to their main points and proposed move. Vigorously defend and reaffirm your own initial proposal, or adjust it slightly if their points are overwhelmingly compelling from your perspective, explaining why. All moves discussed MUST be legal for the FEN.",
        3: "The goal is now to start finding common ground, though full agreement isn't necessary yet. Acknowledge any valid points your opponent has made. Discuss potential compromises or alternative moves that might address concerns from both perspectives However if an original move is still the best move, do not change it. Re-evaluate your proposed move based on the discussion so far. All moves discussed MUST be legal for the FEN.",
        4: "This is the final round. Based on the entire discussion, what is your definitive final proposed move? State your final selected move very clearly and distinctly in UCI format. For example, write: 'My final proposed move is: [UCI_MOVE_ONLY]' (replace [UCI_MOVE_ONLY] with the actual UCI move like 'e2e4'). This move MUST be legal for the FEN. Provide a concise justification for your choice, summarizing key arguments or synthesis. Do not change your final move from Round 1 if it is still the best move."
    }
    # ---------------------------------------------

    def __init__(self, openai_client: OpenAI, model_name: str = "gpt-4o"):
        """
        Initializes the TwoAgentDebateMethod.

        Args:
            openai_client (OpenAI): An initialized OpenAI client instance.
            model_name (str): The OpenAI model to be used for the debate.
        """
        if not openai_client:
            raise ValueError("OpenAI client must be provided.")
        self.client = openai_client
        self.model_name = model_name
        self.agent_personas = {
            "Alpha": self.SYSTEM_PROMPT_ALPHA,
            "Beta": self.SYSTEM_PROMPT_BETA
        }

    def _extract_uci_move(self, text: str) -> str | None:
        """
        A simple heuristic to extract a potential UCI move from text.
        Looks for 4 or 5 character alphanumeric strings that resemble UCI moves.
        Prefers moves like e2e4 or e7e8q.
        """
        if not text: return None
        
        matches = re.findall(r'\b([a-h][1-8][a-h][1-8][qrbn]?)\b', text)
        if matches:
            return matches[-1]

        potential_moves = re.findall(r'\b([a-zA-Z0-9]{4,5})\b', text)
        if potential_moves:
            for move in reversed(potential_moves):
                if len(move) == 4 and move[0].isalpha() and move[1].isdigit() and move[2].isalpha() and move[3].isdigit():
                    return move
                if len(move) == 5 and move[0].isalpha() and move[1].isdigit() and move[2].isalpha() and move[3].isdigit() and move[4].isalpha(): # promotion
                    return move
        return None

    def run_debate(self, fen_position: str, num_rounds: int = 4) -> tuple[str | None, list[dict]]:
        """
        Runs the 1v1 debate for the given FEN position.

        Args:
            fen_position (str): The FEN string of the chess position to debate.
            num_rounds (int): Number of debate rounds (default is 4).

        Returns:
            tuple[str | None, list[dict]]: 
                - The final proposed UCI move (or None if not extractable).
                - The full debate transcript (list of turn dictionaries).
        """
        debate_transcript = []
        messages_alpha = [{"role": "system", "content": self.agent_personas["Alpha"]}]
        messages_beta = [{"role": "system", "content": self.agent_personas["Beta"]}]

        latest_alpha_statement = ""
        latest_beta_statement = ""

        print(f"--- Starting Debate on FEN: {fen_position} ---")

        for i in range(1, num_rounds + 1):
            print(f"\n--- Round {i} ---")

            # Alpha's Turn
            if i == 1:
                user_prompt_alpha = f"The FEN is: {fen_position}. Round {i}: {self.ROUND_INSTRUCTIONS[i]}"
            else:
                user_prompt_alpha = f"The FEN is: {fen_position}. It's Round {i}. Review Beta's last statement: '{latest_beta_statement}'. Your task: {self.ROUND_INSTRUCTIONS[i]}"
            
            messages_alpha.append({"role": "user", "content": user_prompt_alpha})
            try:
                resp_alpha_obj = self.client.chat.completions.create(model=self.model_name, messages=messages_alpha)
                latest_alpha_statement = resp_alpha_obj.choices[0].message.content
            except Exception as e:
                latest_alpha_statement = f"Error: Alpha could not generate a response: {e}"
                print(f"Error for Alpha in Round {i}: {e}")
            
            messages_alpha.append({"role": "assistant", "content": latest_alpha_statement})
            debate_transcript.append({"round": i, "speaker": "Alpha", "text": latest_alpha_statement})
            print(f"Alpha (Round {i}):\n{latest_alpha_statement}")

            # Beta's Turn
            if i == 1:
                user_prompt_beta = f"The FEN is: {fen_position}. Round {i}: {self.ROUND_INSTRUCTIONS[i]}"
            else:
                user_prompt_beta = f"The FEN is: {fen_position}. It's Round {i}. Review Alpha's statement this round: '{latest_alpha_statement}'. Your task: {self.ROUND_INSTRUCTIONS[i]}"
            
            messages_beta.append({"role": "user", "content": user_prompt_beta})
            try:
                resp_beta_obj = self.client.chat.completions.create(model=self.model_name, messages=messages_beta)
                latest_beta_statement = resp_beta_obj.choices[0].message.content
            except Exception as e:
                latest_beta_statement = f"Error: Beta could not generate a response: {e}"
                print(f"Error for Beta in Round {i}: {e}")

            messages_beta.append({"role": "assistant", "content": latest_beta_statement})
            debate_transcript.append({"round": i, "speaker": "Beta", "text": latest_beta_statement})
            print(f"Beta (Round {i}):\n{latest_beta_statement}")

        # --- Post-Debate Move Determination ---
        # Simple strategy: Try to extract move from Beta's final statement (or Alpha's if Beta fails)
        final_move_uci = self._extract_uci_move(latest_beta_statement)
        if not final_move_uci:
            final_move_uci = self._extract_uci_move(latest_alpha_statement)
        
        if final_move_uci:
            # Changed print message slightly to indicate it's pre-external-validation
            print(f"\n--- Debate Concluded. Extracted Final Proposed Move: {final_move_uci} ---")
        else:
            print(f"\n--- Debate Concluded. Could not reliably extract a final UCI move. ---")
            # Only print these if no move was extracted at all, to reduce noise if a move WAS extracted but might fail later
            # This check is now redundant if the print above already confirmed final_move_uci is None, but harmless.
            if not final_move_uci: 
                print(f"Alpha's final statement: {latest_alpha_statement}")
                print(f"Beta's final statement: {latest_beta_statement}")

        # The python-chess validation block that was previously here has been removed.
        # The extracted final_move_uci (which could be None, or an invalid/illegal move)
        # will be returned directly.
        # main.py's Evaluator will be the first point of rigorous legality checking.

        return final_move_uci, debate_transcript

if __name__ == '__main__':
    print("Testing TwoAgentDebateMethod...")
    
    try:
        client = OpenAI()
        debate_method = TwoAgentDebateMethod(openai_client=client, model_name="gpt-4o")

        test_fen = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1" # Starting position
        
        final_move, transcript = debate_method.run_debate(test_fen)
        
        print("\n--- Full Transcript --- ")
        for entry in transcript:
            print(f"Round {entry['round']} - {entry['speaker']}: {entry['text']}")
        
        if final_move:
            print(f"\nRecommended Move from Debate: {final_move}")
        else:
            print("\nNo definitive move could be extracted from the debate.")
            
    except Exception as e:
        print(f"An error occurred during the test run: {e}")
        print("Please ensure your OPENAI_API_KEY environment variable is set and valid.") 