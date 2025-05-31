import json
from openai import OpenAI

# Tool 1: Consult Positional Analyst
CHECK_LEGALITY_TOOL_NAME = "consult_positional_analyst_for_legality"
CHECK_LEGALITY_TOOL_SCHEMA = {
    "type": "function",
    "function": {
        "name": CHECK_LEGALITY_TOOL_NAME,
        "description": "Consults the Positional Analyst to check if a given chess move (UCI format) is legal for the current FEN position. The Positional Analyst will return structured feedback.",
        "parameters": {
            "type": "object",
            "properties": {
                "move_uci": {
                    "type": "string",
                    "description": "The chess move in UCI format (e.g., e2e4, or e7e8q for pawn promotion) to be checked."
                }
            },
            "required": ["move_uci"]
        }
    }
}

# Tool 2: Submit Final Approved Move
SUBMIT_FINAL_MOVE_TOOL_NAME = "submit_final_approved_move"
SUBMIT_FINAL_MOVE_TOOL_SCHEMA = {
    "type": "function",
    "function": {
        "name": SUBMIT_FINAL_MOVE_TOOL_NAME,
        "description": "Submits the final chess move that has been confirmed as legal by the Positional Analyst, along with a brief justification.",
        "parameters": {
            "type": "object",
            "properties": {
                "move_uci": {
                    "type": "string",
                    "description": "The final legal UCI chess move (e.g., e2e4, e7e8q for promotion)."
                },
                "justification": {
                    "type": "string",
                    "description": "A brief justification from the manager for choosing this move."
                }
            },
            "required": ["move_uci", "justification"]
        }
    }
}

#System Prompts for Each Role

SYSTEM_PROMPT_MANAGER = f"""
You are a methodical and precise Chess Manager and a chess grandmaster. Your goal is to decide the best chess move for a given FEN position.
You will receive:
1. The current FEN string.
2. A report from a Risk Analyst.
3. A report from a Strategy Analyst.

Your process MUST be as follows:
1. Review all the provided information carefully.
2. Formulate a candidate chess move in UCI format. Ensure pawn promotions are lowercase (e.g., e7e8q).
3. **Crucial Step:** You MUST use the tool '{CHECK_LEGALITY_TOOL_NAME}' to ask the Positional Analyst to verify if your proposed move is legal for the given FEN.
4. The Positional Analyst (via the tool) will return a JSON response with 'is_legal' (boolean), 'checked_move_uci' (string, potentially corrected by PA), and 'reason' (string).
5. If the 'is_legal' field in the tool's JSON response is 'false', you MUST formulate a NEW, DIFFERENT move and use the '{CHECK_LEGALITY_TOOL_NAME}' tool again. Repeat this iterative process until the 'is_legal' field is 'true'.
6. Once a move is confirmed as 'is_legal: true' by the Positional Analyst tool, and you are confident it's the best strategic choice, you MUST use the tool '{SUBMIT_FINAL_MOVE_TOOL_NAME}' to provide this validated legal move and your brief justification.
Only use the provided tools for legality checking and final submission. Do not state the move in your textual response to the user unless it's part of a tool call.
"""

SYSTEM_PROMPT_RISK_ANALYST = """
You are a Chess Risk Analyst. Given a FEN position, provide a concise textual summary identifying potential risks, tactical dangers, or immediate threats for the side to move.
"""

SYSTEM_PROMPT_STRATEGY_ANALYST = """
You are a Chess Strategy Analyst. Given a FEN position, provide a concise textual summary outlining potential short-term and long-term strategic plans or goals for the side to move.
"""

POSITIONAL_ANALYST_EXPECTED_JSON_PROPERTIES = {
    "is_legal": {"type": "boolean"},
    "checked_move_uci": {"type": "string"},
    "reason": {"type": "string"}
}
SYSTEM_PROMPT_POSITIONAL_ANALYST_SERVICE = f"""
You are a highly precise Chess Positional Analyst. Your ONLY task is to determine the legality of a given chess move (in UCI format) for a specific FEN position.
You MUST respond ONLY with a single, valid JSON object. Do not add any text before or after the JSON object.
The JSON object must contain these exact keys with values of the specified type:
- "is_legal": boolean (true if the move is legal, false otherwise).
- "checked_move_uci": string (the move you analyzed. If the input UCI for promotion was uppercase like 'E7E8Q', correct it to lowercase 'e7e8q' if the base move is valid. If input is malformed but intent clear, return corrected UCI. If unfixably illegal, return original input or an empty string).
- "reason": string (if illegal, a brief, clear explanation. If legal, state "Move is legal.").

Example for a legal move: {json.dumps({"is_legal": True, "checked_move_uci": "e2e4", "reason": "Move is legal."})}
Example for an illegal move: {json.dumps({"is_legal": False, "checked_move_uci": "e2e5", "reason": "Pawn on e2 cannot move to e5 becuase of...."})}
Example for legal promotion: Input 'a7a8Q', FEN allows. Output: {json.dumps({"is_legal": True, "checked_move_uci": "a7a8q", "reason": "Move is legal."})}
DO NOT SAY THAT IT IS LEGAL IF IT IS NOT.
"""

class ManagerAnalystsMethod:
    def __init__(self, openai_client: OpenAI, manager_model: str = "gpt-4.1", analyst_model: str = "gpt-4.1"):
        self.openai_client = openai_client
        self.manager_model_name = manager_model
        self.analyst_model_name = analyst_model 

    def _get_llm_text_response(self, system_prompt: str, user_prompt: str) -> str:
        """Helper for simple text-based analyst calls (Risk, Strategy)."""
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ]
        try:
            completion = self.openai_client.chat.completions.create(
                model=self.analyst_model_name, 
                messages=messages,
                #temperature=0.3,
                #max_tokens=500
            )
            return completion.choices[0].message.content
        except Exception as e:
            print(f"    Error during LLM text response call: {e}")
            return f"Error: LLM call failed - {e}"

    def _call_positional_analyst_llm_service(self, fen: str, move_to_check: str) -> dict:
        print(f"    _call_positional_analyst_llm_service: Checking FEN='{fen}', Move='{move_to_check}'")
        user_prompt_pa = (
            f"FEN: '{fen}'. Proposed move (UCI): '{move_to_check}'. "
            f"Analyze legality and respond ONLY with the specified JSON object. Start by analyzing the board so you understand it."
        )
        messages_pa = [
            {"role": "system", "content": SYSTEM_PROMPT_POSITIONAL_ANALYST_SERVICE},
            {"role": "user", "content": user_prompt_pa}
        ]
        try:
            completion = self.openai_client.chat.completions.create(
                model=self.analyst_model_name, # PA can use the analyst model
                messages=messages_pa,
                response_format={"type": "json_object"}, # Crucial for forcing JSON output
                #temperature=0.0 # For deterministic JSON generation
            )
            raw_json_response = completion.choices[0].message.content
            print(f"    Positional Analyst LLM raw JSON: {raw_json_response}")
            
            pa_data = json.loads(raw_json_response)

            # Validate structure and normalize promotion in checked_move_uci
            if not isinstance(pa_data.get("is_legal"), bool):
                pa_data["is_legal"] = False # Default to false if type is wrong
                pa_data["reason"] = pa_data.get("reason", "") + " (Corrected: is_legal was not boolean)"
            
            checked_uci = pa_data.get("checked_move_uci", move_to_check)
            if isinstance(checked_uci, str) and len(checked_uci) == 5 and checked_uci[4] in "QRNB":
                pa_data["checked_move_uci"] = checked_uci[:4] + checked_uci[4].lower()
                print(f"    Normalized PA 'checked_move_uci' for promotion: {pa_data['checked_move_uci']}")
            
            return pa_data

        except json.JSONDecodeError as e_json:
            print(f"    Error: Positional Analyst LLM didn't return valid JSON: {e_json}. Response: '{raw_json_response if 'raw_json_response' in locals() else 'response not captured'}'")
            return {"is_legal": False, "checked_move_uci": move_to_check, "reason": "Positional Analyst response was not valid JSON."}
        except Exception as e_llm:
            print(f"    Error during Positional Analyst LLM call: {e_llm}")
            return {"is_legal": False, "checked_move_uci": move_to_check, "reason": f"Positional Analyst LLM call failed: {e_llm}"}

    def decide_move(self, fen: str) -> str:
        print(f"\n--- ManagerAnalystsMethod (Agent-Only Validation) starting for FEN: {fen} ---")

        # 1. Get initial analyst reports
        risk_report = self._get_llm_text_response(SYSTEM_PROMPT_RISK_ANALYST, f"FEN: {fen}. Analyze risks.")
        print(f"  Risk Analyst Report:\n{risk_report}")
        strategy_report = self._get_llm_text_response(SYSTEM_PROMPT_STRATEGY_ANALYST, f"FEN: {fen}. Analyze strategy.")
        print(f"  Strategy Analyst Report:\n{strategy_report}")

        if "Error:" in risk_report or "Error:" in strategy_report:
            return "Error: Initial analyst reports failed."

        # 2. Manager's decision loop
        manager_history = [{"role": "system", "content": SYSTEM_PROMPT_MANAGER}]
        initial_prompt_for_manager = (
            f"Current FEN: {fen}\n\n"
            f"Risk Analyst's Report:\n{risk_report}\n\n"
            f"Strategy Analyst's Report:\n{strategy_report}\n\n"
            f"Based on these reports, formulate a candidate move. Use the '{CHECK_LEGALITY_TOOL_NAME}' tool to verify. "
            f"Iterate if illegal. Once a legal move is confirmed, use '{SUBMIT_FINAL_MOVE_TOOL_NAME}'."
            f"Start by analyzing the board so you understand it."
        )
        manager_history.append({"role": "user", "content": initial_prompt_for_manager})

        MAX_ITERATIONS = 5 # Prevent infinite loops
        for i in range(MAX_ITERATIONS):
            print(f"\n  Manager Iteration {i + 1}/{MAX_ITERATIONS}")
            
            try:
                manager_api_response = self.openai_client.chat.completions.create(
                    model=self.manager_model_name,
                    messages=manager_history,
                    tools=[CHECK_LEGALITY_TOOL_SCHEMA, SUBMIT_FINAL_MOVE_TOOL_SCHEMA],
                    tool_choice="auto" # Let Manager decide which tool or if it needs to think more
                )
            except Exception as e_api:
                print(f"    Error during Manager API call: {e_api}")
                manager_history.append({"role": "user", "content": f"An API error occurred: {e_api}. Please try to use a tool or state your reasoning."})
                continue # Allow manager to try again if API call failed

            response_message = manager_api_response.choices[0].message
            manager_history.append(response_message) # Add assistant's response (text or tool_call)

            if response_message.tool_calls:
                tool_call = response_message.tool_calls[0]
                tool_name = tool_call.function.name
                
                try:
                    tool_args = json.loads(tool_call.function.arguments)
                except json.JSONDecodeError:
                    print(f"    Error: Manager tool call arguments for '{tool_name}' not valid JSON: {tool_call.function.arguments}")
                    tool_feedback_content = json.dumps({"error": "Tool arguments were not valid JSON."})
                    manager_history.append({"tool_call_id": tool_call.id, "role": "tool", "name": tool_name, "content": tool_feedback_content})
                    continue

                print(f"    Manager wants to call tool: '{tool_name}' with args: {tool_args}")

                if tool_name == CHECK_LEGALITY_TOOL_NAME:
                    move_to_check = tool_args.get("move_uci")
                    if not move_to_check or not isinstance(move_to_check, str):
                        print(f"    Manager called '{CHECK_LEGALITY_TOOL_NAME}' without a valid 'move_uci' string.")
                        pa_feedback_json = {"is_legal": False, "checked_move_uci": str(move_to_check or ""), "reason": "Invalid 'move_uci' string provided to tool."}
                    else:
                        pa_feedback_json = self._call_positional_analyst_llm_service(fen, move_to_check)
                    
                    manager_history.append({
                        "tool_call_id": tool_call.id,
                        "role": "tool",
                        "name": tool_name,
                        "content": json.dumps(pa_feedback_json)
                    })

                elif tool_name == SUBMIT_FINAL_MOVE_TOOL_NAME:
                    final_move_uci = tool_args.get("move_uci")
                    justification = tool_args.get("justification", "No justification provided.")
                    
                    if not final_move_uci or not isinstance(final_move_uci, str):
                        print(f"    Manager called '{SUBMIT_FINAL_MOVE_TOOL_NAME}' with invalid 'move_uci': {final_move_uci}")
                        manager_history.append({
                            "tool_call_id": tool_call.id, "role": "tool", "name": tool_name,
                            "content": json.dumps({"error": "Invalid 'move_uci' for final submission. It must be a UCI string."})
                        })
                        continue # Let Manager try to submit again correctly

                    if len(final_move_uci) == 5 and final_move_uci[4] in "QRNB":
                        final_move_uci = final_move_uci[:4] + final_move_uci[4].lower()
                    
                    print(f"  ManagerAnalystsMethod determined move: '{final_move_uci}' (Justification: '{justification}')")
                    return final_move_uci # SUCCESS

                else:
                    print(f"    Warning: Manager called unknown tool '{tool_name}'.")
                    manager_history.append({
                        "tool_call_id": tool_call.id, "role": "tool", "name": tool_name,
                        "content": json.dumps({"error": "Unknown tool name called."})
                    })
            else:
                
                manager_text_response = response_message.content if response_message.content else ""
                print(f"    Manager text response (was expecting tool call or leading to one): {manager_text_response[:150]}...")
            
                manager_history.append({
                    "role": "user",
                    "content": "Please continue the process. If you have a move to check, use "
                               f"'{CHECK_LEGALITY_TOOL_NAME}'. If you have a final legal move, use "
                               f"'{SUBMIT_FINAL_MOVE_TOOL_NAME}'. Otherwise, state your reasoning if you need to think more."
                })
        
        print(f"  ManagerAnalystsMethod failed to submit a final move after {MAX_ITERATIONS} iterations.")
        return "Error: Manager failed to submit a final move after iterations."

