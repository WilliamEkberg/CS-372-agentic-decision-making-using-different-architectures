# Chess AI Management Experimentation Framework

This project explores different AI-driven management and agent architectures for selecting chess moves. It provides a framework to define various AI methods, test them against a set of chess positions (FENs), and evaluate their performance by comparing their chosen moves.

## Project Goal

The primary goal is to experiment with and benchmark various AI agent configurations for chess move selection. This includes:
- Single agent decision-making.
- Multi-agent debate/discussion models (e.g., 1v1 debate, manager-analyst hierarchies).
- Utilizing OpenAI's GPT models as the core intelligence for the agents.

## Core Components

- **`chess_ai_management/`**: Main package directory.
    - **`main.py`**: The central script for running experiments. It loads FENs, initializes selected AI methods, processes each FEN through the methods, and scores the results.
    - **`config.py`**: Handles configuration, primarily the OpenAI API key and the path to the Stockfish chess engine executable.
    - **`agents/`**: Contains the definitions for different types of AI agents.
        - `MoveProposingAgent`: A simple agent that proposes a single move using an OpenAI tool call.
    - **`methods/`**: Defines the different AI strategies/architectures for move selection.
        - `SingleAgentMethod`: Uses one `MoveProposingAgent`.
        - `TwoAgentDebateMethod`: Simulates a 4-round debate between two LLM personas to arrive at a move.
        - `ManagerAnalystsMethod`: Implements a manager overseeing risk and strategy analysts, with a positional analyst (as a tool) to validate move legality before the manager submits a final move.
    - **`evaluation/`**: 
        - `Evaluator.py`: Uses the Stockfish chess engine (via `python-stockfish` and `python-chess`) to evaluate the board positions resulting from an agent's move. This is used for the competitive scoring mechanism.
    - **`data/`**: 
        - `fen_loader.py`: Loads chess positions (FEN strings) from a text file.
        - `100-chess-to-solve.txt`: Sample FENs for testing.
- **`stockfish/`**: Directory where the Stockfish executable should be placed (or the path in `config.py` updated accordingly).

## How it Works

1.  **FEN Loading**: `main.py` loads a list of FENs from `data/100-chess-to-solve.txt`.
2.  **Agent Initialization**: Selected AI methods (e.g., `SingleAgentMethod`, `TwoAgentDebateMethod`, `ManagerAnalystsMethod`) are initialized. These methods encapsulate different ways of using LLMs to propose a chess move.
3.  **Experiment Loop**: For each FEN:
    *   Each AI method is asked to propose a move.
    *   The board position *resulting* from each method's proposed move is evaluated using Stockfish (via the `Evaluator` class).
4.  **Scoring**: 
    *   Methods compete against each other for each FEN.
    *   The method(s) whose proposed move leads to the objectively best Stockfish evaluation of the *resulting position* (compared to other methods for that FEN) receives 1 point.
    *   An `illegal_move_count` is also tracked for methods that propose moves deemed invalid by the `Evaluator`.
5.  **Results**: Final scores (percentage of "best relative moves" among competitors) and illegal move counts are printed for each method.

## Setup

1.  **Clone the repository.**
2.  **Install dependencies:**
    ```bash
    pip install openai python-chess stockfish
    ```
3.  **Stockfish Engine:**
    *   Download the Stockfish chess engine executable appropriate for your system.
    *   Place the executable in the `stockfish/` directory at the project root, or update `STOCKFISH_PATH` in `chess_ai_management/config.py` to point to its location.
4.  **OpenAI API Key:**
    *   Set your OpenAI API key in `chess_ai_management/config.py` by replacing `"YOUR_API_KEY_HERE"` with your actual key, or ensure the `OPENAI_API_KEY` environment variable is set.

## Running Experiments

Navigate to the project root directory (`Final project/`) and run:

```bash
python chess_ai_management/main.py
```

-   You can control the number of FENs to test by modifying `max_fens_to_test` in the `run_experiment()` call at the bottom of `main.py`.
-   To test specific methods in isolation, dedicated test scripts like `test_single_agent.py`, `test_two_agent_debate.py`, and `test_manager_analysts.py` are available in the `chess_ai_management/` directory.

## Future Development

-   Explore different "manager" decision-making logics.
-   Refine LLM prompts for better move quality and legality.
-   Add so that the debating method does not debate as many rounds if they agree (for  faster runs)
-   Expand the dataset of FENs for larger runs. 