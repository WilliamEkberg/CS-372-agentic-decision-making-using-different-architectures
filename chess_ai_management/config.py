import os

# Configuration settings
OPENAI_API_KEY = "YOUR_API_KEY_HERE"


CONFIG_DIR = os.path.dirname(os.path.abspath(__file__))

PROJECT_ROOT_DIR = os.path.dirname(CONFIG_DIR)

# Name of the Stockfish executable
STOCKFISH_EXECUTABLE_NAME = "stockfish" 
STOCKFISH_PATH = os.path.join(PROJECT_ROOT_DIR, STOCKFISH_EXECUTABLE_NAME)

FEN_SOURCE_URL = "URL_TO_YOUR_FEN_LIST_ON_GITHUB" 