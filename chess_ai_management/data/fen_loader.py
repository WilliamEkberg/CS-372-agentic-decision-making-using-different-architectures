import os

def load_fens_from_file(filename: str = "100-chess-to-solve.txt") -> list[str]:

    current_dir = os.path.dirname(os.path.abspath(__file__))
    file_path = os.path.join(current_dir, filename)
    
    fens = []
    try:
        with open(file_path, 'r') as f:
            for line in f:
                fen = line.strip()
                if fen:  # Ensure the line is not empty
                    fens.append(fen)
    except FileNotFoundError:
        print(f"Error: FEN file '{file_path}' not found.")
        return []
    except Exception as e:
        print(f"An error occurred while reading '{file_path}': {e}")
        return []
    return fens

if __name__ == '__main__':
    print(f"Attempting to load FENs from '100-chess-to-solve.txt'...")
    loaded_fens = load_fens_from_file() # Uses default filename
    if loaded_fens:
        print(f"Successfully loaded {len(loaded_fens)} FENs.")
        print("First 5 FENs:")
        for i, fen_str in enumerate(loaded_fens[:5]):
            print(f"{i+1}: {fen_str}")
    else:
        print("No FENs loaded or an error occurred.") 