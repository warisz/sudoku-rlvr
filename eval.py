from sudoku import *
from model import *

ALL_VALID_BOARDS = generate_all_valid() 

random.seed(42)

puzzles = {} # key: int (number of empty cells), value: list of puzzles with those many empty cells
puzzle_solve_history = {} # key: int (number of empty cells), value: list[bool] (list of true/false showing if model solved correctly)

for i in range(1,16): # 1 to 15 hints  
    puzzles[i] = []
    puzzle_solve_history[i] = []

for i in puzzles.keys():
    # generate puzzles per possible empty space
    for board in ALL_VALID_BOARDS:
        puzzles[i].append(generate_puzzle(board, i))
    
for i in puzzles.keys():
    for p in puzzles[i]:
        # run the model on that puzzle
        model_output = run_model(p)
        extracted_attempt = extract(model_output)
        verification = verify(p, extracted_attempt)
        puzzle_solve_history[i].append(verification)
    


