from sudoku import *
from model import *
import json

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
    print(f"evaluating {i} empty cells...")
    for batch in chunks(puzzles[i], 64):
        outputs = run_model(batch)
        for p, output in zip(batch, outputs):
            attempt = extract(output)
            print('------------------------------------------------------------------------')
            print(attempt)
            print('------------------------------------------------------------------------')
            puzzle_solve_history[i].append(verify(p, attempt))
    
results = {}
for i in puzzle_solve_history:
    solved = sum(puzzle_solve_history[i]) #counts Trues
    total = len(puzzle_solve_history[i])
    results[i] = {"solved": solved, "total": total, "rate": solved/total}
    print(f"{i} blanks: {solved}/{total} = {solved/total:.1%}")

with open("eval_results.json", "w") as f:
    json.dump(results, f, indent=2)

