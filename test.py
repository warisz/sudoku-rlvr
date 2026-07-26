from sudoku import *
from model import *

boards = generate_all_valid()
test_puzzles = [generate_puzzle(boards[0], 3) for _ in range(4)]
outputs = run_model(test_puzzles)
print(f"got {len(outputs)} outputs")   # MUST be 4, not 1
for p, o in zip(test_puzzles, outputs):
    print("PUZZLE:", p)
    print("RAW:", repr(o[:200]))
    print("EXTRACTED:", extract(o))
    print("SOLVED:", verify(p, extract(o)))
    print("---")