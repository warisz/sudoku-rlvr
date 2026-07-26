#### Util functions for generating and checking sudoku grids
import random 
from ast import literal_eval
import re

def to_2d(flat_board):
    width = 4
    rows = []
    for i in range(4):
        start = i * width
        row = flat_board[start : start + width]
        rows.append(row)
    return rows

def generate_all_valid(): 
    """
    Generates all valid sudoku returns 4x4 sudoku as 2d array
    0 represents empty space
    """

    all_boards = []
    flat_board = [0]*16

    def dfs(pos):
        if pos == 16:
            all_boards.append(to_2d(flat_board))
            return
        for num in [1,2,3,4]:
            if legal_move(pos, num):
                flat_board[pos] = num
                dfs(pos+1)
                flat_board[pos] = 0

    def legal_move(pos, num):
        r, c = divmod(pos, 4)
        # row: check all cells in row r
        for i in range(4):
            if flat_board[r * 4 + i] == num:
                return False
        # column: check all cells in column c
        for i in range(4):
            if flat_board[i * 4 + c] == num:
                return False
        # 2x2 box: find the box's top left corner, check its 4 cells
        br, bc = (r // 2) * 2, (c // 2) * 2 # lowest multiple of 2 for both r and c, either 0 or 2 
        for i in range(2):
            for j in range(2):
                if flat_board[(br + i) * 4 + (bc + j)] == num:
                    return False
        return True   # no conflicts


    dfs(0)
    return all_boards

def generate_puzzle(board, n_empty_cells):
    puzzle = [row[:] for row in board]
    flat_positions = random.sample(range(16), n_empty_cells)
    for pos in flat_positions:
        r, c = divmod(pos, 4)
        puzzle[r][c] = 0
    return puzzle


def extract(text): 
    # finds the last 2D array in text 
    matches = re.findall(r'\[\s*\[.*?\]\s*\]', text, re.DOTALL)
    if not matches:
        return None
    # take the last array
    try:
        arr = literal_eval(matches[-1])
    except (ValueError, SyntaxError):
        arr = None
    return arr


def verify(puzzle, attempt):
    # checks if extracted attempt is a valid solution 
    if attempt is None:
        return False

    # check shape, attempt must be 4x4
    if len(attempt) != 4:
        return False
    if not all(isinstance(row, list) and len(row) == 4 for row in attempt):
        return False

    # make sure original clues are preserved 
    for r in range(4):
        for c in range(4):
            if puzzle[r][c] != 0 and puzzle[r][c] != attempt[r][c]:
                return False

    # make sure it is a valid solution 
    target = {1, 2, 3, 4}
    for i in range(4):
        row = attempt[i]
        col = [attempt[r][i] for r in range(4)]
        if set(row) != target or set(col) != target:
            return False

    for br in (0, 2): #box row
        for bc in (0, 2): #box col
            box = []
            for i in range(2):
                for j in range(2):
                    box.append(attempt[br + i][bc + j])

            if set(box) != target:
                return False

    return True


