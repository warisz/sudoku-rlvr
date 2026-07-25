#### Util functions for generating and checking sudoku grids

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

    all_puzzles = []
    flat_board = [0]*16

    def dfs(pos):
        if pos == 16:
            all_puzzles.append(to_2d(flat_board))
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
        br, bc = (r // 2) * 2, (c // 2) * 2
        for i in range(2):
            for j in range(2):
                if flat_board[(br + i) * 4 + (bc + j)] == num:
                    return False
        return True   # no conflicts

    dfs(0)

    return all_puzzles


print(generate_all_valid())
