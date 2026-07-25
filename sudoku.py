#### Util functions for generating and checking sudoku grids

def generate(): 
    """
    Generates and returns 4x4 sudoku as 2d array
    0 represents empty space
    """

    sudoku = []
    for i in range(4): 
        sudoku.append([0]*4)

    

    return sudoku


print(generate())
