import numpy as np

def check_horizontal_symmetry(grid):
    n = len(grid)
    for i in range(n):
        for j in range(n // 2):
            if grid[i][j] != grid[i][n - j - 1]:
                return 0
    return 1

def check_vertical_symmetry(grid):
    n = len(grid)
    for i in range(n // 2):
        for j in range(n):
            if grid[i][j] != grid[n - i - 1][j]:
                return 0
    return 1

def check_rotational_symmetry_90(grid):
    rotated_grid = np.rot90(grid)
    return int(np.array_equal(grid, rotated_grid))

def check_rotational_symmetry_180(grid):
    rotated_grid = np.rot90(grid, 2)
    return int(np.array_equal(grid, rotated_grid))

def check_rotational_symmetry_270(grid):
    rotated_grid = np.rot90(grid, 3)
    return int(np.array_equal(grid, rotated_grid))

def check_diagonal_symmetry(grid):
    n = len(grid)
    for i in range(n):
        for j in range(n):
            if grid[i][j] != grid[j][i]:
                return 0
    return 1

def check_anti_diagonal_symmetry(grid):
    n = len(grid)
    for i in range(n):
        for j in range(n):
            if grid[i][j] != grid[n-j-1][n-i-1]:
                return 0
    return 1

def extract_features(grid):
    return [
        check_horizontal_symmetry(grid),
        check_vertical_symmetry(grid),
        check_rotational_symmetry_90(grid),
        check_rotational_symmetry_180(grid),
        check_rotational_symmetry_270(grid),
        check_diagonal_symmetry(grid),
        check_anti_diagonal_symmetry(grid)
    ]