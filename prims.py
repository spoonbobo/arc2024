from typing import Tuple

def expand(grid: Tuple[Tuple[int]], factor: int) -> Tuple[Tuple[int]]:
    """Expands the grid by the given factor."""
    n = len(grid)
    new_size = factor * n
    new_grid = [[0] * new_size for _ in range(new_size)]
    
    for i in range(n):
        for j in range(n):
            for di in range(factor):
                for dj in range(factor):
                    new_grid[factor * i + di][factor * j + dj] = grid[i][j]
    
    return tuple(tuple(row) for row in new_grid)

def checkered(grid: Tuple[Tuple[int]], factor: int) -> Tuple[Tuple[int]]:
    """Creates a checkered pattern in the grid by the given factor."""
    n = len(grid) // factor
    new_grid = [[0] * len(grid) for _ in range(len(grid))]
    
    for i in range(n):
        for j in range(n):
            for di in range(factor):
                for dj in range(factor):
                    if di == factor // 2:
                        new_grid[factor * i + di][factor * j + dj] = grid[factor * i + di][factor * (n - 1 - j) + dj]
                    else:
                        new_grid[factor * i + di][factor * j + dj] = grid[factor * i + di][factor * j + dj]
    
    return tuple(tuple(row) for row in new_grid)

def hmirror(grid: Tuple[Tuple[int]]) -> Tuple[Tuple[int]]:
    """ mirroring along horizontal """
    return tuple(tuple(row) for row in grid[::-1])

def vmirror(grid: Tuple[Tuple[int]]) -> Tuple[Tuple[int]]:
    """ Mirroring along vertical axis """
    return tuple([tuple(row[::-1]) for row in grid])

def rot90(grid: Tuple[Tuple[int]]) -> Tuple[Tuple[int]]:
    """ clockwise rotation by 90 degrees """
    return tuple([tuple(row) for row in zip(*grid[::-1])])

def rot180(grid: Tuple[Tuple[int]]) -> Tuple[Tuple[int]]:
    """ Rotates the grid by 180 degrees """
    return tuple([tuple(row[::-1]) for row in grid[::-1]])

def fill_color(grid: Tuple[Tuple[int]], color: int) -> Tuple[Tuple[int]]:
    """ Fills the grid with the specified color"""
    return tuple([tuple(color for cell in row) for row in grid])
