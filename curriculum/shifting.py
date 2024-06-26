from typing import Tuple
from collections import Counter
import numpy as np

def shift_down(grid: Tuple[Tuple[int]], steps: int) -> Tuple[Tuple[int]]:
    """ Shifts the grid down by a specified number of steps """
    if not grid or not grid[0]:
        return grid
    if steps <= 0:
        return grid
    steps = min(steps, len(grid))  # Ensure steps do not exceed grid height
    return tuple([tuple([0] * len(grid[0])) for _ in range(steps)] + [tuple(row) for row in grid[:-steps]])

def shift_right(grid: Tuple[Tuple[int]], steps: int) -> Tuple[Tuple[int]]:
    """ Shifts the grid to the right by a specified number of steps """
    if not grid or not grid[0]:
        return grid
    if steps <= 0:
        return grid
    steps = min(steps, len(grid[0]))  # Ensure steps do not exceed grid width
    return tuple([tuple([0] * steps + list(row[:-steps])) for row in grid])

def shift_left(grid: Tuple[Tuple[int]], steps: int) -> Tuple[Tuple[int]]:
    """ Shifts the grid to the left by a specified number of steps """
    if not grid or not grid[0]:
        return grid
    if steps <= 0:
        return grid
    steps = min(steps, len(grid[0]))  # Ensure steps do not exceed grid width
    return tuple([tuple(list(row[steps:]) + [0] * steps) for row in grid])

def shift_up(grid: Tuple[Tuple[int]], steps: int) -> Tuple[Tuple[int]]:
    """ Shifts the grid up by a specified number of steps """
    if not grid or not grid[0]:
        return grid
    if steps <= 0:
        return grid
    steps = min(steps, len(grid))  # Ensure steps do not exceed grid height
    return tuple([tuple(row) for row in grid[steps:]] + [tuple([0] * len(grid[0])) for _ in range(steps)])

def diagonal_mirror(grid: Tuple[Tuple[int]]) -> Tuple[Tuple[int]]:
    """ Mirrors the grid along the main diagonal """
    return tuple([tuple(row) for row in zip(*grid)])

def tophalf(grid: Tuple[Tuple[int]]) -> Tuple[Tuple[int]]:
    """ upper half """
    return tuple(tuple(row) for row in grid[:len(grid) // 2])

def bottomhalf(grid: Tuple[Tuple[int]]) -> Tuple[Tuple[int]]:
    """ lower half """
    return tuple(tuple(row) for row in grid[len(grid) // 2:])

def lefthalf(grid: Tuple[Tuple[int]]) -> Tuple[Tuple[int]]:
    """ left half """
    return tuple([tuple(row[:len(row) // 2]) for row in grid])

def trim(grid: Tuple[Tuple[int]]) -> Tuple[Tuple[int]]:
    """ removes border """
    return tuple([tuple(r[1:-1]) for r in grid[1:-1]])

def compress(grid: Tuple[Tuple[int]]) -> Tuple[Tuple[int]]:
    """ removes frontiers """
    ri = [i for i, r in enumerate(grid) if len(set(r)) == 1]
    ci = [j for j, c in enumerate(zip(*grid)) if len(set(c)) == 1]
    return tuple([tuple(v for j, v in enumerate(r) if j not in ci) for i, r in enumerate(grid) if i not in ri])

def dilate(grid: Tuple[Tuple[int]]) -> Tuple[Tuple[int]]:
    """ dilates the shapes in the grid """
    new_grid = [[0] * len(row) for row in grid]
    for i in range(len(grid)):
        for j in range(len(grid[0])):
            if grid[i][j] == 1:
                new_grid[i][j] = 1
                if i > 0:
                    new_grid[i-1][j] = 1
                if i < len(grid) - 1:
                    new_grid[i+1][j] = 1
                if j > 0:
                    new_grid[i][j-1] = 1
                if j < len(grid[0]) - 1:
                    new_grid[i][j+1] = 1
    return tuple([tuple(row) for row in new_grid])

# convolution
def convolve(grid: Tuple[Tuple[int]], kernel: Tuple[Tuple[int]]) -> Tuple[Tuple[int]]:
    """Applies a convolution operation to the grid using the given kernel."""
    grid_array = np.array(grid)
    kernel_array = np.array(kernel)
    
    # Ensure kernel is 2D and has valid dimensions
    if grid_array.ndim != 2 or kernel_array.ndim != 2 or kernel_array.size == 0:
        return grid
    
    kernel_height, kernel_width = kernel_array.shape
    grid_height, grid_width = grid_array.shape
    
    # Ensure kernel dimensions do not exceed grid dimensions
    if kernel_height > grid_height or kernel_width > grid_width:
        return grid
    
    # Calculate the dimensions of the output grid
    output_height = grid_height - kernel_height + 1
    output_width = grid_width - kernel_width + 1
    
    # Initialize the output grid
    output_grid = np.zeros((output_height, output_width), dtype=int)
    
    # Perform the convolution
    for i in range(output_height):
        for j in range(output_width):
            region = grid_array[i:i+kernel_height, j:j+kernel_width]
            output_grid[i, j] = np.sum(region * kernel_array)
    
    output_grid_tuple = tuple(tuple(int(cell) for cell in row) for row in output_grid)
    
    return output_grid_tuple

# transformation
def transpose(grid: Tuple[Tuple[int]]) -> Tuple[Tuple[int]]:
    """ Transposes the grid (swaps rows and columns) """
    return tuple([tuple(row) for row in zip(*grid)])

def extract_outline(grid: Tuple[Tuple[int]]) -> Tuple[Tuple[int]]:
    """ Extracts the outline of shapes in the grid """
    outline_grid = [[0]*len(row) for row in grid]
    for i in range(1, len(grid)-1):
        for j in range(1, len(grid[0])-1):
            if grid[i][j] != grid[i+1][j] or grid[i][j] != grid[i-1][j] or grid[i][j] != grid[i][j+1] or grid[i][j] != grid[i][j-1]:
                outline_grid[i][j] = 1
    return tuple([tuple(row) for row in outline_grid])

def blur(grid: Tuple[Tuple[int]]) -> Tuple[Tuple[int]]:
    """ Applies a simple blur effect to the grid """
    def get_neighbors(i, j):
        neighbors = [(i-1, j-1), (i-1, j), (i-1, j+1), (i, j-1), (i, j), (i, j+1), (i+1, j-1), (i+1, j), (i+1, j+1)]
        return [(x, y) for x, y in neighbors if 0 <= x < len(grid) and 0 <= y < len(grid[0])]
    
    new_grid = [[0] * len(row) for row in grid]
    for i in range(len(grid)):
        for j in range(len(grid[0])):
            neighbors = get_neighbors(i, j)
            neighbor_values = [grid[x][y] for x, y in neighbors]
            new_grid[i][j] = sum(neighbor_values) // len(neighbor_values)
    return tuple([tuple(row) for row in new_grid])

def erode(grid: Tuple[Tuple[int]]) -> Tuple[Tuple[int]]:
    """Erodes the shapes in the grid."""
    new_grid = [[0] * len(row) for row in grid]
    for i in range(1, len(grid) - 1):
        for j in range(1, len(grid[0]) - 1):
            if grid[i][j] == 1 and all(grid[i+di][j+dj] == 1 for di, dj in [(-1, 0), (1, 0), (0, -1), (0, 1)]):
                new_grid[i][j] = 1
    return tuple([tuple(row) for row in new_grid])

def invert_colors(grid: Tuple[Tuple[int]]) -> Tuple[Tuple[int]]:
    """ Inverts the colors in the grid (assuming binary colors 0 and 1), ignoring None values """
    return tuple([tuple(1 - cell for cell in row) for row in grid])

def fill_color(grid: Tuple[Tuple[int]], color: int) -> Tuple[Tuple[int]]:
    """ Fills the grid with the specified color, replacing all non-None values """
    return tuple([tuple(color for cell in row) for row in grid])

def grid_mean(grid: Tuple[Tuple[int]]) -> int:
    """Calculates the mean of all values in the grid."""
    flat_list = [item for sublist in grid for item in sublist]
    if not flat_list:
        return 0
    return int(sum(flat_list) / len(flat_list))

def grid_mode(grid: Tuple[Tuple[int]]) -> int:
    """Finds the mode (most frequent value) in the grid."""
    flat_list = [item for sublist in grid for item in sublist]
    if not flat_list:
        return 0
    counter = Counter(flat_list)
    mode, _ = counter.most_common(1)[0]
    return mode

def grid_max(grid: Tuple[Tuple[int]]) -> int:
    """Finds the maximum value in the grid."""
    flat_list = [item for sublist in grid for item in sublist]
    if not flat_list:
        return 0
    return max(flat_list)

def grid_min(grid: Tuple[Tuple[int]]) -> int:
    """Finds the minimum value in the grid."""
    flat_list = [item for sublist in grid for item in sublist]
    if not flat_list:
        return 0
    return min(flat_list)

def count_nonzero(grid: Tuple[Tuple[int]]) -> int:
    """Counts the number of non-zero values in the grid."""
    return sum(1 for row in grid for cell in row if cell != 0)

def count_value(grid: Tuple[Tuple[int]], value: int) -> int:
    """Counts the number of occurrences of a specific value in the grid."""
    return sum(row.count(value) for row in grid)
