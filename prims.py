from collections import Counter, deque

import numpy as np

from arc_types import *

# rotation
def rot45(grid: Grid, fill_value: Integer) -> Grid:
    """ Rotates the grid by 45 degrees, filling empty spaces with fill_value """
    rows = len(grid)
    cols = len(grid[0]) if rows > 0 else 0
    new_size = rows + cols - 1
    new_grid = [[fill_value] * new_size for _ in range(new_size)]
    
    for i in range(rows):
        for j in range(cols):
            new_i = i + j
            new_j = cols - 1 - i + j
            if 0 <= new_i < new_size and 0 <= new_j < new_size:
                new_grid[new_i][new_j] = grid[i][j]
    
    return tuple([tuple(row) for row in new_grid])

def rot90(grid: Grid) -> Grid:
    """ clockwise rotation by 90 degrees """
    return tuple([tuple(row) for row in zip(*grid[::-1])])

def rot135(grid: Grid, fill_value: Integer) -> Grid:
    """ Rotates the grid by 135 degrees, filling empty spaces with fill_value """
    return rot45(rot90(grid), fill_value)

def rot180(grid: Grid) -> Grid:
    """ Rotates the grid by 180 degrees """
    return tuple([tuple(row[::-1]) for row in grid[::-1]])


# shift
def shift_down(grid: Grid, steps: Integer, fill_value: Integer) -> Grid:
    """ Shifts the grid down by a specified number of steps """
    if not grid or not grid[0]:
        return grid
    if steps <= 0:
        return grid
    steps = min(steps, len(grid))  # Ensure steps do not exceed grid height
    return tuple([tuple([fill_value] * len(grid[0])) for _ in range(steps)] + [tuple(row) for row in grid[:-steps]])

def shift_right(grid: Grid, steps: Integer, fill_value: Integer) -> Grid:
    """ Shifts the grid to the right by a specified number of steps """
    if not grid or not grid[0]:
        return grid
    if steps <= 0:
        return grid
    steps = min(steps, len(grid[0]))  # Ensure steps do not exceed grid width
    return tuple([tuple([fill_value] * steps + list(row[:-steps])) for row in grid])

def shift_left(grid: Grid, steps: Integer, fill_value: Integer) -> Grid:
    """ Shifts the grid to the left by a specified number of steps """
    if not grid or not grid[0]:
        return grid
    if steps <= 0:
        return grid
    steps = min(steps, len(grid[0]))  # Ensure steps do not exceed grid width
    return tuple([tuple(list(row[steps:]) + [fill_value] * steps) for row in grid])

def shift_up(grid: Grid, steps: Integer, fill_value: Integer) -> Grid:
    """ Shifts the grid up by a specified number of steps """
    if not grid or not grid[0]:
        return grid
    if steps <= 0:
        return grid
    steps = min(steps, len(grid))  # Ensure steps do not exceed grid height
    return tuple([tuple(row) for row in grid[steps:]] + [tuple([fill_value] * len(grid[0])) for _ in range(steps)])

# mirroring
def hmirror(grid: Grid) -> Grid:
    """ mirroring along horizontal """
    return tuple(tuple(row) for row in grid[::-1])

def vmirror(grid: Grid) -> Grid:
    """ Mirroring along vertical axis """
    return tuple([tuple(row[::-1]) for row in grid])

def diagonal_mirror(grid: Grid) -> Grid:
    """ Mirrors the grid along the main diagonal """
    return tuple([tuple(row) for row in zip(*grid)])

# scaling
def scale_up(grid: Grid, factor: Integer) -> Grid:
    """ Scales up the grid by a given factor """
    if factor <= 0:
        return grid
    return tuple([tuple(cell for cell in row for _ in range(factor)) for row in grid for _ in range(factor)])

def scale_down(grid: Grid, factor: Integer) -> Grid:
    """ Scales down the grid by a given factor """
    if factor <= 0:
        return grid
    return tuple([tuple(row[::factor]) for row in grid[::factor]])

# crop/ trimming
def tophalf(grid: Grid) -> Grid:
    """ upper half """
    return tuple(tuple(row) for row in grid[:len(grid) // 2])

def bottomhalf(grid: Grid) -> Grid:
    """ lower half """
    return tuple(tuple(row) for row in grid[len(grid) // 2:])

def lefthalf(grid: Grid) -> Grid:
    """ left half """
    return tuple([tuple(row[:len(row) // 2]) for row in grid])

def trim(grid: Grid) -> Grid:
    """ removes border """
    return tuple([tuple(r[1:-1]) for r in grid[1:-1]])

def crop(grid: Grid, top: Integer, bottom: Integer, left: Integer, right: Integer) -> Grid:
    """Crops the grid by removing specified number of rows and columns from each side."""
    if not grid or not grid[0]:
        return grid
    return tuple([tuple(row[left:len(row)-right]) for row in grid[top:len(grid)-bottom]])

# Interpolation
def compress(grid: Grid) -> Grid:
    """ removes frontiers """
    ri = [i for i, r in enumerate(grid) if len(set(r)) == 1]
    ci = [j for j, c in enumerate(zip(*grid)) if len(set(c)) == 1]
    return tuple([tuple(v for j, v in enumerate(r) if j not in ci) for i, r in enumerate(grid) if i not in ri])

def dilate(grid: Grid) -> Grid:
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
def convolve(grid: Grid, kernel: Grid) -> Grid:
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
def transpose(grid: Grid) -> Grid:
    """ Transposes the grid (swaps rows and columns) """
    return tuple([tuple(row) for row in zip(*grid)])

# special effects
def checkered_mask(grid: Grid) -> Grid:
    """ Applies a checkered mask to the grid, replacing None with 0 """
    return tuple([tuple(cell if (i + j) % 2 == 0 else 0 for j, cell in enumerate(row)) for i, row in enumerate(grid)])

def extract_outline(grid: Grid) -> Grid:
    """ Extracts the outline of shapes in the grid """
    outline_grid = [[0]*len(row) for row in grid]
    for i in range(1, len(grid)-1):
        for j in range(1, len(grid[0])-1):
            if grid[i][j] != grid[i+1][j] or grid[i][j] != grid[i-1][j] or grid[i][j] != grid[i][j+1] or grid[i][j] != grid[i][j-1]:
                outline_grid[i][j] = 1
    return tuple([tuple(row) for row in outline_grid])

def extract_subgrid(grid: Grid, start_row: Integer, start_col: Integer, num_rows: Integer, num_cols: Integer) -> Grid:
    """Extracts a subgrid starting from (start_row, start_col) with specified number of rows and columns."""
    return tuple([tuple(row[start_col:start_col + num_cols]) for row in grid[start_row:start_row + num_rows]])

def blur(grid: Grid) -> Grid:
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

def erode(grid: Grid) -> Grid:
    """Erodes the shapes in the grid."""
    new_grid = [[0] * len(row) for row in grid]
    for i in range(1, len(grid) - 1):
        for j in range(1, len(grid[0]) - 1):
            if grid[i][j] == 1 and all(grid[i+di][j+dj] == 1 for di, dj in [(-1, 0), (1, 0), (0, -1), (0, 1)]):
                new_grid[i][j] = 1
    return tuple([tuple(row) for row in new_grid])

# magic number

# coloring
def invert_colors(grid: Grid) -> Grid:
    """ Inverts the colors in the grid (assuming binary colors 0 and 1), ignoring None values """
    return tuple([tuple(1 - cell for cell in row) for row in grid])

def fill_color(grid: Grid, color: Integer) -> Grid:
    """ Fills the grid with the specified color, replacing all non-None values """
    return tuple([tuple(color for cell in row) for row in grid])

def replace_value(grid: Grid, old_value: Integer, new_value: Integer) -> Grid:
    """Replaces all instances of old_value with new_value in the grid"""
    return tuple([tuple(new_value if cell == old_value else cell for cell in row) for row in grid])


def flood_fill(grid: Grid, start_row: Integer, start_col: Integer, new_color: Integer) -> Grid:
    rows, cols = len(grid), len(grid[0])
    
    # Check if the starting coordinates are within the grid's bounds
    if not (0 <= start_row < rows and 0 <= start_col < cols):
        return grid
    
    color_to_replace = grid[start_row][start_col]
    if color_to_replace == new_color:
        return grid
    
    q = deque([(start_row, start_col)])
    new_grid = [list(row) for row in grid]
    
    while q:
        x, y = q.popleft()
        if new_grid[x][y] == color_to_replace:
            new_grid[x][y] = new_color
            for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                nx, ny = x + dx, y + dy
                if 0 <= nx < rows and 0 <= ny < cols:
                    q.append((nx, ny))
    
    return tuple(tuple(row) for row in new_grid)

def normalize_grid(grid: Grid, new_min: Integer, new_max: Integer) -> Grid:
    """ Normalizes the grid values to a new range [new_min, new_max]. """
    old_min = min(min(row) for row in grid)
    old_max = max(max(row) for row in grid)
    range_old = old_max - old_min
    if range_old == 0:
        # If all values are the same, return the grid filled with new_min
        return tuple([tuple(new_min for _ in row) for row in grid])
    range_new = new_max - new_min
    return tuple([tuple(int((cell - old_min) / range_old * range_new + new_min) for cell in row) for row in grid])

# border manipulations
def add_border(grid: Grid, border_value: Integer) -> Grid:
    """ adds a border around the grid """
    if not grid or not grid[0]:
        return grid
    if not (0 <= border_value <= 9): 
        return grid
    
    new_grid = [[border_value] * (len(grid[0]) + 2)]
    for row in grid:
        new_grid.append([border_value] + list(row) + [border_value])
    new_grid.append([border_value] * (len(grid[0]) + 2))
    return tuple([tuple(row) for row in new_grid])

# logic operators
def boolean_and(grid1: Grid, grid2: Grid) -> Grid:
    """Performs element-wise AND operation between two grids."""
    return tuple([tuple(cell1 and cell2 for cell1, cell2 in zip(row1, row2)) for row1, row2 in zip(grid1, grid2)])

def boolean_or(grid1: Grid, grid2: Grid) -> Grid:
    """Performs element-wise OR operation between two grids."""
    return tuple([tuple(cell1 or cell2 for cell1, cell2 in zip(row1, row2)) for row1, row2 in zip(grid1, grid2)])

def is_square_grid(grid: Grid) -> Boolean:
    """Checks if the grid is square."""
    return len(grid) == len(grid[0]) if grid else False

def contains_value(grid: Grid, value: Integer) -> Boolean:
    """Checks if the grid contains a specific value."""
    return any(value in row for row in grid)

# heuristics
def grid_mean(grid: Grid) -> Integer:
    """Calculates the mean of all values in the grid."""
    flat_list = [item for sublist in grid for item in sublist]
    if not flat_list:
        return 0
    return Integer(sum(flat_list) / len(flat_list))

def grid_mode(grid: Grid) -> Integer:
    """Finds the mode (most frequent value) in the grid."""
    flat_list = [item for sublist in grid for item in sublist]
    if not flat_list:
        return 0
    counter = Counter(flat_list)
    mode, _ = counter.most_common(1)[0]
    return mode

def grid_max(grid: Grid) -> Integer:
    """Finds the maximum value in the grid."""
    flat_list = [item for sublist in grid for item in sublist]
    if not flat_list:
        return 0
    return max(flat_list)

def grid_min(grid: Grid) -> Integer:
    """Finds the minimum value in the grid."""
    flat_list = [item for sublist in grid for item in sublist]
    if not flat_list:
        return 0
    return min(flat_list)

def grid_sum(grid: Grid) -> Integer:
    """Calculates the sum of all values in the grid."""
    return sum(sum(row) for row in grid)

def count_nonzero(grid: Grid) -> Integer:
    """Counts the number of non-zero values in the grid."""
    return sum(1 for row in grid for cell in row if cell != 0)

def count_value(grid: Grid, value: Integer) -> Integer:
    """Counts the number of occurrences of a specific value in the grid."""
    return sum(row.count(value) for row in grid)

def get_value(grid: Grid, row: Integer, col: Integer) -> Integer:
    """Get the value at a specific cell in the grid."""
    if 0 <= row < len(grid) and 0 <= col < len(grid[0]):
        return grid[row][col]
    else:
        return 0

def column_sums(grid: Grid) -> IntegerList:
    """Calculates the sum of each column in the grid."""
    return tuple(sum(row[i] for row in grid) for i in range(len(grid[0])))

# rows
def sort_integer_list(int_list: IntegerList) -> IntegerList:
    """Sorts the list of integers in ascending order."""
    return tuple(sorted(int_list))

# numbers
def sum_integer_list(int_list: IntegerList) -> Integer:
    """Calculates the sum of all integers in the list."""
    return sum(int_list)

def min_integer_list(int_list: IntegerList) -> Integer:
    """Finds the minimum value in the list."""
    return min(int_list) if int_list else 0

def max_integer_list(int_list: IntegerList) -> Integer:
    """Finds the maximum value in the list."""
    return max(int_list) if int_list else 0

