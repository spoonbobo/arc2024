from arc_types import *
from collections import Counter
import numpy as np

# rotation
def rot45(grid: Grid, fill_value: Integer) -> Grid:
    """ Rotates the grid by 45 degrees, filling empty spaces with fill_value """
    grid_array = np.array(grid)
    
    if grid_array.ndim != 2:
        return grid
    
    rows, cols = grid_array.shape
    new_size = rows + cols - 1
    new_grid = np.full((new_size, new_size), fill_value)
    
    for i in range(rows):
        for j in range(cols):
            new_i = i + j
            new_j = cols - 1 - i + j
            if 0 <= new_i < new_size and 0 <= new_j < new_size:
                new_grid[new_i, new_j] = grid_array[i, j]
    
    return tuple(tuple(int(cell) for cell in row) for row in new_grid)

def rot90(grid: Grid) -> Grid:
    """ clockwise rotation by 90 degrees """
    grid_array = np.array(grid)
    rotated = np.rot90(grid_array, k=-1)
    return tuple(tuple(int(cell) for cell in row) for row in rotated)

def rot135(grid: Grid, fill_value: Integer) -> Grid:
    """ Rotates the grid by 135 degrees, filling empty spaces with fill_value """
    rotated_90 = rot90(grid)
    return rot45(rotated_90, fill_value)

def rot180(grid: Grid) -> Grid:
    """ Rotates the grid by 180 degrees """
    grid_array = np.array(grid)
    rotated = np.rot90(grid_array, k=2)
    return tuple(tuple(int(cell) for cell in row) for row in rotated)

def rotate_mirror(grid: Grid) -> Grid:
    """ Rotates the grid by 90 degrees and then mirrors it horizontally """
    rotated = rot90(grid)
    mirrored = hmirror(rotated)
    return mirrored

# shift
def shift_down(grid: Grid, steps: Integer, fill_value: Integer) -> Grid:
    """ Shifts the grid down by a specified number of steps """
    grid_array = np.array(grid)
    if steps <= 0:
        return grid
    steps = min(steps, grid_array.shape[0])
    new_grid = np.full_like(grid_array, fill_value)
    new_grid[steps:] = grid_array[:-steps]
    return tuple(tuple(int(cell) for cell in row) for row in new_grid)

def shift_right(grid: Grid, steps: Integer, fill_value: Integer) -> Grid:
    """ Shifts the grid to the right by a specified number of steps """
    grid_array = np.array(grid)
    if steps <= 0:
        return grid
    steps = min(steps, grid_array.shape[1])
    new_grid = np.full_like(grid_array, fill_value)
    new_grid[:, steps:] = grid_array[:, :-steps]
    return tuple(tuple(int(cell) for cell in row) for row in new_grid)

def shift_left(grid: Grid, steps: Integer, fill_value: Integer) -> Grid:
    """ Shifts the grid to the left by a specified number of steps """
    grid_array = np.array(grid)
    if steps <= 0:
        return grid
    steps = min(steps, grid_array.shape[1])
    new_grid = np.full_like(grid_array, fill_value)
    new_grid[:, :-steps] = grid_array[:, steps:]
    return tuple(tuple(int(cell) for cell in row) for row in new_grid)

def shift_up(grid: Grid, steps: Integer, fill_value: Integer) -> Grid:
    """ Shifts the grid up by a specified number of steps """
    grid_array = np.array(grid)
    if steps <= 0:
        return grid
    steps = min(steps, grid_array.shape[0])
    new_grid = np.full_like(grid_array, fill_value)
    new_grid[:-steps] = grid_array[steps:]
    return tuple(tuple(int(cell) for cell in row) for row in new_grid)

def shift_diagonal(grid: Grid, steps: Integer, fill_value: Integer) -> Grid:
    """ Shifts the grid diagonally (down-right) by a specified number of steps """
    grid_array = np.array(grid)
    if steps <= 0:
        return grid
    steps = min(steps, min(grid_array.shape))
    new_grid = np.full_like(grid_array, fill_value)
    new_grid[steps:, steps:] = grid_array[:-steps, :-steps]
    return tuple(tuple(int(cell) for cell in row) for row in new_grid)

# mirroring
def anti_diagonal_mirror(grid: Grid) -> Grid:
    """ Mirrors the grid along the anti-diagonal """
    grid_array = np.array(grid)
    mirrored = np.fliplr(np.flipud(grid_array)).T
    return tuple(tuple(int(cell) for cell in row) for row in mirrored)

def hmirror(grid: Grid) -> Grid:
    """ mirroring along horizontal """
    grid_array = np.array(grid)
    mirrored = np.flipud(grid_array)
    return tuple(tuple(int(cell) for cell in row) for row in mirrored)

def vmirror(grid: Grid) -> Grid:
    """ Mirroring along vertical axis """
    grid_array = np.array(grid)
    mirrored = np.fliplr(grid_array)
    return tuple(tuple(int(cell) for cell in row) for row in mirrored)

def diagonal_mirror(grid: Grid) -> Grid:
    """ Mirrors the grid along the main diagonal """
    grid_array = np.array(grid)
    mirrored = grid_array.T
    return tuple(tuple(int(cell) for cell in row) for row in mirrored)

# scaling
def scale_up_non_uniform(grid: Grid, row_factor: Integer, col_factor: Integer) -> Grid:
    """ Scales up the grid by different factors for rows and columns """
    if row_factor <= 0 or col_factor <= 0:
        return grid
    grid_array = np.array(grid)
    scaled = np.kron(grid_array, np.ones((row_factor, col_factor), dtype=int))
    return tuple(tuple(int(cell) for cell in row) for row in scaled)

def scale_up(grid: Grid, factor: Integer) -> Grid:
    """ Scales up the grid by a given factor """
    if factor <= 0:
        return grid
    grid_array = np.array(grid)
    scaled = np.kron(grid_array, np.ones((factor, factor), dtype=int))
    return tuple(tuple(int(cell) for cell in row) for row in scaled)

def scale_down(grid: Grid, factor: Integer) -> Grid:
    """ Scales down the grid by a given factor """
    if factor <= 0:
        return grid
    grid_array = np.array(grid)
    scaled = grid_array[::factor, ::factor]
    return tuple(tuple(int(cell) for cell in row) for row in scaled)

# crop/ trimming
def tophalf(grid: Grid) -> Grid:
    """ upper half """
    grid_array = np.array(grid)
    half = grid_array[:len(grid_array) // 2]
    return tuple(tuple(int(cell) for cell in row) for row in half)

def bottomhalf(grid: Grid) -> Grid:
    """ lower half """
    grid_array = np.array(grid)
    half = grid_array[len(grid_array) // 2:]
    return tuple(tuple(int(cell) for cell in row) for row in half)

def lefthalf(grid: Grid) -> Grid:
    """ left half """
    grid_array = np.array(grid)
    half = grid_array[:, :grid_array.shape[1] // 2]
    return tuple(tuple(int(cell) for cell in row) for row in half)

def righthalf(grid: Grid) -> Grid:
    """ right half """
    grid_array = np.array(grid)
    half = grid_array[:, grid_array.shape[1] // 2:]
    return tuple(tuple(int(cell) for cell in row) for row in half)

def trim(grid: Grid) -> Grid:
    """ removes border """
    grid_array = np.array(grid)
    trimmed = grid_array[1:-1, 1:-1]
    return tuple(tuple(int(cell) for cell in row) for row in trimmed)

def crop(grid: Grid, top: Integer, bottom: Integer, left: Integer, right: Integer) -> Grid:
    """Crops the grid by removing specified number of rows and columns from each side."""
    grid_array = np.array(grid)
    cropped = grid_array[top:grid_array.shape[0]-bottom, left:grid_array.shape[1]-right]
    return tuple(tuple(int(cell) for cell in row) for row in cropped)

# Interpolation
def compress(grid: Grid) -> Grid:
    """ removes frontiers """
    grid_array = np.array(grid)
    ri = [i for i, r in enumerate(grid_array) if len(set(r)) == 1]
    ci = [j for j, c in enumerate(grid_array.T) if len(set(c)) == 1]
    compressed = np.delete(grid_array, ri, axis=0)
    compressed = np.delete(compressed, ci, axis=1)
    return tuple(tuple(int(cell) for cell in row) for row in compressed)

def dilate(grid: Grid) -> Grid:
    """ dilates the shapes in the grid """
    grid_array = np.array(grid)
    new_grid = np.copy(grid_array)
    for i in range(grid_array.shape[0]):
        for j in range(grid_array.shape[1]):
            if grid_array[i, j] == 1:
                if i > 0:
                    new_grid[i-1, j] = 1
                if i < grid_array.shape[0] - 1:
                    new_grid[i+1, j] = 1
                if j > 0:
                    new_grid[i, j-1] = 1
                if j < grid_array.shape[1] - 1:
                    new_grid[i, j+1] = 1
    return tuple(tuple(int(cell) for cell in row) for row in new_grid)

# convolution
def convolve(grid: Grid, kernel: Grid) -> Grid:
    """Applies a convolution operation to the grid using the given kernel."""
    grid_array = np.array(grid)
    kernel_array = np.array(kernel)
    
    if grid_array.ndim != 2 or kernel_array.ndim != 2 or kernel_array.size == 0:
        return grid
    
    kernel_height, kernel_width = kernel_array.shape
    grid_height, grid_width = grid_array.shape
    
    if kernel_height > grid_height or kernel_width > grid_width:
        return grid
    
    output_height = grid_height - kernel_height + 1
    output_width = grid_width - kernel_width + 1
    
    output_grid = np.zeros((output_height, output_width), dtype=int)
    for i in range(output_height):
        for j in range(output_width):
            region = grid_array[i:i+kernel_height, j:j+kernel_width]
            output_grid[i, j] = np.sum(region * kernel_array)
    
    return tuple(tuple(int(cell) for cell in row) for row in output_grid)

# transformation
def transpose(grid: Grid) -> Grid:
    """ Transposes the grid (swaps rows and columns) """
    grid_array = np.array(grid)
    transposed = grid_array.T
    return tuple(tuple(int(cell) for cell in row) for row in transposed)

# special effects
def checkered_mask(grid: Grid) -> Grid:
    """ Applies a checkered mask to the grid, replacing None with 0 """
    grid_array = np.array(grid)
    mask = np.indices(grid_array.shape).sum(axis=0) % 2 == 0
    masked = np.where(mask, grid_array, 0)
    return tuple(tuple(int(cell) for cell in row) for row in masked)

def extract_outline(grid: Grid) -> Grid:
    """ Extracts the outline of shapes in the grid """
    grid_array = np.array(grid)
    outline_grid = np.zeros_like(grid_array, dtype=int)
    
    for i in range(1, grid_array.shape[0] - 1):
        for j in range(1, grid_array.shape[1] - 1):
            if (grid_array[i, j] != grid_array[i + 1, j] or
                grid_array[i, j] != grid_array[i - 1, j] or
                grid_array[i, j] != grid_array[i, j + 1] or
                grid_array[i, j] != grid_array[i, j - 1]):
                outline_grid[i, j] = 1
    
    return tuple(tuple(int(cell) for cell in row) for row in outline_grid)

def extract_subgrid(grid: Grid, start_row: Integer, start_col: Integer, num_rows: Integer, num_cols: Integer) -> Grid:
    """Extracts a subgrid starting from (start_row, start_col) with specified number of rows and columns."""
    grid_array = np.array(grid)
    subgrid = grid_array[start_row:start_row + num_rows, start_col:start_col + num_cols]
    return tuple(tuple(int(cell) for cell in row) for row in subgrid)

def blur(grid: Grid) -> Grid:
    """ Applies a simple blur effect to the grid """
    grid_array = np.array(grid)
    padded_grid = np.pad(grid_array, pad_width=1, mode='constant', constant_values=0)
    new_grid = np.zeros_like(grid_array)
    
    for i in range(1, padded_grid.shape[0] - 1):
        for j in range(1, padded_grid.shape[1] - 1):
            neighbors = padded_grid[i-1:i+2, j-1:j+2]
            new_grid[i-1, j-1] = np.mean(neighbors)
    
    return tuple(tuple(int(cell) for cell in row) for row in new_grid)

def erode(grid: Grid) -> Grid:
    """ Erodes the shapes in the grid """
    grid_array = np.array(grid)
    padded_grid = np.pad(grid_array, pad_width=1, mode='constant', constant_values=0)
    new_grid = np.zeros_like(grid_array)
    
    for i in range(1, padded_grid.shape[0] - 1):
        for j in range(1, padded_grid.shape[1] - 1):
            if padded_grid[i, j] == 1 and np.all(padded_grid[i-1:i+2, j-1:j+2] == 1):
                new_grid[i-1, j-1] = 1
    
    return tuple(tuple(int(cell) for cell in row) for row in new_grid)

# coloring

def fill_color(grid: Grid, color: Integer) -> Grid:
    """ Fills the grid with the specified color, replacing all non-None values """
    grid_array = np.array(grid)
    filled = np.where(grid_array != None, color, grid_array)
    return tuple(tuple(int(cell) for cell in row) for row in filled)

def replace_value(grid: Grid, old_value: Integer, new_value: Integer) -> Grid:
    """Replaces all instances of old_value with new_value in the grid"""
    grid_array = np.array(grid)
    replaced = np.where(grid_array == old_value, new_value, grid_array)
    return tuple(tuple(int(cell) for cell in row) for row in replaced)

def fill_region(grid: Grid, start_row: Integer, start_col: Integer, color: Integer) -> Grid:
    """Fills a region starting from (start_row, start_col) with the specified color."""
    grid_array = np.array(grid)
    
    # Check if start_row and start_col are within the bounds of the grid
    if not (0 <= start_row < grid_array.shape[0] and 0 <= start_col < grid_array.shape[1]):
        return grid
    
    if grid_array[start_row, start_col] == color:
        return grid
    
    target_value = grid_array[start_row, start_col]
    stack = [(start_row, start_col)]
    while stack:
        r, c = stack.pop()
        if grid_array[r, c] == target_value:
            grid_array[r, c] = color
            if r > 0:
                stack.append((r-1, c))
            if r < grid_array.shape[0] - 1:
                stack.append((r+1, c))
            if c > 0:
                stack.append((r, c-1))
            if c < grid_array.shape[1] - 1:
                stack.append((r, c+1))
    
    return tuple(tuple(int(cell) for cell in row) for row in grid_array)

def count_color(grid: Grid, color: Integer) -> Integer:
    """Counts the number of cells with the specified color in the grid."""
    grid_array = np.array(grid)
    count = np.sum(grid_array == color)
    return Integer(count)

# border manipulations
def add_border(grid: Grid, border_value: Integer) -> Grid:
    """Adds a border around the grid."""
    grid_array = np.array(grid, dtype=int)  # Ensure the grid is of integer type
    new_grid = np.pad(grid_array, pad_width=1, mode='constant', constant_values=border_value)
    
    # Ensure new_grid is 2D
    if new_grid.ndim == 1:
        new_grid = np.expand_dims(new_grid, axis=0)
    
    return tuple(tuple(int(cell) for cell in row) for row in new_grid.tolist())

# logic operators
def boolean_and(grid1: Grid, grid2: Grid) -> Grid:
    """Performs element-wise AND operation between two grids."""
    grid1_array = np.array(grid1)
    grid2_array = np.array(grid2)
    
    if grid1_array.shape != grid2_array.shape:
        return tuple()
    
    result = np.logical_and(grid1_array, grid2_array).astype(int)
    return tuple(tuple(int(cell) for cell in row) for row in result)

def boolean_or(grid1: Grid, grid2: Grid) -> Grid:
    """Performs element-wise OR operation between two grids."""
    grid1_array = np.array(grid1)
    grid2_array = np.array(grid2)
    
    if grid1_array.shape != grid2_array.shape:
        return tuple()
    
    result = np.logical_or(grid1_array, grid2_array).astype(int)
    return tuple(tuple(int(cell) for cell in row) for row in result)

def is_symmetric(grid: Grid, axis: str = 'horizontal') -> Boolean:
    """Checks if the grid is symmetric along the specified axis ('horizontal' or 'vertical')."""
    grid_array = np.array(grid)
    if axis == 'horizontal':
        return Boolean(np.array_equal(grid_array, np.flipud(grid_array)))
    elif axis == 'vertical':
        return Boolean(np.array_equal(grid_array, np.fliplr(grid_array)))
    else:
        return Boolean(False)

# heuristics
def grid_mean(grid: Grid) -> Integer:
    """Calculates the mean of all values in the grid."""
    grid_array = np.array(grid)
    mean_value = np.mean(grid_array)
    return Integer(mean_value)

def grid_mode(grid: Grid) -> Integer:
    """Finds the mode (most frequent value) in the grid."""
    grid_array = np.array(grid)
    flat_list = grid_array.flatten()
    counter = Counter(flat_list)
    mode, _ = counter.most_common(1)[0]
    return Integer(mode)

def grid_max(grid: Grid) -> Integer:
    """Finds the maximum value in the grid."""
    grid_array = np.array(grid)
    max_value = np.max(grid_array)
    return Integer(max_value)

def grid_min(grid: Grid) -> Integer:
    """Finds the minimum value in the grid."""
    grid_array = np.array(grid)
    min_value = np.min(grid_array)
    return Integer(min_value)

def grid_sum(grid: Grid) -> Integer:
    """Calculates the sum of all values in the grid."""
    grid_array = np.array(grid)
    sum_value = np.sum(grid_array)
    return Integer(sum_value)

def count_nonzero(grid: Grid) -> Integer:
    """Counts the number of non-zero values in the grid."""
    grid_array = np.array(grid)
    nonzero_count = np.count_nonzero(grid_array)
    return Integer(nonzero_count)

# object
def extract_objects(grid: Grid) -> Objects:
    """Extracts distinct objects from the grid."""
    grid_array = np.array(grid)
    objects = set()
    visited = np.zeros_like(grid_array, dtype=bool)
    
    def dfs(r, c, obj):
        if (r < 0 or r >= grid_array.shape[0] or
            c < 0 or c >= grid_array.shape[1] or
            visited[r, c] or grid_array[r, c] == 0):
            return
        visited[r, c] = True
        obj.add((r, c))
        dfs(r+1, c, obj)
        dfs(r-1, c, obj)
        dfs(r, c+1, obj)
        dfs(r, c-1, obj)
    
    for i in range(grid_array.shape[0]):
        for j in range(grid_array.shape[1]):
            if grid_array[i, j] != 0 and not visited[i, j]:
                obj = set()
                dfs(i, j, obj)
                objects.add(frozenset(obj))
    
    return frozenset(objects)

def object_to_grid(obj: Object, fill_value: Integer) -> Grid:
    """Converts an object to a grid representation."""
    if not obj:
        return tuple()
    
    # Extract rows and columns from the object cells
    rows = [cell[0] for cell in obj]
    cols = [cell[1] for cell in obj]
    
    # Determine the bounding box of the object
    min_row, max_row = min(rows), max(rows)
    min_col, max_col = min(cols), max(cols)
    
    # Create a grid with the appropriate size, filled with the fill_value
    grid = np.full((max_row - min_row + 1, max_col - min_col + 1), fill_value)
    
    # Fill the grid with the object cells
    for (r, c) in obj:
        grid[r - min_row, c - min_col] = 1
    
    # Convert the numpy array to a tuple of tuples
    return tuple(tuple(int(cell) for cell in row) for row in grid)

def object_area(obj: Object) -> Integer:
    """Calculates the area (number of cells) of an object."""
    return Integer(len(obj))

def object_perimeter(obj: Object) -> Integer:
    """Calculates the perimeter of an object."""
    perimeter = 0
    for (r, c) in obj:
        if (r-1, c) not in obj:
            perimeter += 1
        if (r+1, c) not in obj:
            perimeter += 1
        if (r, c-1) not in obj:
            perimeter += 1
        if (r, c+1) not in obj:
            perimeter += 1
    return Integer(perimeter)

def largest_object(objects: Objects) -> Object:
    """Finds the largest object by area."""
    if not objects:
        return frozenset()
    
    largest = max(objects, key=len)
    return largest

def smallest_object(objects: Objects) -> Object:
    """Finds the smallest object by area."""
    if not objects:
        return frozenset()
    
    smallest = min(objects, key=len)
    return smallest