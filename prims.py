from arc_types import *

# rotation
def rot45(grid: Grid, fill_value: int) -> Grid:
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

def rot135(grid: Grid, fill_value: int) -> Grid:
    """ Rotates the grid by 135 degrees, filling empty spaces with fill_value """
    return rot45(rot90(grid), fill_value)

def rot180(grid: Grid) -> Grid:
    """ Rotates the grid by 180 degrees """
    return tuple([tuple(row[::-1]) for row in grid[::-1]])

def rot225(grid: Grid, fill_value: int) -> Grid:
    """ Rotates the grid by 225 degrees, filling empty spaces with fill_value """
    return rot45(rot180(grid), fill_value)

def rot270(grid: Grid) -> Grid:
    """ clockwise rotation by 270 degrees """
    return tuple([tuple(row) for row in zip(*grid)][::-1])

def rot315(grid: Grid, fill_value: int) -> Grid:
    """ Rotates the grid by 315 degrees, filling empty spaces with fill_value """
    return rot45(rot270(grid), fill_value)

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
def scale_up(grid: Grid, factor: int) -> Grid:
    """ Scales up the grid by a given factor """
    if factor <= 0:
        return grid
    return tuple([tuple(cell for cell in row for _ in range(factor)) for row in grid for _ in range(factor)])

def scale_down(grid: Grid, factor: int) -> Grid:
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

# interpolation

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

# special effects
def checkered_mask(grid: Grid) -> Grid:
    """ Applies a checkered mask to the grid """
    return tuple([tuple(cell if (i + j) % 2 == 0 else None for j, cell in enumerate(row)) for i, row in enumerate(grid)])

def extract_outline(grid: Grid) -> Grid:
    """ Extracts the outline of shapes in the grid """
    outline_grid = [[0]*len(row) for row in grid]
    for i in range(1, len(grid)-1):
        for j in range(1, len(grid[0])-1):
            if grid[i][j] != grid[i+1][j] or grid[i][j] != grid[i-1][j] or grid[i][j] != grid[i][j+1] or grid[i][j] != grid[i][j-1]:
                outline_grid[i][j] = 1
    return tuple([tuple(row) for row in outline_grid])

# coloring
def invert_colors(grid: Grid) -> Grid:
    """ Inverts the colors in the grid (assuming binary colors 0 and 1), ignoring None values """
    return tuple([tuple(1 - cell if cell is not None else None for cell in row) for row in grid])

# border manipulations
def add_border(grid: Grid, border_value: int) -> Grid:
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

# heuristics
def grid_mean(grid: Grid) -> int:
    """Calculates the mean of all values in the grid."""
    flat_list = [item if item is not None else 0 for sublist in grid for item in sublist]
    if not flat_list:
        return 0
    return int(sum(flat_list) / len(flat_list))

def grid_max(grid: Grid) -> int:
    """Finds the maximum value in the grid."""
    flat_list = [item for sublist in grid for item in sublist if item is not None]
    if not flat_list:
        return 0
    return max(flat_list)

def grid_min(grid: Grid) -> int:
    """Finds the minimum value in the grid."""
    flat_list = [item for sublist in grid for item in sublist if item is not None]
    if not flat_list:
        return 0
    return min(flat_list)

# transformation
def transpose(grid: Grid) -> Grid:
    """ Transposes the grid (swaps rows and columns) """
    return tuple([tuple(row) for row in zip(*grid)])
