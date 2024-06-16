import ast
from arc_types import *
from typing import get_type_hints

# Raw text containing the function definitions
raw_text = """
def count_nonzero(grid: Grid) -> Integer:
    return sum(1 for row in grid for cell in row if cell != 0)

def pad_grid(grid: Grid) -> Grid:
    max_size = max(len(row) for row in grid)
    return [[0] * max_size for _ in range(max_size)]

def get_nonzero_cells(grid: Grid) -> List[Cell]:
    return [(row[i],) for i, row in enumerate(grid) if any(cell != 0 for cell in row)]

def get_max_length(grid: Grid) -> Integer:
    return max(len(row) for row in grid)

def is_symmetric(grid: Grid) -> Boolean:
    return grid == [[cell for cell in reversed(row)] for row in reversed(list(zip(*grid)))]

def get_center_cell(grid: Grid) -> Cell:
    center_row = len(grid) // 2
    center_col = len(grid[0]) // 2
    return (grid[center_row][center_col],)

def get_nonzero_cells_in_radius(grid: Grid, radius: Integer) -> List[Cell]:
    result = []
    for i in range(len(grid)):
        for j in range(len(grid[0])):
            if all(abs(i - k) + abs(j - m) <= radius for k, row in enumerate(grid) for m, cell in enumerate(row) if cell != 0):
                result.append((grid[i][j],))
    return result

def get_nonzero_cells_in_orthogonal_direction(grid: Grid, direction: Integer) -> List[Cell]:
    result = []
    for i in range(len(grid)):
        for j in range(len(grid[0])):
            if all(abs(i - k) == abs(j - m) and (k, m) != (i, j) for k, row in enumerate(grid) for m, cell in enumerate(row) if cell != 0):
                result.append((grid[i][j],))
    return result

def get_nonzero_cells_in_diagonal_direction(grid: Grid, direction: Integer) -> List[Cell]:
    result = []
    for i in range(len(grid)):
        for j in range(len(grid[0])):
            if all(abs(i - k) == abs(j - m) and (k, m) != (i, j) and ((direction == 1 and i < k) or (direction == -1 and i > k)) for k, row in enumerate(grid) for m, cell in enumerate(row) if cell != 0):
                result.append((grid[i][j],))
    return result
"""

# Parse the raw text to extract function definitions
module = ast.parse(raw_text)

functions_info = {}
for node in module.body:
    if isinstance(node, ast.FunctionDef):
        func_name = node.name
        func_code = compile(ast.Module(body=[node], type_ignores=[]), filename="<ast>", mode="exec")
        func = {}
        exec(func_code, globals(), func)
        func = func[func_name]
        type_hints = get_type_hints(func)
        return_type = type_hints.pop('return', None)
        functions_info[func_name] = {
            'func': func,
            'return_type': return_type,
            'input_types': type_hints
        }

# Print the extracted information
for name, info in functions_info.items():
    print(f"Function Name: {name}")
    print(f"Function: {info['func']}")
    print(f"Return Type: {info['return_type']}")
    print(f"Parameter Types: {info['input_types']}")
    print()
