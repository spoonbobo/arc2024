from typing import Tuple, FrozenSet

def flood_fill(grid: Tuple[Tuple[int]], start: Tuple[int, int]) -> FrozenSet[Tuple[int, int]]:
    """Performs flood fill from a starting cell and returns the connected region as a frozen set."""
    sr, sc = start
    rows, cols = len(grid), len(grid[0])
    original_value = grid[sr][sc]
    stack = [(sr, sc)]
    connected_cells = set()
    visited = set()

    while stack:
        r, c = stack.pop()
        if (r, c) not in visited:
            visited.add((r, c))
            connected_cells.add((r, c))
            for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                nr, nc = r + dr, c + dc
                if 0 <= nr < rows and 0 <= nc < cols and grid[nr][nc] == original_value and (nr, nc) not in visited:
                    stack.append((nr, nc))
    
    return frozenset(connected_cells)

def flood_fill_all(grid: Tuple[Tuple[int]]) -> Tuple[FrozenSet[Tuple[int, int]], ...]:
    """Finds all connected regions in the grid and returns them as a tuple of frozen sets."""
    rows, cols = len(grid), len(grid[0])
    visited = set()
    regions = []

    for r in range(rows):
        for c in range(cols):
            if (r, c) not in visited:
                region = flood_fill(grid, (r, c))
                if region:
                    regions.append(region)
                    visited.update(region)
    
    return tuple(regions)

def count_connected_components(grid: Tuple[Tuple[int]]) -> int:
    """Counts the number of distinct connected components in the grid."""
    rows, cols = len(grid), len(grid[0])
    visited = set()
    component_count = 0

    for r in range(rows):
        for c in range(cols):
            if (r, c) not in visited and grid[r][c] != 0:
                region = flood_fill(grid, (r, c))
                visited.update(region)
                component_count += 1

    return component_count

def paint_regions(grid: Tuple[Tuple[int]], regions: Tuple[FrozenSet[Tuple[int, int]], ...], symbol: int) -> Tuple[Tuple[int]]:
    """Paints each region with the provided integer symbol and returns the new grid."""
    rows, cols = len(grid), len(grid[0])
    new_grid = [[0 for _ in range(cols)] for _ in range(rows)]

    for region in regions:
        for r, c in region:
            new_grid[r][c] = symbol
    
    return tuple(tuple(row) for row in new_grid)

def detect_edges(grid: Tuple[Tuple[int]], background_value: int) -> FrozenSet[Tuple[int, int]]:
    """Detects boundary edges of regions in the grid and returns them as a frozen set of coordinates."""
    rows, cols = len(grid), len(grid[0])
    edges = set()
    
    for r in range(rows):
        for c in range(cols):
            if grid[r][c] != background_value:
                for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                    nr, nc = r + dr, c + dc
                    if not (0 <= nr < rows and 0 <= nc < cols) or grid[nr][nc] == background_value:
                        edges.add((r, c))
                        break
    
    return frozenset(edges)

def cells_surrounded_by_edges(grid: Tuple[Tuple[int]], edges: FrozenSet[Tuple[int, int]]) -> FrozenSet[Tuple[int, int]]:
    """Finds cells that are completely surrounded by edges."""
    rows, cols = len(grid), len(grid[0])
    surrounded_cells = set()

    for r in range(1, rows - 1):
        for c in range(1, cols - 1):
            if all((r + dr, c + dc) in edges for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]):
                surrounded_cells.add((r, c))
    
    return frozenset(surrounded_cells)

def paint_edges(grid: Tuple[Tuple[int]], edges: FrozenSet[Tuple[int, int]], symbol: int) -> Tuple[Tuple[int]]:
    """Paints the edges with the provided integer symbol and returns the new grid."""
    new_grid = [list(row) for row in grid]

    for r, c in edges:
        new_grid[r][c] = symbol
    
    return tuple(tuple(row) for row in new_grid)

def paint_surrounded_cells(grid: Tuple[Tuple[int]], surrounded_cells: FrozenSet[Tuple[int, int]], symbol: int) -> Tuple[Tuple[int]]:
    """Paints the cells surrounded by edges with the provided integer symbol and returns the new grid."""
    new_grid = [list(row) for row in grid]

    for r, c in surrounded_cells:
        new_grid[r][c] = symbol
    
    return tuple(tuple(row) for row in new_grid)

def form_grid_from_region(region: FrozenSet[Tuple[int, int]], symbol: int) -> Tuple[Tuple[int]]:
    """Forms a new grid that includes the region, marked with the symbol, and fills the rest with 0."""
    if not region:
        return tuple()

    min_r = min(r for r, _ in region)
    max_r = max(r for r, _ in region)
    min_c = min(c for _, c in region)
    max_c = max(c for _, c in region)

    height = max_r - min_r + 1
    width = max_c - min_c + 1

    new_grid = [[0 for _ in range(width)] for _ in range(height)]

    for r, c in region:
        new_grid[r - min_r][c - min_c] = symbol
    
    return tuple(tuple(row) for row in new_grid)

def grid_mean(grid: Tuple[Tuple[int]]) -> int:
    """Calculates the mean value of the grid and returns the floor of the mean."""
    flat_grid = [cell for row in grid for cell in row]
    return int(sum(flat_grid) / len(flat_grid))

def grid_mode(grid: Tuple[Tuple[int]]) -> int:
    """Calculates the mode value of the grid without using the statistics library."""
    flat_grid = [cell for row in grid for cell in row]
    frequency = {}
    for value in flat_grid:
        if value in frequency:
            frequency[value] += 1
        else:
            frequency[value] = 1
    mode_value = max(frequency, key=frequency.get)
    return mode_value

def grid_max(grid: Tuple[Tuple[int]]) -> int:
    """Finds the maximum value in the grid."""
    flat_grid = [cell for row in grid for cell in row]
    return max(flat_grid)

def grid_min(grid: Tuple[Tuple[int]]) -> int:
    """Finds the minimum value in the grid."""
    flat_grid = [cell for row in grid for cell in row]
    return min(flat_grid)

def grid_sum(grid: Tuple[Tuple[int]]) -> int:
    """Calculates the sum of all values in the grid."""
    return sum(cell for row in grid for cell in row)

def grid_median(grid: Tuple[Tuple[int]]) -> float:
    """Calculates the median value of the grid."""
    flat_grid = sorted(cell for row in grid for cell in row)
    n = len(flat_grid)
    mid = n // 2
    if n % 2 == 0:
        return int((flat_grid[mid - 1] + flat_grid[mid]) / 2)
    else:
        return flat_grid[mid]

def grid_unique_values(grid: Tuple[Tuple[int]]) -> int:
    """Counts the number of unique values in the grid."""
    flat_grid = [cell for row in grid for cell in row]
    return len(set(flat_grid))

def grid_range(grid: Tuple[Tuple[int]]) -> int:
    """Calculates the range of the grid (max value - min value)."""
    flat_grid = [cell for row in grid for cell in row]
    return max(flat_grid) - min(flat_grid)

def grid_non_zero_count(grid: Tuple[Tuple[int]]) -> int:
    """Counts the number of non-zero elements in the grid."""
    return sum(1 for row in grid for cell in row if cell != 0)


# def test_detect_edges():
#     test_cases = [
#         {
#             "grid": (
#                 (1, 1, 0, 0),
#                 (1, 0, 0, 1),
#                 (0, 0, 1, 1),
#                 (0, 1, 1, 1)
#             ),
#             "background_value": 0,
#             "expected_edges": frozenset({
#                 (0, 0), (0, 1), (0, 3), (1, 0), (1, 3), (2, 2), (2, 3), (3, 1), (3, 2), (3, 3)
#             })
#         },
#         {
#             "grid": (
#                 (2, 2, 2, 2),
#                 (2, 0, 0, 2),
#                 (2, 0, 0, 2),
#                 (2, 2, 2, 2)
#             ),
#             "background_value": 0,
#             "expected_edges": frozenset({
#                 (0, 0), (0, 1), (0, 2), (0, 3), (1, 0), (1, 3), (2, 0), (2, 3), (3, 0), (3, 1), (3, 2), (3, 3)
#             })
#         },
#         {
#             "grid": (
#                 (3, 3, 3, 3),
#                 (3, 1, 1, 3),
#                 (3, 1, 1, 3),
#                 (3, 3, 3, 3)
#             ),
#             "background_value": 3,
#             "expected_edges": frozenset({
#                 (1, 1), (1, 2), (2, 1), (2, 2)
#             })
#         },
#         {
#             "grid": (
#                 (0, 0, 0, 0),
#                 (0, 4, 4, 0),
#                 (0, 4, 4, 0),
#                 (0, 0, 0, 0)
#             ),
#             "background_value": 0,
#             "expected_edges": frozenset({
#                 (1, 1), (1, 2), (2, 1), (2, 2)
#             })
#         },
#         {
#             "grid": (
#                 (5, 5, 5, 5),
#                 (5, 5, 5, 5),
#                 (5, 5, 5, 5),
#                 (5, 5, 5, 5)
#             ),
#             "background_value": 5,
#             "expected_edges": frozenset()
#         }
#     ]

#     for i, case in enumerate(test_cases):
#         grid = case["grid"]
#         background_value = case["background_value"]
#         expected_edges = case["expected_edges"]

#         result = detect_edges(grid, background_value)
#         print(f"Test case {i + 1}: Detected edges: {result}")
#         print(set(result) == set(expected_edges))

# # Run the tests
# test_detect_edges()
