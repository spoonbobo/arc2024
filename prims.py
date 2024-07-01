from typing import Tuple, FrozenSet
import math

def make_coordinate(x: int, y: int) -> Tuple[int, int]:
    """Make a grid coordinate with x and y"""
    return (x, y)

def flood_fill(grid: Tuple[Tuple[int]], start: Tuple[int, int]) -> FrozenSet[Tuple[int, int]]:
    """Performs flood fill from a starting cell and returns the connected region as a frozen set."""
    sr, sc = start
    rows, cols = len(grid), len(grid[0])
    stack = [(sr, sc)]
    connected_cells = set()
    visited = set()

    while stack:
        r, c = stack.pop()
        if (r, c) not in visited:
            visited.add((r, c))
            if grid[r][c] != 0:  # Consider all non-zero cells as part of the region
                connected_cells.add((r, c))
                for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                    nr, nc = r + dr, c + dc
                    if 0 <= nr < rows and 0 <= nc < cols and (nr, nc) not in visited:
                        stack.append((nr, nc))
    
    return frozenset(connected_cells)

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

def paint_region_with_mean(grid: Tuple[Tuple[int]], region: FrozenSet[Tuple[int, int]]) -> Tuple[Tuple[int]]:
    """Paints the region with the mean value of the region."""
    values = [grid[r][c] for r, c in region]
    mean_value = sum(values) // len(values)
    new_grid = [list(row) for row in grid]
    for r, c in region:
        new_grid[r][c] = mean_value
    return tuple(tuple(row) for row in new_grid)

def paint_region_with_mode(grid: Tuple[Tuple[int]], region: FrozenSet[Tuple[int, int]]) -> Tuple[Tuple[int]]:
    """Paints the region with the mode value of the region."""
    values = [grid[r][c] for r, c in region]
    frequency = {}
    for value in values:
        if value in frequency:
            frequency[value] += 1
        else:
            frequency[value] = 1
    mode_value = max(frequency, key=frequency.get)
    new_grid = [list(row) for row in grid]
    for r, c in region:
        new_grid[r][c] = mode_value
    return tuple(tuple(row) for row in new_grid)

def paint_region_with_max(grid: Tuple[Tuple[int]], region: FrozenSet[Tuple[int, int]]) -> Tuple[Tuple[int]]:
    """Paints the region with the max value of the region."""
    values = [grid[r][c] for r, c in region]
    max_value = max(values)
    new_grid = [list(row) for row in grid]
    for r, c in region:
        new_grid[r][c] = max_value
    return tuple(tuple(row) for row in new_grid)

def paint_region_with_min(grid: Tuple[Tuple[int]], region: FrozenSet[Tuple[int, int]]) -> Tuple[Tuple[int]]:
    """Paints the region with the min value of the region."""
    values = [grid[r][c] for r, c in region]
    min_value = min(values)
    new_grid = [list(row) for row in grid]
    for r, c in region:
        new_grid[r][c] = min_value
    return tuple(tuple(row) for row in new_grid)

def paint_region_with_median(grid: Tuple[Tuple[int]], region: FrozenSet[Tuple[int, int]]) -> Tuple[Tuple[int]]:
    """Paints the region with the median value of the region."""
    values = sorted([grid[r][c] for r, c in region])
    n = len(values)
    mid = n // 2
    if n % 2 == 0:
        median_value = (values[mid - 1] + values[mid]) // 2
    else:
        median_value = values[mid]
    new_grid = [list(row) for row in grid]
    for r, c in region:
        new_grid[r][c] = median_value
    return tuple(tuple(row) for row in new_grid)

def rotate_region_90(region: FrozenSet[Tuple[int, int]], center: Tuple[int, int]) -> FrozenSet[Tuple[int, int]]:
    """Rotates the region by 90 degrees clockwise around a center point."""
    cx, cy = center
    return frozenset((cy - y + cx, x - cx + cy) for x, y in region)

def rotate_region_180(region: FrozenSet[Tuple[int, int]], center: Tuple[int, int]) -> FrozenSet[Tuple[int, int]]:
    """Rotates the region by 180 degrees around a center point."""
    cx, cy = center
    return frozenset((2 * cx - x, 2 * cy - y) for x, y in region)

def rotate_region_270(region: FrozenSet[Tuple[int, int]], center: Tuple[int, int]) -> FrozenSet[Tuple[int, int]]:
    """Rotates the region by 270 degrees clockwise around a center point."""
    cx, cy = center
    return frozenset((cy + y - cx, cx - x + cy) for x, y in region)

def reflect_region_horizontal(region: FrozenSet[Tuple[int, int]]) -> FrozenSet[Tuple[int, int]]:
    """Reflects the region horizontally (over the y-axis)."""
    return frozenset((-x, y) for x, y in region)

def reflect_region_vertical(region: FrozenSet[Tuple[int, int]]) -> FrozenSet[Tuple[int, int]]:
    """Reflects the region vertically (over the x-axis)."""
    return frozenset((x, -y) for x, y in region)

def reflect_region_diagonal(region: FrozenSet[Tuple[int, int]]) -> FrozenSet[Tuple[int, int]]:
    """Reflects the region along the main diagonal (y = x)."""
    return frozenset((y, x) for x, y in region)

def apply_region_to_grid(grid: Tuple[Tuple[int]], region: FrozenSet[Tuple[int, int]], symbol: int) -> Tuple[Tuple[int]]:
    """Applies the region to the grid with the given symbol."""
    new_grid = [list(row) for row in grid]
    for r, c in region:
        if 0 <= r < len(grid) and 0 <= c < len(grid[0]):  # Ensure within bounds
            new_grid[r][c] = symbol
    return tuple(tuple(row) for row in new_grid)

# def move(region: FrozenSet[Tuple[int, int]], dx: int, dy: int) -> FrozenSet[Tuple[int, int]]:
#     """Moves a region by a given factor (dx, dy)."""
#     return frozenset((r + dx, c + dy) for r, c in region)

def move_by_one(region: FrozenSet[Tuple[int, int]]) -> FrozenSet[Tuple[int, int]]:
    """Moves a region by one unit in both x and y directions."""
    def move(region: FrozenSet[Tuple[int, int]], dx: int, dy: int) -> FrozenSet[Tuple[int, int]]:
        return frozenset((r + dx, c + dy) for r, c in region)
    
    return move(region, 1, 1)

def move_by_two(region: FrozenSet[Tuple[int, int]]) -> FrozenSet[Tuple[int, int]]:
    """Moves a region by two units in both x and y directions."""
    def move(region: FrozenSet[Tuple[int, int]], dx: int, dy: int) -> FrozenSet[Tuple[int, int]]:
        return frozenset((r + dx, c + dy) for r, c in region)
    
    return move(region, 2, 2)

def grid_to_region(grid: Tuple[Tuple[int]]) -> FrozenSet[Tuple[int, int]]:
    """Converts a grid to a region by identifying all cells."""
    region = set()
    for r, row in enumerate(grid):
        for c, _ in enumerate(row):
            region.add((r, c))
    return frozenset(region)

def union_regions(region1: FrozenSet[Tuple[int, int]], region2: FrozenSet[Tuple[int, int]]) -> FrozenSet[Tuple[int, int]]:
    """Returns the union of two regions."""
    return region1 | region2

def intersect_regions(region1: FrozenSet[Tuple[int, int]], region2: FrozenSet[Tuple[int, int]]) -> FrozenSet[Tuple[int, int]]:
    """Returns the intersection of two regions."""
    return region1 & region2

def difference_regions(region1: FrozenSet[Tuple[int, int]], region2: FrozenSet[Tuple[int, int]]) -> FrozenSet[Tuple[int, int]]:
    """Returns the difference of two regions (region1 - region2)."""
    return region1 - region2

def symmetric_difference_regions(region1: FrozenSet[Tuple[int, int]], region2: FrozenSet[Tuple[int, int]]) -> FrozenSet[Tuple[int, int]]:
    """Returns the symmetric difference of two regions."""
    return region1 ^ region2

def bounding_box(region: FrozenSet[Tuple[int, int]]) -> Tuple[Tuple[int, int], Tuple[int, int]]:
    """Calculates the smallest rectangle that can contain the entire region."""
    min_r = min(r for r, _ in region)
    max_r = max(r for r, _ in region)
    min_c = min(c for _, c in region)
    max_c = max(c for _, c in region)
    return (min_r, min_c), (max_r, max_c)

# def region_perimeter(region: FrozenSet[Tuple[int, int]]) -> int:
#     """Calculates the perimeter of the region."""
#     perimeter = 0
#     for r, c in region:
#         for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
#             if (r + dr, c + dc) not in region:
#                 perimeter += 1
#     return perimeter

# def region_area(region: FrozenSet[Tuple[int, int]]) -> int:
#     """Calculates the area (number of cells) of the region."""
#     return len(region)

def scale_region(region: FrozenSet[Tuple[int, int]], scale: float, center: Tuple[int, int] = (0, 0)) -> FrozenSet[Tuple[int, int]]:
    """Scales the region by a given factor relative to a center point."""
    cx, cy = center
    return frozenset((round(cx + (r - cx) * scale), round(cy + (c - cy) * scale)) for r, c in region)

def rotate_region(region: FrozenSet[Tuple[int, int]], angle: float, center: Tuple[int, int]) -> FrozenSet[Tuple[int, int]]:
    """Rotates the region by an arbitrary angle (in degrees) around a center point."""
    angle_rad = math.radians(angle)
    cx, cy = center
    return frozenset((
        round(cx + (r - cx) * math.cos(angle_rad) - (c - cy) * math.sin(angle_rad)),
        round(cy + (r - cx) * math.sin(angle_rad) + (c - cy) * math.cos(angle_rad))
    ) for r, c in region)


def convex_hull(region: FrozenSet[Tuple[int, int]]) -> FrozenSet[Tuple[int, int]]:
    """Calculates the convex hull of the region using the Graham scan algorithm."""
    points = sorted(region)
    if len(points) <= 1:
        return region

    def cross(o, a, b):
        return (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])

    lower = []
    for p in points:
        while len(lower) >= 2 and cross(lower[-2], lower[-1], p) <= 0:
            lower.pop()
        lower.append(p)

    upper = []
    for p in reversed(points):
        while len(upper) >= 2 and cross(upper[-2], upper[-1], p) <= 0:
            upper.pop()
        upper.append(p)

    return frozenset(lower[:-1] + upper[:-1])

def contour_extraction(region: FrozenSet[Tuple[int, int]]) -> FrozenSet[Tuple[int, int]]:
    """Extracts the contour (boundary) of the region."""
    contour = set()
    for r, c in region:
        for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            if (r + dr, c + dc) not in region:
                contour.add((r, c))
                break
    return frozenset(contour)

def dilate_region(region: FrozenSet[Tuple[int, int]]) -> FrozenSet[Tuple[int, int]]:
    """Expands the region by one cell in all directions."""
    dilated = set(region)
    for r, c in region:
        for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            dilated.add((r + dr, c + dc))
    return frozenset(dilated)

def erode_region(region: FrozenSet[Tuple[int, int]]) -> FrozenSet[Tuple[int, int]]:
    """Shrinks the region by one cell in all directions."""
    eroded = set()
    for r, c in region:
        if all((r + dr, c + dc) in region for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]):
            eroded.add((r, c))
    return frozenset(eroded)

def translate_region(region: FrozenSet[Tuple[int, int]], dx: int, dy: int) -> FrozenSet[Tuple[int, int]]:
    """Translates the region by a specified amount in both x and y directions."""
    return frozenset((r + dx, c + dy) for r, c in region)

def fill_diagonal_with_zeros(grid: Tuple[Tuple[int]], start: Tuple[int, int], steps: int) -> Tuple[Tuple[int]]:
    """Fills the diagonal starting from a point with zeros by steps."""
    new_grid = [list(row) for row in grid]
    r, c = start
    for _ in range(steps):
        if 0 <= r < len(grid) and 0 <= c < len(grid[0]):
            new_grid[r][c] = 0
        r += 1
        c += 1
    return tuple(tuple(row) for row in new_grid)

def fill_row_with_zeros(grid: Tuple[Tuple[int]], start: Tuple[int, int], steps: int) -> Tuple[Tuple[int]]:
    """Fills the row starting from a point with zeros by steps."""
    new_grid = [list(row) for row in grid]
    r, c = start
    for _ in range(steps):
        if 0 <= r < len(grid) and 0 <= c < len(grid[0]):
            new_grid[r][c] = 0
        c += 1
    return tuple(tuple(row) for row in new_grid)

def fill_column_with_zeros(grid: Tuple[Tuple[int]], start: Tuple[int, int], steps: int) -> Tuple[Tuple[int]]:
    """Fills the column starting from a point with zeros by steps."""
    new_grid = [list(row) for row in grid]
    r, c = start
    for _ in range(steps):
        if 0 <= r < len(grid) and 0 <= c < len(grid[0]):
            new_grid[r][c] = 0
        r += 1
    return tuple(tuple(row) for row in new_grid)
