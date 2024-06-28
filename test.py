from prims import *
from utils import *

grid = [
                    [
                        0,
                        7,
                        7
                    ],
                    [
                        7,
                        7,
                        7
                    ],
                    [
                        0,
                        7,
                        7
                    ]
                ]
s1 = detect_edges(grid, 2)
print(s1)
s2 = paint_edges(grid, s1, 5)
print(s2)

plot_single_grid(s2)