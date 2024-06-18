# import numpy as np
import json

from utils import plot_grids

class VQCSolver:

    def __init__(self, data, labels):
        self.data = data
        self.labels = labels

    def solve(self, grid, target):
        pass


if __name__ == "__main__":
    base_path = '../arc-prize-2024'
    testing_data = json.load(open(f'{base_path}/arc-agi_test_challenges.json', 'r'))
    
    for key, task in testing_data.items():
        test_data = task['train']
        for pair in test_data:
            grid = pair['input']
            target = pair['output']
            plot_grids([grid], [''])
            print(grid)
            print(target)
            break
        break
