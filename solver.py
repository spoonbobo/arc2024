from itertools import product
from typing import Any, List, Tuple, Set, get_type_hints, Dict
from collections import defaultdict
import inspect
import types
import ast

import prims
import arc_types
from arc_types import *
# from typeguard import typeguard

class InstructedDSL:
    
    """
    This class is used to create a DSL for solving puzzles using primitives under LLM instructions
    """
    
    def __init__(self,
                 max_depth: int = 2,
                 instruction: str = '',
                 use_instruction: bool = False,
                 use_beam: bool = False,
                 beam_width: int = 2):

        self.max_depth = max_depth
        self.use_beam = use_beam
        self.beam_width = beam_width
        self.use_instruction = use_instruction

        self.primitives: Dict[str, Dict[str, Callable | type | Dict[str, type]]] = {}
        if use_instruction:
            self.load_prims(instruction)
        else:
            self.load_prims(prims)
        # print(self.primitives.keys())
    
    def load_prims(self, primitives: types.ModuleType | List[str]) -> None:
        if isinstance(primitives, types.ModuleType):
            self.primitives = {
                name: {
                    'func': func,
                    'return_type': get_type_hints(func).get('return', None),
                    'input_types': {k: v for k, v in get_type_hints(func).items() if k != 'return'}
                }
                for name, func in inspect.getmembers(prims, inspect.isfunction)
            }
        
        else:
            func_env = {}

            for name, obj in inspect.getmembers(arc_types):
                if not inspect.isbuiltin(obj):
                    func_env[name] = obj

            for prim in primitives:
                module = ast.parse(prim)
                for node in module.body:
                    if isinstance(node, ast.FunctionDef):
                        func_name = node.name
                        try:
                            func_code = compile(ast.Module(body=[node], type_ignores=[]), filename="<ast>", mode="exec")
                            exec(func_code, func_env)
                            func = func_env[func_name]
                            type_hints = get_type_hints(func)
                            return_type = type_hints.pop('return', None)
                            self.primitives[func_name] = {
                                'func': func,
                                'return_type': return_type,
                                'input_types': type_hints
                            }
                            print(self.primitives[func_name])
                        except Exception as e:
                            print(f"Failed to parse function {func_name}: {e} ({prim})")

    def generate_chains(self, input_pools: Dict[str, Set[Any]]):
        keys = input_pools.keys()
        values_product = product(*input_pools.values())
        return [{key: value for key, value in zip(keys, values)} for values in values_product]
    
    def solve(self, grid, target):
        chaining_pool = defaultdict(set)
        trace_pool = defaultdict(list)

        # Initialize chaining pool with depth=1 results
        for primitive, details in self.primitives.items():
            if not details['input_types'] or \
                (len(details['input_types']) == 1 and Grid in details['input_types'].values()) or \
                    (len(details['input_types']) == 1 and Tuple[Tuple[int]] in details['input_types'].values()):
                args = {param_name: grid for param_name in details['input_types']}
                result = details['func'](**args)
                if result == target:
                    return True, result, [(primitive, args)]
                chaining_pool[details['return_type']].add(result)
                trace_pool[details['return_type']].append([(primitive, {param_name: {"from": None} for param_name in details['input_types']})])

        for depth in range(2, self.max_depth + 1):
            new_traces = defaultdict(list)
            for primitive, details in self.primitives.items():
                input_pools = {param_name: chaining_pool[param_type] for param_name, param_type in details['input_types'].items()}
                input_traces = {param_name: trace_pool[param_type] for param_name, param_type in details['input_types'].items()}
                candidate_chains = self.generate_chains(input_pools)
                
                if self.use_beam:
                    candidate_chains = sorted(candidate_chains, key=lambda x: self.h(x, target))[:self.beam_width]
                
                for candidate_chain in candidate_chains:
                    result = details['func'](**candidate_chain)
                    if result == target:
                        trace = []
                        for param_name, param_type in details['input_types'].items():
                            param_value = candidate_chain[param_name]
                            param_index = {v: i for i, v in enumerate(input_pools[param_name])}[param_value]
                            trace.extend(input_traces[param_name][param_index])
                        trace.append((primitive, {param_name: {"from": input_traces[param_name][param_index][-1][0]} for param_name in details['input_types']}))
                        return True, result, trace
                    current_trace = []
                    for param_name in details['input_types'].keys():
                        param_value = candidate_chain[param_name]
                        param_index = {v: i for i, v in enumerate(input_pools[param_name])}[param_value]
                        current_trace.extend(input_traces[param_name][param_index])
                    current_trace.append((primitive, {param_name: {"from": input_traces[param_name][param_index][-1][0]} for param_name in details['input_types']}))
                    new_traces[details['return_type']].append((result, current_trace))

            for ret_type, traces in new_traces.items():
                for result, trace in traces:
                    chaining_pool[ret_type].add(result)
                    trace_pool[ret_type].append(trace)

        return False, None, None


    def h(self, candidate_chain, target_grid):
        sim = 0.5
        if 'grid' in candidate_chain:
            grid = candidate_chain['grid']
            sim = grid_similarity(grid, target_grid)
        return sim

def grid_similarity(grid1, grid2):
    # Calculate the similarity between two grids of varying sizes
    # Example: Intersection over Union (IoU) or a custom similarity metric
    if len(grid1) == 0 or len(grid2) == 0:
        return float('inf')  # Handle empty grids

    # Calculate the dimensions of the grids
    rows1, cols1 = len(grid1), len(grid1[0])
    rows2, cols2 = len(grid2), len(grid2[0])

    # Calculate the overlapping area
    overlap_rows = min(rows1, rows2)
    overlap_cols = min(cols1, cols2)
    overlap_area = sum(1 for i in range(overlap_rows) for j in range(overlap_cols) if grid1[i][j] == grid2[i][j])

    # Calculate the total area of both grids
    total_area = rows1 * cols1 + rows2 * cols2 - overlap_area

    # Return the similarity measure (higher is better, so we return the inverse for the heuristic)
    return 1 - (overlap_area / total_area)


if __name__ == '__main__':
    idsl = InstructedDSL()
