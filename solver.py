from itertools import product
from typing import Any, get_origin, get_args, List, Tuple, Set, FrozenSet, Union, get_type_hints, Dict
from collections import defaultdict
import inspect
import types
import ast

import prims
import arc_types
from arc_types import *

def typeguard(inp: Any, return_type: type):
    origin = get_origin(return_type)
    args = get_args(return_type)
    
    if origin is list:
        if isinstance(inp, list):
            inner_type = args[0]
            return tuple(typeguard(i, inner_type) for i in inp)
    elif origin is tuple:
        if isinstance(inp, (list, tuple)):
            return tuple(typeguard(i, args[0]) for i in inp)
    elif origin is set:
        if isinstance(inp, set):
            inner_type = args[0]
            return frozenset(typeguard(i, inner_type) for i in inp)
    elif origin is frozenset:
        if isinstance(inp, set):
            inner_type = args[0]
            return frozenset(typeguard(i, inner_type) for i in inp)
    elif origin is Union:
        for arg in args:
            try:
                return typeguard(inp, arg)
            except:
                continue
    elif return_type == Grid:
        if isinstance(inp, (list, tuple)):
            return tuple(tuple(i) if isinstance(i, list) else i for i in inp)
    elif return_type == IntegerList:
        if isinstance(inp, (list, tuple)):
            return tuple(inp)
    elif return_type == IntegerSet:
        if isinstance(inp, set):
            return frozenset(inp)
    elif return_type == Cell:
        if isinstance(inp, (list, tuple)):
            return tuple(inp)
    elif return_type == Object:
        if isinstance(inp, set):
            return frozenset(inp)
    elif return_type == Objects:
        if isinstance(inp, set):
            return frozenset(inp)
    elif return_type == Indices:
        if isinstance(inp, set):
            return frozenset(inp)
    elif return_type == IndicesSet:
        if isinstance(inp, set):
            return frozenset(inp)
    elif return_type == Patch:
        if isinstance(inp, (set, list, tuple)):
            return frozenset(inp) if isinstance(inp, set) else tuple(inp)
    elif return_type == Element:
        if isinstance(inp, (set, list, tuple)):
            return frozenset(inp) if isinstance(inp, set) else tuple(inp)
    elif return_type == Piece:
        if isinstance(inp, (set, list, tuple)):
            return frozenset(inp) if isinstance(inp, set) else tuple(inp)
    elif return_type == ListList:
        if isinstance(inp, (list, tuple)):
            return tuple(tuple(i) if isinstance(i, list) else i for i in inp)
    elif return_type == ContainerContainer:
        if isinstance(inp, Container):
            return inp
    return inp

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
        chaining_pool = defaultdict(set)  # Use set for chaining pool
        trace_pool = defaultdict(list)  # Dictionary to store traces of primitives and their input parameters

        # Initialize chaining pool with depth=1 results
        for primitive, details in self.primitives.items():
            print('----primitive----', primitive, details['input_types'], Tuple[Tuple[int]] in details['input_types'].values(), Tuple[Tuple[int]] in [Grid])
            if not details['input_types'] or \
            (len(details['input_types']) == 1 and (Grid in details['input_types'].values() or Tuple[Tuple[int]] in details['input_types'].values())):
                args = {param_name: grid for param_name in details['input_types']}
                result = details['func'](**args)
                if result == target:
                    return True, result, [(primitive, args)]  # Return the primitive and its args as part of the trace
                chaining_pool[details['return_type']].add(result)
                trace_pool[details['return_type']].append([(primitive, {param_name: {"value": grid, "from": None} for param_name in details['input_types']})])  # Store initial trace with args

        # Chain combinations of previous results for deeper levels
        for depth in range(2, self.max_depth + 1):
            new_traces = defaultdict(list)  # New dictionary for updated traces
            for primitive, details in self.primitives.items():
                input_pools = {param_name: chaining_pool[param_type] 
                            for param_name, param_type in details['input_types'].items()}
                input_traces = {param_name: trace_pool[param_type] 
                                for param_name, param_type in details['input_types'].items()}
                candidate_chains = self.generate_chains(input_pools)
                
                # Apply beam search: keep only the top `beam_width` candidate chains
                if self.use_beam:
                    candidate_chains = sorted(candidate_chains, key=lambda x: self.h(x, target))[:self.beam_width]
                
                for candidate_chain in candidate_chains:
                    result = details['func'](**candidate_chain)
                    if result == target:
                        # Gather the trace leading to the successful result
                        print('result', result, 'target', target)
                        trace = []
                        for param_name, param_type in details['input_types'].items():
                            trace.extend(input_traces[param_name][list(input_pools[param_name]).index(candidate_chain[param_name])])
                        trace.append((primitive, {param_name: {"value": candidate_chain[param_name], "from": input_traces[param_name][list(input_pools[param_name]).index(candidate_chain[param_name])][-1][0]} for param_name in details['input_types']}))  # Append the current primitive and its args
                        return True, result, trace  # Return the successful trace
                    # Update the trace with the current primitive and its args
                    current_trace = []
                    for param_name in details['input_types'].keys():
                        current_trace.extend(input_traces[param_name][list(input_pools[param_name]).index(candidate_chain[param_name])])
                    current_trace.append((primitive, {param_name: {"value": candidate_chain[param_name], "from": input_traces[param_name][list(input_pools[param_name]).index(candidate_chain[param_name])][-1][0]} for param_name in details['input_types']}))
                    new_traces[details['return_type']].append((result, current_trace))  # Append the result and the current trace

            # Update chaining pool and trace pool with new traces
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
