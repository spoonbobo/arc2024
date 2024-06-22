from itertools import product
from typing import Any, List, Tuple, Set, get_type_hints, Dict, Callable
from typing import get_origin, get_args
from collections import defaultdict
from functools import lru_cache
import inspect
import types
import ast
import os
import json
import uuid

import prims
import arc_types
from arc_types import *

class InstructedDSL:
    """
    This class is used to create a DSL for solving puzzles using primitives under LLM instructions
    """
    
    def __init__(self,
                 max_depth: int = 2,
                 instruction: str = '',
                 use_instruction: bool = False,
                 use_beam: bool = False,
                 beam_width: int = 2,
                 bootstrap_data: bool = False):

        self.max_depth = max_depth
        self.use_beam = use_beam
        self.beam_width = beam_width
        self.use_instruction = use_instruction
        self.bootstrap_data = bootstrap_data

        self.primitives: Dict[str, Dict[str, Callable | type | Dict[str, type]]] = {}
        self.memo = {}
        if use_instruction:
            self.load_prims(instruction)
        else:
            self.load_prims(prims)
    
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
            func_env = {name: obj for name, obj in inspect.getmembers(arc_types) if not inspect.isbuiltin(obj)}

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
                        except Exception as e:
                            print(f"Failed to parse function {func_name}: {e} ({prim})")

    def generate_chains(self, input_pools: Dict[str, Set[Any]]):
        keys = input_pools.keys()
        for values in product(*input_pools.values()):
            yield {key: value for key, value in zip(keys, values)}

    def make_hashable(self, obj):
        if isinstance(obj, list):
            return tuple(self.make_hashable(e) for e in obj)
        elif isinstance(obj, dict):
            return frozenset((k, self.make_hashable(v)) for k, v in obj.items())
        elif isinstance(obj, set):
            return frozenset(self.make_hashable(e) for e in obj)
        return obj

    @lru_cache(maxsize=500)
    def memoized_func(self, func, **kwargs):
        # Convert all values in kwargs to a hashable type
        hashable_kwargs = {k: self.make_hashable(v) for k, v in kwargs.items()}
        key = (func.__name__, frozenset(hashable_kwargs.items()))
        if key not in self.memo:
            self.memo[key] = func(**kwargs)
        return self.memo[key]

    def solve(self, grid, target, key, grid_id, bootstrap=True):
        def make_serializable(obj):
            if isinstance(obj, (set, frozenset)):
                return [make_serializable(e) for e in obj]
            if isinstance(obj, dict):
                return {k: make_serializable(v) for k, v in obj.items()}
            if isinstance(obj, list):
                return [make_serializable(e) for e in obj]
            if isinstance(obj, tuple):
                return tuple(make_serializable(e) for e in obj)
            if isinstance(obj, (int, float, str, bool)) or obj is None:
                return obj
            if hasattr(obj, '__dict__'):
                return make_serializable(vars(obj))
            return str(obj)  # Fallback for other types
        chaining_pool = defaultdict(set)
        trace_pool = defaultdict(list)
        solutions  = []
        if bootstrap:
            data = []
        grid = self.make_hashable(grid)
        target = self.make_hashable(target)

        # Initialize static params
        for symbol in range(10):
            chaining_pool[Integer].add(symbol)
            trace_pool[Integer].append([(f'symbol_{symbol}', {'void': {'from': None}})])
        
        # Initialize primitives
        for primitive, details in self.primitives.items():
            if not details['input_types'] or \
                (len(details['input_types']) == 1 and Grid in details['input_types'].values()) or \
                    (len(details['input_types']) == 1 and Tuple[Tuple[int]] in details['input_types'].values()):
                args = {param_name: grid for param_name in details['input_types']}
                # print(key, grid, grid_id, target)
                # exit()
                try:
                    result = self.memoized_func(details['func'], **args)
                except:
                    continue
                if result == target:
                    solutions.append((True, result, [(primitive, args)]))
                if bootstrap:
                    if isinstance(result, tuple) and all(isinstance(item, tuple) and all(isinstance(i, int) for i in item) for item in result):
                        if len(result) and len(result[0]) and all(len(row) == len(result[0]) for row in result):
                            data.append({'result': result, 'trace': [(primitive, args)]})
                
                chaining_pool[details['return_type']].add(result)
                trace_pool[details['return_type']].append([(primitive, {param_name: {"from": None} for param_name in details['input_types']})])

        for depth in range(2, self.max_depth + 1):
            print('at depth', depth)
            new_traces = defaultdict(list)
            for primitive, details in self.primitives.items():
                input_pools = {param_name: chaining_pool[param_type] for param_name, param_type in details['input_types'].items()}
                input_traces = {param_name: trace_pool[param_type] for param_name, param_type in details['input_types'].items()}
                candidate_chains = self.generate_chains(input_pools)
                
                if self.use_beam:
                    candidate_chains = sorted(candidate_chains, key=lambda x: self.h(x, target))[:self.beam_width]

                for candidate_chain in candidate_chains:
                    try:
                        result = self.memoized_func(details['func'], **candidate_chain)
                    except:
                        # print(primitive, 'returns', None)
                        continue
                    if result is None:
                        continue
                    if result == target:
                        trace = self.build_trace(candidate_chain, details, input_pools, input_traces)
                        solutions.append((True, result, trace))
                    if bootstrap:
                        # print(f"Result type: {type(result)}, Origin: {get_origin(result)}, Args: {get_args(result)}")
                        if isinstance(result, tuple) and all(isinstance(item, tuple) and all(isinstance(i, int) for i in item) for item in result):
                            if len(result) and len(result[0]) and all(len(row) == len(result[0]) for row in result):
                                trace = self.build_trace(candidate_chain, details, input_pools, input_traces)
                                data.append({'result': result, 'trace': trace})
                    current_trace = self.build_trace(candidate_chain, details, input_pools, input_traces)
                    new_traces[details['return_type']].append((result, current_trace))

            for ret_type, traces in new_traces.items():
                for result, trace in traces:
                    chaining_pool[ret_type].add(result)
                    trace_pool[ret_type].append(trace)

        if bootstrap:
            os.makedirs('dataset', exist_ok=True) 
            json.dump(data, open(f'dataset/{key}_{grid_id}.json', 'w'))

        return solutions

    def generate_chains(self, input_pools: Dict[str, Set[Any]]):
        keys = input_pools.keys()
        for values in product(*input_pools.values()):
            yield {key: value for key, value in zip(keys, values)}

    def build_trace(self, candidate_chain, details, input_pools, input_traces):
        trace = []
        param_indices = {param_name: {v: i for i, v in enumerate(input_pools[param_name])} for param_name in details['input_types']}
        
        for param_name, param_type in details['input_types'].items():
            param_value = candidate_chain[param_name]
            param_index = param_indices[param_name][param_value]
            
            # Debugging statements
            if param_name not in input_traces:
                print(f"Error: {param_name} not in input_traces")
                continue
            if param_index >= len(input_traces[param_name]):
                print(f"Error: param_index {param_index} out of range for {param_name}")
                continue
            if not input_traces[param_name][param_index]:
                print(f"Error: input_traces[{param_name}][{param_index}] is empty")
                continue
            
            trace.extend(input_traces[param_name][param_index])
        
        # Ensure all param_indices are valid before appending to trace
        valid_trace = True
        trace_dict = {}
        for param_name in details['input_types']:
            param_index = param_indices[param_name][candidate_chain[param_name]]
            if param_index >= len(input_traces[param_name]) or not input_traces[param_name][param_index]:
                valid_trace = False
                break
            trace_dict[param_name] = {"from": input_traces[param_name][param_index][-1][0]}
        
        if valid_trace:
            trace.append((details['func'].__name__, trace_dict))
        
        return trace

    def h(self, candidate_chain, target_grid):
        # Create a hashable key from the candidate chain
        # chain_key = frozenset(candidate_chain.items())
        sim = 0.5
        if 'grid' in candidate_chain:
            grid = candidate_chain['grid']
            sim = grid_similarity(grid, target_grid)
            # self.last_heuristics[chain_key] = sim  # Store the heuristic value
        return sim

def grid_similarity(grid1, grid2):
    if not grid1 or not grid2:
        return float('inf')  # Handle empty grids

    rows1, cols1 = len(grid1), len(grid1[0]) if grid1 else 0
    rows2, cols2 = len(grid2), len(grid2[0]) if grid2 else 0

    overlap_rows = min(rows1, rows2)
    overlap_cols = min(cols1, cols2)
    
    overlap_area = 0
    for i in range(overlap_rows):
        for j in range(overlap_cols):
            if i < rows1 and j < len(grid1[i]) and i < rows2 and j < len(grid2[i]) and grid1[i][j] == grid2[i][j]:
                overlap_area += 1

    total_area = rows1 * cols1 + rows2 * cols2 - overlap_area

    return 1 - (overlap_area / total_area)
