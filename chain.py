from itertools import product
from typing import get_type_hints
from collections import defaultdict
import inspect

import prims
from arc_types import *

# for efficient predictions of less trace
# TODO: implement solver without trace (or generalise to allow trace)

prims_with_return_types = {
    name: {
        'func': func,
        'return_type': get_type_hints(func).get('return', None),
        'input_types': {k: v for k, v in get_type_hints(func).items() if k != 'return'}
    }
    for name, func in inspect.getmembers(prims, inspect.isfunction)
}

def find_common_items(input_pools: dict):
    keys = input_pools.keys()
    values_product = product(*input_pools.values())
    return [{key: value for key, value in zip(keys, values)} for values in values_product]

def solver_with_trace(grid, target, use_beam=True, max_depth=2, beam_width=3):
    chaining_pool = defaultdict(set)  # Use set for chaining pool
    trace_pool = defaultdict(list)  # Dictionary to store traces of primitives and their input parameters

    # Initialize chaining pool with depth=1 results
    for primitive, details in prims_with_return_types.items():
        if not details['input_types'] or (len(details['input_types']) == 1 and Grid in details['input_types'].values()):
            args = {param_name: grid for param_name in details['input_types']}
            result = details['func'](**args)
            if result == target:
                return True, result, [(primitive, args)]  # Return the primitive and its args as part of the trace
            chaining_pool[details['return_type']].add(result)
            trace_pool[details['return_type']].append([(primitive, {param_name: {"value": grid, "from": None} for param_name in details['input_types']})])  # Store initial trace with args

    # Chain combinations of previous results for deeper levels
    for depth in range(2, max_depth + 1):
        new_traces = defaultdict(list)  # New dictionary for updated traces
        for primitive, details in prims_with_return_types.items():
            input_pools = {param_name: chaining_pool[param_type] 
                           for param_name, param_type in details['input_types'].items()}
            input_traces = {param_name: trace_pool[param_type] 
                            for param_name, param_type in details['input_types'].items()}
            candidate_chains = find_common_items(input_pools)
            
            # Apply beam search: keep only the top `beam_width` candidate chains
            if use_beam:
                candidate_chains = sorted(candidate_chains, key=lambda x: h(x, target))[:beam_width]
            
            for candidate_chain in candidate_chains:
                result = details['func'](**candidate_chain)
                if result == target:
                    # Gather the trace leading to the successful result
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

def h(candidate_chain, target_grid):
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
