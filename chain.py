from itertools import product
import prims
from typing import get_type_hints
from collections import defaultdict
from copy import deepcopy

import inspect
from arc_types import *

# for efficient predictions of less trace
# TODO: use set to store chain pools
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

def solver_with_trace(grid, target, max_depth=2):
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
        new_chains = defaultdict(set)  # Use set for new chains
        new_traces = defaultdict(list)  # New dictionary for updated traces
        for primitive, details in prims_with_return_types.items():
            input_pools = {param_name: deepcopy(chaining_pool[param_type]) 
                           for param_name, param_type in details['input_types'].items()}
            input_traces = {param_name: deepcopy(trace_pool[param_type]) 
                            for param_name, param_type in details['input_types'].items()}
            for candidate_chain in find_common_items(input_pools):
                result = details['func'](**candidate_chain)
                if result == target:
                    # Gather the trace leading to the successful result
                    trace = []
                    for param_name, param_type in details['input_types'].items():
                        trace.extend(input_traces[param_name][list(input_pools[param_name]).index(candidate_chain[param_name])])
                    trace.append((primitive, {param_name: {"value": candidate_chain[param_name], "from": input_traces[param_name][list(input_pools[param_name]).index(candidate_chain[param_name])][-1][0]} for param_name in details['input_types']}))  # Append the current primitive and its args
                    return True, result, trace  # Return the successful trace
                new_chains[details['return_type']].add(result)
                # Update the trace with the current primitive and its args
                current_trace = []
                for param_name in details['input_types'].keys():
                    current_trace.extend(input_traces[param_name][list(input_pools[param_name]).index(candidate_chain[param_name])])
                current_trace.append((primitive, {param_name: {"value": candidate_chain[param_name], "from": input_traces[param_name][list(input_pools[param_name]).index(candidate_chain[param_name])][-1][0]} for param_name in details['input_types']}))
                new_traces[details['return_type']].append(current_trace)

        # Update chaining pool and trace pool with new chains and traces
        for ret_type, chains in new_chains.items():
            chaining_pool[ret_type].update(chains)
            trace_pool[ret_type].extend(new_traces[ret_type])

    return False, None, None