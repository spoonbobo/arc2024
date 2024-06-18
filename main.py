import os
import json
import uuid
from tqdm import tqdm
from multiprocessing import Pool, Manager
import traceback

from arc_types import *
from prims import *
from utils import load_json, plot_grids, save_image
from solvers.dsl import InstructedDSL
# from llm import PrimitiveInstructor

# for development, enable solver with trace to see primitives and visualize results

base_path = 'arc-prize-2024/'
max_depth = 5
use_beam = True
beam_width = 5

# data
train_challenges = load_json(base_path + 'arc-agi_training_challenges.json')
train_solutions = load_json(base_path + 'arc-agi_training_solutions.json')

# TODO: hypothesis in training phases
# TODO: improve efficiency at scaling up

def evaluate_task(args):
    try:
        key, task, train_solutions, experiment_path = args
        train_inputs = [example['input'] for example in task['train']]
        train_outputs = [example['output'] for example in task['train']]
        test_input = task['test'][0]['input']
        test_output = train_solutions[key][0]
        # convert test_output to tuple of tuples
        test_output = tuple(tuple(row) for row in test_output)
        # primitive_instructor = PrimitiveInstructor(None)
        idsl = InstructedDSL(max_depth=max_depth, use_beam=use_beam, beam_width=beam_width)
        res, result, trace = idsl.solve(test_input, 
                                        test_output)

        # Determine result folder based on success or failure
        result_folder = "success" if res else "failed"
        exp_path = f'{experiment_path}/{result_folder}/{key}'
        os.makedirs(exp_path, exist_ok=True)
        
        if res:
            # Save primitives as JSON
            primitives_data = {p[0]: p[1] for p in trace}
            with open(f'{exp_path}/primitives_trace.json', 'w') as f:
                json.dump(primitives_data, f, indent=4)
            
            print(key, f'{[p[0] for p in trace]}')
            
            with open(f'{exp_path}/primitives_trace.txt', 'w') as f:
                for p in trace:
                    f.write(f"{p[0]}\n")

        # Save predictions
        with open(f'{exp_path}/result.json', 'w') as f:
            json.dump({'input': test_input, 'result': result, 'ground_truth': test_output}, f)
        
        # Prepare titles for each grid
        train_input_titles = [f'Train Input {i+1}' for i in range(len(train_inputs))]
        train_output_titles = [f'Train Output {i+1}' for i in range(len(train_outputs))]
        titles = train_input_titles + train_output_titles + ['Test Input', 'Result', 'Ground Truth']
        
        # Combine all grids and plot
        all_grids = train_inputs + train_outputs + [test_input, result, test_output]
        save_image(plot_grids(all_grids, titles), f'{exp_path}/grid.png')
        
        return res
    except Exception as e:
        traceback.print_exc()
        # exit()
        return False

if __name__ == '__main__':
    with Manager() as manager:
        correct_guess = manager.Value('i', 0)
        total_tasks = len(train_challenges)
        
        experiment_id = str(uuid.uuid4())
        print(experiment_id)
        experiment_path = f'exp/{experiment_id}'
        os.makedirs(experiment_path, exist_ok=True)
        
        with Pool() as pool:
            results = list(tqdm(pool.imap(evaluate_task, [(key, task, train_solutions, experiment_path) for key, task in train_challenges.items()]), total=total_tasks, desc="Evaluating tasks"))
        
        # for key, task in tqdm(train_challenges.items(), total=total_tasks, desc="Evaluating tasks"):
        #     result = evaluate_task((key, task, train_solutions, experiment_path))
        #     results.append(result)
        #     # exit()
    
        correct_guess = sum(results)
        print(f'\nMade correct guesses for {correct_guess} out of {total_tasks} tasks')
