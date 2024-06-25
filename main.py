import os
import json
import uuid
from tqdm import tqdm
from multiprocessing import Pool, Manager
import traceback
import shutil  # Add this import


from arc_types import *
from prims import *
from utils import load_json, plot_grids, save_image
from solvers.dsl import InstructedDSL
from llm import ARCAgent

# for development, enable solver with trace to see primitives and visualize results

base_path = 'arc-prize-2024/'
max_depth = 1
use_beam = True
beam_width = 3

SUBMISSION = False
BOOTSTRAP_DATA = True
USE_AI_AGENT = False

if BOOTSTRAP_DATA:
    use_beam = False

# data
train_challenges = load_json(base_path + 'arc-agi_training_challenges.json')
train_solutions = load_json(base_path + 'arc-agi_training_solutions.json')
test_challenges = load_json(base_path + 'arc-agi_test_challenges.json')

if USE_AI_AGENT:
    arc_agent = ARCAgent(return_primitives=True, model_path='microsoft/Phi-3-mini-4k-instruct', tokenizer_path='microsoft/Phi-3-mini-4k-instruct')
max_llm_retry = 3

if os.path.exists('dataset'):
    shutil.rmtree('dataset')

def evaluate_task(args):
    try:
        key, task, train_solutions, experiment_path = args
        train_inputs = [example['input'] for example in task['train']]
        train_outputs = [example['output'] for example in task['train']]
        test_input = task['test'][0]['input']
        test_output = train_solutions[key][0]
        test_output = tuple(tuple(row) for row in test_output)
        # solution = arc_agent.solve(train_inputs, train_outputs, test_input, key)
        idsl = InstructedDSL(max_depth=max_depth, use_beam=use_beam, beam_width=beam_width)
        solutions = []
        for grid_id, (train_inp, train_out) in enumerate(zip(train_inputs, train_outputs)):
            if BOOTSTRAP_DATA:
                solutions.append(idsl.solve(train_inp, train_out, key, grid_id, bootstrap=True))
                # exit()
            else:
                solutions.append(idsl.solve(test_input, test_output, key, grid_id))
    
        # # Determine result folder based on success or failure
        result_folder = "success" if solutions else "failed"
        exp_path = f'{experiment_path}/{result_folder}/{key}'
        os.makedirs(exp_path, exist_ok=True)
        
        # if candidate_traces:
        #     for i, trace in enumerate(candidate_traces):
        #         # Save primitives as JSON
        #         primitives_data = {p[0]: p[1] for p in trace}
        #         with open(f'{exp_path}/primitives_trace_{i}.json', 'w') as f:
        #             json.dump(primitives_data, f, indent=4)
            
        #         with open(f'{exp_path}/primitives_trace_{i}.txt', 'w') as f:
        #             for p in trace:
        #                 f.write(f"{p[0]}\n")

        # Save predictions
        # with open(f'{exp_path}/result.json', 'w') as f:
        #     json.dump({'input': test_input, 'result': candidate_traces[0][-1][1] if candidate_traces else None, 'ground_truth': test_output}, f)
        # Combine all grids and plot
        if not SUBMISSION:
            train_input_titles = [f'Train Input {i+1}' for i in range(len(train_inputs))]
            train_output_titles = [f'Train Output {i+1}' for i in range(len(train_outputs))]
            titles = train_input_titles + train_output_titles + ['Test Input', 'Prediction', 'Ground Truth']
            all_grids = train_inputs + train_outputs + [test_input, test_input, test_output]
            # todo: fix it save_image(plot_grids(all_grids, titles), f'{exp_path}/grid.png')
        
        return bool(solutions)
    except Exception as e:
        traceback.print_exc()
        return False

if __name__ == '__main__':
    with Manager() as manager:
        correct_guess = manager.Value('i', 0)
        total_tasks = len(train_challenges)
        
        experiment_id = str(uuid.uuid4())
        print(experiment_id)
        experiment_path = f'exp/{experiment_id}'
        os.makedirs(experiment_path, exist_ok=True)
        
        results = []
        # with Pool() as pool:
        #     if not SUBMISSION:
        #         results = list(tqdm(pool.imap(evaluate_task, [(key, task, train_solutions, experiment_path) for key, task in train_challenges.items()]), total=total_tasks, desc="Evaluating tasks"))
        #     else:
        #         results = list(tqdm(pool.imap(evaluate_task, [(key, task, train_solutions, experiment_path) for key, task in test_challenges.items()]), total=total_tasks, desc="Evaluating tasks"))
        
        ct = 0
        for key, task in tqdm(train_challenges.items(), total=total_tasks, desc="Evaluating tasks"):
            result = evaluate_task((key, task, train_solutions, experiment_path))
            results.append(result)
            
            if ct > 3:
                break
            ct += 1
        
    
        # correct_guess = sum(results)
        print(f'\nMade correct guesses for {correct_guess} out of {total_tasks} tasks')

