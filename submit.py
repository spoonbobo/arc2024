import os
import json
from tqdm import tqdm
import uuid

from arc_types import *
from prims import *
from utils import load_json, plot_grids, save_image
from chain import solver, solver_with_trace

base_path = 'arc-prize-2024/'
max_depth = 3

# data
test_challenges = load_json(base_path + 'arc-agi_test_challenges.json')

# TODO: param bootstraping

exist_hypothesis = 0

# Generate a unique ID for the entire experiment
experiment_id = str(uuid.uuid4())
experiment_path = f'exp/{experiment_id}'

# Initialize tqdm with the total number of tasks for proper progress calculation
# let's try to make a submission

submission = dict()
# iterate over all tasks
with tqdm(test_challenges.items(), desc="Evaluating tasks", total=len(test_challenges)) as pbar:
    for key, task in pbar:
        train_inputs = [example['input'] for example in task['train']]
        train_outputs = [example['output'] for example in task['train']]
        test_inputs = [example['input'] for example in task['test']]
        # exit()
        continue
        hypothesis = []
        for inp, outp in zip(train_inputs, train_outputs):
            res, result, primitives = solver_with_trace(inp, outp, max_depth)
            if res:
                hypothesis.append(primitives)

        print(hypothesis)

        if len(hypothesis):
            exist_hypothesis += 1  # Increment correct_guess if the result is True

        # Update the progress bar with current accuracy
        pbar.set_postfix(exist_chance=f"{exist_hypothesis}/{len(test_challenges)} ({exist_hypothesis/len(test_challenges)*100:.2f}%)")



with open('submission.json', 'w') as fp:
    json.dump(submission, fp)
