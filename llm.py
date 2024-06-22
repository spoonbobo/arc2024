from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
import torch
import re
from utils import load_json, plot_grids, save_image
from transformers import Trainer, TrainingArguments
import json
from torch.utils.data import Dataset
import ast

def get_functions_from_file(file_path):
    with open(file_path, 'r') as file:
        file_content = file.read()
    
    tree = ast.parse(file_content)
    functions = []
    
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef):
            start_line = node.lineno - 1
            end_line = node.end_lineno
            function_code = file_content.splitlines()[start_line:end_line]
            functions.append('\n'.join(function_code))
    
    return functions

# torch.set_grad_enabled(False)
torch.random.manual_seed(0)

class GridDataset(Dataset):
    def __init__(self, inputs, outputs, tokenizer):
        self.inputs = inputs
        self.outputs = outputs
        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.inputs)

    def __getitem__(self, idx):
        input_grid = self.inputs[idx]
        output_grid = self.outputs[idx]
        
        # Convert grids to JSON strings
        input_grid_str = json.dumps(input_grid)
        output_grid_str = json.dumps(output_grid)
        
        input_encodings = self.tokenizer(input_grid_str, truncation=True, padding='max_length', return_tensors='pt')
        output_encodings = self.tokenizer(output_grid_str, truncation=True, padding='max_length', return_tensors='pt')
        
        return {
            'input_ids': input_encodings['input_ids'].squeeze(),
            'attention_mask': input_encodings['attention_mask'].squeeze(),
            'labels': output_encodings['input_ids'].squeeze()
        }

class ARCAgent:
    
    # prompt templates
    ROLE: str = """
    You are an advanced AI who has strong reasoning skills. You can solve Abstract reasoning corpus tasks by analyzing and uncovering the symmetrical relationships between input grid and output grid.
    All the grids you are going to see follow the same symmetric pattern.
    The number values of the grids represent colors range from 0 to 9.
    Grids can be of any shape with smallest dimension of 1 by 1 to largest 30 by 30.
    The pattern could be simple or complex, including the combinations of reflection, rotation, color fill, scaling, translation, etc.
    Try to learn the pattern with few learning samples and transform an unseen grid to match the learnt pattern.
    """
    
    GRID_PROMPT: str = """
    grid pair {curr}/{total}
    
    input grid
    {input_grid}
    
    output grid
    {output_grid}
    """
    
    SOLUTION_PROMPT: str = """
    apply learnt pattern to the following unseen grid
    {grid}
    
    return the grid after applying the learnt pattern in python list
    Trim your response to only include the grid, no other text, narratives, or context is needed
    """
    
    ROLE_PRIM: str = """
    You are an advanced AI who has strong reasoning skills. You can solve Abstract reasoning corpus tasks by analyzing and uncovering the symmetrical relationships between input grid and output grid.
    All the grids you are going to see follow the same symmetric pattern.
    The number values of the grids represent colors range from 0 to 9.
    Grids can be of any shape with smallest dimension of 1 by 1 to largest 30 by 30.
    The pattern could be simple or complex, including the combinations of reflection, rotation, color fill, scaling, translation, etc.
    Try to learn the pattern with few learning samples and suggest program synthesises to transform an unseen grid to match the learnt pattern.
    """
    
    SOLUTION_PROMPT_PRIM: str = """
    suggest program synthesises to transform the following unseen grid to match the learnt pattern
    {grid}
    
    
    only 1 thing to note about is you have to make sure your primitvies supply return types that are needed by other suggested primitives
    primitives example:
    
    # rotation
    def rot90(
        grid: Tuple[Tuple[int]]
    ) -> Tuple[Tuple[int]]:
        return tuple(row for row in zip(*grid[::-1]))
    
    # reflective
    def hmirror(
        piece: Piece
    ) -> Piece:
        if isinstance(piece, tuple):
            return piece[::-1]
        d = ulcorner(piece)[0] + lrcorner(piece)[0]
        if isinstance(next(iter(piece))[1], tuple):
            return frozenset((v, (d - i, j)) for v, (i, j) in piece)
        return frozenset((d - i, j) for i, j in piece)
    
    you can also design primitives for scaling, translation, mirroing, as long as you think they recover underlying pattern from the input and output grids
    return 10 useful primitives as what as they originally are. do not modify the primitive name, type annotations, as well as the function body
    Trim your response to only include the python program synthesises, no other text, narratives, or context is needed:
    
    ```python
    <wrap-your-answers-here-for-user-to-parse>
    ```
    """

    
    def __init__(self, 
                 model_path='microsoft/Phi-3-mini-4k-instruct', 
                 tokenizer_path='microsoft/Phi-3-mini-4k-instruct',
                 return_primitives=True):
        self.rc = {}
        self.model = AutoModelForCausalLM.from_pretrained(model_path, trust_remote_code=True, load_in_4bit=True, torch_dtype=torch.float16, device_map="cuda")
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
        self.pipe = pipeline(
            "text-generation",
            model=self.model,
            tokenizer=self.tokenizer
        )
        self.return_primitives = return_primitives
    
    def solve(self, grid_inputs, grid_outputs, test_grid, key):
        prompt = ''
        for i in range(len(grid_inputs)):
            if i == 0:
                prompt += self.ROLE if not self.return_primitives else self.ROLE_PRIM
            prompt += self.GRID_PROMPT.format(curr=i+1, total=len(grid_inputs), input_grid=grid_inputs[i], output_grid=grid_outputs[i])
            if i != 0:
                prompt += f'this pair and all previous {i} grid pair follow same symmetrical pattern.'
            
        fig = plot_grids(grid_inputs+[test_grid], grid_outputs, [f'grid pair {i+1}/{len(grid_inputs)} input' for i in range(len(grid_inputs))]+[f'test grid'], [f'grid pair {i+1}/{len(grid_outputs)} output' for i in range(len(grid_outputs))])
        save_image(fig, f'{key}.png')
        # rgb_arr = plot_to_array(fig)
        prompt = self.SOLUTION_PROMPT.format(grid=test_grid) if not self.return_primitives else self.SOLUTION_PROMPT_PRIM.format(grid=test_grid)
        msgs = [{'role': 'user', 'content': prompt}]

        generation_args = {
            "max_new_tokens": 4000,
            "return_full_text": False,
            "temperature": 0.0,
            "do_sample": False,
        }
        
        solution = self.pipe(msgs, **generation_args)[0]['generated_text']
        return solution
    
    def feedback_fn(self, g_truth, solution):
        solution = parse_grid(solution)
        print(solution, type(solution), g_truth)
        if solution != g_truth:
            return g_truth
        return None
    
    def active_learning(self, grid_input, grid_output, test_grid, test_labels, keys):
        for l_i in range(len(grid_input)):
            solution = self.solve(grid_input[l_i], grid_output[l_i], test_grid[l_i], keys[l_i])
            feedback = self.feedback_fn(test_labels[l_i], solution)
            if feedback:
                grid_input.append(test_grid[l_i])
                grid_output.append(feedback)
                print('not good enough, work harder!')
                self.fine_tune_model(grid_input[l_i], grid_output[l_i])
                
    
    def fine_tune_model(self, grid_inputs, grid_outputs):
        dataset = GridDataset(grid_inputs, grid_outputs, self.tokenizer)
        
        training_args = TrainingArguments(
            output_dir='./results',
            num_train_epochs=3,
            per_device_train_batch_size=4,
            save_steps=10_000,
            save_total_limit=2,
        )
        
        trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=dataset
        )
        
        trainer.train()

def parse_grid(grid_str):
    # Use regular expression to find all lists in the string
    list_pattern = re.compile(r'\[\s*(?:\d+\s*,\s*)*\d+\s*\]')
    matches = list_pattern.findall(grid_str)
    
    # Convert the matched strings to actual lists
    parsed_grid = [eval(match) for match in matches]
    
    return parsed_grid    

if __name__ == '__main__':
    base_path = 'arc-prize-2024/'
    train_challenges = load_json(base_path + 'arc-agi_training_challenges.json')
    train_solutions = load_json(base_path + 'arc-agi_training_solutions.json')
    arc_agent = ARCAgent()

    keys = []
    grid_inputs = []
    grid_outputs = []
    test_grids = []
    test_label = []
    
    for key, task in train_challenges.items():
        keys.append(key)
        train_inputs = [example['input'] for example in task['train']]
        train_outputs = [example['output'] for example in task['train']]
        test_input = task['test'][0]['input']
        test_output = train_solutions[key][0]
        grid_inputs.append(train_inputs)
        grid_outputs.append(train_outputs)
        test_grids.append(test_input)
        test_label.append(test_output)

    arc_agent.active_learning(grid_inputs, grid_outputs, test_grids, test_label, keys)
