import ast
import re
from utils import load_json

import ollama

class PrimitiveInstructor:
    """
    llm agent who suggested primitives at runtime to help solving ARC tasks
    """
    
    # prompt templates
    ROLE: str = """
    You are an advanced AI who has strong reasoning skills, and you are capable to solve abstract reasoning tasks. You analyze symmetric relationships between input grids and output grids, and suggest primitives as building blocks to uncover the symmetrical patterns.
    
    You observe a set of grid pairs, each pairs consists of input grid and output grid, grid pair may or may not follow same dimensions, the smallest grid's size is 1x1, while largest is 30x30. After your observation of the grid pairs, you suggest primitives (python functions) based on your observed symmetrical patterns in observed grid pairs. You generalise the sysmetrical patterns from the grid pairs.
    """

    OBSERVE_ARC: str = """ 
    grid pair {current_pair}/{total_pairs}:
    
    input grid:
    {grid}
    
    output grid:
    {output_grid}
    """
    
    PRIMITIVE_REQUEST: str = """ 
    Now suggest no more than 10 primitives in python functions as the basis for your reasoning of the symmetrical pattern.
    
    Important: your primitives must have input params and return types annotated, otherwise the system will break
    
    Cautious: the variables you used in primitives must be either assigned, parameters, or other compatible suggested primitives. Accessing undefined variables will break the system.
    
    
    Invalid example:
    def count_cells(grid):
        return sum(sum(row) for row in grid)
        
    valid example:
    def count_cells(grid: Tuple[Tuple[int]]): -> int
        return sum(sum(row) for row in grid)
        
    Do not return None/

    
    Trim your response to include only annotatedprimitives, their code implementations, and nothing else.
    """

    def __init__(self, llm=None, train_data=None):
        self.llm = llm
        self.rc = {}
        self.primitives = None
        self.load_primitives('prims.py')
        
        # build chain of thought
        self.COT = ''
        curr_pair = 0
        cot_sample = train_data['d9fac9be']['train']
        for sample_idx in range(len(cot_sample)):
            inp = cot_sample[sample_idx]['input']
            outp = cot_sample[sample_idx]['output']
            self.COT += self.OBSERVE_ARC.format(current_pair=curr_pair,
                                                              total_pairs=len(cot_sample),
                                                              grid=inp,
                                                              output_grid=outp)
            curr_pair += 1
        self.COT += self.PRIMITIVE_REQUEST.format(total_pairs=len(cot_sample))
    
    def load_primitives(self, path):
        with open(path, "r") as file:
            source = file.read()
            tree = ast.parse(source, filename=path)
        
        function_definitions = []
        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef):
                start_line = node.lineno - 1
                end_line = node.end_lineno
                function_code = source.splitlines()[start_line:end_line]
                
                # Normalize indentation
                min_indent = min(len(line) - len(line.lstrip()) for line in function_code if line.strip())
                function_code = [line[min_indent:] for line in function_code]
                
                function_definitions.append("\n".join(function_code))
                
        self.primitives = '\n\n'.join(function_definitions)

    def parse_prims(self, response):
        pattern = r'```(?:python)?\n(.*?)\n```'
        matches = re.findall(pattern, response, re.DOTALL)
        return matches
    
    def ask(self, grid_inputs, grid_outputs):
        prompt = ''
        total_pairs = len(grid_inputs)
        for i in range(total_pairs):
            prompt += self.OBSERVE_ARC.format(current_pair=i+1,
                                              total_pairs=total_pairs,
                                              grid=grid_inputs[i], 
                                              output_grid=grid_outputs[i])
        prompt += self.PRIMITIVE_REQUEST.format(total_pairs=total_pairs)
        
        response = ollama.chat(model='llama3:8b', messages=[
        {
            'role': 'user',
            'content': self.ROLE,
        },
        {
            'role': 'assistant',
            'content': 'Sure! I will observe a set of pairs of input and output grids, generalise the symmetrical patterns between the pairs, and suggested primitives: ',
        },
        {
            'role': 'user',
            'content': self.COT,
        },
        {
            'role': 'assistant',
            'content': """
            Here are my suggested primitives based on my reasoning of patterns of these grid pairs
            ```python
            def blur(grid: Tuple[Tuple[int]]) -> Tuple[Tuple[int]]:
                def get_neighbors(i, j):
                    neighbors = [(i-1, j-1), (i-1, j), (i-1, j+1), (i, j-1), (i, j), (i, j+1), (i+1, j-1), (i+1, j), (i+1, j+1)]
                    return [(x, y) for x, y in neighbors if 0 <= x < len(grid) and 0 <= y < len(grid[0])]
            
                new_grid = [[0] * len(row) for row in grid]
                for i in range(len(grid)):
                    for j in range(len(grid[0])):
                        neighbors = get_neighbors(i, j)
                    neighbor_values = [grid[x][y] for x, y in neighbors]
                    new_grid[i][j] = sum(neighbor_values) // len(neighbor_values)
                return tuple([tuple(row) for row in new_grid])
            
            def boolean_or(grid1: Tuple[Tuple[int]], grid2: Tuple[Tuple[int]]) -> Tuple[Tuple[int]]:
                return tuple([tuple(cell1 or cell2 for cell1, cell2 in zip(row1, row2)) for row1, row2 in zip(grid1, grid2)])
            
            def count_nonzero(grid: Tuple[Tuple[int]]) -> Integer:
                return sum(1 for row in grid for cell in row if cell != 0)

            def scale_down(grid: Tuple[Tuple[int]], factor: Integer) -> Tuple[Tuple[int]]:
                if factor <= 0:
                    return grid
                return tuple([tuple(row[::factor]) for row in grid[::factor]])
            ```
            """,
        },
        {
            'role': 'user',
            'content': "great job!" + prompt,
        },
        ], options={'temperature': 0.1, 'top_k': 10})
        
        resp = response['message']['content']
        print(resp)
        return self.parse_prims(resp)

if __name__ == '__main__':
    base_path = 'arc-prize-2024/'
    train_challenges = load_json(base_path + 'arc-agi_training_challenges.json')
    instructor = PrimitiveInstructor(train_data=train_challenges)
