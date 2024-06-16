import ast

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
    
    PRIMITIVES: str = """
    Here are primitive examples:
    {primitives}
    """

    OBSERVE_ARC: str = """ 
    grid pair {current_pair}/{total_pairs}:
    
    input grid:
    {grid}
    
    output grid:
    {output_grid}
    """
    
    PRIMITIVE_REQUEST: str = """ 
    You observed a general sysmetrical pattern from these {total_pairs} pairs.
    Now suggest no more than 10 primitives in python functions (```python <your-primitives>```) as the basis for your reasoning of the symmetrical pattern.
    Your suggested primitives could be from examples or from your own defined primitives.
        
    To help you familiarize with the requirements, I will give you examples of primitives

    def count_nonzero(grid: Grid) -> Integer:
        # Counts the number of non-zero values in the grid
        return sum(1 for row in grid for cell in row if cell != 0)
        
    def pad_grid(grid: Grid) -> Grid:
        # Pads the grid with zeros to the same size as the largest grid
        max_size = max(len(row) for row in grid)
        return [[0] * max_size for _ in range(max_size)]
        
    Available types: Boolean, Integer, IntegerList, Numerical, IntegerSet, Grid, Cell, Object, Objects, Indices, IndicesSet, Patch, Element, Piece, ListList, ContainerContainer, Container
    
    parameters and return type of your suggested primitives must be from above available types, no exception.
    
    For definitions of above available types, please read following:
    Boolean  (bool)
    Integer  (int)
    IntegerList  (Tuple[Integer, ...])
    Numerical  (Union[Integer, IntegerList])
    IntegerSet  (FrozenSet[Integer])
    Grid  (Tuple[Tuple[Integer, ...], ...])
    Cell  (Tuple[Union[Integer, IntegerList], ...])
    Object  (FrozenSet[Cell])
    Objects  (FrozenSet[Object])
    Indices  (FrozenSet[IntegerList])
    IndicesSet  (FrozenSet[Indices])
    Patch  (Union[Object, Indices])
    Element  (Union[Object, Grid]
    Piece  (Union[Grid, Patch])
    ListList  (Tuple[Tuple[Any, ...], ...])
    ContainerContainer  (Container[Container])
    
    Trim your response to include only primitives, and nothing else.
    """

    def __init__(self, llm):
        self.llm = llm
        self.rc = {}
        self.primitives = None
        self.load_primitives('prims.py')
    
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
    
    def ask(self, grid_inputs, grid_outputs):
        prompt = ''
        prompt += self.ROLE
        prompt += self.PRIMITIVES.format(primitives=self.primitives)
        total_pairs = len(grid_inputs)
        for i in range(total_pairs):
            prompt += self.OBSERVE_ARC.format(current_pair=i+1,
                                              total_pairs=total_pairs,
                                              grid=grid_inputs[i], 
                                              output_grid=grid_outputs[i])
        prompt += self.PRIMITIVE_REQUEST.format(total_pairs=total_pairs)
        
        # no need creativity.
        response = ollama.chat(model='llama3:8b', messages=[
        {
            'role': 'user',
            'content': prompt,
        },
        ], options={'temperature': 0.0, 'top_k': 10})
        
        return response['message']['content']

if __name__ == '__main__':
    instructor = PrimitiveInstructor(llm=None)
