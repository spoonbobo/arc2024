import os
import json
import shutil
from tqdm import tqdm
import uuid

# Define the input and output directories
input_dir = 'dataset'
output_dir = 'NewDataset2'

# Ensure the output directory is clean
if os.path.exists(output_dir):
    shutil.rmtree(output_dir)
os.makedirs(output_dir, exist_ok=True)

# Function to process the trace and create the new directory structure
def process_trace(trace):
    # Extract unique operations and sort them alphabetically
    unique_operations = sorted(set(op for op, _ in trace))
    # Join them with underscores
    return '-'.join(unique_operations)

# Iterate over all JSON files in the input directory
for filename in tqdm(os.listdir(input_dir)):
    if filename.endswith('.json'):
        input_path = os.path.join(input_dir, filename)
        
        filename = os.path.splitext(filename)[0]
        task_key, grid_id = filename.split('_')
        
        # Read the JSON file
        with open(input_path, 'r') as file:
            data = json.load(file)
        
        # Process each entry in the JSON file
        for entry in data:
            trace = entry['trace']
            result = entry['result']
            new_dir_name = process_trace(trace)
            
            # Create the new directory if it doesn't exist
            new_dir_path = os.path.join(output_dir, new_dir_name)
            os.makedirs(new_dir_path, exist_ok=True)
            
            # Create a new JSON file with a single entry
            new_entry = {
                "result": result,
                "trace": trace,
                "task_key": task_key,
                "grid_id": grid_id
            }
            new_filename = f"{uuid.uuid4()}.json"
            output_path = os.path.join(new_dir_path, new_filename)
            with open(output_path, 'w') as file:
                json.dump(new_entry, file, indent=4)
    # break

# https://www.mlexpert.io/blog/alpaca-fine-tuning
# TODO: group combs across training grids for dataset loading (3/4 training input)
# TODO: wrap all into a single JSON file
print("Transformation complete.")