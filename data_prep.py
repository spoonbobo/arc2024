import os
import json
import uuid
from pathlib import Path
import shutil
from concurrent.futures import ProcessPoolExecutor, as_completed
from tqdm import tqdm
from multiprocessing import Lock, Manager

def merge_json_files(json_files, folder_name):
    merged_data = {
        "result": [],
        "trace": folder_name,  # Store folder name as a single string
        "task_key": [],
        "grid_id": []
    }
    
    for file in json_files:
        with open(file, 'r') as f:
            data = json.load(f)
            # Convert all fields to strings
            merged_data["result"].append(str(data["result"]))
            merged_data["task_key"].append(str(data["task_key"]))
            merged_data["grid_id"].append(str(data["grid_id"]))
    
    return merged_data

def save_merged_jsonl(merged_data, output_file, lock):
    with lock:
        with open(output_file, 'a', buffering=1) as f:  # Line buffered
            json.dump(merged_data, f)
            f.write('\n')

def reset_output_file(output_file):
    if output_file.exists():
        output_file.unlink()

def process_folder(folder, output_file, lock):
    json_files = list(folder.glob('*.json'))
    folder_name = folder.name  # Get the folder name
    for i in range(0, len(json_files), 3):
        chunk = json_files[i:i+3]
        if chunk:
            merged_data = merge_json_files(chunk, folder_name)  # Pass folder name
            save_merged_jsonl(merged_data, output_file, lock)

def main():
    input_base_dir = Path('NewDataset')
    output_file = Path('train_data.jsonl')
    
    reset_output_file(output_file)
    
    folders = [folder for folder in input_base_dir.iterdir() if folder.is_dir()]
    
    manager = Manager()
    lock = manager.Lock()
    
    with ProcessPoolExecutor() as executor:
        futures = {executor.submit(process_folder, folder, output_file, lock): folder for folder in folders}
        for future in tqdm(as_completed(futures), total=len(futures), desc="Processing folders"):
            future.result()  # Ensure any exceptions are raised

if __name__ == "__main__":
    main()