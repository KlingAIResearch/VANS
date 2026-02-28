import json
import os
import subprocess
import argparse
import multiprocessing
from multiprocessing import Pool
import math

def extract_video_segments(json_file_path, video_dir, output_dir, num_processes=None):
    """
    Process the JSON annotation file and extract video segments using ffmpeg.
    
    Args:
        json_file_path (str): Path to the dataset JSON file.
        video_dir (str): Base directory containing the original videos.
        output_dir (str): Base directory to save the extracted video segments.
        num_processes (int): Number of processes to use. Defaults to CPU core count.
    """
    
    with open(json_file_path, 'r') as f:
        data = json.load(f)
    
    tasks = []
    for video_id, video_info in data['database'].items():
        recipe_type = video_info['recipe_type']
        annotations = video_info['annotation']
        
        # Construct the original video path based on the user-provided directory
        original_video_path = os.path.join(video_dir, str(recipe_type), f"{video_id}.mp4")
        
        if not os.path.exists(original_video_path):
            print(f"Warning: Original video not found: {original_video_path}")
            continue
        
        for annotation in annotations:
            segment = annotation['segment']
            annotation_id = annotation['id']
            start_time = segment
            end_time = segment
            
            # Construct the output path
            output_path = os.path.join(output_dir, str(video_id), f"{annotation_id}.mp4")
            
            tasks.append({
                'input_path': original_video_path,
                'start_time': start_time,
                'end_time': end_time,
                'output_path': output_path
            })
    
    if num_processes is None:
        num_processes = multiprocessing.cpu_count()
    num_processes = min(num_processes, len(tasks))
    
    print(f"Starting to process {len(tasks)} video segments using {num_processes} processes...")
    
    # Split tasks into chunks for the multiprocessing pool
    chunk_size = math.ceil(len(tasks) / num_processes)
    if chunk_size == 0:
        print("No tasks to process. Exiting.")
        return

    task_chunks = [tasks[i:i + chunk_size] for i in range(0, len(tasks), chunk_size)]
    
    with Pool(processes=num_processes) as pool:
        results = pool.map(process_task_chunk, task_chunks)
    
    successful = sum(results)
    print(f"Processing complete! Success: {successful}, Total: {len(tasks)}")

def process_task_chunk(task_chunk):
    """
    Process a chunk of tasks (executed by a single worker process).
    
    Args:
        task_chunk (list): List of task dictionaries.
        
    Returns:
        int: Number of successfully processed tasks in this chunk.
    """
    successful_count = 0
    for task in task_chunk:
        output_dir = os.path.dirname(task['output_path'])
        os.makedirs(output_dir, exist_ok=True)
        
        if extract_single_segment(
            task['input_path'],
            task['start_time'],
            task['end_time'],
            task['output_path']
        ):
            successful_count += 1
            
    return successful_count

def extract_single_segment(input_path, start_time, end_time, output_path):
    """
    Extract a single video segment using ffmpeg without re-encoding.
    
    Args:
        input_path (str): Path to the input video.
        start_time (float): Start time in seconds.
        end_time (float): End time in seconds.
        output_path (str): Path to save the extracted video.
        
    Returns:
        bool: True if extraction was successful, False otherwise.
    """
    if os.path.exists(output_path):
        print(f"File already exists, skipping: {output_path}")
        return True
    
    duration = end_time - start_time
    cmd = [
        'ffmpeg',
        '-ss', str(start_time),  
        '-i', input_path,        
        '-t', str(duration),     
        '-c', 'copy',            
        '-y',                    
        output_path              
    ]
    
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
        
        if result.returncode == 0:
            print(f"Successfully extracted: {output_path} (Duration: {duration:.2f}s)")
            return True
        else:
            print(f"Extraction failed: {output_path}")
            print(f"Error message: {result.stderr}")
            return False
            
    except subprocess.TimeoutExpired:
        print(f"Extraction timeout: {output_path}")
        return False
    except Exception as e:
        print(f"Extraction exception: {output_path}, Error: {e}")
        return False

def main():
    parser = argparse.ArgumentParser(description="Extract video segments based on dataset JSON annotations.")
    parser.add_argument(
        "--json_path", 
        type=str, 
        required=True, 
        help="Path to the JSON annotation file (e.g., COIN_OK.json)."
    )
    parser.add_argument(
        "--video_dir", 
        type=str, 
        required=True, 
        help="Base directory containing the original full-length videos."
    )
    parser.add_argument(
        "--output_dir", 
        type=str, 
        required=True, 
        help="Directory where the extracted video segments will be saved."
    )
    parser.add_argument(
        "--num_processes", 
        type=int, 
        default=None, 
        help="Number of CPU processes to use. Defaults to all available cores."
    )
    
    args = parser.parse_args()
    
    if not os.path.exists(args.json_path):
        print(f"Error: JSON file does not exist: {args.json_path}")
        return
        
    extract_video_segments(args.json_path, args.video_dir, args.output_dir, args.num_processes)

if __name__ == "__main__":
    main()