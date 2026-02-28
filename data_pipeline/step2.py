import json
import os
import time
import random
import argparse
import google.generativeai as genai
from tqdm import tqdm

def chat_with_multi_modal(model_name: str, prompt: str, video_files: list):
    """
    Interact with the Gemini multi-modal model.
    """
    safety_settings = {
        'HATE': 'BLOCK_NONE',
        'HARASSMENT': 'BLOCK_NONE',
        'SEXUAL': 'BLOCK_NONE',
        'DANGEROUS': 'BLOCK_NONE'
    }

    model = genai.GenerativeModel(model_name=model_name)
    input_list = []
    
    for i, video_file in enumerate(video_files):
        input_list.append(f"Video {i+1}:")
        input_list.append(video_file)
        
    input_list.append(prompt)
    
    print("############### Generating Response ###############")
    ret = model.generate_content(
        input_list,
        request_options={"timeout": 600},
        safety_settings=safety_settings
    )
    return ret.text

def predict_gemini(api_keys: list, model_name: str, query: str, file_paths: list):
    """
    Upload files to Gemini, generate a response, and clean up files from the cloud.
    """
    # Randomly select an API key to balance load
    key_now = random.randint(0, len(api_keys) - 1)
    genai.configure(api_key=api_keys[key_now])

    video_files = []
    response = "WRONG"

    try:
        # Upload videos
        for file_path in file_paths:
            print(f"Uploading {file_path} to Gemini...")
            video_file = genai.upload_file(path=file_path)
            
            # Wait for processing to complete
            while video_file.state.name == "PROCESSING":
                print('.', end='', flush=True)
                time.sleep(20)
                video_file = genai.get_file(video_file.name)
                
            if video_file.state.name == "FAILED":
                print(f"\nFailed to process video: {file_path}")
                continue 
                
            video_files.append(video_file)
            print(" Upload complete.")
            
        # Run inference
        if video_files:
            response = chat_with_multi_modal(model_name, query, video_files)

    finally:
        # ALWAYS clean up uploaded files to avoid hitting API storage quotas
        for video_file in video_files:
            try:
                genai.delete_file(video_file.name)
                print(f"Deleted temporary cloud file: {video_file.name}")
            except Exception as e:
                print(f"Failed to delete {video_file.name}: {e}")

    return response

def process_dataset_annotations(json_path: str, input_video_dir: str, output_txt_dir: str, api_keys: list, start_idx=0.0, end_idx=1.0):
    """
    Process dataset annotations, find matching videos, and use Gemini to generate dense captions.
    """
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    database_items = list(data['database'].items())
    
    start_pos = int(start_idx * len(database_items))
    end_pos = int(end_idx * len(database_items))
    print(f"Processing videos from index {start_pos} to {end_pos - 1} (Total: {len(database_items)})")
    
    for video_id, video_info in database_items[start_pos:end_pos]:
        video_folder = os.path.join(input_video_dir, video_id)

        if not os.path.exists(video_folder):
            print(f"Warning: Video folder {video_folder} does not exist, skipping...")
            continue
        
        print(f"\n=== Processing Video ID: {video_id} ===")   
        mp4_files = [f for f in os.listdir(video_folder) if f.endswith('.mp4')]
    
        output_path = os.path.join(output_txt_dir, video_id)
        os.makedirs(output_path, exist_ok=True)
        
        for annotation in video_info.get('annotation', []):
            annotation_id = annotation['id']
            sentence = annotation['label']
            
            print(f"ID: {annotation_id} | Caption: {sentence}")

            output_txt_path = os.path.join(output_path, f"{annotation_id}.txt")
            if os.path.exists(output_txt_path):
                print("Already processed, skipping.")
                continue

            matching_files = []
            
            # Find exact match (e.g., "0.mp4")
            exact_match = f"{annotation_id}.mp4"
            if exact_match in mp4_files:
                matching_files.append(exact_match)
            
            # Find sub-ID matches (e.g., "2_0.mp4", "2_1.mp4")
            pattern = f"{annotation_id}_"
            for file in mp4_files:
                if file.startswith(pattern) and file.endswith('.mp4'):
                    matching_files.append(file)
            
            video_paths = [os.path.join(video_folder, f) for f in matching_files]

            if not video_paths:
                print(f"No matching videos found for annotation {annotation_id}.")
                continue

            # Note: Kept the prompt in Chinese to maintain compatibility with downstream text parsers
            optimized_prompt = f'''
            Your task is to align the videos with the provided annotation. Please follow this standardized process:

            1. First, evaluate which of the {len(video_paths)} input videos are high-quality: they should be full-screen videos without picture-in-picture windows, large blank borders, or auxiliary elements like purely voiceover commentary screens. If no video meets this high-quality standard, ignore all subsequent instructions and return None.

            2. Next, assess whether the caption ("{sentence}") describes an action step. If it does not, please condense it to extract the most critical core action.

            3. Finally, perform the matching: determine which specific time segment of which video best matches the step's description. Based on the caption, provide a detailed visual description of that segment. The return format must be: [Step Name][video-n][00:04-00:09][Detailed description of the video scene] (Each segment should be between 3 and 5 seconds long, or the full length of the video if it is shorter than 3 seconds). If no matching video segment is found, return [Step Name][None].

            Output Format:
            [Video 1 Evaluation]: Description of Video 1, followed by a judgment on whether it meets the quality requirements.
            [Video n Evaluation]: Description of Video n, followed by a judgment on whether it meets the quality requirements.
            [Caption Evaluation]: Analysis of the caption and whether it needs to be summarized into a new core step.
            [Matching Analysis]: Analysis of which video best matches the step, concluding with the single best match.
            [Final Conclusion]: [Step Name][video-n][00:01-00:04][Detailed visual description, including main subjects, actions, and background]
            '''
            
            model_list = ["gemini-2.5-pro"]
            predicted_answer = "WRONG"
            
            # Retry mechanism across different models
            for i in range(3):
                try:
                    model_id_now = i % len(model_list)
                    predicted_answer = predict_gemini(api_keys, model_list[model_id_now], optimized_prompt, video_paths)
                    time.sleep(5) # Rate limiting buffer
                    break
                except Exception as e:
                    print(f"Error during API call: {e}")
                    if 'PROHIBITED_CONTENT' in str(e):
                        predicted_answer = "PROHIBITED_CONTENT"
                        break
                    time.sleep(5)
                    continue 

            print(f"Response: {predicted_answer}")
            
            if predicted_answer != "WRONG":
                with open(output_txt_path, "w", encoding="utf-8") as file:
                    file.write(predicted_answer + "\n")
                    file.write(str(video_paths))

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Generate dense captions using Gemini Vision API.')
    parser.add_argument('--json_path', type=str, required=True, help='Path to dataset JSON (e.g., COIN_OK.json)')
    parser.add_argument('--input_video_dir', type=str, required=True, help='Directory containing the split mp4 chunks')
    parser.add_argument('--output_txt_dir', type=str, required=True, help='Directory to save the generated text responses')
    parser.add_argument('--api_keys', type=str, help='Comma-separated list of Gemini API keys. Or use GEMINI_API_KEYS env var.')
    parser.add_argument('--start_idx', type=float, default=0.0, help='Start index fraction (0.0 to 1.0)')
    parser.add_argument('--end_idx', type=float, default=1.0, help='End index fraction (0.0 to 1.0)')
    
    args = parser.parse_args()
    
    # Securely load API keys
    keys_string = args.api_keys or os.environ.get("GEMINI_API_KEYS", "")
    api_key_list = [k.strip() for k in keys_string.split(',') if k.strip()]
    
    if not api_key_list:
        raise ValueError("No API keys provided! Use --api_keys or set GEMINI_API_KEYS environment variable.")
    
    process_dataset_annotations(
        json_path=args.json_path, 
        input_video_dir=args.input_video_dir, 
        output_txt_dir=args.output_txt_dir, 
        api_keys=api_key_list,
        start_idx=args.start_idx, 
        end_idx=args.end_idx
    )