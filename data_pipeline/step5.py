import os
import json
import csv
import ast
import random
import argparse
import pandas as pd
from typing import List, Dict, Any
from pathlib import Path

# Import Google Generative AI SDK
import google.generativeai as genai

# ==========================================
# Prompt Templates (Translated to English)
# ==========================================

prompt_template_with_skipped_steps = """You are an experienced life skills mentor and video content analysis expert. You excel at accurately inferring the progress of a task based on the provided video content and operation descriptions, and giving professional, easy-to-understand guidance for the next steps.

Please generate a structured guidance Q&A based on the following task information:

* **Ongoing Task**: {recipe_type}
* **Initial video content uploaded by the user (invisible to you)**: {input_dense_caption}
* **Other steps completed by the user after the initial video**: {skipped_steps_text}
* **Next action to be executed (requires logical inference)**: {target_sentence}
* **Ideal state description after completing the action**: {target_dense_caption}

Please strictly follow the three-part structure and requirements below to generate your response:

1.  **[Instruction]**:
    * Simulate a user's question to the AI assistant.
    * The user states they are currently doing {recipe_type}, and mentions having uploaded an initial video (Note: do not describe any visual content of this video).
    * The user also states that after the initial video, they have completed the following steps: {skipped_steps_text}.
    * The user asks what operation should be performed next.
    * **Requirement**: Natural, conversational tone, fitting the persona of someone actively performing the task.

2.  **[Think]**:
    * Simulate the AI assistant's internal reasoning process.
    * **First**, based on the initial video content, infer what the user "initially completed".
    * **Then**, incorporate the other completed steps ({skipped_steps_text}).
    * **Finally**, based on all completed steps and your understanding of the standard {recipe_type} workflow, logically deduce the "most reasonable next step", pointing directly to the action described in {target_sentence}.
    * **Requirement**: Reasoning must be logically coherent and reflect the causal relationship between steps.

3.  **[GT Caption]**:
    * Simulate the AI assistant's final response to the user.
    * **Must** be a declarative sentence directly describing the ideal visual state of the scene after completing the next action.
    * **Core content** must be built upon {target_dense_caption}.
    * **Requirement**: Specific, objective, using declarative sentences, avoiding colloquialisms, allowing the user to clearly visualize the successful operation.

Please ensure all parts (Instruction, Think, GT Caption) are written in fluent and natural English.

Response Format:
[Instruction] ...
[Think] ...
[GT Caption] ...
"""

prompt_template_without_skipped_steps = """You are an experienced life skills mentor and video content analysis expert. You excel at accurately inferring the progress of a task based on the provided video content and operation descriptions, and giving professional, easy-to-understand guidance for the next steps.

Please generate a structured guidance Q&A based on the following task information:

* **Ongoing Task**: {recipe_type}
* **Initial video content uploaded by the user (invisible to you)**: {input_dense_caption}
* **Next action to be executed (requires logical inference)**: {target_sentence}
* **Ideal state description after completing the action**: {target_dense_caption}

Please strictly follow the three-part structure and requirements below to generate your response:

1.  **[Instruction]**:
    * Simulate a user's question to the AI assistant.
    * The user states they are currently doing {recipe_type}, and mentions having uploaded an initial video (Note: do not describe any visual content of this video).
    * The user asks what operation should be performed next.
    * **Requirement**: Natural, conversational tone, fitting the persona of someone actively performing the task.

2.  **[Think]**:
    * Simulate the AI assistant's internal reasoning process.
    * Based on the initial video content, analyze what the user "just completed".
    * Then, based on the current state and your understanding of the standard {recipe_type} workflow, logically deduce the "most reasonable next step", pointing directly to the action described in {target_sentence}.
    * **Requirement**: Reasoning must be logically coherent and reflect the causal relationship between steps.

3.  **[GT Caption]**:
    * Simulate the AI assistant's final response to the user.
    * **Must** be a declarative sentence directly describing the ideal visual state of the scene after completing the next action.
    * **Core content** must be built upon {target_dense_caption}.
    * **Requirement**: Specific, objective, using declarative sentences, avoiding colloquialisms, allowing the user to clearly visualize the successful operation.

Please ensure all parts (Instruction, Think, GT Caption) are written in fluent and natural English.

Response Format:
[Instruction] ...
[Think] ...
[GT Caption] ...
"""

# ==========================================
# Data Processing Utilities
# ==========================================

def parse_skipped_info(skipped_info):
    """
    Parse the skipped_info field, handling string or list formats.
    """
    if pd.isna(skipped_info) or skipped_info in ['[]', '']:
        return []
    
    try:
        if isinstance(skipped_info, str):
            return ast.literal_eval(skipped_info)
        elif isinstance(skipped_info, list):
            return skipped_info
        else:
            return []
    except (ValueError, SyntaxError):
        print(f"Failed to parse skipped_info: {skipped_info}")
        return []

def format_skipped_steps(skipped_info):
    """
    Format the skipped steps information into readable strings.
    """
    skipped_steps = parse_skipped_info(skipped_info)
    if not skipped_steps:
        return "No intermediate steps", "", ""
    
    # Generate steps description text
    steps_text = ", ".join([f"'{step['sentence']}'" for step in skipped_steps if 'sentence' in step])
    
    # Generate texts for Instruction and Think parts
    instruction_text = f"The user explains that after the initial video, they completed the following steps: {steps_text}."
    think_text = f"**Next**, the user completed the intermediate steps: {steps_text}, indicating the progress has advanced."
    
    return steps_text, instruction_text, think_text

def read_csv_and_prepare_prompts(csv_file_path: str) -> List[Dict[str, Any]]:
    """
    Read the CSV file and prepare Prompt data for each row.
    """
    df = pd.read_csv(csv_file_path)
    
    required_columns = ['recipe_type', 'input_dense_caption', 'target_sentence', 'target_dense_caption', 'skipped_info']
    for col in required_columns:
        if col not in df.columns:
            raise ValueError(f"Missing required column in CSV: {col}")
    
    prompts_data = []
    for index, row in df.iterrows():
        skipped_steps_text, skipped_instruction_part, skipped_think_part = format_skipped_steps(row['skipped_info'])
        
        if skipped_steps_text != "No intermediate steps":
            formatted_prompt = prompt_template_with_skipped_steps.format(
                recipe_type=row['recipe_type'],
                input_dense_caption=row['input_dense_caption'],
                skipped_steps_text=skipped_steps_text,
                target_sentence=row['target_sentence'],
                target_dense_caption=row['target_dense_caption']
            )
        else:
            formatted_prompt = prompt_template_without_skipped_steps.format(
                recipe_type=row['recipe_type'],
                input_dense_caption=row['input_dense_caption'],
                target_sentence=row['target_sentence'],
                target_dense_caption=row['target_dense_caption']
            )
        
        prompt_info = {
            'row_index': index,
            'name': row.get('name', ''),
            'recipe_type': row['recipe_type'],
            'input_id': row.get('input_id', ''),
            'input_video_path': row.get('input_video_path', ''),
            'input_sentence': row.get('input_sentence', ''),
            'input_dense_caption': row['input_dense_caption'],
            'target_id': row.get('target_id', ''),
            'target_video_path': row.get('target_video_path', ''),
            'target_sentence': row['target_sentence'],
            'target_dense_caption': row['target_dense_caption'],
            'skipped_ids': row.get('skipped_ids', []),
            'skipped_info': row['skipped_info'],
            'skipped_steps_text': skipped_steps_text,
            'formatted_prompt': formatted_prompt
        }
        
        prompts_data.append(prompt_info)
    
    print(f"Successfully processed {len(prompts_data)} rows of data.")
    return prompts_data

def save_results(results: List[Dict[str, Any]], output_file: str):
    """
    Save processing results to a CSV file.
    """
    try:
        df_results = pd.DataFrame(results)
        
        columns_to_save = [
            'row_index', 'name', 'recipe_type', 'input_sentence', 
            'target_sentence', 'skipped_steps_text', 'llm_response'
        ]
        
        if all(col in df_results.columns for col in columns_to_save):
            df_results[columns_to_save].to_csv(output_file, index=False, encoding='utf-8-sig')
        else:
            df_results.to_csv(output_file, index=False, encoding='utf-8-sig')
            
        print(f"Result saved to: {output_file}")
    except Exception as e:
        print(f"Error saving results: {e}")

# ==========================================
# Core LLM Processing (Gemini 2.5 Pro)
# ==========================================

def process_all_prompts(prompts_data: List[Dict[str, Any]], output_dir: str):
    """
    Process all prepared Prompts using Gemini-2.5-Pro.
    """
    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)
    
    # Configure Gemini API
    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        raise ValueError("Please set the GEMINI_API_KEY environment variable before running the script.")
    
    genai.configure(api_key=api_key)
    
    # Initialize the specific Gemini model
    model = genai.GenerativeModel(model_name="gemini-2.5-pro")
    
    safety_settings = {
        'HATE': 'BLOCK_NONE',
        'HARASSMENT': 'BLOCK_NONE',
        'SEXUAL': 'BLOCK_NONE',
        'DANGEROUS': 'BLOCK_NONE'
    }

    for prompt_info in prompts_data:
        output_file_name = f"{prompt_info['name']}_{prompt_info['input_id']}_{prompt_info['target_id']}.csv"
        output_file_path = os.path.join(output_dir, output_file_name)
        
        if os.path.exists(output_file_path):
            print(f"File {output_file_name} already exists. Skipping...")
            continue 

        prompt = prompt_info['formatted_prompt']
        print(f"############### Generating for {prompt_info['name']} ###############")
        
        try:
            # Call Gemini API
            response = model.generate_content(
                prompt,
                request_options={"timeout": 600},
                safety_settings=safety_settings
            )
            llm_response = response.text
            
            print("Response:", llm_response[:100], "...") # Print snippet for tracking
            
            # Save results
            result = {
                **prompt_info,
                'llm_response': llm_response
            }
            save_results([result], output_file_path) 
            
        except Exception as e:
            print(f"An error occurred while calling the API for row {prompt_info['row_index']}: {e}")

# ==========================================
# Main Execution
# ==========================================

def main(args):
    # Validate paths
    if not os.path.exists(args.input_csv):
        raise FileNotFoundError(f"Input CSV not found: {args.input_csv}")
    
    # Read CSV and prepare prompts
    prompts_data = read_csv_and_prepare_prompts(args.input_csv)
    
    # Slice data based on indices
    start_index = int(len(prompts_data) * args.start_idx)
    end_index = int(len(prompts_data) * args.end_idx)
    prompts_data = prompts_data[start_index:end_index]

    if prompts_data:
        process_all_prompts(prompts_data, args.output_dir)
    else:
        print("No data to process in the specified range.")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Open-Source Data Generation Script using Gemini 2.5 Pro')
    parser.add_argument('--input_csv', type=str, required=True, help='Path to the input CSV file')
    parser.add_argument('--output_dir', type=str, required=True, help='Directory to save the processed output CSVs')
    parser.add_argument('--start_idx', type=float, default=0.0, help='Start index fraction (0.0 to 1.0)')
    parser.add_argument('--end_idx', type=float, default=1.0, help='End index fraction (0.0 to 1.0)')
    parser.add_argument('--api_keys', type=str, help='Comma-separated list of Gemini API keys. Or use GEMINI_API_KEYS env var.')
    
    args = parser.parse_args()
    keys_string = args.api_keys or os.environ.get("GEMINI_API_KEYS", "") or os.environ.get("GEMINI_API_KEY", "")
    api_key_list = [k.strip() for k in keys_string.split(',') if k.strip()]
    
    if not api_key_list:
        raise ValueError("No API keys provided! Please use --api_keys or set the GEMINI_API_KEYS environment variable.")
    
    genai.configure(api_key=api_key_list)
    # ==========================================

    main(args)