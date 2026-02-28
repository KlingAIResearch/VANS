import pandas as pd
import os
import re
import argparse

def parse_llm_response(response_text):
    """
    Extract the content of each section from the LLM response text.
    Matches the tags defined in the updated English prompt templates.
    """
    # Initialize dictionary to store results
    result = {
        'Think': '',
        'GT_Caption': '',
        'Instruction': ''
    }
    
    # Define regex patterns for loose matching
    patterns = {
        'Think': r'\[Think\](.*?)(?=\[|$)',
        'GT_Caption': r'\[GT Caption\](.*?)(?=\[|$)',
        'Instruction': r'\[Instruction\](.*?)(?=\[|$)'
    }
    
    # Apply each pattern to the response text
    for key, pattern in patterns.items():
        matches = re.findall(pattern, str(response_text), re.DOTALL | re.IGNORECASE)
        if matches:
            # Take the last match (in case the model repeated tags)
            content = matches[-1].strip()
            # Clean up content: remove leading/trailing quotes and spaces
            content = re.sub(r'^["\']+|\s+$', '', content)
            result[key] = content
            
    return result

def process_csv(input_csv_path, instruction_dir, output_csv_path):
    """
    Process the pairs CSV file, extract parsed LLM answers from individual files,
    append the new columns, and drop any rows containing empty values.
    """
    print(f"Loading base pairs CSV from: {input_csv_path}")
    df = pd.read_csv(input_csv_path)
    
    # Initialize the new columns with empty strings
    new_columns = ['Instruction', 'Think', 'GT_Caption']
    for col in new_columns:
        df[col] = ''
        
    rows_to_drop = []
    
    # Process each row
    for index, row in df.iterrows():
        name = row.get('name', '')
        input_id = row.get('input_id', '')
        target_id = row.get('target_id', '')
        
        # Construct the expected individual instruction file path
        file_name = f"{name}_{input_id}_{target_id}.csv"
        instruction_file_path = os.path.join(instruction_dir, file_name)
        
        if input_id != target_id and os.path.exists(instruction_file_path):
            try:
                # Read the individual generated instruction file
                instruction_df = pd.read_csv(instruction_file_path)
                
                # Check if 'llm_response' column exists and is not empty
                if 'llm_response' in instruction_df.columns and len(instruction_df) > 0:
                    llm_response = instruction_df['llm_response'].iloc
                    
                    # Parse the raw response text
                    parsed_data = parse_llm_response(llm_response)
                    
                    # Update the corresponding row in the main DataFrame
                    for col in new_columns:
                        df.at[index, col] = parsed_data[col]
                        
                else:
                    print(f"Warning: {file_name} is missing 'llm_response' column or is empty.")
                    rows_to_drop.append(index)
                    
            except Exception as e:
                print(f"Error processing {file_name}: {str(e)}")
                rows_to_drop.append(index)
        else:
            rows_to_drop.append(index)
            
    # Drop rows that had no corresponding file or encountered processing errors
    original_count = len(df)
    df = df.drop(rows_to_drop).reset_index(drop=True)
    removed_missing_count = len(rows_to_drop)
    
    print(f"\n--- Cleaning Statistics ---")
    print(f"Original row count: {original_count}")
    print(f"Rows dropped (missing file or processing error): {removed_missing_count}")
    
    # Further filter: drop any rows that have empty strings in the newly parsed columns
    before_empty_clean_count = len(df)
    
    # Create a mask for rows where any of the new columns are effectively empty
    empty_mask = df[new_columns].apply(lambda x: x.astype(str).str.strip().eq('')).any(axis=1)
    
    # Keep only rows that do NOT match the empty mask
    df = df[~empty_mask].reset_index(drop=True)
    
    after_clean_count = len(df)
    empty_removed_count = before_empty_clean_count - after_clean_count
    
    print(f"Rows dropped (empty parsed values): {empty_removed_count}")
    print(f"Final valid row count: {after_clean_count}")
    
    # Save the finalized CSV
    df.to_csv(output_csv_path, index=False, encoding='utf-8-sig')
    print(f"\nProcessing complete! Final dataset saved to: {output_csv_path}")
    
    return df

def main():
    parser = argparse.ArgumentParser(description='Parse LLM responses into structured columns and clean the dataset.')
    parser.add_argument('--input_csv', type=str, required=True, help='Path to the base pairs CSV (e.g., COIN_caption_pair.csv)')
    parser.add_argument('--instruction_dir', type=str, required=True, help='Directory containing the LLM generated CSVs')
    parser.add_argument('--output_csv', type=str, required=True, help='Path to save the final merged and cleaned CSV')
    
    args = parser.parse_args()
    
    result_df = process_csv(args.input_csv, args.instruction_dir, args.output_csv)
    
    if not result_df.empty:
        print("\nPreview of the first few rows:")
        preview_cols = ['name', 'input_id', 'target_id', 'Instruction']
        existing_cols = [col for col in preview_cols if col in result_df.columns]
        print(result_df[existing_cols].head())

if __name__ == "__main__":
    main()