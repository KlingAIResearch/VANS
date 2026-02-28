import argparse
import pandas as pd
import numpy as np

def process_data_pairs(df):
    """
    Process the data and build pairs of adjacent steps where video_path is not null.
    
    Args:
        df (pd.DataFrame): The input dataframe containing parsed COIN data.
        
    Returns:
        list: A list of dictionaries, where each dictionary represents a data pair.
    """
    # Group by video name
    grouped = df.groupby('name')
    
    pairs = []
    
    for name, group in grouped:
        # Get recipe_type
        recipe_type = group['recipe_type'].iloc
        
        # Filter rows where video_path is not null
        valid_rows = group[group['video_path'].notna()].copy()
        valid_rows = valid_rows.sort_values('id')
        
        # If there are fewer than 2 valid rows, we cannot form a pair, so skip
        if len(valid_rows) < 2:
            continue
        
        # Get the complete list of IDs (including those with empty video_paths)
        all_ids = group.sort_values('id')['id'].tolist()
        valid_ids = valid_rows['id'].tolist()
        
        # Build adjacent valid ID pairs
        for i in range(len(valid_ids) - 1):
            current_id = valid_ids[i]
            next_id = valid_ids[i + 1]
            
            # Get information for current and target rows
            current_row = valid_rows[valid_rows['id'] == current_id].iloc
            target_row = valid_rows[valid_rows['id'] == next_id].iloc
            
            # Identify skipped IDs in between
            skipped_ids = []
            skipped_info = []
            
            # Find the indices of current_id and next_id in the complete list
            start_index = all_ids.index(current_id)
            end_index = all_ids.index(next_id)
            
            # If there are elements between them, they were skipped
            if end_index - start_index > 1:
                skipped_ids = all_ids[start_index + 1:end_index]
                # Get the sentence/description for the skipped IDs
                for skipped_id in skipped_ids:
                    skipped_row = group[group['id'] == skipped_id]
                    if not skipped_row.empty:
                        skipped_info.append({
                            'id': skipped_id,
                            'sentence': skipped_row['sentence'].iloc,
                        })
            
            # Construct the data pair
            pair = {
                'name': name,
                'recipe_type': recipe_type,
                'input_id': current_id,
                'input_video_path': current_row['video_path'],
                'input_sentence': current_row['sentence'],
                'input_dense_caption': current_row['dense_caption'] if pd.notna(current_row['dense_caption']) else '',
                'target_id': next_id,
                'target_video_path': target_row['video_path'],
                'target_sentence': target_row['sentence'],
                'target_dense_caption': target_row['dense_caption'] if pd.notna(target_row['dense_caption']) else '',
                'skipped_ids': skipped_ids,
                'skipped_info': skipped_info
            }
            pairs.append(pair)
    
    return pairs

def main():
    parser = argparse.ArgumentParser(description="Generate input-target sequence pairs from the master CSV.")
    parser.add_argument("--input_csv", type=str, required=True, help="Path to the input COIN_caption.csv.")
    parser.add_argument("--output_csv", type=str, required=True, help="Path to save the generated COIN_caption_pair.csv.")
    
    args = parser.parse_args()
    
    # Read the CSV file
    print(f"Loading data from {args.input_csv}...")
    df = pd.read_csv(args.input_csv)
    
    # Process data pairs
    print("Processing data pairs and identifying skipped steps...")
    data_pairs = process_data_pairs(df)
    
    # Convert to DataFrame for viewing and saving
    pairs_df = pd.DataFrame(data_pairs)
    
    # Print results summary
    print(f"\nSuccessfully constructed {len(pairs_df)} data pairs.")
    print("\nPreview of the first few data pairs:")
    print(pairs_df.head().to_string())
    
    # Save to CSV
    pairs_df.to_csv(args.output_csv, index=False, encoding='utf-8-sig')
    print(f"\nPairs saved successfully to: {args.output_csv}")

if __name__ == "__main__":
    main()