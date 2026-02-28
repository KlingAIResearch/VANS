# VAND-Data Pipeline

This repository provides the VANS-DATA pipeline. It processes raw long videos, leverages Gemini Vision models to generate dense captions, constructs adjacent action pairs, and finally prompts the Gemini LLM to generate structured instructional Q&A pairs.

Take the **COIN dataset** as an example, you can download the vanilla videos and annotations via [this link](https://github.com/coin-dataset/annotations.git).

---

## 🛠️ Prerequisites

1. **System Requirements (FFmpeg)**: 
   Ensure `ffmpeg` is installed on your system for fast, lossless video splitting.
   ```bash
   # Ubuntu/Debian
   sudo apt update && sudo apt install ffmpeg  
   
   # MacOS
   brew install ffmpeg      
   ```

2. **Python Dependencies**:
   Install the required Python packages:
   ```bash
   pip install pandas tqdm pillow google-generativeai
   ```

---

## 📁 Data Preparation

Please organize your working directory as follows before running the pipeline:

- Original data file: `./data/COIN.json` (or `COIN_OK.json`)
- Recipe mapping file: `./data/label_foodtype_coin.csv`
- Original video folder: `./data/videos/`

---

## 🚀 Pipeline Steps

### Step 1: Split Videos
Split the full-length videos into short clips according to the ground-truth steps provided in the JSON.

```bash
python step1.py \
    --json_path ./data/COIN.json \
    --video_dir ./data/videos \
    --output_dir ./data/processed_videos \
    --num_processes 8
```

### Step 2: Generate Dense Captions
Check if each split is good as data and generate its dense visual description using the Gemini Vision API. 

*Note: You can pass multiple API keys separated by commas for load balancing.*

```bash
export GEMINI_API_KEYS="AIzaSyBd_gg5M-VMNAz...,AIzaSyBWrfGXqN29v..."

python step2.py \
    --json_path ./data/COIN.json \
    --input_video_dir ./data/processed_videos \
    --output_txt_dir ./data/processed_videos_final_caption \
    --start_idx 0.0 \
    --end_idx 1.0
```

### Step 3: Merge Captions into CSV
Merge the generated dense captions and the original annotations into a single structured CSV file.

```bash
python step3.py \
    --json_path ./data/COIN.json \
    --output_csv ./data/COIN_caption.csv \
    --txt_folder ./data/processed_videos_final_caption \
    --recipe_csv ./data/label_foodtype_coin.csv
```

### Step 4: Construct Sequence Pairs
Convert the merged CSV into a sequential pair CSV (Input Step -> Target Step) to prepare for instruction generation.

```bash
python step4.py \
    --input_csv ./data/COIN_caption.csv \
    --output_csv ./data/COIN_caption_pair.csv
```

### Step 5: Generate Instructions
Feed the sequence pairs into the Gemini LLM to reason about the user's progress and generate structured instructional Q&A pairs. 

*Note: This step automatically utilizes the same `GEMINI_API_KEYS` environment variable you configured in Step 2.*

```bash
python step5.py \
    --input_csv ./data/COIN_caption_pair.csv \
    --output_dir ./data/processed_videos_instruction_OK \
    --start_idx 0.0 \
    --end_idx 1.0
```

### Step 6: Collect Final Dataset
Parse the LLM outputs, clean the data (drop empty rows), and compile everything into the final instruction tuning dataset.

```bash
python step6.py \
    --input_csv ./data/COIN_caption_pair.csv \
    --instruction_dir ./data/processed_videos_instruction_OK \
    --output_csv ./data/COIN_final_dataset_OK.csv
```

🎉 **Congratulations!** Your dataset is now ready at `./data/COIN_final_dataset_OK.csv`.