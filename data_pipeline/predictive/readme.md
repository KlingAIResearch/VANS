# Predictive Pipeline

The predictive pipeline evaluates two temporally adjacent clips. It forces the MLLM to learn the logical leap from `Video 1` to `Video 2` *without* explicitly revealing what happens in `Video 2`, thereby generating highly deterministic, reasoning-heavy generation instructions.

## 🚀 How It Works (Generate & Verify)

Instead of a single-pass generation, this workflow uses a two-step agentic approach:

1. **Generation (`step5_1.py`)**: The Vision model watches both videos, assesses their visual/logical consistency, and drafts a raw predictive instruction and reasoning process.
2. **Verification & Translation (`step5_2.py`)**: A strict language model reviews the draft. It actively penalizes "information leaks" (e.g., explicitly mentioning Video 2), fixes weak logic, and outputs a clean, bilingual (English & Chinese) dataset.

---

## 🛠️ Usage Instructions

### Step 1: Generate Raw Instructions
First, generate the initial drafts using your sequence pair CSV (which should contain `video1_path` and `video2_path`).

```bash
export GEMINI_API_KEYS="your_api_key_1,your_api_key_2"

python step5_1.py \
    --input_csv ./data/predictive_pairs.csv \
    --output_dir ./results/predictive_raw \
    --api_keys $GEMINI_API_KEYS
```

### Step 2: Strict Review & Rewrite
Next, run the generated texts through the rigorous review agent to rewrite them into high-quality, bilingual formats.

```bash
python step5_2.py \
    --input_dir ./results/predictive_raw \
    --output_dir ./results/predictive_final \
    --api_keys $GEMINI_API_KEYS \
    --start_idx 0.0 \
    --end_idx 1.0
```

Once completed, `./results/predictive_final` will contain your validated, bilingual predictive instruction data!