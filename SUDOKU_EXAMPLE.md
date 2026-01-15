# Running the Sudoku GRPO Example Locally

This guide explains how to run the TRL sudoku example with GRPO (Group Relative Policy Optimization) locally, without requiring cloud resources or HuggingFace authentication.

## Setup

```bash
# Clone the repository
git clone https://github.com/JoeHowarth/trl.git
cd trl

# Create and activate virtual environment
python3 -m venv venv
source venv/bin/activate

# Install dependencies
pip install --upgrade pip
pip install -e .
pip install git+https://huggingface.co/spaces/openenv/sudoku
pip install vllm trackio
```

## Running the Example

### Option 1: Local Environment (Recommended)

Use the local Sudoku environment - no external services needed:

```bash
source venv/bin/activate

python examples/scripts/openenv/sudoku.py \
    --env-mode local \
    --vllm-mode colocate \
    --model-id Qwen/Qwen2.5-0.5B \
    --dataset-size 20 \
    --max-turns 30 \
    --difficulty easy \
    --num-generations 2 \
    --gradient-accumulation-steps 4 \
    --vllm-gpu-memory-utilization 0.5 \
    --save-interval 1000
```

### Option 2: HuggingFace Space (may be unreliable)

```bash
python examples/scripts/openenv/sudoku.py \
    --env-mode space \
    --vllm-mode colocate \
    --model-id Qwen/Qwen2.5-0.5B \
    --dataset-size 5 \
    --max-turns 15 \
    --difficulty easy
```

## SFT Warmup (Recommended)

Before GRPO training, run supervised fine-tuning to teach the model the output format:

```bash
# Step 1: SFT warmup (generates local puzzles, trains model)
python examples/scripts/openenv/sudoku_sft_warmup.py \
    --model-id Qwen/Qwen2.5-0.5B \
    --num-examples 200 \
    --num-epochs 2 \
    --batch-size 2

# Step 2: GRPO with warmed model
python examples/scripts/openenv/sudoku.py \
    --env-mode local \
    --model-id outputs/sudoku-sft-warmup-TIMESTAMP/final \
    --vllm-mode colocate \
    --dataset-size 20 \
    --max-turns 30 \
    --difficulty easy
```

**SFT warmup benefits:**
- Model learns output format with 100% accuracy in ~1 minute
- Eliminates "cold start" problem
- GRPO can focus on learning Sudoku strategy

## Results & Issues Encountered

### Initial GPU Memory Problem

**Issue:** With `device_map="auto"`, the training model consumed all GPU memory before vLLM could allocate its KV cache.

**Solution:** Added `max_memory` constraint in `sudoku.py`:
```python
model_init_kwargs={"max_memory": {0: "10GiB", "cpu": "30GiB"}}
```

### Training Results

#### Before Structured Output (Empty Outputs)

```
Step 1:
Step 2:
Episode: empty_cell=0.00, valid=-0.50, repetition=-0.50, progress=0.00 (0 cells), correct=0.00
```

The model generated empty outputs instead of valid moves.

#### After Structured Output (Valid Format)

```
Step 1: [1 2 3]
Step 2: [5 3 7]
Episode: empty_cell=0.00, valid=-0.50, repetition=0.00, progress=0.00 (0 cells), correct=0.00
```

Every output is now a valid `[row col number]` format.

#### With Local Environment (Actual Progress!)

```
Step 1: [5 3 7]
...
Episode: empty_cell=1.00, valid=-0.50, repetition=-7.93, progress=0.00 (0 cells), correct=-0.10
Episode: empty_cell=0.87, valid=-0.45, repetition=-7.20, progress=0.02 (1 cells), correct=0.10
```

**Key improvements with local env:**
- `empty_cell=1.00` - model correctly targeting empty cells
- `progress=0.02 (1 cells)` - actually filling cells correctly!
- Real reward signal for learning

### Why It Struggled

1. **Tiny model:** Qwen2.5-0.5B (494M parameters) is very small for complex reasoning
2. **Minimal training:** Only 2 epochs with 5 samples
3. **Cold start:** Model needs to learn format + Sudoku rules + strategy

### Hardware Requirements

- **Minimum:** RTX 3090 (24GB VRAM) or equivalent
- **Tested on:** RTX 3090
- **Model + vLLM memory split:** ~10GB for model, ~14GB for vLLM KV cache

## Changes Made to Original Example

1. **`vllm_structured_outputs_regex`** - Force valid move format with regex
2. **`max_memory` constraint** - Prevent OOM by limiting training model memory
3. **`report_to="none"`** - Disable cloud logging for offline use
4. **`--env-mode local`** - New local Sudoku environment option
5. **`sudoku_sft_warmup.py`** - New SFT warmup script

### Structured Output (Key Improvement)

The regex constraint `\[[1-9] [1-9] [1-9]\]` forces valid moves:

```python
vllm_structured_outputs_regex=r"\[[1-9] [1-9] [1-9]\]"
```

**Benefits:**
- **No empty outputs** - model must produce tokens matching the pattern
- **No malformed moves** - only valid `[row col number]` format allowed
- **Reduced learning burden** - model learns WHICH digits, not format
- **Constrained output space** - only 729 valid moves (9³) possible

### Local Environment

The `--env-mode local` option uses `local_sudoku_env.py`:
- Generates Sudoku puzzles locally
- No Docker or external services needed
- Proper reward signal for valid/invalid moves
- Works offline

## Files Added/Modified

| File | Description |
|------|-------------|
| `sudoku.py` | Added structured output regex, memory constraints, local env support |
| `sudoku_sft_warmup.py` | New script for supervised fine-tuning warmup |
| `local_sudoku_env.py` | Local Sudoku environment (no external services) |
| `SUDOKU_EXAMPLE.md` | This documentation |
