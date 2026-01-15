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

The example has been modified to work locally with limited GPU memory:

```bash
source venv/bin/activate

python examples/scripts/openenv/sudoku.py \
    --vllm-mode colocate \
    --model-id Qwen/Qwen2.5-0.5B \
    --dataset-size 5 \
    --max-turns 15 \
    --difficulty easy \
    --num-generations 2 \
    --gradient-accumulation-steps 4 \
    --vllm-gpu-memory-utilization 0.5
```

## Results & Issues Encountered

### Initial GPU Memory Problem

**Issue:** With `device_map="auto"`, the training model consumed all GPU memory before vLLM could allocate its KV cache, causing:
```
ValueError: To serve at least one request with the models's max seq len (40960),
4.38 GiB KV cache is needed, which is larger than the available KV cache memory (0.85 GiB)
```

**Solution:** Added `max_memory` constraint in `sudoku.py`:
```python
model_init_kwargs={"max_memory": {0: "10GiB", "cpu": "30GiB"}}
```

### Poor Training Results

Our training run completed but the model performed poorly:

```
Step 1:
Step 2:
Episode: empty_cell=0.00, valid=-0.50, repetition=-0.50, progress=0.00 (0 cells), correct=0.00
```

**Key metrics:**
- **Cells filled:** 0 (no valid Sudoku moves made)
- **Valid moves reward:** -0.5 (penalty for invalid moves)
- **Progress reward:** 0.0
- **Total reward:** -1.0

The model generated empty outputs instead of valid moves in `[row col number]` format.

### Why It Struggled

1. **Tiny model:** Qwen2.5-0.5B (494M parameters) is very small for complex reasoning tasks like Sudoku
2. **Minimal training:** Only 2 epochs with 5 samples each (~20 seconds total)
3. **Cold start:** The model needs to learn both:
   - The output format `[row col number]`
   - Sudoku rules and valid move logic
   - Strategic solving approaches

### Recommendations for Better Results

To achieve meaningful learning on this task:

```bash
# Use a larger model
python examples/scripts/openenv/sudoku.py \
    --model-id Qwen/Qwen3-1.7B \
    --dataset-size 100 \
    --max-turns 50 \
    --difficulty easy \
    --num-generations 4 \
    --gradient-accumulation-steps 16
```

**Suggested changes:**
- **Model size:** Use Qwen3-1.7B or larger (requires ~3-4GB GPU memory)
- **Dataset size:** 100+ episodes for meaningful gradient updates
- **Max turns:** 50+ to allow full puzzle attempts
- **Training time:** Expect several hours for convergence

### Hardware Requirements

- **Minimum:** RTX 3090 (24GB VRAM) or equivalent
- **Tested on:** RTX 3090
- **Model + vLLM memory split:** ~10GB for model, ~14GB for vLLM KV cache

## Changes Made to Original Example

1. **Line 736:** Added memory constraints to prevent OOM
2. **Line 753:** Changed `report_to="trackio"` to `report_to="none"` for offline use

These allow running without HuggingFace authentication while managing GPU memory properly.
