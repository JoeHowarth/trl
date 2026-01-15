# OpenEnv: GRPO Training for Interactive Environments

Training language models with GRPO (Group Relative Policy Optimization) on interactive environments like games.

## Goal

Train a small base model with RL to play games, ultimately targeting a simplified Rimworld-like game where the model assigns tasks and pawn work priorities. Using Catch as a simpler proxy to validate the pipeline.

## Current Status

**Working pipeline established** with the following configuration:

```python
GRPOConfig(
    use_vllm=True,
    vllm_mode="colocate",
    vllm_importance_sampling_correction=False,  # Key fix for logprob mismatch
    temperature=1.0,  # Minimizes logprob divergence
    ...
)
```

## Key Findings

### vLLM + GRPO Logprob Mismatch Issue

When using vLLM for generation, there's a fundamental mismatch between:
- vLLM's logprobs (from inference model)
- Training model's recomputed logprobs

This causes importance sampling ratios near zero (~1e-27), resulting in no gradients.

**Solutions explored:**
1. Use supported vLLM version (0.10.2-0.12.0) - helps but doesn't fully solve
2. Set `temperature=1.0` - minimizes but doesn't eliminate mismatch
3. Set `vllm_importance_sampling_correction=False` - **works!** Gradients flow (grad_norm: 20-50)

### SFT Warmup is Essential

Without SFT warmup:
- Base model outputs garbage text ("Current grid", "Hello user", etc.)
- Garbage parses to default action
- Gradients flow but model doesn't learn meaningful behavior

With SFT warmup on **optimal actions**:
- Model achieves 99% catch rate immediately
- No room for RL improvement

With SFT warmup on **random actions** (format-only):
- Model outputs proper digits (0, 1, 2)
- Random policy that GRPO can improve
- **This is the right starting point for RL**

## Files

- `local_catch_env.py` - Local Catch environment (no external dependencies)
- `catch_sft_warmup.py` - SFT warmup script with `--random-actions` flag
- `catch_local.py` - GRPO training with vLLM colocate mode
- `catch_no_vllm.py` - GRPO without vLLM (doesn't work - trainer ignores rollout_func)

## Usage

```bash
# Step 1: SFT warmup with random actions (teaches format only)
python catch_sft_warmup.py --random-actions --num-examples 200 --num-epochs 1

# Step 2: GRPO training
python catch_local.py \
    --model-id outputs/catch-sft-warmup-*/final \
    --dataset-size 300 \
    --learning-rate 2e-5 \
    --gradient-accumulation-steps 4
```

## Next Steps

1. Run longer GRPO training to see reward improvement (needs more disk space)
2. Experiment with learning rates and batch sizes
3. Once Catch is solved, build proto-Rimworld environment

## References

- TRL vLLM integration: https://huggingface.co/docs/trl/main/en/vllm_integration
- GRPO Trainer: https://huggingface.co/docs/trl/main/en/grpo_trainer
- vLLM importance sampling: https://github.com/huggingface/trl/issues/4159
