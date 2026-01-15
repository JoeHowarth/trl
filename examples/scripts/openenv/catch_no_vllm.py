#!/usr/bin/env python3
"""
GRPO training for Catch WITHOUT vLLM.

Uses the training model directly for generation - slower but correct logprobs.

Usage:
    python examples/scripts/openenv/catch_no_vllm.py \
        --model-id outputs/catch-sft-warmup-*/final \
        --dataset-size 20
"""

import argparse
import re
import torch
from datetime import datetime

from datasets import Dataset
from transformers import AutoTokenizer, AutoModelForCausalLM

from trl import GRPOConfig, GRPOTrainer

from local_catch_env import LocalCatchEnv, OpenSpielAction


def parse_args():
    parser = argparse.ArgumentParser(description="GRPO training for Catch (no vLLM)")
    parser.add_argument("--model-id", default="Qwen/Qwen2.5-0.5B")
    parser.add_argument("--dataset-size", type=int, default=50)
    parser.add_argument("--max-turns", type=int, default=15)
    parser.add_argument("--num-generations", type=int, default=4)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=8)
    parser.add_argument("--learning-rate", type=float, default=5e-6)
    parser.add_argument("--save-interval", type=int, default=100)
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--debug", action="store_true")
    return parser.parse_args()


SYSTEM_PROMPT = """You are playing Catch. A ball falls from the top of a 10x5 grid. You control a paddle at the bottom.

Actions:
- 0 = move left
- 1 = stay
- 2 = move right

Respond with ONLY a single digit: 0, 1, or 2."""


def grid_to_visual(info_state: list[float], rows: int = 10, cols: int = 5) -> str:
    """Convert flat info_state to visual grid."""
    lines = []
    for r in range(rows):
        line = ""
        for c in range(cols):
            val = info_state[r * cols + c]
            if val > 0.5:
                if r == rows - 1:
                    line += "="
                else:
                    line += "O"
            else:
                line += "."
        lines.append(line)
    return "\n".join(lines)


def extract_action(text: str) -> int:
    """Extract action (0, 1, or 2) from model output."""
    match = re.search(r"\b([0-2])\b", text)
    if match:
        return int(match.group(1))
    return 1  # Default to stay


def generate_with_model(model, tokenizer, prompt_text: str, max_new_tokens: int = 4, temperature: float = 0.8):
    """Generate completion using the model directly and return logprobs."""
    device = next(model.parameters()).device

    # Tokenize prompt
    inputs = tokenizer(prompt_text, return_tensors="pt").to(device)
    prompt_ids = inputs.input_ids[0].tolist()

    # Generate with logprobs
    with torch.no_grad():
        outputs = model.generate(
            inputs.input_ids,
            attention_mask=inputs.attention_mask,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            do_sample=True,
            return_dict_in_generate=True,
            output_scores=True,
            pad_token_id=tokenizer.pad_token_id,
        )

    # Extract completion ids
    full_ids = outputs.sequences[0]
    completion_ids = full_ids[len(prompt_ids):].tolist()

    # Compute logprobs from scores
    logprobs = []
    for i, score in enumerate(outputs.scores):
        # Apply temperature scaling (same as training)
        scaled_logits = score[0] / temperature
        log_softmax = torch.log_softmax(scaled_logits, dim=-1)
        token_id = completion_ids[i] if i < len(completion_ids) else 0
        logprobs.append(log_softmax[token_id].item())

    # Decode completion
    completion_text = tokenizer.decode(completion_ids, skip_special_tokens=True)

    return {
        "prompt_ids": prompt_ids,
        "completion_ids": completion_ids,
        "logprobs": logprobs,
        "text": completion_text,
    }


def rollout_once(
    model,
    tokenizer,
    env: LocalCatchEnv,
    max_turns: int,
    temperature: float = 0.8,
    debug: bool = False,
) -> dict:
    """Run one episode and return data for training."""
    result = env.reset()
    obs = result.observation

    all_prompt_ids = []
    all_completion_ids = []
    all_logprobs = []
    total_reward = 0.0
    steps = 0

    while not obs.done and steps < max_turns:
        # Create prompt with current state
        grid_visual = grid_to_visual(obs.info_state)
        user_content = f"Current grid:\n{grid_visual}\n\nYour action (0=left, 1=stay, 2=right):"

        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_content},
        ]
        prompt_text = tokenizer.apply_chat_template(
            messages, add_generation_prompt=True, tokenize=False
        )

        # Generate action using model directly
        rollout_output = generate_with_model(model, tokenizer, prompt_text, temperature=temperature)

        completion_text = rollout_output["text"]
        action = extract_action(completion_text)

        if debug:
            print(f"Step {steps + 1}: output='{completion_text.strip()}' -> action={action}")

        # Store for training (accumulate all steps)
        all_prompt_ids.extend(rollout_output["prompt_ids"])
        all_completion_ids.extend(rollout_output["completion_ids"])
        all_logprobs.extend(rollout_output["logprobs"])

        # Step environment
        result = env.step(OpenSpielAction(action_id=action))
        obs = result.observation
        if result.reward is not None:
            total_reward += result.reward
        steps += 1

    outcome = "CATCH" if total_reward > 0 else "MISS"
    print(f"Episode: {steps} steps, reward={total_reward:.1f} ({outcome})")

    return {
        "prompt_ids": all_prompt_ids,
        "completion_ids": all_completion_ids,
        "logprobs": all_logprobs,
        "env_reward": total_reward,
    }


def reward_from_env(completions, **kwargs):
    """Reward function that returns environment rewards."""
    rewards = kwargs.get("env_reward", [])
    if rewards:
        return [float(r) for r in rewards]
    return [0.0] * len(completions)


def main():
    args = parse_args()

    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    output_dir = f"outputs/catch-grpo-novllm-{timestamp}"

    print("=== Catch GRPO Training (No vLLM) ===")
    print(f"Model: {args.model_id}")
    print(f"Dataset size: {args.dataset_size}")
    print(f"Output: {output_dir}")

    # Create environment
    env = LocalCatchEnv()

    # Load tokenizer for rollouts
    tokenizer = AutoTokenizer.from_pretrained(args.model_id)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Create dataset
    dataset = Dataset.from_dict({"prompt": ["Play Catch"] * args.dataset_size})

    # Training config - NO VLLM
    config = GRPOConfig(
        output_dir=output_dir,
        use_vllm=False,  # Disable vLLM!
        num_train_epochs=1,
        learning_rate=args.learning_rate,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        per_device_train_batch_size=1,
        num_generations=args.num_generations,
        max_completion_length=4,
        temperature=args.temperature,
        logging_steps=1,
        save_strategy="steps",
        save_steps=args.save_interval,
        report_to="none",
        gradient_checkpointing=True,
    )

    # We need access to the model for rollouts, so we load it separately
    # The trainer will also load it, but we need it before trainer.train()
    print("Loading model for rollouts...")
    rollout_model = AutoModelForCausalLM.from_pretrained(
        args.model_id,
        torch_dtype=torch.bfloat16,
        device_map="auto",
    )
    rollout_model.eval()

    def rollout_func(prompts: list[str], trainer: GRPOTrainer) -> dict[str, list]:
        """Generate rollouts using the model directly."""
        all_prompt_ids = []
        all_completion_ids = []
        all_logprobs = []
        all_rewards = []

        # Use trainer's model if available, otherwise use our loaded model
        model = trainer.model if hasattr(trainer, 'model') and trainer.model is not None else rollout_model

        for _ in prompts:
            episode = rollout_once(
                model=model,
                tokenizer=tokenizer,
                env=env,
                max_turns=args.max_turns,
                temperature=args.temperature,
                debug=args.debug,
            )
            all_prompt_ids.append(episode["prompt_ids"])
            all_completion_ids.append(episode["completion_ids"])
            all_logprobs.append(episode["logprobs"])
            all_rewards.append(episode["env_reward"])

        return {
            "prompt_ids": all_prompt_ids,
            "completion_ids": all_completion_ids,
            "logprobs": all_logprobs,
            "env_reward": all_rewards,
        }

    trainer = GRPOTrainer(
        model=args.model_id,
        reward_funcs=[reward_from_env],
        train_dataset=dataset,
        args=config,
        rollout_func=rollout_func,
    )

    print(f"\nStarting training (no vLLM - using model directly)...")
    print(f"Episodes per step: {args.num_generations}")
    print(f"Gradient accumulation: {args.gradient_accumulation_steps}")

    try:
        trainer.train()
    finally:
        env.close()

    print("\nTraining complete!")


if __name__ == "__main__":
    main()
