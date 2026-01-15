#!/usr/bin/env python3
"""
GRPO training for Catch with local environment.

Usage:
    python examples/scripts/openenv/catch_local.py \
        --model-id Qwen/Qwen2.5-0.5B \
        --dataset-size 50 \
        --max-turns 20

The Catch game:
- 10x5 grid, ball falls from top, paddle at bottom
- Actions: 0=left, 1=stay, 2=right
- Reward: +1 catch, -1 miss
- Episode length: 9 steps (ball falls 9 rows)
"""

import argparse
import re
from datetime import datetime

from datasets import Dataset
from transformers import AutoTokenizer

from trl import GRPOConfig, GRPOTrainer
from trl.experimental.openenv import generate_rollout_completions

from local_catch_env import LocalCatchEnv, OpenSpielAction


def parse_args():
    parser = argparse.ArgumentParser(description="GRPO training for Catch")
    parser.add_argument("--model-id", default="Qwen/Qwen2.5-0.5B")
    parser.add_argument("--dataset-size", type=int, default=100)
    parser.add_argument("--max-turns", type=int, default=15)
    parser.add_argument("--num-generations", type=int, default=4)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=16)
    parser.add_argument("--learning-rate", type=float, default=5e-6)
    parser.add_argument("--vllm-gpu-memory-utilization", type=float, default=0.5)
    parser.add_argument("--save-interval", type=int, default=100)
    parser.add_argument("--temperature", type=float, default=1.0)  # 1.0 minimizes logprob mismatch
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
                    line += "="  # Paddle
                else:
                    line += "O"  # Ball
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


def rollout_once(
    trainer: GRPOTrainer,
    env: LocalCatchEnv,
    tokenizer: AutoTokenizer,
    max_turns: int,
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

        # Generate action
        rollout_output = generate_rollout_completions(trainer, [prompt_text])[0]

        completion_text = tokenizer.decode(
            rollout_output["completion_ids"], skip_special_tokens=True
        )
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
    output_dir = f"outputs/catch-grpo-{timestamp}"

    print("=== Catch GRPO Training ===")
    print(f"Model: {args.model_id}")
    print(f"Dataset size: {args.dataset_size}")
    print(f"Output: {output_dir}")

    # Create environment
    env = LocalCatchEnv()

    # Create dataset (prompts are just placeholders, rollout_func handles actual prompts)
    dataset = Dataset.from_dict({"prompt": ["Play Catch"] * args.dataset_size})

    # Training config
    config = GRPOConfig(
        output_dir=output_dir,
        use_vllm=True,
        vllm_mode="colocate",
        vllm_gpu_memory_utilization=args.vllm_gpu_memory_utilization,
        # vllm_structured_outputs_regex=r"[0-2]",  # Disabled - causes importance sampling issues
        model_init_kwargs={"max_memory": {0: "10GiB", "cpu": "30GiB"}},
        num_train_epochs=1,
        learning_rate=args.learning_rate,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        per_device_train_batch_size=1,
        num_generations=args.num_generations,
        max_completion_length=2,  # Only need 1 token for action
        temperature=args.temperature,
        vllm_importance_sampling_correction=False,  # Disable IS to see if training works (biased update)
        logging_steps=1,
        save_strategy="steps",
        save_steps=args.save_interval,
        report_to="none",
        gradient_checkpointing=True,
    )

    def rollout_func(prompts: list[str], trainer: GRPOTrainer) -> dict[str, list]:
        """Generate rollouts for a batch of prompts."""
        all_prompt_ids = []
        all_completion_ids = []
        all_logprobs = []
        all_rewards = []

        for _ in prompts:
            episode = rollout_once(
                trainer=trainer,
                env=env,
                tokenizer=trainer.processing_class,
                max_turns=args.max_turns,
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

    print(f"\nStarting training...")
    print(f"Episodes per step: {args.num_generations}")
    print(f"Gradient accumulation: {args.gradient_accumulation_steps}")

    try:
        trainer.train()
    finally:
        env.close()

    print("\nTraining complete!")


if __name__ == "__main__":
    main()
