#!/usr/bin/env python3
"""
SFT warmup for Catch - teach model to output single digits 0, 1, 2.

This primes the model so GRPO can work with structured outputs.

Usage:
    python examples/scripts/openenv/catch_sft_warmup.py --num-examples 500

Then run GRPO:
    python examples/scripts/openenv/catch_local.py --model-id outputs/catch-sft-warmup-*/final
"""

import argparse
import random
from datetime import datetime
from pathlib import Path

from datasets import Dataset
from transformers import AutoTokenizer
from trl import SFTTrainer, SFTConfig

from local_catch_env import LocalCatchEnv


def parse_args():
    parser = argparse.ArgumentParser(description="SFT warmup for Catch")
    parser.add_argument("--model-id", default="Qwen/Qwen2.5-0.5B")
    parser.add_argument("--num-examples", type=int, default=500)
    parser.add_argument("--num-epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--learning-rate", type=float, default=2e-5)
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--random-actions", action="store_true",
                        help="Use random actions instead of optimal (teaches format only)")
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


def get_optimal_action(ball_col: int, paddle_col: int) -> int:
    """Get the optimal action to catch the ball."""
    if paddle_col < ball_col:
        return 2  # Move right
    elif paddle_col > ball_col:
        return 0  # Move left
    else:
        return 1  # Stay


def generate_sft_examples(tokenizer, num_examples: int, random_actions: bool = False) -> Dataset:
    """Generate supervised examples: (board_state, action).

    If random_actions=True, uses random actions (teaches format only).
    If random_actions=False, uses optimal actions (teaches optimal policy).
    """
    env = LocalCatchEnv()
    examples = []

    mode = "random" if random_actions else "optimal"
    print(f"Generating {num_examples} SFT examples with {mode} actions...", flush=True)

    while len(examples) < num_examples:
        result = env.reset()
        obs = result.observation

        # Play through the episode, collecting examples
        while not obs.done:
            # Get current state
            grid_visual = grid_to_visual(obs.info_state)

            # Get action (random or optimal)
            if random_actions:
                action = random.randint(0, 2)
            else:
                action = get_optimal_action(env.ball_col, env.paddle_col)

            # Create prompt
            user_content = f"Current grid:\n{grid_visual}\n\nYour action (0=left, 1=stay, 2=right):"
            messages = [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_content},
            ]

            prompt_text = tokenizer.apply_chat_template(
                messages, add_generation_prompt=True, tokenize=False
            )

            # The completion is just the digit
            completion = str(action)

            examples.append({
                "text": prompt_text + completion,
            })

            if len(examples) >= num_examples:
                break

            # Take the action to continue episode
            from local_catch_env import OpenSpielAction
            result = env.step(OpenSpielAction(action_id=action))
            obs = result.observation

    # Show action distribution
    actions = [int(ex["text"][-1]) for ex in examples]
    print(f"Action distribution: 0={actions.count(0)}, 1={actions.count(1)}, 2={actions.count(2)}")

    return Dataset.from_list(examples)


def main():
    args = parse_args()

    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    output_dir = args.output_dir or f"outputs/catch-sft-warmup-{timestamp}"

    print("=== Catch SFT Warmup ===")
    print(f"Model: {args.model_id}")
    print(f"Examples: {args.num_examples}")
    print(f"Random actions: {args.random_actions} (teaches format only)" if args.random_actions else f"Optimal actions (teaches policy)")
    print(f"Output: {output_dir}")

    # Load tokenizer
    print("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(args.model_id)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Generate dataset
    dataset = generate_sft_examples(tokenizer, args.num_examples, args.random_actions)

    # Configure SFT
    sft_config = SFTConfig(
        output_dir=output_dir,
        num_train_epochs=args.num_epochs,
        per_device_train_batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        logging_steps=10,
        save_strategy="no",  # Don't checkpoint during training
        report_to="none",
        bf16=True,
        gradient_checkpointing=True,
    )

    # Create trainer
    trainer = SFTTrainer(
        model=args.model_id,
        args=sft_config,
        train_dataset=dataset,
    )

    print(f"\n=== Starting SFT Training ===")
    print(f"Dataset size: {len(dataset)}")
    print(f"Epochs: {args.num_epochs}")
    print(f"Batch size: {args.batch_size}")

    # Train
    trainer.train()

    # Save final model
    final_path = Path(output_dir) / "final"
    trainer.save_model(str(final_path))
    tokenizer.save_pretrained(str(final_path))

    print(f"\n=== Training Complete ===")
    print(f"Model saved to: {final_path}")
    print(f"\nNext step - run GRPO with warmed model:")
    print(f"  python examples/scripts/openenv/catch_local.py \\")
    print(f"      --model-id {final_path} \\")
    print(f"      --dataset-size 100")


if __name__ == "__main__":
    main()
