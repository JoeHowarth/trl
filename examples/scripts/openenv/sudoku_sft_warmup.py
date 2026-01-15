#!/usr/bin/env python3
"""
Supervised Fine-Tuning warmup for Sudoku.

Generates a dataset of (board_state, correct_move) pairs and trains the model
to output correct moves before running GRPO reinforcement learning.

This version generates puzzles LOCALLY without needing the remote environment.

Usage:
    python examples/scripts/openenv/sudoku_sft_warmup.py --num-examples 500

Then run GRPO with the warmed model:
    python examples/scripts/openenv/sudoku.py --model-id ./outputs/sudoku-sft-warmup/final ...
"""

import argparse
import random
import sys
from datetime import datetime
from pathlib import Path

from datasets import Dataset
from transformers import AutoTokenizer
from trl import SFTTrainer, SFTConfig

# Import helper functions from sudoku.py
sys.path.insert(0, str(Path(__file__).parent))
from sudoku import (
    resolve_system_prompt,
    get_valid_numbers,
    make_compact_prompt,
)


def parse_args():
    parser = argparse.ArgumentParser(description="SFT warmup for Sudoku")
    parser.add_argument("--model-id", default="Qwen/Qwen3-1.7B")
    parser.add_argument("--num-examples", type=int, default=500)
    parser.add_argument("--num-epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--learning-rate", type=float, default=2e-5)
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--system-prompt-path", default="sudoku_prompt.txt")
    return parser.parse_args()


# Simple Sudoku generator
def generate_complete_sudoku() -> list[list[int]]:
    """Generate a complete valid Sudoku grid using backtracking."""
    grid = [[0] * 9 for _ in range(9)]

    def is_valid(grid, row, col, num):
        # Check row
        if num in grid[row]:
            return False
        # Check column
        if num in [grid[r][col] for r in range(9)]:
            return False
        # Check 3x3 box
        box_row, box_col = 3 * (row // 3), 3 * (col // 3)
        for r in range(box_row, box_row + 3):
            for c in range(box_col, box_col + 3):
                if grid[r][c] == num:
                    return False
        return True

    def solve(grid):
        for row in range(9):
            for col in range(9):
                if grid[row][col] == 0:
                    nums = list(range(1, 10))
                    random.shuffle(nums)
                    for num in nums:
                        if is_valid(grid, row, col, num):
                            grid[row][col] = num
                            if solve(grid):
                                return True
                            grid[row][col] = 0
                    return False
        return True

    solve(grid)
    return grid


def create_puzzle(solution: list[list[int]], num_clues: int = 30) -> list[list[int]]:
    """Create a puzzle by removing numbers from a complete solution."""
    puzzle = [row[:] for row in solution]
    cells = [(r, c) for r in range(9) for c in range(9)]
    random.shuffle(cells)

    cells_to_remove = 81 - num_clues
    for r, c in cells[:cells_to_remove]:
        puzzle[r][c] = 0

    return puzzle


def grid_to_board_string(grid: list[list[int]]) -> str:
    """Convert a 9x9 grid to the board string format used in prompts."""
    lines = []
    lines.append("   C1 C2 C3   C4 C5 C6   C7 C8 C9  ")
    for row_idx in range(9):
        row_label = f"R{row_idx + 1}"
        cells = []
        for col_idx in range(9):
            val = grid[row_idx][col_idx]
            cells.append(str(val) if val != 0 else ".")
        # Format with separators
        line = f"{row_label}  {cells[0]}  {cells[1]}  {cells[2]} | {cells[3]}  {cells[4]}  {cells[5]} | {cells[6]}  {cells[7]}  {cells[8]}"
        lines.append(line)
        if row_idx in [2, 5]:
            lines.append("   - - - - - - - - - - - - - - - - ")
    return "\n".join(lines)


def find_cells_with_candidates(grid: list[list[int]]) -> list[tuple[int, int, set[int]]]:
    """Find empty cells with their valid candidate numbers."""
    cells = []
    for row in range(9):
        for col in range(9):
            if grid[row][col] == 0:
                candidates = get_valid_numbers(grid, row, col)
                cells.append((row + 1, col + 1, candidates))  # 1-indexed
    # Sort by number of candidates (easiest first)
    cells.sort(key=lambda x: len(x[2]))
    return cells


def generate_sft_dataset(
    tokenizer: AutoTokenizer,
    system_prompt: str,
    num_examples: int,
) -> Dataset:
    """Generate supervised training examples from locally generated Sudoku puzzles."""

    examples = []
    puzzles_generated = 0

    print(f"Generating {num_examples} SFT examples locally...", flush=True)

    while len(examples) < num_examples:
        # Generate a puzzle
        solution = generate_complete_sudoku()
        # More clues = easier puzzle = more naked singles
        num_clues = random.randint(35, 45)
        puzzle = create_puzzle(solution, num_clues)
        puzzles_generated += 1

        # Convert to board string
        board_string = grid_to_board_string(puzzle)

        # Find cells with candidates
        cells = find_cells_with_candidates(puzzle)

        # Extract examples from this puzzle
        examples_from_puzzle = 0
        for row, col, candidates in cells:
            if examples_from_puzzle >= 5:  # Max 5 per puzzle
                break
            if len(examples) >= num_examples:
                break

            # Only use cells with 1-3 candidates
            if len(candidates) == 0 or len(candidates) > 3:
                continue

            # Get the correct answer from the solution
            correct_num = solution[row - 1][col - 1]  # Convert back to 0-indexed

            # Create the prompt
            user_prompt = make_compact_prompt(
                board=board_string,
                step=1,
                successful_moves=[],
                failed_moves=[],
                difficulty="easy",
            )

            # Format as chat
            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ]

            prompt_text = tokenizer.apply_chat_template(
                messages,
                add_generation_prompt=True,
                tokenize=False,
                enable_thinking=False,
            )

            # The correct completion
            completion = f"[{row} {col} {correct_num}]"

            examples.append({
                "text": prompt_text + completion,
            })
            examples_from_puzzle += 1

        if puzzles_generated % 20 == 0:
            print(f"  Puzzles: {puzzles_generated}, Examples: {len(examples)}", flush=True)

    print(f"Generated {len(examples)} examples from {puzzles_generated} puzzles", flush=True)

    # Show some examples
    print("\nSample completions:", flush=True)
    for i, ex in enumerate(examples[:5]):
        # Extract just the completion part
        completion = ex['text'].split("[")[-1]
        print(f"  {i+1}. [{completion}", flush=True)

    return Dataset.from_list(examples)


def main():
    args = parse_args()

    # Setup
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    output_dir = args.output_dir or f"outputs/sudoku-sft-warmup-{timestamp}"

    print(f"=== Sudoku SFT Warmup ===", flush=True)
    print(f"Model: {args.model_id}", flush=True)
    print(f"Examples: {args.num_examples}", flush=True)
    print(f"Output: {output_dir}", flush=True)

    # Load tokenizer
    print("Loading tokenizer...", flush=True)
    tokenizer = AutoTokenizer.from_pretrained(args.model_id)
    print("Tokenizer loaded.", flush=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Load system prompt
    print("Loading system prompt...", flush=True)
    system_prompt = resolve_system_prompt(args.system_prompt_path)
    print(f"System prompt loaded ({len(system_prompt)} chars)", flush=True)

    # Generate dataset locally
    dataset = generate_sft_dataset(
        tokenizer=tokenizer,
        system_prompt=system_prompt,
        num_examples=args.num_examples,
    )

    # Configure SFT
    sft_config = SFTConfig(
        output_dir=output_dir,
        num_train_epochs=args.num_epochs,
        per_device_train_batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        logging_steps=10,
        save_strategy="epoch",
        save_total_limit=2,
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

    print(f"\n=== Starting SFT Training ===", flush=True)
    print(f"Dataset size: {len(dataset)}", flush=True)
    print(f"Epochs: {args.num_epochs}", flush=True)
    print(f"Batch size: {args.batch_size}", flush=True)

    # Train
    trainer.train()

    # Save final model
    final_path = Path(output_dir) / "final"
    trainer.save_model(str(final_path))
    tokenizer.save_pretrained(str(final_path))

    print(f"\n=== Training Complete ===", flush=True)
    print(f"Model saved to: {final_path}", flush=True)
    print(f"\nNext step - run GRPO with warmed model:", flush=True)
    print(f"  python examples/scripts/openenv/sudoku.py \\", flush=True)
    print(f"      --model-id {final_path} \\", flush=True)
    print(f"      --vllm-mode colocate \\", flush=True)
    print(f"      --dataset-size 100 \\", flush=True)
    print(f"      --difficulty easy", flush=True)


if __name__ == "__main__":
    main()
