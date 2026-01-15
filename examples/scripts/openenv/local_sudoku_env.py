"""
Local Sudoku environment that doesn't require Docker or external services.
Drop-in replacement for TextArenaEnv for testing GRPO training locally.
"""

import random
from dataclasses import dataclass, field
from typing import Any


@dataclass
class Message:
    content: str
    sender_id: int = -1


@dataclass
class Observation:
    messages: list[Message] = field(default_factory=list)


@dataclass
class StepResult:
    observation: Observation
    reward: float = 0.0
    done: bool = False


@dataclass
class TextArenaAction:
    message: str


def generate_complete_sudoku() -> list[list[int]]:
    """Generate a complete valid Sudoku grid using backtracking."""
    grid = [[0] * 9 for _ in range(9)]

    def is_valid(grid, row, col, num):
        if num in grid[row]:
            return False
        if num in [grid[r][col] for r in range(9)]:
            return False
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


def create_puzzle(solution: list[list[int]], num_clues: int = 35) -> list[list[int]]:
    """Create a puzzle by removing numbers from a complete solution."""
    puzzle = [row[:] for row in solution]
    cells = [(r, c) for r in range(9) for c in range(9)]
    random.shuffle(cells)
    cells_to_remove = 81 - num_clues
    for r, c in cells[:cells_to_remove]:
        puzzle[r][c] = 0
    return puzzle


def grid_to_board_string(grid: list[list[int]]) -> str:
    """Convert grid to the standard board string format."""
    lines = []
    lines.append("   C1 C2 C3   C4 C5 C6   C7 C8 C9  ")
    for row_idx in range(9):
        row_label = f"R{row_idx + 1}"
        cells = []
        for col_idx in range(9):
            val = grid[row_idx][col_idx]
            cells.append(str(val) if val != 0 else ".")
        line = f"{row_label}  {cells[0]}  {cells[1]}  {cells[2]} | {cells[3]}  {cells[4]}  {cells[5]} | {cells[6]}  {cells[7]}  {cells[8]}"
        lines.append(line)
        if row_idx in [2, 5]:
            lines.append("   - - - - - - - - - - - - - - - - ")
    return "\n".join(lines)


class LocalSudokuEnv:
    """Local Sudoku environment - no external services needed."""

    def __init__(self, num_clues: int = 35):
        self.num_clues = num_clues
        self.solution = None
        self.puzzle = None
        self.current_grid = None
        self.moves_made = 0
        self.max_moves = 100

    def reset(self) -> StepResult:
        """Start a new puzzle."""
        self.solution = generate_complete_sudoku()
        self.puzzle = create_puzzle(self.solution, self.num_clues)
        self.current_grid = [row[:] for row in self.puzzle]
        self.moves_made = 0

        board_str = grid_to_board_string(self.current_grid)
        message = f"[GAME] Sudoku puzzle started. Fill in the empty cells.\n\n{board_str}"

        return StepResult(
            observation=Observation(messages=[Message(content=message, sender_id=-1)]),
            reward=0.0,
            done=False,
        )

    def step(self, action: TextArenaAction) -> StepResult:
        """Process a move."""
        self.moves_made += 1
        move = action.message.strip()

        # Parse move [row col num]
        import re
        match = re.search(r"\[(\d)\s*(\d)\s*(\d)\]", move)

        if not match:
            return StepResult(
                observation=Observation(messages=[
                    Message(content="Invalid move format. Use [row col number].", sender_id=-1)
                ]),
                reward=-0.1,
                done=False,
            )

        row, col, num = int(match.group(1)), int(match.group(2)), int(match.group(3))

        # Validate coordinates (1-indexed)
        if not (1 <= row <= 9 and 1 <= col <= 9 and 1 <= num <= 9):
            return StepResult(
                observation=Observation(messages=[
                    Message(content="Invalid coordinates. Row, col, and number must be 1-9.", sender_id=-1)
                ]),
                reward=-0.1,
                done=False,
            )

        # Convert to 0-indexed
        row_idx, col_idx = row - 1, col - 1

        # Check if cell is empty
        if self.puzzle[row_idx][col_idx] != 0:
            board_str = grid_to_board_string(self.current_grid)
            return StepResult(
                observation=Observation(messages=[
                    Message(content=f"Cannot modify pre-filled cell at ({row}, {col}). Please resubmit.\n\n{board_str}", sender_id=-1)
                ]),
                reward=-0.1,
                done=False,
            )

        if self.current_grid[row_idx][col_idx] != 0:
            board_str = grid_to_board_string(self.current_grid)
            return StepResult(
                observation=Observation(messages=[
                    Message(content=f"Cell ({row}, {col}) already filled. Please resubmit.\n\n{board_str}", sender_id=-1)
                ]),
                reward=-0.1,
                done=False,
            )

        # Check if move is correct
        correct_num = self.solution[row_idx][col_idx]
        if num != correct_num:
            board_str = grid_to_board_string(self.current_grid)
            return StepResult(
                observation=Observation(messages=[
                    Message(content=f"Invalid move: {num} violates Sudoku rules at ({row}, {col}). Please resubmit to avoid penalties.\n\n{board_str}", sender_id=-1)
                ]),
                reward=-0.1,
                done=False,
            )

        # Valid move!
        self.current_grid[row_idx][col_idx] = num
        board_str = grid_to_board_string(self.current_grid)

        # Check if puzzle is complete
        is_complete = all(
            self.current_grid[r][c] != 0
            for r in range(9) for c in range(9)
        )

        if is_complete:
            return StepResult(
                observation=Observation(messages=[
                    Message(content=f"Congratulations! Puzzle solved!\n\n{board_str}", sender_id=-1)
                ]),
                reward=1.0,
                done=True,
            )

        return StepResult(
            observation=Observation(messages=[
                Message(content=f"Good move!\n\n{board_str}", sender_id=-1)
            ]),
            reward=0.1,
            done=False,
        )

    def close(self):
        """Cleanup (no-op for local env)."""
        pass


# Alias to match the original import
TextArenaEnv = LocalSudokuEnv
