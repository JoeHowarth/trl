"""
Local Catch environment - no external services needed.
Drop-in replacement for OpenSpielEnv for the Catch game.

The game:
- 10 rows x 5 columns grid
- Ball starts at top, random column
- Paddle at bottom row
- Ball falls one row per step
- Actions: 0=left, 1=stay, 2=right
- +1 reward if caught, -1 if missed
"""

import random
from dataclasses import dataclass, field
from typing import List, Optional


@dataclass
class OpenSpielObservation:
    """Observation from the Catch game."""
    info_state: List[float]  # Flattened 10x5 grid
    legal_actions: List[int] = field(default_factory=lambda: [0, 1, 2])
    game_phase: str = "playing"
    current_player_id: int = 0
    opponent_last_action: Optional[int] = None
    done: bool = False
    reward: Optional[float] = None
    metadata: dict = field(default_factory=dict)


@dataclass
class StepResult:
    """Result from reset() or step()."""
    observation: OpenSpielObservation
    reward: Optional[float] = None
    done: bool = False


@dataclass
class OpenSpielAction:
    """Action for the Catch game."""
    action_id: int
    game_name: str = "catch"
    game_params: dict = field(default_factory=dict)


class LocalCatchEnv:
    """
    Local Catch environment.

    Grid is 10 rows x 5 columns.
    Ball starts at row 0 (top), paddle at row 9 (bottom).
    """

    def __init__(self, rows: int = 10, cols: int = 5):
        self.rows = rows
        self.cols = cols
        self.ball_row = 0
        self.ball_col = 0
        self.paddle_col = cols // 2  # Start in middle

    def _get_observation(self, done: bool = False, reward: float = None) -> OpenSpielObservation:
        """Create observation as flattened grid."""
        # Create 10x5 grid of zeros
        grid = [[0.0] * self.cols for _ in range(self.rows)]

        # Place ball (if not done)
        if not done:
            grid[self.ball_row][self.ball_col] = 1.0

        # Place paddle (always at bottom row)
        grid[self.rows - 1][self.paddle_col] = 1.0

        # Flatten
        info_state = [cell for row in grid for cell in row]

        return OpenSpielObservation(
            info_state=info_state,
            legal_actions=[0, 1, 2],
            game_phase="terminal" if done else "playing",
            done=done,
            reward=reward,
        )

    def reset(self) -> StepResult:
        """Start a new episode."""
        self.ball_row = 0
        self.ball_col = random.randint(0, self.cols - 1)
        self.paddle_col = self.cols // 2

        obs = self._get_observation()
        return StepResult(observation=obs, reward=None, done=False)

    def step(self, action: OpenSpielAction) -> StepResult:
        """Take a step in the environment."""
        action_id = action.action_id

        # Move paddle
        if action_id == 0:  # Left
            self.paddle_col = max(0, self.paddle_col - 1)
        elif action_id == 2:  # Right
            self.paddle_col = min(self.cols - 1, self.paddle_col + 1)
        # action_id == 1 means stay

        # Move ball down
        self.ball_row += 1

        # Check if ball reached bottom
        if self.ball_row >= self.rows - 1:
            # Episode ends
            caught = (self.ball_col == self.paddle_col)
            reward = 1.0 if caught else -1.0
            obs = self._get_observation(done=True, reward=reward)
            return StepResult(observation=obs, reward=reward, done=True)

        # Episode continues
        obs = self._get_observation(done=False, reward=0.0)
        return StepResult(observation=obs, reward=0.0, done=False)

    def close(self):
        """Cleanup (no-op for local env)."""
        pass

    def render(self) -> str:
        """Render the grid as ASCII for debugging."""
        lines = []
        for row in range(self.rows):
            line = ""
            for col in range(self.cols):
                if row == self.ball_row and col == self.ball_col:
                    line += "O"  # Ball
                elif row == self.rows - 1 and col == self.paddle_col:
                    line += "="  # Paddle
                else:
                    line += "."
            lines.append(line)
        return "\n".join(lines)


# Alias to match the original import
OpenSpielEnv = LocalCatchEnv
