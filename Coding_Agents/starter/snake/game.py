"""Snake rules. Nothing in here draws or reads keys, so it is easy to test.

The rules are in README.md. The factory builds the rest of this file.
"""

import random

Point = tuple[int, int]

UP: Point = (0, -1)
DOWN: Point = (0, 1)
LEFT: Point = (-1, 0)
RIGHT: Point = (1, 0)


class Game:
    def __init__(self, width: int = 20, height: int = 15, seed: int | None = None):
        self.width = width
        self.height = height
        self.rng = random.Random(seed)
        # TODO: the snake (a list of squares, head first), its direction,
        # the score, whether the game is over, and where the food is.

    def turn(self, direction: Point) -> None:
        """Steer the snake. It moves this way on the next step."""
        raise NotImplementedError

    def step(self) -> None:
        """Move the snake one square."""
        raise NotImplementedError
