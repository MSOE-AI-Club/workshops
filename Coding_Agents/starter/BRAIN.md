# Factory brain

What earlier runs learned about this project. Every station reads this file first. Only `factory-close` adds to it, after a person approves.

Keep each note short, specific, and true. Delete a note when it stops being true.

## Build and test

- Run tests with `uv run pytest`. On Windows, plain `python` can open the Microsoft Store instead of Python.

## Gotchas

- Food lands on a random square. In tests, set `game.food` yourself (or pass `Game(seed=...)`) so the snake doesn't eat by accident and the test gives the same result every run.

## Findings the audit caught more than once

- (none yet)
