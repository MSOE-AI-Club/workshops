# Snake

The classic Snake game, for the MSOE AI Club's Coding Agents workshop. Right now it's only the bones: a window with an empty board. The factory builds the game from the rules below.

## Play

```powershell
uv run python -m snake
```

Steer with the arrow keys (or WASD). Press R to play again after a game over, and Esc to quit. To make it faster or slower, add `--speed 12` (moves per second, default 8).

## Rules

- The board is 20 squares wide and 15 tall. The snake starts in the middle, 3 squares long, moving right.
- The snake moves one square per tick in the direction it's facing.
- Eating food makes the snake one square longer and adds 1 point. New food appears right away on an empty square, never on the snake.
- Hitting a wall or the snake's own body ends the game.
- The snake can never turn back on itself. If it's moving right, it can't start moving left, no matter how fast you press the keys.
- The head may move into the square the tail is leaving, because the tail moves out of the way.

## Code

- `snake/game.py` has the rules. It doesn't draw anything, so it's easy to test.
- `snake/app.py` draws the window and reads the keys.
- `requirements.txt` lists the packages: `pygame-ce` and `pytest`. The setup script installs them into `.venv`. To install them again: `uv pip install -r requirements.txt`.

## Checks

```powershell
uv run pytest
```

## How we work

This project is built by a software factory: agent skills that each do one station of the work. See `AGENTS.md`, `BRAIN.md`, and the `factory` folder.
