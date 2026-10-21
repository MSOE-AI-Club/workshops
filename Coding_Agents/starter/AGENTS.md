# Agent instructions

This is the classic Snake game in Python, drawn with pygame. The rules it must follow are in `README.md`. It starts as bones: `snake/game.py` and `snake/app.py` have TODOs for the factory to fill in.

## Commands

- Run the tests: `uv run pytest`
- Play the game: `uv run python -m snake` (opens a window)
- Packages are in `requirements.txt`, installed in `.venv`. Reinstall with `uv pip install -r requirements.txt`.

Use `uv run`. Do not use `pip` or plain `python`.

## The factory brain

Read `BRAIN.md` before you start any task. It holds what earlier runs learned about this project.

Do not edit `BRAIN.md` yourself. Only the `factory-close` skill adds to it, and only after a person approves.

## The factory

Work moves through five stations. Each station is a skill in `.agents/skills`. Each one reads the files the station before it wrote:

| Station | Skill | Writes |
| --- | --- | --- |
| 1. Intake | `$factory-intake` | `factory/current/TICKET.md` |
| 2. Shape | `$factory-shape` | `factory/current/SPEC.md`, approved by a person |
| 3. Build | `$factory-build` | code, tests, and evidence in `SPEC.md` |
| 4. Audit | `$factory-audit`, in a new chat (the team writes this skill) | `factory/current/REVIEW.md` |
| 5. Close | `$factory-close` | lessons in `BRAIN.md`, the run in `factory/done/`, a commit |

When a person asks for a station, do only that station, then say which station comes next.

## Rules

- Keep game rules in `snake/game.py`, with no pygame in it. `snake/app.py` only draws and reads keys.
- You can't see the game window, so test the rules in `snake/game.py` instead of playing.
- Keep changes small. Change only what the task needs.
- Do not add a dependency without asking.
