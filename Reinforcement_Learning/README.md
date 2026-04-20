# workshop-ideas

## Reinforcement Learning Workshop

This repository contains a custom Gymnasium environment for a 2D drone navigation task, along with training and evaluation helpers built with Stable Baselines3. The drone must navigate through checkpoints while managing battery usage, staying airborne, and avoiding ground collisions.

The workshop is designed to walk through the full reinforcement learning loop:

1. Define the environment, action space, and observation space.
2. Train a PPO agent against the drone environment.
3. Evaluate the trained agent and review the rendered episode video.
4. Compare multiple saved models in a shared evaluation run.

## Workshop Notebook

The `rl_workshop.ipynb` notebook contains the step-by-step code for defining the environment, training a model, and evaluating it. It also includes explanations of the reward function and the design choices made in the environment.

Use this notebook as a guide to understand how the environment works and how to train and evaluate agents. You can run the cells in order to see the results and modify the code to experiment with different reward functions, training parameters, or evaluation settings. Or if you've cloned the repository, you can run the helper scripts directly as shown in the next sections.

## Project Layout

- `drone_env/env.py`: the custom drone environment.
- `drone_env/train.py`: helper for training or continuing a PPO model.
- `drone_env/eval.py`: single-model evaluation helper that writes a video.
- `drone_env/eval_many.py`: multi-model evaluation helper for leaderboard-style comparisons.
- `main.py`: example entry point that runs `eval_many()` over the contents of `models/`.

## Setup

Install the required packages:

```bash
pip install gymnasium numpy opencv-python-headless stable_baselines3 imageio[ffmpeg]
```

## Training and Single-Model Evaluation

Train a model by running the helper in `drone_env/train.py`. The first run creates a new PPO model if no checkpoint exists, and later runs continue training from the saved file.

```python
from drone_env.train import train

train(100_000)
```

To evaluate a single model and render a video of one episode:

```python
from drone_env.eval import eval

eval()
```

This saves a rendered episode video to `episode.mp4` by default.

## Comparing Multiple Models

The `eval_many()` helper is intended for competition-style comparisons between several saved checkpoints. Put multiple `.zip` model files in the `models/` directory, then evaluate them together with the same reset seeds and the same number of episodes per model.

```python
from drone_env.eval_many import eval_many

eval_many(
    num_episodes=10,
    models_dir="models",
    render=True,
)
```

What this does:

1. Loads every `.zip` model file found in `models/`.
2. Runs each model through the same set of episode seeds for a fair comparison.
3. Ranks models by cumulative score and collected checkpoints.
4. Optionally writes a side-by-side style leaderboard video.

If rendering is enabled and no custom output path is provided, the video is saved as `episode_many.mp4`.

### Competition-Style Workflow

If you want to turn this into a small internal competition, use a shared ruleset like this:

1. Give every participant the same environment and training budget.
2. Ask each person to export their best checkpoint into `models/` with a unique filename.
3. Run `eval_many()` with a fixed `seed_base` so everyone is tested on identical scenarios.
4. Compare the printed ranking and the generated video to see which policy performs best.

Example:

```python
from drone_env.eval_many import eval_many

eval_many(
    num_episodes=20,
    models_dir="models",
    seed_base=12345,
    render=True,
    output_path="competition_results.mp4",
)
```

## Entry Point

The `main.py` file shows one simple way to run a batch comparison directly:

```python
seed_base = int(time.time() * 1_000_000) % 1_000_000_000
eval_many(num_episodes=10, models_dir="models", render=True, seed_base=seed_base)
```

## Notes

- Model checkpoints are stored as `.zip` files.
- `eval_many()` assumes the environment has already been registered by importing `drone_env`.
- The evaluation helpers are meant to be run after you have at least one trained checkpoint in `models/`.
