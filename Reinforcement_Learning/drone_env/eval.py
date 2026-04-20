from pathlib import Path

import gymnasium as gym
import imageio
from stable_baselines3 import PPO

import drone_env  # Registers DroneDelivery-v0
from drone_env.env import MODEL_PATH, OUTPUT_PATH

def eval():
    print("Starting evaluation...")
    env = gym.make("DroneDelivery-v0")
    model = PPO.load(MODEL_PATH)

    observation, _ = env.reset()
    terminated = False
    truncated = False
    total_reward = 0.0
    step_count = 0
    total_collected = 0
    score = 0

    frame = env.render()
    if frame is None:
        env.close()
        raise RuntimeError("Environment returned no frame.")

    fps = int(60)
    output_file = Path(OUTPUT_PATH)
    output_file.parent.mkdir(parents=True, exist_ok=True)
    output_path = str(output_file.resolve())
    writer = imageio.get_writer(output_path, fps=fps, macro_block_size=1, codec="libx264")

    try:
        while not (terminated or truncated):
            action, _ = model.predict(observation, deterministic=True)
            observation, reward, terminated, truncated, info = env.step(action)
            total_reward += float(reward)
            step_count += 1
            total_collected = info.get("total_collected", total_collected)
            score = info.get("score", score)

            frame = env.render()
            if frame is None:
                break
            writer.append_data(frame)
    finally:
        writer.close()
        env.close()

    print("Evaluation complete.")
    print(f"Episode steps:          {step_count}")
    print(f"Episode reward:         {total_reward:.2f}")
    print(f"Collected checkpoints:  {total_collected}")
    print(f"Score:                  {score}")