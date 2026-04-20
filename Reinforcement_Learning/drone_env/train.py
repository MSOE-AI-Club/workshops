import gymnasium as gym
from stable_baselines3 import PPO
import drone_env # To register the environment
import os
from pathlib import Path
from drone_env.env import MODEL_PATH

def train(timesteps):
    env = gym.make("DroneDelivery-v0")

    model_path = Path(MODEL_PATH)

    try:
        model = PPO.load(str(model_path), env=env)
        # model.learning_rate = 3e-4
        # You can adjust hyperparameters here if you want to try other values
    except:
        model = PPO("MlpPolicy", env, verbose=1, learning_rate=3e-4, stats_window_size=10)

    if timesteps > 0:
        model.learn(total_timesteps=timesteps, reset_num_timesteps=False)
        print(f"Training for {timesteps} timesteps completed.")
    model.save(MODEL_PATH)
    print(f"Model saved to {MODEL_PATH}.")