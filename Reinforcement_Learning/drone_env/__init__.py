from gymnasium.envs.registration import register
from drone_env.env import DroneEnv, MAX_EPISODE_STEPS

register(
    id="DroneDelivery-v0",
    entry_point="drone_env.env:DroneEnv",
    max_episode_steps=MAX_EPISODE_STEPS,
)
