from drone_env.train import train
from drone_env.eval import eval
from drone_env.eval_many import eval_many
import time

TIMESTEPS = 100_000

if __name__ == "__main__":
    # train(TIMESTEPS)
    # eval()
    seed_base = int(time.time() * 1_000_000) % 1_000_000_000
    eval_many(num_episodes=10, models_dir="models", render=True, seed_base=seed_base)
