# %%
from parked_car_scenic_env import ScenicGymEnv
from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor

# %%

log_dir = "logs/"
env = ScenicGymEnv("car.scenic", network_file="maps/Town01.xodr", max_steps=100, render=True)
env = Monitor(env, log_dir)
model = PPO("MlpPolicy", env, verbose=1, n_steps=256)
model.learn(total_timesteps=100_000, progress_bar=True)

# %%
