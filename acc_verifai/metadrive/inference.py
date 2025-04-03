# %%
import pathlib

import numpy as np
import scenic
from gymnasium import spaces
from scenic.gym import ScenicGymEnv
from scenic.simulators.metadrive import MetaDriveSimulator
from stable_baselines3 import PPO
from stable_baselines3.common import results_plotter
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.monitor import Monitor


# %%

results_plotter.plot_results(
    ["logs/tmp6"],
    num_timesteps=100_000,
    x_axis=results_plotter.X_TIMESTEPS,
    task_name="PPO IDM Attacker",
)

# %%

scenario = scenic.scenarioFromFile(
        "idm.scenic",
        model="scenic.simulators.metadrive.model",
        mode2D=True,
    )

env = ScenicGymEnv(
    scenario,
    MetaDriveSimulator(timestep=0.1, sumo_map=pathlib.Path("../maps/Town06.net.xml"), render=True, real_time=True),
    observation_space=spaces.Box(low=-np.inf, high=np.inf, shape=(4,4)),
    action_space=spaces.Box(low=-1, high=1, shape=(1,)),
    max_steps=300,
)

log_dir="logs/inference"
env = Monitor(env, log_dir, info_keywords=("attacker_crashed", "counter_example_found", "timeout", "dense_reward_signals"))


# %%
model = PPO("MlpPolicy", env, verbose=1)
model.load("models/ppo_idm_attacker5", env=env)

# %%
mean_reward, std_reward = evaluate_policy(model, model.get_env(), n_eval_episodes=10)

# %%
