# %%
import pathlib

import numpy as np
import scenic
from gymnasium import spaces
from scenic.gym import ScenicGymEnv
from scenic.simulators.metadrive import MetaDriveSimulator
from stable_baselines3 import PPO


# %%

scenario = scenic.scenarioFromFile(
        "idm.scenic",
        model="scenic.simulators.metadrive.model",
        mode2D=True,
    )

env = ScenicGymEnv(
    scenario,
    MetaDriveSimulator(sumo_map=pathlib.Path("../maps/Town06.net.xml"), render=True, real_time=True),
    observation_space=spaces.Box(low=-np.inf, high=np.inf, shape=(4,4)),
    action_space=spaces.Box(low=-1, high=1, shape=(1,)),
)

# %%
model = PPO("MlpPolicy", env, verbose=1)
model.load("models/ppo_idm_attacker", env=env)

# %%
# Enjoy trained agent
env = model.get_env()
obs = env.reset()
for _ in range(1000):
    action, _states = model.predict(obs, deterministic=True)
    obs, rewards, dones, info = env.step(action)

# %%
