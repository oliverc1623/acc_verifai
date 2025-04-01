# %%
import pathlib

import numpy as np
import scenic
from gymnasium import spaces
from scenic.gym import ScenicGymEnv
from scenic.simulators.metadrive import MetaDriveSimulator
from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.utils import set_random_seed


# %%
def main() -> None:
    """Run RL training."""
    set_random_seed(0)

    scenario = scenic.scenarioFromFile(
        "idm.scenic",
        model="scenic.simulators.metadrive.model",
        mode2D=True,
    )

    log_dir="tmp2/"

    env = ScenicGymEnv(
        scenario,
        MetaDriveSimulator(sumo_map=pathlib.Path("../maps/Town06.net.xml"), render=False, real_time=False),
        observation_space=spaces.Box(low=-np.inf, high=np.inf, shape=(16,)),
        action_space=spaces.Box(low=-1, high=1, shape=(1,)),
    )
    env = Monitor(env, log_dir)

    obs, info = env.reset()

    model = PPO("MlpPolicy", env, verbose=1)
    model.learn(total_timesteps=100_000, progress_bar=True)


if __name__== "__main__":
    main()
