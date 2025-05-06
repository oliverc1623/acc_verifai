# %%
import pathlib

import gymnasium
import numpy as np
import scenic
from gymnasium import spaces
from scenic.gym import ScenicGymEnv
from scenic.simulators.metadrive import MetaDriveSimulator
from stable_baselines3.common.utils import set_random_seed


# %%


def make_env() -> callable:
    """Create a function that returns a new environment instance."""

    def thunk() -> ScenicGymEnv:
        scenario = scenic.scenarioFromFile(
            "idm.scenic",
            model="scenic.simulators.metadrive.model",
            mode2D=True,
        )

        env = ScenicGymEnv(
            scenario,
            MetaDriveSimulator(timestep=0.05, sumo_map=pathlib.Path("../maps/Town06.net.xml"), render=True, real_time=False),
            observation_space=spaces.Box(low=-np.inf, high=np.inf, shape=(7, 5)),
            action_space=spaces.Box(low=-1, high=1, shape=(2,)),
            max_steps=700,
        )
        return env

    return thunk


# %%
def main() -> None:
    """Run RL training."""
    set_random_seed(0)

    envs = [make_env() for i in range(4)]
    env = gymnasium.vector.AsyncVectorEnv(envs)
    env.reset()

    # only run this part on local machine
    while 1:
        action = env.action_space.sample()
        obs, reward, done, truc, info = env.step(action)
        print(done)

    env.close()


if __name__ == "__main__":
    main()
