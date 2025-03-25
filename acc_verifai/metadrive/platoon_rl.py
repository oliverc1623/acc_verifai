# %%
import pathlib

import numpy as np
import scenic
from gymnasium import spaces
from scenic.gym import ScenicGymEnv
from scenic.simulators.metadrive import MetaDriveSimulator


# %%

scenario = scenic.scenarioFromFile(
    "idm.scenic",
    model="scenic.simulators.metadrive.model",
    mode2D=True,
)

# %%

env = ScenicGymEnv(
    scenario,
    MetaDriveSimulator(sumo_map=pathlib.Path("../maps/Town06.net.xml")),
    action_space=spaces.Box(low=-1, high=1, shape=(2,)),
)

# %%
env.reset()

# %%
for _ in range(100):
    a = env.action_space.sample()
    env.step(np.array([1,1]))

# %%
spaces.Box(low=-1, high=1, shape=(2,)).sample()

# %%
env.action_space.sample()
