# %%
import pathlib

import numpy as np
import scenic
import torch
from gymnasium import spaces
from multiprocessing_ppo import ActorCritic
from scenic.gym import ScenicGymEnv
from scenic.simulators.metadrive import MetaDriveSimulator


# %%

scenario = scenic.scenarioFromFile(
    "lane_follow.scenic",
    model="scenic.simulators.metadrive.model",
    mode2D=True,
)

env = ScenicGymEnv(
    scenario,
    MetaDriveSimulator(timestep=0.05, sumo_map=pathlib.Path("../maps/Town06.net.xml"), render=True, real_time=False),
    observation_space=spaces.Box(low=-np.inf, high=np.inf, shape=(1, 5)),
    action_space=spaces.Box(low=-1, high=1, shape=(1,)),
    max_steps=700,
)

# %%
obs_space_shape = env.observation_space.shape
action_space = env.action_space
obs_dim = np.prod(obs_space_shape) if isinstance(obs_space_shape, tuple) else obs_space_shape[0]
model = ActorCritic(obs_dim, action_space)

# %%
model.load_state_dict(torch.load("models/ppo_lane_follow_model.pth"))

# %%
n_iterations = 1
obs, _ = env.reset()  # reset returns (obs, info)

for _ in range(n_iterations):
    done = False
    while not done:
        # Prepare observation as a Torch tensor and add batch dimension
        state = torch.FloatTensor(obs).unsqueeze(0)

        with torch.no_grad():
            # Obtain action from the model (assumes model returns (action, value) tuple)
            mean, log_std, _ = model(state)
            std = log_std.exp()
            normal = torch.distributions.Normal(mean, std)
            x_t = normal.rsample()
            y_t = torch.tanh(x_t)
            action = y_t * model.action_scale + model.action_bias

        # Remove batch dimension and convert to numpy array for the environment
        action = action.squeeze(0).numpy()

        # Step the environment (gymnasium returns: next_obs, reward, done, truncated, info)
        next_obs, reward, done, truncated, info = env.step(action)

    obs, _ = env.reset()

# %%
