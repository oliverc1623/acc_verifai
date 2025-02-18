# %%
import gymnasium as gym
from gymnasium import spaces
import numpy as np
import scenic
from simulator import NewtonianSimulator
from scenic.domains.driving.actions import SetSteerAction, SetThrottleAction
from scenic.domains.driving.roads import Network


# %% Gymnasium environment

class ScenicGymEnv(gym.Env):
    def __init__(self, scene_file: str, network_file: str, max_steps=100, timestep=0.1, render=False, seed=1):
        super().__init__()
        self.scene_file = scene_file
        self.network_file = network_file
        self.max_steps = max_steps
        self.timestep = timestep
        self.render_mode = render
        self.current_step = 0
        self.seed = None

        # just steer for now, throttle will be constant
        self.action_space = spaces.Box(
            low=np.array([-1.0]),  # steer
            high=np.array([1.0]),
            dtype=np.float32
        )

        # [ego_x, ego_y, ego_heading].
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(3,), dtype=np.float32
        )
        self.network = Network.fromFile(self.network_file)
        # Instantiate the simulator; you may pass a road network if available.
        self.simulator = NewtonianSimulator(network=self.network, timestep=self.timestep, render=self.render_mode)
        self.simulation = None
        self.scene = None

    def reset(self, seed:int=1):
        self.seed = seed

        # Sample a new scenario from Scenic.
        scenario = scenic.scenarioFromFile(
            self.scene_file,
            model='scenic.simulators.newtonian.driving_model'
        )
        self.scene, _ = scenario.generate()
        
        # Create a new simulation from the scene.
        self.simulation = self.simulator.createSimulation(self.scene, verbosity=0)
        self.current_step = 0
        ego = self.scene.objects[0]
        observation = np.array([ego.position.x, ego.position.y, ego.heading], dtype=np.float32)
        return observation, {}
    
    def step(self, action):
        # Apply the action to the simulation.
        ego = self.scene.objects[0]
        SetSteerAction(action).applyTo(ego, self.simulator)
        SetThrottleAction(1.0).applyTo(ego, self.simulator) # constant for now
        
        self.simulation.step()
        self.current_step += 1
        observation = np.array([ego.position.x, ego.position.y, ego.heading], dtype=np.float32)
        trunacted = self.current_step >= self.max_steps
        reward = self.simulation.compute_lane_reward()
        terminated = reward == -1 or trunacted
        return observation, reward, terminated, trunacted, {}
    
    def render(self):
        self.scene.show()

    def close(self):
        pass

# %%
