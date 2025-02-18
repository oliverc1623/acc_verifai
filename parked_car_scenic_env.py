# %%
import gymnasium as gym
from gymnasium import spaces
import numpy as np
import scenic
from simulator import NewtonianSimulator
from scenic.domains.driving.actions import SetSteerAction, SetThrottleAction
from scenic.domains.driving.roads import Network
from stable_baselines3.common.env_checker import check_env

# %%
network = Network.fromFile('maps/Town01.xodr')
# Sample a new scenario from Scenic.
scenario = scenic.scenarioFromFile(
    'car.scenic',
    model='scenic.simulators.newtonian.driving_model'
)
scene, _ = scenario.generate()
simulator = NewtonianSimulator(network=network, render=True)
simulation = simulator.createSimulation(scene, verbosity=0)

# %%
img = scene.show()

# %%

simulation.step()
scene.show(zoom=0.5)
print(simulation.compute_lane_reward())
print(f"ego position: {simulation.ego.position},\nprev position: {simulation.prev_ego_position}")

SetThrottleAction(1.0).applyTo(scene.objects[0], simulator)
SetSteerAction(-1.0).applyTo(scene.objects[0], simulator)

# %% Gymnasium environment

class ScenicGymEnv(gym.Env):
    def __init__(self, scene_file: str, max_steps=100, timestep=0.1, render=False, seed=1):
        super().__init__()
        self.scene_file = scene_file
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

        # Instantiate the simulator; you may pass a road network if available.
        self.simulator = NewtonianSimulator(timestep=self.timestep, render=self.render_mode)
        self.simulation = None
        self.scene = None

    def reset(self, seed:int=1):
        self.seed = seed

        # Sample a new scenario from Scenic.
        scenario = scenic.scenarioFromFile(
            'car.scenic',
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
        terminated = reward == -1
        return observation, reward, terminated, trunacted, {}
    
    def render(self):
        scene.show()

    def close(self):
        pass
    
# %%
env = ScenicGymEnv("car.scenic", render=False)
check_env(env)

# %%
obs, _ = env.reset()
print("Initial observation:", obs)
done = False
for i in range(10):
    action = env.action_space.sample()  # Replace with agent's action in practice.
    obs, reward, terminated, truncated, info = env.step(action)
    done = terminated or truncated
    print(f"Step {env.current_step}: Observation: {obs}, Reward: {reward}")
env.close()

# %%
