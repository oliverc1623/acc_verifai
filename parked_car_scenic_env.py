# %%
import gymnasium as gym
import scenic
from simulator import NewtonianSimulator

# %%
scenario = scenic.scenarioFromFile('../Scenic-2.1.0/examples/gta/badlyParkedCar2.scenic',
                                   model='scenic.simulators.newtonian.driving_model')
scene, _ = scenario.generate()
simulator = NewtonianSimulator()
simulation = simulator.simulate(scene, maxSteps=10)
if simulation:  # `simulate` can return None if simulation fails
    result = simulation.result
    for i, state in enumerate(result.trajectory):
            egoPos, parkedCarPos = state
            print(f'Time step {i}: ego at {egoPos}; parked car at {parkedCarPos}')
            scene.show(zoom=0.95)
# %%

simulation = simulator.createSimulation(scene, verbosity=0)
simulation

# %%
simulation.objects


# %% Gymnasium environment

class ScenicGymEnv(gym.Env):
    def __init__(self, scene_file: str, max_steps=50, lane_center_y=0.0, timestep=0.1, render=False):
        super().__init__()
        self.scene_file = scene_file
        self.max_steps = max_steps
        self.lane_center_y = lane_center_y
        self.timestep = timestep
        self.render_mode = render
        self.current_step = 0

        # Define action space for the RL agent (e.g., throttle, steer, brake).
        # TODO: define action space most likekly multi discrete/continuous

        # Define action space for the RL agent (e.g., throttle, steer, brake).
        # TODO: define observation space.... 
        
        # Instantiate the simulator; you may pass a road network if available.
        self.simulator = NewtonianSimulator(timestep=self.timestep, render=self.render_mode)
        self.simulation = None
        self.scene = None

    def reset(self, seed=None, options=None):
        if seed is not None:
                self.seed(seed)

        # Sample a new scenario from Scenic.
        scenario = scenic.scenarioFromFile(
            'car.scenic',
            model='scenic.simulators.newtonian.driving_model'
        )
        self.scene, _ = scenario.generate()
        
        # Create a new simulation from the scene.
        self.simulation = self.simulator.createSimulation(self.scene, verbosity=0)
        self.current_step = 0
        
        observation = self._get_observation()
        return observation, {}
