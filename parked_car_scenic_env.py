# %%
import gymnasium as gym
from gymnasium import spaces
import numpy as np
# The following imports assume that Scenic and its Carla simulator provide
# these modules/classes. Adjust these imports to match your actual Scenic API.
from scenic.core.scenarios import Scenario
from scenic.simulators.newtonian.simulator import NewtonianSimulator
import scenic.syntax.translator as translator
# import scenic.simulators.newtonian.simulator as newtonian

# %%

scene = translator.scenarioFromFile("badlyParkedCarPullingIn.scenic")

#%%

# 2. Define simulation parameters.
network = None       # Provide a Network object if your scenario includes one.
render = True        # Set True to display a graphical window.
debug_render = False # Set True to enable debug rendering.
export_gif = False   # Set True to export a GIF of the simulation.
timestep = 1.0 / 10  # The simulation time step (0.1 seconds).

# 3. Create a NewtonianSimulator instance.
simulator = NewtonianSimulator(
    network=network,
    render=render,
    debug_render=debug_render,
    export_gif=export_gif
)

# 4. Create a NewtonianSimulation instance using the simulator.
simulation = simulator.simulate(scene)

# %%

class ScenicNewtonianEnv(gym.Env):
    """
    Gym environment wrapping a Scenic scenario for the Newtonian simulator.
    
    This environment loads a Scenic scenario (e.g., badlyParkedCarPullingIn.scenic)
    and allows an RL agent to control the ego vehicle via continuous steering and throttle commands.
    
    The observation is a vector (e.g., [x, y, heading, speed]) representing the ego vehicle’s state.
    """
    metadata = {'render.modes': ['human']}

    def __init__(self, scenario_file: str, time_step: float = 0.1):
        """
        Initialize the Scenic Newtonian environment.
        
        Parameters
        ----------
        scenario_file : str
            Path to the Scenic scenario file.
        time_step : float, optional
            Simulation time step in seconds (default is 0.1).
        """
        super(ScenicNewtonianEnv, self).__init__()
        self.time_step = time_step
        self.scenario_file = scenario_file

        # Load and parse the Scenic scenario.
        with open(scenario_file, 'r') as f:
            scenario_code = f.read()
        self.scenario = Scenario.parse(scenario_code)

        # Instantiate the Newtonian simulator.
        self.simulator = NewtonianSimulator(None, render=True)

        # Define the continuous action space: [steering, throttle].
        self.action_space = spaces.Box(low=np.array([-1.0, 0.0], dtype=np.float32),
                                       high=np.array([1.0, 1.0], dtype=np.float32),
                                       dtype=np.float32)

        # Define the observation space (example: [x, y, heading, speed]).
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(4,), dtype=np.float32)

        self.simulation_time = 0.0

env = ScenicNewtonianEnv('badlyParkedCarPullingIn.scenic')

# %% 