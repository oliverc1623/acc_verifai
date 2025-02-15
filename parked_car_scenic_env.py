# %%
import scenic
from scenic.simulators.newtonian import NewtonianSimulator


scenario = scenic.scenarioFromFile('../Scenic-2.1.0/examples/driving/badlyParkedCarPullingIn.scenic',
                                   model='scenic.simulators.newtonian.driving_model')
scene, _ = scenario.generate()
simulator = NewtonianSimulator()
simulation = simulator.simulate(scene, maxSteps=10)
if simulation:  # `simulate` can return None if simulation fails
        result = simulation.result
        for i, state in enumerate(result.trajectory):
                egoPos, parkedCarPos = state
                print(f'Time step {i}: ego at {egoPos}; parked car at {parkedCarPos}')
