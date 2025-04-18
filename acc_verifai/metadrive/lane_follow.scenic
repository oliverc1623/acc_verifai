'''
To run this file using the MetaDrive simulator:
    scenic examples/driving/car.scenic --2d --model scenic.simulators.metadrive.model --simulate

To run this file using the Carla simulator:
    scenic examples/driving/car.scenic --2d --model scenic.simulators.carla.model --simulate
'''

param map = localPath('../maps/Town05.xodr')

model scenic.domains.driving.model

behavior dummy_attacker():
    while True:
        print(self.lane.orientation.value)
        print(self.heading)
        take SetThrottleAction(1.0), SetBrakeAction(0.0), SetSteerAction(-0.3)

ego = new Car with behavior dummy_attacker()
