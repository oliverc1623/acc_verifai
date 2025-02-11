'''
To run this file using the Carla simulator:
    scenic examples/driving/car.scenic --2d --model scenic.simulators.carla.model --simulate
'''

param map = localPath('maps/Town01.xodr')

model scenic.domains.driving.model

ego = new Car