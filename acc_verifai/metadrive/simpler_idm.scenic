#SET MAP AND MODEL (i.e. definitions of all referenceable vehicle types, road library, etc)
# Imports
import math
import numpy as np
from metadrive.policy.idm_policy import IDMPolicy
from controllers.lateral_control import LateralControl

param map = localPath('../maps/Town05.xodr')
param carla_map = 'Town05'
param time_step = 1.0/10
model scenic.simulators.metadrive.model
param verifaiSamplerType = 'ce' # TODO: use scenic/random/uniform/halton sampler to train from scratch; then use ce for fine-tuning

#CONSTANTS
TERMINATE_TIME = 25 / globalParameters.time_step

# Parameters of the scenario.
inter_vehivle_disance = Range(30, 60)

# platoon placement 
LEADCAR_TO_EGO = C1_TO_C2 = C2_TO_C3 = -inter_vehivle_disance

def not_zero(x: float, eps: float = 1e-2) -> float:
    if abs(x) > eps:
        return x
    elif x > 0:
        return eps
    else:
        return -eps

def first_vehicle_ahead(agent, laneSec):
	"""Return nearest vehicle ahead of *agent* in *laneSec* (or None)."""
	best = None
	bestDist = 1e9
	for obj in simulation().objects:
		if obj is agent or not obj.isVehicle:
			continue
		if obj._laneSection is not laneSec:
			continue
		d = (distance from agent to obj)
		if 0 < d < bestDist:
			bestDist = d
			best = obj
	return best

def first_vehicle_behind(agent, laneSec):
	"""Return nearest vehicle behind *agent* in *laneSec* (or None)."""
	best = None
	bestDist = 1e9
	for obj in simulation().objects:
		if obj is agent or not obj.isVehicle:
			continue
		if obj._laneSection is not laneSec:
			continue
		d = (distance from obj to agent)   # reverse order
		if 0 < d < bestDist:
			bestDist = d
			best = obj
	return best


def acc_idm(agent, vehicle_in_front):
	########## IDM PARAMETERS #################################################
	# Max speed is 22.5 m/s = 80 kmh = 50 mph
	# normal_speed is 19.4 m/s = 70 kmh
	ACC_FACTOR = 1.0
	DEACC_FACTOR = Range(-10,-4)
	target_speed = Range(20, 22.5)
	DISTANCE_WANTED = Range(0.5, 1.0)
	TIME_WANTED = Range(0.1, 1.5)
	delta =Range(2, 6)

	acceleration = ACC_FACTOR * (1-np.power(max(agent.speed, 0) / target_speed, delta))
	gap = (distance from agent to vehicle_in_front) - agent.length
	d0 = DISTANCE_WANTED
	tau = TIME_WANTED
	ab = -ACC_FACTOR * DEACC_FACTOR
	dv = agent.speed - vehicle_in_front.speed
	d_star = d0 + agent.speed * tau + vehicle_in_front.speed * dv / (2 * np.sqrt(ab))
	speed_diff = d_star / not_zero(gap)
	acceleration -= ACC_FACTOR * (speed_diff**2)
	return acceleration

## Longitudinal IDM BEHAVIOR
behavior Longitudinal_IDM(id):
	########## MOBIL PARAMETERS ###############################################
	p_polite = Range(0.2, 0.5)               # politeness factor
	b_safe = 4                             # max comfortable brake (m/s²)
	a_thresh = 0.1                           # incentive threshold (m/s²)
	a_bias = 0.3                           # keep‑right bias (positive favours right)

	lat_control  = LateralControl(globalParameters.time_step)

	while True:
		curLS = self.laneSection
		v_self = self.speed
		if id == 1: 
			print(f"id: {id}, speed: {v_self}, lane: {curLS}")
		leader_cur = first_vehicle_ahead(self, curLS)
		follower_cur = first_vehicle_behind(self, curLS)
		if id == 1:
			print(f"leader: {leader_cur}, follower: {follower_cur}")

		acceleration = acc_idm(self, leader_cur) if leader_cur else 0

		if acceleration > 0:
			throttle = min(acceleration, 1)
			brake = 0
		else:
			throttle = 0
			brake = min(-acceleration, 1)

		take SetThrottleAction(throttle), SetBrakeAction(brake), SetSteerAction(0)

#PLACEMENT
# initLane = network.roads[13].forwardLanes.lanes[0]
# spawnPt = initLane.centerline.pointAlongBy(0)
# spawnPt = (-100 @ -48.87)

id = 0
ego = new Car with velocity(22.5,0)

id = 1
c1 = new Car at ego.position offset by (LEADCAR_TO_EGO, 0),
	with behavior Longitudinal_IDM(id), with velocity(22.5,0)

id = 2
c2 = new Car at c1.position offset by (C1_TO_C2, 0),
	with behavior Longitudinal_IDM(id), with velocity(22.5,0)

id = 3
c3 = new Car at c2.position offset by (C2_TO_C3, 0),
	with behavior Longitudinal_IDM(id), with velocity(22.5,0)


'''
require always (distance from ego.position to c1.position) > 4.99
terminate when ego.lane == None 
'''
# terminate when (simulation().currentTime > TERMINATE_TIME) 
terminate when (distance from ego to c1) < 4.99
terminate when (distance from c1 to c2) < 4.99
terminate when (distance from c2 to c3) < 4.99