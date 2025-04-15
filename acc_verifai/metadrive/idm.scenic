#SET MAP AND MODEL (i.e. definitions of all referenceable vehicle types, road library, etc)
# Imports
import math
import numpy as np
from metadrive.policy.idm_policy import IDMPolicy
from controllers.lateral_control import LateralControl

param map = localPath('../maps/Town06.xodr')
param carla_map = 'Town06'
param time_step = 1.0/10
model scenic.simulators.metadrive.model
param verifaiSamplerType = 'ce' # TODO: use scenic/random/uniform/halton sampler to train from scratch; then use ce for fine-tuning

#CONSTANTS
TERMINATE_TIME = 20 / globalParameters.time_step

# Parameters of the scenario.
inter_vehivle_disance = 40 # Range(30, 60)

# platoon placement 
LEADCAR_TO_EGO = C1_TO_C2 = C2_TO_C3 = -inter_vehivle_disance

def not_zero(x: float, eps: float = 1e-2) -> float:
    if abs(x) > eps:
        return x
    elif x > 0:
        return eps
    else:
        return -eps

def get_vehicle_ahead(id, vehicle, lane):
	""" Returns the closest object in front of the vehicle that is:
	(1) visible,
	(2) on the same lane (or intersection),
	within the thresholdDistance.
	Returns the object if found, or None otherwise. """
	closest = None
	minDistance = float('inf')
	objects = simulation().objects
	for obj in objects:
		if not (vehicle can see obj):
			continue
		d = abs(vehicle.position.x - obj.position.x) # (distance from vehicle.position to obj.position)
		if vehicle == obj or d < 0.1:
			continue
		if lane != obj.lane:
			continue
		if d < minDistance:
			minDistance = d
			closest = obj
	return closest

def get_vehicle_behind(id, vehicle, lane):
	""" Returns the closest object behind the vehicle that is:
	(1) visible,
	(2) on the same lane (or intersection),
	within the thresholdDistance.
	Returns the object if found, or None otherwise. """
	closest = None
	minDistance = float('inf')
	objects = simulation().objects
	for obj in objects:
		if not (obj can see vehicle):
			continue
		d = abs(obj.position.x - vehicle.position.x)
		if vehicle == obj or d < 0.1:
			continue
		if lane != obj.lane:
			continue
		if d < minDistance:
			minDistance = d
			closest = obj
	return closest

def get_adjacent_lane(id, vehicle, direction):
	"""Get the adjacent lane in the specified direction (left or right) from the current lane."""
	lane_section = vehicle.laneSection
	left_lane = lane_section.laneToLeft.lane
	right_lane = lane_section.laneToRight.lane
	if direction == "left" and left_lane:
		return left_lane
	elif direction == "right" and right_lane:
		return right_lane
	else:
		raise ValueError("Direction must be 'left' or 'right'.")

def map_acc_to_throttle_brake(acc):
	if acc > 0:
		throttle = min(acc, 1)
		brake = 0
	else:
		throttle = 0
		brake = min(abs(acc), 1)
	return throttle, brake

def regulateSteering(steer, past_steer, max_steer=0.8):
	# Steering regulation: changes cannot happen abruptly, can't steer too much.
	if steer > past_steer + 0.1:
		steer = past_steer + 0.1
	elif steer < past_steer - 0.1:
		steer = past_steer - 0.1
	if steer >= 0:
		steer = min(max_steer, steer)
	else:
		steer = max(-max_steer, steer)
	return steer

def idm_acc(agent, vehicle_in_front):
	# IDM params
	ACC_FACTOR = 1.0
	DEACC_FACTOR = -4 # Range(-6,-4)
	target_speed = 15 # Range(20, 22.5)
	DISTANCE_WANTED = 4.5 # Range(1.0, 2.0)
	TIME_WANTED = 1.5 # Range(0.1, 1.5)
	delta = 2 # Range(2, 6)      # Acceleration exponent

	acceleration = ACC_FACTOR * (1-np.power(max(agent.speed, 0) / target_speed, delta))
	if vehicle_in_front is None:
		return map_acc_to_throttle_brake(acceleration)

	gap = (vehicle_in_front.position.x - agent.position.x) - agent.length
	d0 = DISTANCE_WANTED
	tau = TIME_WANTED
	ab = -ACC_FACTOR * DEACC_FACTOR
	dv = agent.speed - vehicle_in_front.speed
	d_star = d0 + agent.speed * tau + vehicle_in_front.speed * dv / (2 * np.sqrt(ab))
	speed_diff = d_star / not_zero(gap)
	acceleration -= ACC_FACTOR * (speed_diff**2)
	return map_acc_to_throttle_brake(acceleration)

behavior IDM_MOBIL(id, target_speed=12, politeness=0.3, acceleration_threshold=0.2, safe_braking=-4):
	_lon_controller_follow, _lat_controller_follow = simulation().getLaneFollowingControllers(self)
	past_steer_angle = 0
	current_lane = self.lane
	current_centerline = current_lane.centerline

	while True:	
		# Lateral: MOBIL
		best_change_advantage = -float('inf')
		target_lane_for_change = None

		for direction in ["left", "right"]:
			adjacent_lane = get_adjacent_lane(id, self, direction)
			if adjacent_lane is None or adjacent_lane == current_lane:
				continue

			# find relevant vehicles for MOBIL calculation
			ego_leader = get_vehicle_ahead(id, self, current_lane)
			ego_follower = get_vehicle_behind(id, self, current_lane)
			adjacent_leader = get_vehicle_ahead(id, self, adjacent_lane)
			adjacent_follower = get_vehicle_behind(id, self, adjacent_lane)

		current_lane = self.lane
		current_centerline = current_lane.centerline
		nearest_line_points = current_centerline.nearestSegmentTo(self.position)

		vehicle_front = get_vehicle_ahead(id, self, current_lane)
		vehicle_behind = get_vehicle_behind(id, self, current_lane)

		throttle, brake = idm_acc(self, vehicle_front)

		if target_lane_for_change:
			pass
		else:
			nearest_line_points = current_centerline.nearestSegmentTo(self.position)
			nearest_line_segment = PolylineRegion(nearest_line_points)
			cte = nearest_line_segment.signedDistanceTo(self.position)
			current_steer_angle = _lat_controller_follow.run_step(cte) # Use the lane following lateral controller
			current_steer_angle = regulateSteering(current_steer_angle, past_steer_angle)

		take SetThrottleAction(throttle), SetBrakeAction(brake), SetSteerAction(current_steer_angle)
		past_steer_angle = current_steer_angle

#PLACEMENT
spawnPt = (200 @ -48.87)

id = 0
ego = new Car at spawnPt #, with behavior FollowLaneBehavior() # IDM_MOBIL(target_speed=22, politeness=0.3, acceleration_threshold=0.2, safe_braking=-4)

id = 1
c1 = new Car at ego.position offset by (LEADCAR_TO_EGO, 0),
	with behavior IDM_MOBIL(id, target_speed=15) #IDM_MOBIL(target_speed=22, politeness=0.3, acceleration_threshold=0.2, safe_braking=-4)

id = 2
c2 = new Car at c1.position offset by (C1_TO_C2, 0),
	with behavior IDM_MOBIL(id, target_speed=15) #IDM_MOBIL(target_speed=22, politeness=0.3, acceleration_threshold=0.2, safe_braking=-4)

id = 3
c3 = new Car at c2.position offset by (C2_TO_C3, 4),
	with behavior IDM_MOBIL(id, target_speed=15)


'''
require always (distance from ego.position to c1.position) > 4.99
terminate when ego.lane == None 
'''
# terminate when (simulation().currentTime > TERMINATE_TIME) 
terminate when (distance from ego to c1) < 4.5
# terminate when (distance from c1 to c2) < 4.5
# terminate when (distance from c2 to c3) < 4.5