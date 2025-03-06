from metadrive.manager import BaseManager
from metadrive.component.vehicle.vehicle_type import DefaultVehicle
from metadrive.policy.idm_policy import IDMPolicy


class AccTrafficManager(BaseManager):
    """Manager class for ACC theat model with configurable number of follower vehicles."""
    def __init__(self, ego_init_position=(100, 0), num_followers=2,):
        super(AccTrafficManager, self).__init__()
        self.num_followers = num_followers
        self.ego_init_position = ego_init_position
            
    def before_step(self):
        for id, obj in self.spawned_objects.items():
            p = self.get_policy(id)
            obj.before_step(p.act()) # set action

    def reset(self):
        for i in range(self.num_followers):
            position = (self.ego_init_position[0] - (i + 1) * 30, 0)
            obj = self.spawn_object(DefaultVehicle, 
                            vehicle_config=dict(), 
                            position=position, 
                            heading=0)
            self.add_policy(obj.id, IDMPolicy, obj, self.generate_seed())

    def after_step(self):
        for obj in self.spawned_objects.values():
            obj.after_step()
        if self.episode_step == 180:
            self.clear_objects(list(self.spawned_objects.keys()))
