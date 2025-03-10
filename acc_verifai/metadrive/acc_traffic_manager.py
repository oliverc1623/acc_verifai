from metadrive.component.vehicle.vehicle_type import DefaultVehicle
from metadrive.manager import BaseManager
from metadrive.policy.idm_policy import IDMPolicy


MAX_EPISODE_STEP = 180


class AccTrafficManager(BaseManager):
    """Manager class for ACC theat model with configurable number of follower vehicles."""

    def __init__(self, ego_init_position: tuple[int, int] = (100, 0), num_followers: int = 2):
        super().__init__()
        self.num_followers = num_followers
        self.ego_init_position = ego_init_position

    def before_step(self) -> None:
        """Set actions for all spawned objects before each simulation step."""
        for i, obj in self.spawned_objects.items():
            p = self.get_policy(i)
            obj.before_step(p.act())  # set action

    def reset(self) -> None:
        """Reset the traffic manager for a new episode."""
        for i in range(self.num_followers):
            position = (self.ego_init_position[0] - (i + 1) * 30, 0)
            obj = self.spawn_object(DefaultVehicle, vehicle_config=dict(), position=position, heading=0)
            self.add_policy(obj.id, IDMPolicy, obj, self.generate_seed())

    def after_step(self) -> None:
        """Perform any necessary actions after each simulation step."""
        for obj in self.spawned_objects.values():
            obj.after_step()
        if self.episode_step == MAX_EPISODE_STEP:
            self.clear_objects(list(self.spawned_objects.keys()))
