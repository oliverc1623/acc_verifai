from metadrive.envs import MetaDriveEnv
from acc_traffic_manager import AccTrafficManager

class AccEnv(MetaDriveEnv):
    """Environment class for ACC theat model with configurable number of follower vehicles."""
    def __init__(self, config, ego_init_position: tuple[int,int], num_followers=2):
        super(AccEnv, self).__init__(config)
        # TODO: Access the parameters in engine.global_config instead of storing them in the class
        self.ego_init_position = ego_init_position
        self.num_followers = num_followers

    def setup_engine(self):
        super(AccEnv, self).setup_engine()
        # replace existing traffic manager
        self.engine.update_manager(
            "traffic_manager", 
            AccTrafficManager(self.ego_init_position, num_followers=self.num_followers
        )
    ) 
        
    # def reward_function(*args, **kwargs):
        
    #     return reward, kwargs
