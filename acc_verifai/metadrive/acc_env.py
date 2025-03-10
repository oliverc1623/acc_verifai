from acc_traffic_manager import AccTrafficManager

from metadrive.envs import MetaDriveEnv


class AccEnv(MetaDriveEnv):
    """Environment class for ACC theat model with configurable number of follower vehicles."""

    def __init__(self, config: dict, ego_init_position: tuple[int, int], num_followers: int = 2):
        super().__init__(config)
        # TODO: Access the parameters in engine.global_config instead of storing them in the class
        self.ego_init_position = ego_init_position
        self.num_followers = num_followers

    def setup_engine(self) -> None:
        """Set up the engine with a custom traffic manager."""
        super().setup_engine()
        # replace existing traffic manager
        self.engine.update_manager(
            "traffic_manager",
            AccTrafficManager(
                self.ego_init_position,
                num_followers=self.num_followers,
            ),
        )
