
# %%
from IPython.display import clear_output, Image
from metadrive.component.map.base_map import BaseMap
from metadrive.component.map.pg_map import MapGenerateMethod
from acc_env import AccEnv

# %%

# Define straight road map
map_config={BaseMap.GENERATE_TYPE: MapGenerateMethod.BIG_BLOCK_SEQUENCE,
            BaseMap.GENERATE_CONFIG: "SSSSS",
            BaseMap.LANE_WIDTH: 10, 
            BaseMap.LANE_NUM: 1}

# Create environment
ego_init_position=(100, 0)
env = AccEnv(
    dict(
        map_config=map_config, 
        vehicle_config=dict(spawn_longitude=ego_init_position[0], spawn_lateral=ego_init_position[1],)
    ), 
    ego_init_position=ego_init_position,
    num_followers=2
)

try:
    env.reset()
    for _ in range(100):
        env.step([0, 0.0]) # ego car is static
        env.render(mode="topdown", 
                   window=False,
                   screen_size=(400, 400),
                   camera_position=(120, 7),
                   scaling=2,
                   screen_record=True,
                   text={"Has vehicle": bool(len(env.engine.traffic_manager.spawned_objects)),
                         "Timestep": env.episode_step})
    assert env
    env.top_down_renderer.generate_gif()
finally:
    env.close()
    clear_output()
Image(open("demo.gif", 'rb').read())


# %%

