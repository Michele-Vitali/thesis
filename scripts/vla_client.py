# This file is the client for OpenVLA inference and action decisions.

import robosuite as suite
import settings
from robosuite import load_controller_config, macros
from robosuite.environments.manipulation.single_arm_env import SingleArmEnv


def main():

    print("VLA client script started!")

    # Create the OSC_POSE controller (not the custom, as the model should already be trained to avoid 15 joint limit!)
    config = load_controller_config(default_controller="OSC_POSE")
    config.update(settings.osc_config)

    # Create the simulation environment
    create_env(config)

def create_env(config: dict | None):
    """This function returns the built simulation environment with some configs chosen in the 'settings' file and the passed controller configs.

    Args:
        config (dict | None): The controller configurations. Defaults to None.
    
    Returns:
        SingleArmEnv: The built simulation environment.
    """

    # Create the environment.
    env = suite.make(
        env_name=settings.env_name,
        robots=settings.robot,
        gripper_types=settings.gripper_types,
        has_renderer=settings.render_on_screen, # On-screen renderer, usefule for debug, but slow for dataset gathering...
        has_offscreen_renderer=True, # Necessary for the dataset!
        use_camera_obs=True, # Also for dataset...
        camera_names=settings.dataset_camera_name,
        camera_heights=settings.dataset_image_height,
        camera_widths=settings.dataset_image_width,
        camera_depths=False,
        control_freq=settings.dataset_fps,
        controller_configs=config,
        hard_reset=False, # Avoids segfault on macos or glfw error on Linux (per docs...)
        horizon=1000, # So we are sure that all the pick and places terminate
        reward_shaping=True, # Not useful in our case but had implemented it anyway...
    )

    # Move the robot into the starting position.

if __name__ == "__main__":
    main()