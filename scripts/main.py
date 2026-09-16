import robosuite as suite
import numpy as np

# Import the custom robot and environment so they are initialized
import custom_robot
import custom_env

# Create the base environment
env = suite.make(
    env_name="CustomTask",
    robots="CustomUR5e",
    has_renderer=True,
    has_offscreen_renderer=False,
    use_camera_obs=False,
    control_freq=20 # Limits the robot to 20 actions/second
)

env.reset()

# Fake action to trick the engine into rendering, but maintaining the robot still
fake_action = np.zeros(env.action_dim)

print("Simulation started!")

while True:
    env.render()
    env.step(fake_action)