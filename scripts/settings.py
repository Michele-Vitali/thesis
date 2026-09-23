# Setup values
env_name = "CustomTask"
robot = "CustomUR5e"
gripper_types = "default"

# Movements values
translation_k = 10.0 # Useful for deciding how strong the movement should be.
stationary_k = 1.5
rotation_k = 0.5
rotation_ramp_steps = 20 # Number of steps to bring rotational velocity to itx max (avoids sudden bursts)
max_rot_speed = 0.3

# Other values
n_objects = 1
hold_steps = 5