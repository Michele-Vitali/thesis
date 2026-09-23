# Setup values
env_name = "CustomTask"
robot = "PRob3"
gripper_types = None

# Movements values
translation_k = 10.0 # Useful for deciding how strong the movement should be.
stationary_k = 1.5 # Determines how strong the gripper should close (avoids slipping!)
rotation_k = 0.5 # Influences the rollercoaster issue a lot!
rotation_ramp_steps = 20 # Number of steps to bring rotational velocity to itx max (avoids sudden bursts)
max_rot_speed = 0.3

# Reward and penalty factors
closeness = 0.1
grasp = 0.1
lift = 0.25
placing = 0.5
sudden_movements_penalty = 0.02
z_target = 0.3

# Other values
n_objects = 1
hold_steps = 5