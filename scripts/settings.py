# Setup values
env_name = "CustomTask"
robot = "PRob3"
gripper_types = "PRob3Gripper"

# Important variables
safe_offset_gripper = [0.0, 0.0, 0.1]
desired_elevation = 0.1 # 10cm
elevation_step = 0.02 # 2 cm

# Perturbation variables
initial_radius = 0.01 # 1cm
radius_step = 0.01
max_radius = 0.05
n_candidates = 8

# Starting pose for the robot sim
starting_pose = [0.0, -0.366, 0.800, 0.0, 1.137, 0.0]

# Reward and penalty factors
closeness = 0.1
grasp = 0.1
lift = 0.25
placing = 0.5
sudden_movements_penalty = 0.02
z_target = 0.3

# Other values
n_objects = 1