# Setup values
env_name = "CustomTask"
robot = "PRob3"
gripper_types = "PRob3Gripper"

# Important variables
safe_offset_gripper = [0.0, 0.0, 0.1]
# Via empirical tests, we found out that the maximum height our robot can elevate is approximately 35cm TOTAL (so the elevation is 35cm - object_height)!
# BUT, t move the cube to the center, we found out that we can move the cube to a maximum height of 25cm (in reality 27cm, but we keep a safe margin from the actual limit!)
# => We want to elevate our cube to this 25cm!
ideal_z = 0.25
elevation_step = 0.02 # 2 cm

# Perturbation variables
initial_radius = 0.01 # 1cm
radius_step = 0.01
max_radius = 0.05
n_candidates = 8

# Local IK preview variables
preview_local_seeds = 2
preview_seed_perturbation = 0.03
preview_position_iterations = 30
preview_pose_iterations = 120

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