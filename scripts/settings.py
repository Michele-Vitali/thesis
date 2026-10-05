# Setup values
env_name = "CustomTask"
robot = "PRob3"
gripper_types = "PRob3Gripper"


# Important variables
safe_offset_gripper = [0.0, 0.0, 0.1]

# Via empirical tests, we found out that the maximum height our robot can elevate is approximately 35cm TOTAL (so the elevation is 35cm - object_height)!
# BUT, to move the cube to the center, we found out that we can move the cube to a maximum height of 25cm (in reality 27cm, but we keep a safe margin from the actual limit!)
# => We want to elevate our cube to this 25cm!
ideal_z = 0.25
elevation_step = 0.02 # 2 cm

# Central drop point
drop_point = [0.0, 0.0]
drop_clearance = 0.01

# Lower the original front camera without changing its orientation
camera_lowering = 0.10 # 10 cm


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

# LeRobot dataset
dataset_root = "data"
dataset_name = "prob3_pick_place"

dataset_fps = 20

dataset_camera_name = "frontview"
dataset_camera_observation_key = "frontview_image"

dataset_image_width = 256
dataset_image_height = 256

dataset_use_videos = False

dataset_task = "Pick up the cube and place it at the center of the table"

gripper_q_min = -0.001
gripper_q_max = 1.0472

# Controller configurations
osc_config = {
    "impednace_mode": "fixed",
    "kp": 150,
    "damping_ratio": 1,
    "control_delta": True,
    "input_min": -1.0,
    "input_max": 1.0,
    "output_min": [-0.05, -0.05, -0.05, -0.5, -0.5, -0.5],
    "output_max": [+0.05, +0.05, +0.05, +0.5, +0.5, +0.5]
}