# Setup values
env_name = "CustomTask"
robot = "PRob3"
gripper_types = "PRob3Gripper"

# Environment variables
max_reachable_x = 0.55 # It is 60cm, but we take a margin and set it to 55cm
rob_x = 0.45 # It is a minimum of 29cm, taking a margin of 1cm...
table_width = 0.30 # We make a bigger table than reachable, then define a suitable spawning algorithm!
robot_position = [-rob_x, 0.0, 0.0]

# Object geometry
object_half_size_min = 0.035
object_half_size_max = 0.035

max_object_height = 2.0 * object_half_size_max

# Object spawning variables
forbidden_spawning_pos = (0.0, 0.0) # As it is the drop point!
forbidden_spawning_radius = 0.08
radial_range = [0.35, 0.50] # Stable pre-waypoint working range

# Important variables
safe_offset_gripper = [0.0, 0.0, 0.115]

# After some testing we found out that some ideal z values are:
maximum_transport_z = 0.195
transport_bottom_clearance = max_object_height
initial_lift_clearance = 0.010
pregrasp_top_clearance = 0.015

# Integral controller implementation
z_integral_gain = 0.60
z_integral_limit = 0.025
z_integal_deadband = 0.001
z_integral_action_limit = 0.15

# Workspace
workspace_direct_lift_min_radius = 0.40
workspace_direct_lift_max_radius = 0.50
#maximum_distance_from_center = 0.50
#workspace_correction_step = 0.02

# Central drop point
drop_point = [0.0, 0.0]
drop_clearance = 0.01

# Lower the original front camera without changing its orientation (the following values have been collected by testing)
# camera_init_pos = [1.6, 0.0, 1.45]
# camera_init_quat = [0.561, 0.431, 0.431, 0.561]
camera_pos = [1.6, 0.0, 1.05]
camera_quat = [0.561, 0.431, 0.431, 0.561]

# Starting pose for the robot sim
starting_pose = [0.0, -0.366, 0.800, 0.0, 1.137, 0.0] # Top-down grasp

# Reward and penalty factors
closeness = 0.1
grasp = 0.1
lift = 0.25
placing = 0.5
sudden_movements_penalty = 0.02
z_target = 0.3

# Tolerances
position_tolerance = 0.003 # 3mm
rotation_tolerance_deg = 1.0 # 1 degree
final_tolerance = 0.005

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
    "impedance_mode": "fixed",
    "kp": 150,
    "damping_ratio": 1,
    "control_delta": True,
    "input_min": -1.0,
    "input_max": 1.0,
    "output_min": [-0.05, -0.05, -0.05, -0.5, -0.5, -0.5],
    "output_max": [+0.05, +0.05, +0.05, +0.5, +0.5, +0.5],
    # P-Rob3 q5 joint-limit avoidance
    "q5_index": 4,
    "q5_avoidance_start": 1.94,
    "q5_avoidance_full": 1.985,
    "q5_avoidance_max_torque": 4.0,
    "q5_avoidance_damping": 0.75,
}