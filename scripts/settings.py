import numpy as np

# Setup values
env_name = "CustomTask"
robot = "PRob3"
gripper_types = "PRob3Gripper"
render_on_screen = True

# Adaptive motion.
position_far_threshold = 0.03
position_mid_threshold = 0.01

position_gain_far = np.array([0.45, 0.45, 0.45])
position_gain_mid = np.array([0.30, 0.30, 0.30])
position_gain_near = np.array([0.20, 0.20, 0.20])

position_action_far = 0.65
position_action_mid = 0.50
position_action_near = 0.40

rotation_far_threshold_deg = 10.0
rotation_mid_threshold_deg = 3.0

rotation_gain_far = 0.20
rotation_gain_mid = 0.14
rotation_gain_near = 0.10

rotation_action_far = 0.60
rotation_action_mid = 0.50
rotation_action_near = 0.40

# Integral controller implementation
xyz_integral_gain = np.array([0.06, 0.06, 0.36])
xyz_integral_limit = np.array([0.012, 0.012, 0.025])
xyz_integral_deadband = np.array([0.00075, 0.00075, 0.00100])
xyz_integral_action_limit = np.array([0.035, 0.035, 0.090])

# Derivative gains
xyz_derivative_gain = np.array([0.0, 0.0, 0.0])
xyz_derivative_action_limit = np.array([0.03, 0.03, 0.03])

xyz_derivative_filter_alpha = 0.20
xyz_pid_q5_freeze_threshold = 1.97

# Environment variables
max_reachable_x = 0.55 # It is 60cm, but we take a margin and set it to 55cm
rob_x = 0.45 # It is a minimum of 29cm, taking a margin of 1cm...
table_width = 0.30 # We make a bigger table than reachable, then define a suitable spawning algorithm!
robot_position = np.array([-rob_x, 0.0, 0.0])

# Object geometry
object_half_size_min = 0.035
object_half_size_max = 0.035

max_object_height = 2.0 * object_half_size_max

# Object spawning variables
forbidden_spawning_pos = (0.0, 0.0) # As it is the drop point!
forbidden_spawning_radius = 0.08
radial_range = [0.54, 0.55] # Stable pre-waypoint working range

# Important variables
safe_offset_gripper = [0.0, 0.0, 0.115]

# After some testing we found out that some ideal z values are:
maximum_transport_z = 0.195
transport_bottom_clearance = max_object_height
initial_lift_clearance = 0.010
pregrasp_top_clearance = 0.015

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
camera_pos = np.array([1.6, 0.0, 1.05])
camera_quat = np.array([0.561, 0.431, 0.431, 0.561])

# Starting pose for the robot sim
starting_pose = np.array([0.0, -0.366, 0.800, 0.0, 1.137, 0.0]) # Top-down grasp

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
transport_position_tolerance = 0.004
transport_rotation_tolerance = 2.0
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
# kp and damping ratio configurations...
free_kp = np.full(6, float(150))
free_damping = np.full(6, 1.0)

grasp_kp = np.array([250, 250, 300, 150, 150, 150])
grasp_damping = np.array([1.15, 1.15, 1.30, 1.0, 1.0, 1.0])


osc_config = {
    "impedance_mode": "fixed",
    "kp": free_kp,
    "damping_ratio": free_damping,
    "control_delta": True,
    "input_min": -1.0,
    "input_max": 1.0,
    "output_min": np.array([-0.05, -0.05, -0.05, -0.5, -0.5, -0.5]),
    "output_max": np.array([+0.05, +0.05, +0.05, +0.5, +0.5, +0.5]),
    # P-Rob3 q5 joint-limit avoidance
    "q5_index": 4,
    "q5_avoidance_start": 1.96,
    "q5_avoidance_full": 1.985,
    "q5_avoidance_max_torque": 4.0,
    "q5_avoidance_damping": 0.75,
#    "q5_orientation_relax_start": 1.94,
#    "q5_orientation_relax_full": 1.985,
#    "q5_orientation_min_scale": 0.40,
#    "q5_escape_enter": 1.97,
#    "q5_escape_exit": 1.91,
#    "q5_escape_orientation_scale": 0.05,
}