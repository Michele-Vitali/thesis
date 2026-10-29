from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation as R

# Setup values
env_name = "CustomTask"
robot = "PRob3"
gripper_types = "PRob3Gripper"
render_on_screen = True    # Set to false for faster simulations
record = False

# Environment variables
max_reachable_x = 0.55 # It is 60cm, but we take a margin and set it to 55cm
rob_x = 0.45 # It is a minimum of 29cm, taking a margin of 1cm...
table_width = 0.30 # We make a bigger table than reachable, then define a suitable spawning algorithm!
robot_position = np.array([-rob_x, 0.0, 0.0])
table_height = 0.1

# Object geometry
object_half_size_min = 0.010
object_half_size_max = 0.035

max_object_height = 2.0 * object_half_size_max

recreate_env_every = 5

# Object spawning variables
forbidden_spawning_pos = (0.0, 0.0) # As it is the drop point!
forbidden_spawning_radius = 0.08
radial_range = [0.35, 0.55] # Stable pre-waypoint working range

# Important variables
safe_offset_gripper = [0.0, 0.0, 0.115]

# OSC actions variables
gains = {
    "position": {
        "approach": 1.0,
        "descend": 0.14,
        "lift": 0.36,
        "transport": 1.0,
        "lower": 0.14,
    },
    "rotation": {
        "align": 0.57
    }
}

max_action_delta = 0.05 # This ensures we do not "jump" too many bins (useful for ensuring smoother model jumps and predictions!) This is an implicit way of introducing a ramp action, avoids spikes in commands!
max_position_action = 0.50  # Safe maximum output so the robot stays in a controllable speed
max_rotation_action = 0.50

gripper_transition_seconds = 1.0

expert_max_segment_seconds = 20.0

# Integral Z payload compensation
z_integral_gain = 0.36
z_integral_limit = 0.025
z_integral_deadband = 0.001
z_integral_action_limit = 0.09

# After some testing we found out that some ideal z values are:
maximum_transport_z = 0.195
transport_bottom_clearance = max_object_height
initial_lift_clearance = 0.010
pregrasp_top_clearance = 0.015

# Central drop point
drop_point = [0.0, 0.0]
drop_clearance = 0.01

# Lower the original front camera without changing its orientation (the following values have been collected by testing)
# camera_init_pos = [1.6, 0.0, 1.45]
# camera_init_quat = [0.561, 0.431, 0.431, 0.561]
camera_target = np.array([0.0, 0.0, 0.10])
camera_pos = np.array([-0.40, -0.55, 0.64])

#camera_pos = np.array([-1.0, -2.0, -2.0]) Camera for debugging...

# Compute the quaternion as manually setting it is an hell...
def compute_camera_quat(camera_target: np.ndarray, camera_pos: np.ndarray) -> np.ndarray:
    forward = camera_target - camera_pos
    norm = np.linalg.norm(forward)

    if norm < 1e-8:
        raise ValueError("The quaternion is impossible to compute.")

    forward /= norm
    world_up = np.array([0.0, 0.0, 1.0])
    right = np.cross(forward, world_up)
    right_norm = np.linalg.norm(right)

    if right_norm < 1e-8:
        raise ValueError("Camera direction parallel to world z.")

    right /= right_norm

    up = np.cross(right, forward)

    rotation_matrix = np.column_stack([right, up, -forward])
    quat = R.from_matrix(rotation_matrix).as_quat(scalar_first=True)

    return np.asarray(quat)

camera_quat = np.array(compute_camera_quat(camera_target, camera_pos))

# Rendering
camera_shadows = True
camera_shadow_ambient = 0.35
camera_shadow_diffuse = 0.45
camera_light_pos = np.array([-0.20, -0.35, 0.55])

# Starting pose for the robot sim
starting_pose = np.array([0.0, 0.0, 1.5, 0.0, 1.8, 0.0]) # Top-down grasp


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
transport_position_tolerance = 0.005
transport_rotation_tolerance = 2.0
final_tolerance = 0.005

# Other values
n_objects = 1

# RLDS dataset
dataset_root = str(Path(__file__).resolve().parents[1] / "hdf5_data")
dataset_name = "v7_overfit_1k_no_randomization"

dataset_fps = 5

# External fixed camera
dataset_external_camera_name = "frontview"
dataset_external_camera_observation_key = "frontview_image"
#dataset_camera_fovy = 70.0

# Gripper-connected wrist camera
dataset_wrist_camera_name = "wristview"
dataset_wrist_camera_observation_key = "wristview_image"

# Useful for robosuite to register the cameras...
dataset_camera_names = [dataset_external_camera_name, f"all-{dataset_wrist_camera_name}"]

dataset_image_width = 256
dataset_image_height = 256

dataset_use_videos = False

dataset_task = "Pick up the object and place it straight at the center of the table"

gripper_q_min = -0.001
gripper_q_max = 1.0472

# Controller configurations
# kp and damping ratio configurations...
free_kp = np.full(6, float(150))
free_damping = np.full(6, 1.0)

grasp_kp = np.array([250, 250, 300, 150, 150, 150])
grasp_damping = np.array([1.15, 1.15, 1.30, 1.0, 1.0, 1.0])

test_kp = np.array([250, 250, 300, 150, 150, 150])
test_damping = np.array([1.15, 1.15, 1.30, 1.0, 1.0, 1.0])

osc_config = {
    "type": "OSC_POSE",
    "interpolate": True,
    "ramp_ratio": 0.1,
    "interpolation": "linear",
    "impedance_mode": "fixed",
    "kp": test_kp,
    "damping_ratio": test_damping,
    "control_delta": True,
    "input_min": -1.0,
    "input_max": 1.0,
    "output_min": np.array([-0.05, -0.05, -0.05, -0.5, -0.5, -0.5]),
    "output_max": np.array([+0.05, +0.05, +0.05, +0.5, +0.5, +0.5]),
    # P-Rob3 q5 joint-limit avoidance
    "q5_index": 4,
    "q5_avoidance_start": 1.94,
    "q5_avoidance_full": 1.985,
    "q5_avoidance_max_torque": 4.0,
    "q5_avoidance_damping": 0.75,
}

# OpenVLA Client/Server configurations
server_host = "127.0.0.1"
server_port = 8000
server_url = f"http://{server_host}:{server_port}"
unnorm_key = "prob3_pick_place"
server_timeout = 120 # 120 s = 2 mins
openvla_path = "runs/openvla/openvla-7b+prob3_pick_place+b8+lr-5e-05+lora-r64+dropout-0.0--v7_overfit_test_5_demos_second_try--1400_chkpt"