
import numpy as np
from scipy.spatial.transform import Rotation as R

def camera_look_at(position, target):
    position = np.asarray(position, dtype=float)
    target = np.asarray(target, dtype=float)

    forward = target - position
    forward /= np.linalg.norm(forward)

    reference_up = np.array([0.0, 0.0, 1.0])

    right = np.cross(forward, reference_up)
    right /= np.linalg.norm(right)

    up = np.cross(right, forward)

    rotation = np.column_stack([
        right,
        up,
        -forward,
    ])

    return R.from_matrix(rotation).as_quat(scalar_first=True)


camera_pos = [0.10, 0, 0.12]
grasp_target = [0.0, 0.0, 0.30]

quat = camera_look_at(camera_pos, grasp_target)

print("Camera position:", camera_pos)
print("Camera quaternion:", quat)
