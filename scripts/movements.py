import numpy as np
from scipy.spatial.transform import Rotation as R
import settings
import time

def move_to_target(env, target_pos, max_steps=150, tolerance=0.005):
    """
    We always first move over the target and then we descend.
    We add maximum steps and tolerance because perfect positioning can be a complex problem.
    """

    # Fake action just to initalize obs.
    action = np.zeros(env.action_dim)
    obs, reward, done, info = env.step(action)

    steps = 0

    # Now we start to move.
    for step in range(max_steps):
        # eef stands for End-Effector, which in our case is the gripper!
        # 1. Take the gripper position.
        eef_pos = (0,0,0) if not obs else obs["robot0_eef_pos"]

        # 2. Compute the distance.
        delta_pos = target_pos - eef_pos

        # np.linalg.norm just computes the euclidean distance between target_pos and eef_pos.
        distance = np.linalg.norm(delta_pos)

        if distance < tolerance:
            # We reached our target.
            print(f"Target reached within {steps} steps! Final distance: {distance:.2f}cm")
            return True

        # 3.a If not reached the target, compute proportional action
        action[0:3] = np.clip(settings.translation_k * delta_pos, -0.7, 0.7) # We proportionally move, clipped to abs 0.7 to not move at max speed

        # 4. Execute
        obs, reward, done, info = env.step(action)

        steps += 1
        # Logging for debug
        #print(f"Step: {steps}, Distance: {distance*100:.2f} cm, Action done: {action[0:3]}")

    # Eventually after 'max_steps' steps the loop finishes...
    print(f"Target not reached even in {steps} steps...")
    return False

def rotate(env, target_rot, max_steps=150, tolerance=1.0):
    # Similar to move_to_target...
    # Fake action just to initalize obs.
    action = np.zeros(env.action_dim)
    obs, reward, done, info = env.step(action)

    steps = 0

    # Now we start to move.
    for step in range(max_steps):
        # 1. Take eef and object quaternions and convert them to scipy quaternions (they have a different ordering convention...)
        eef_quat = (0,0,0) if not obs else obs["robot0_eef_quat"]
        eef_quat_scipy = [eef_quat[1], eef_quat[2], eef_quat[3], eef_quat[0]]

        object_id = env.sim.model.body_name2id(env.objects[0].root_body)
        object_quat = env.sim.data.body_xquat[object_id]
        object_quat_scipy = [object_quat[1], object_quat[2], object_quat[3], object_quat[0]]

        eef_rotation = R.from_quat(eef_quat_scipy)
        object_rotation = R.from_quat(object_quat_scipy)

        # 2. Orientate the gripper remembering it is over the cube! (Must point below...)
        object_yaw = object_rotation.as_euler('zyx')[0]
        shortest_yaw = (object_yaw + np.pi/4) % (np.pi/2) - np.pi/4 # The robot choose the closest face of the object avoiding rotating more than 45°

        yaw_rotation = R.from_euler('z', shortest_yaw)
        downward_rotation = R.from_euler('y', np.pi)
        # 3. Compute the target orientation
        target_rotation = yaw_rotation * downward_rotation

        # 4. Compute the rotation difference
        delta_rotation = target_rotation * eef_rotation.inv()

        # 5. Convert in a rotation vector (axis-angle) for the OSC_POSE Controller
        # In this way we also avoid 'Gimbal Lock' problem!
        delta_rotvec = delta_rotation.as_rotvec() #[rx, ry, rz] in radiants

        # 6. Check if already aligned
        degree_difference = abs(180 - np.degrees(delta_rotation.magnitude()))

        if step % 10 == 0:
            print(f"Step: {steps}, Degree difference: {degree_difference:.1f}")

        if degree_difference < tolerance:
            print(f"Aligned within {steps} steps! Final difference: {degree_difference:.2f}°")
            return True

        # 7. Generate the action
        action[0:3] = 0.0
        action[3:6] = np.clip(settings.rotation_k * delta_rotvec, -0.7, 0.7)

        # 8. Execute
        obs, reward, done, info = env.step(action)
        
        steps += 1
        # Logging for debug
        #print(f"Step: {steps}, Action done: {action[0:3]}")
        
    # Eventually after 'max_steps' steps the loop finishes...
    print(f"Target not reached even in {steps} steps...")
    return False

def toggle_grab(env, grab, min_steps=30):
    action = np.zeros(env.action_dim)
    if grab:
        action[-1] = 1.0
    else:
        action[-1] = -1.0

    # Keep closing for 'min_steps' otherwise the gripper won't completely close itself and miss the object.
    for i in range(min_steps):
        env.step(action)