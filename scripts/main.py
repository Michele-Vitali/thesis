import mujoco
import numpy as np
import robosuite as suite

from robosuite import load_controller_config, macros
from robosuite.wrappers import DomainRandomizationWrapper
from scipy.spatial.transform import Rotation as R

import custom_env  # noqa: F401
import custom_robot  # noqa: F401

# Import utilities files
import movements

# Import the settings file
import settings


# Compare the old initial pose with a top-down pose already found valid in our kinematic tests.
INIT_POSES = {
    "OLD INIT": np.array([0.0, -0.366, 0.8, 0.0, 1.137, 0.0]),
    "TOP-DOWN INIT": np.array([0.0, 0.05473, 1.23560, 0.0, 1.85127, 0.0]),
}

PREGRASP_HEIGHT = 0.10
POSITION_TOLERANCE = 0.01
ROTATION_TOLERANCE = 1.0
MIN_JOINT_MARGIN = 0.02


def main():
    n_tests = int(input("How many random tests for each init pose? "))

    all_results = {}

    for init_name, init_qpos in INIT_POSES.items():
        print("\n" + "=" * 70)
        print(init_name)
        print("=" * 70)

        config = load_controller_config(default_controller="OSC_POSE")
        env = create_randomized_env(config, init_qpos)

        results = []

        try:
            for i in range(n_tests):
                # Same seed index for both init poses, so the comparison is as fair as possible.
                np.random.seed(i)
                env.reset()

                if env.has_renderer:
                    env.render()

                result = test_osc_pose(env, i)
                results.append(result)
        finally:
            env.close()

        all_results[init_name] = results
        print_report(init_name, results)

    print_comparison(all_results)


def create_randomized_env(config: dict, arm_qpos: np.ndarray) -> DomainRandomizationWrapper:
    # As the docs say, we use this so that entire geom groups are randomized as a whole
    macros.USING_INSTANCE_RANDOMIZATION = True

    # Create the base environment
    env = suite.make(
        env_name=settings.env_name,
        robots=settings.robot,
        gripper_types=settings.gripper_types,
        has_renderer=True,
        has_offscreen_renderer=False,
        use_camera_obs=False,
        control_freq=20, # Limits the robot to 20 actions/second (5 for testing in lab...)
        controller_configs=config,
        hard_reset=False, # Avoids segfault on macos or glfw error on Linux (per docs...)
        horizon=50000, # So we are sure that all the pick and places terminate
        reward_shaping=True, # So we enable RL success check
    )

    # Set the arm initial configuration directly on the instantiated robosuite robot.
    robot_init_qpos = np.asarray(env.robots[0].init_qpos, dtype=float).copy()
    robot_init_qpos[:6] = np.asarray(arm_qpos, dtype=float)
    env.robots[0].init_qpos = robot_init_qpos

    # We use domain randomization to create a more robust dataset
    env = DomainRandomizationWrapper(
        env,
        randomize_color=False,
        randomize_lighting=True,
        randomize_camera=False,
        randomize_dynamics=False, # Breaks the whole robot, but seems useful for the future...
        randomize_on_reset=True,
        randomize_every_n_steps=0, # Randomization must not happen during the episode
    )

    env.reset()

    if env.has_renderer:
        if getattr(env.sim, "_render_context_offscreen", None) is not None:
            env.sim._render_context_offscreen.vopt.flags[mujoco.mjtVisFlag.mjVIS_RANGEFINDER] = 0
        env.render()

    return env


def test_osc_pose(env, test_index: int) -> dict:
    obj = env.objects[0]
    obj_name = obj.root_body
    obj_id = env.sim.model.body_name2id(obj_name)

    obj_pos = env.sim.data.body_xpos[obj_id].copy()
    obj_rot = env.sim.data.body_xquat[obj_id].copy()

    over_obj_pos = obj_pos + np.array([0.0, 0.0, PREGRASP_HEIGHT])

    movement_ctrl = movements.MovementController(env)
    n_faces = movement_ctrl.determine_n_faces(obj)

    initial_obs = env._get_observations()
    initial_q, initial_margins = get_arm_joint_data(env)
    initial_rotation = R.from_quat(initial_obs["robot0_eef_quat"])
    initial_tool_z = initial_rotation.as_matrix()[:, 2]
    initial_tilt = angle_from_down(initial_tool_z)

    print("\n" + "-" * 70)
    print(f"TEST {test_index + 1}")
    print("-" * 70)
    print("OBJECT POS:", np.round(obj_pos, 5))
    print("TARGET POS:", np.round(over_obj_pos, 5))
    print("INITIAL Q:", np.round(initial_q, 5))
    print("INITIAL TOOL +Z:", np.round(initial_tool_z, 5))
    print(f"INITIAL TILT FROM DOWN: {initial_tilt:.3f} deg")
    print(f"INITIAL MIN JOINT MARGIN: {np.nanmin(initial_margins):.5f} rad")

    # 1. Reach the pre-grasp position using the original OSC translation logic.
    movement_ctrl.complex_traslation(
        over_obj_pos,
        "Move over",
        max_steps=300,
        tolerance=0.005
    )

    # 2. Compute exactly the same top-down grasp orientation used by the old MovementController.
    obs = movement_ctrl._obs_init()
    target_rotation = movement_ctrl.optimal_eef_rotation(
        obj_rot,
        obs["robot0_eef_quat"],
        n_faces
    )

    # 3. Reach that orientation using OSC_POSE while actively correcting position.
    rotation_steps, rotation_reward = movement_ctrl.rotation(
        obs,
        target_rotation,
        "Rotate",
        max_steps=300,
        tolerance=ROTATION_TOLERANCE,
    )

    movement_ctrl.steps += rotation_steps
    movement_ctrl.reward += rotation_reward

    # 4. Measure the final state.
    final_obs = movement_ctrl._obs_init()
    final_pos = final_obs["robot0_eef_pos"].copy()
    final_rotation = R.from_quat(final_obs["robot0_eef_quat"])

    position_error = np.linalg.norm(over_obj_pos - final_pos)

    rotation_error = np.degrees(
        (target_rotation * final_rotation.inv()).magnitude()
    )

    tool_z = final_rotation.as_matrix()[:, 2]
    tilt_from_down = angle_from_down(tool_z)

    final_q, final_margins = get_arm_joint_data(env)
    min_joint_margin = np.nanmin(final_margins)

    position_ok = position_error < POSITION_TOLERANCE
    rotation_ok = rotation_error < ROTATION_TOLERANCE
    joints_ok = min_joint_margin > MIN_JOINT_MARGIN

    success = position_ok and rotation_ok and joints_ok

    print("FINAL EEF POS:", np.round(final_pos, 5))
    print(f"POSITION ERROR: {position_error * 1000:.3f} mm")
    print(f"ROTATION ERROR: {rotation_error:.3f} deg")
    print("TOOL +Z:", np.round(tool_z, 5))
    print(f"TILT FROM DOWN: {tilt_from_down:.3f} deg")
    print("FINAL Q:", np.round(final_q, 5))
    print("FINAL JOINT MARGINS:", np.round(final_margins, 5))
    print(f"MIN JOINT MARGIN: {min_joint_margin:.5f} rad")
    print("RESULT:", "SUCCESS" if success else "FAIL")

    return {
        "success": success,
        "position_error": position_error,
        "rotation_error": rotation_error,
        "tilt": tilt_from_down,
        "q": final_q,
        "joint_margins": final_margins,
    }


def get_arm_joint_data(env) -> tuple[np.ndarray, np.ndarray]:
    robot = env.robots[0]

    qpos_indexes = np.asarray(
        robot._ref_joint_pos_indexes,
        dtype=int
    )

    q = env.sim.data.qpos[qpos_indexes].copy()

    margins = []

    for qpos_index, joint_q in zip(qpos_indexes, q):
        joint_ids = np.where(
            np.asarray(env.sim.model.jnt_qposadr) == qpos_index
        )[0]

        if len(joint_ids) == 0:
            margins.append(np.nan)
            continue

        joint_id = int(joint_ids[0])

        lower_limit, upper_limit = env.sim.model.jnt_range[joint_id]

        margin = min(
            joint_q - lower_limit,
            upper_limit - joint_q
        )

        margins.append(margin)

    return q, np.asarray(margins)


def angle_from_down(tool_z: np.ndarray) -> float:
    return np.degrees(
        np.arccos(
            np.clip(
                np.dot(
                    tool_z,
                    np.array([0.0, 0.0, -1.0])
                ),
                -1.0,
                1.0,
            )
        )
    )


def print_report(init_name: str, results: list[dict]):
    successes = sum(
        result["success"]
        for result in results
    )

    position_errors = np.array([
        result["position_error"]
        for result in results
    ])

    rotation_errors = np.array([
        result["rotation_error"]
        for result in results
    ])

    joint_margins = np.array([
        np.nanmin(result["joint_margins"])
        for result in results
    ])

    print("\n" + "=" * 70)
    print(f"REPORT - {init_name}")
    print("=" * 70)

    print(
        f"Successes: {successes}/{len(results)} "
        f"({100 * successes / len(results):.1f}%)"
    )

    print(
        f"Average position error: "
        f"{np.mean(position_errors) * 1000:.3f} mm"
    )

    print(
        f"Maximum position error: "
        f"{np.max(position_errors) * 1000:.3f} mm"
    )

    print(
        f"Average rotation error: "
        f"{np.mean(rotation_errors):.3f} deg"
    )

    print(
        f"Maximum rotation error: "
        f"{np.max(rotation_errors):.3f} deg"
    )

    print(
        f"Minimum joint margin seen: "
        f"{np.min(joint_margins):.5f} rad"
    )

    print("=" * 70)


def print_comparison(all_results: dict[str, list[dict]]):
    print("\n\n" + "=" * 70)
    print("FINAL OSC_POSE COMPARISON")
    print("=" * 70)

    for init_name, results in all_results.items():
        successes = sum(
            result["success"]
            for result in results
        )

        average_rotation_error = np.mean([
            result["rotation_error"]
            for result in results
        ])

        minimum_joint_margin = min(
            np.nanmin(result["joint_margins"])
            for result in results
        )

        print(
            f"{init_name}: "
            f"{successes}/{len(results)} successes | "
            f"avg rot err {average_rotation_error:.3f} deg | "
            f"min joint margin {minimum_joint_margin:.5f} rad"
        )

    print("=" * 70)


if __name__ == "__main__":
    main()