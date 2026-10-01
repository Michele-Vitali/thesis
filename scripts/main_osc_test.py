import time

import custom_env  # noqa: F401
import custom_robot  # noqa: F401
import mujoco
import numpy as np
import robosuite as suite

import settings

from movements_osc import OSCMovementController
from robosuite import load_controller_config, macros
from robosuite.environments.manipulation.single_arm_env import SingleArmEnv
from robosuite.wrappers import DomainRandomizationWrapper


def main():

    # -------------------------------------------------------------
    # OSC_POSE CONTROLLER
    # -------------------------------------------------------------
    config = load_controller_config(
        default_controller="OSC_POSE"
    )

    # Fixed impedance OSC.
    config["impedance_mode"] = "fixed"

    config["kp"] = 150
    config["damping_ratio"] = 1

    # Cartesian actions are interpreted as deltas from the
    # current EEF pose.
    config["control_delta"] = True

    # Decouple translational and rotational operational-space
    # control as much as possible.
    config["uncouple_pos_ori"] = True

    # Explicit action scaling.
    #
    # normalized +/-1 corresponds to:
    #   +/- 5 cm translation
    #   +/- 0.5 rad rotation
    config["input_min"] = -1
    config["input_max"] = +1

    config["output_min"] = [
        -0.05,
        -0.05,
        -0.05,
        -0.5,
        -0.5,
        -0.5,
    ]

    config["output_max"] = [
        +0.05,
        +0.05,
        +0.05,
        +0.5,
        +0.5,
        +0.5,
    ]

    task_loop = True

    while task_loop:

        n_tasks = int(
            input(
                "How many times do you want to "
                "simulate the OSC task? "
            )
        )

        results = []

        for i in range(n_tasks):

            success = pick_and_place_task(
                config,
                i
            )

            results.append(success)

        successes = sum(results)

        print("=" * 60)

        print(
            f"OSC final report: "
            f"{successes}/{n_tasks} successful tasks."
        )

        print("=" * 60)

        choice = input(
            "Want to repeat the tasks? (Y/N): "
        )

        task_loop = (
            choice.upper() == "Y"
        )


def create_randomized_env(
    config: dict
) -> DomainRandomizationWrapper:

    macros.USING_INSTANCE_RANDOMIZATION = True

    env = suite.make(
        env_name=settings.env_name,
        robots=settings.robot,
        gripper_types=settings.gripper_types,
        has_renderer=True,
        has_offscreen_renderer=False,
        use_camera_obs=False,
        control_freq=20,
        controller_configs=config,
        hard_reset=False,
        horizon=1000,
        reward_shaping=True,
    )

    env = DomainRandomizationWrapper(
        env,
        randomize_color=True,
        randomize_lighting=True,
        randomize_camera=True,
        randomize_dynamics=False,
        randomize_on_reset=True,
        randomize_every_n_steps=0,
    )

    env.reset()

    if env.has_renderer:

        if (
            getattr(
                env.sim,
                "_render_context_offscreen",
                None
            )
            is not None
        ):

            env.sim._render_context_offscreen.vopt.flags[
                mujoco.mjtVisFlag.mjVIS_RANGEFINDER
            ] = 0

        env.render()

    # -------------------------------------------------------------
    # IMPORTANT:
    # Check whether the P-Rob3 arm actuators are torque / motor
    # actuators, as required by robosuite's arm controllers.
    # -------------------------------------------------------------
    verify_arm_actuators_for_osc(env)

    print(
        f"Environment action dimension: "
        f"{env.action_dim}"
    )

    print(
        "Initial arm joints:",
        np.round(
            env.robots[0]._joint_positions,
            5
        )
    )

    return env


def verify_arm_actuators_for_osc(
    env: "SingleArmEnv"
):
    """
    Verify that the six P-Rob3 arm actuators are compatible with
    robosuite's torque-producing OSC controller.

    MuJoCo <position> actuators contain a non-zero affine bias term,
    while direct torque / motor actuators do not.
    """

    robot = env.robots[0]

    actuator_indexes = getattr(
        robot,
        "_ref_joint_actuator_indexes",
        None
    )

    if actuator_indexes is None:

        print(
            "WARNING: Could not automatically inspect "
            "the arm actuator types."
        )

        print(
            "Make sure the six PRob3 arm actuators "
            "are <motor>, not <position>."
        )

        return

    actuator_indexes = np.asarray(
        actuator_indexes,
        dtype=int
    )

    position_like_actuators = []

    for actuator_id in actuator_indexes:

        actuator_name = mujoco.mj_id2name(
            env.sim.model._model,
            mujoco.mjtObj.mjOBJ_ACTUATOR,
            int(actuator_id)
        )

        bias_prm = (
            env.sim.model.actuator_biasprm[
                actuator_id
            ]
        )

        # MuJoCo's <position> shortcut generates an affine
        # bias term whose second component is -kp.
        if abs(bias_prm[1]) > 1e-9:

            position_like_actuators.append(
                actuator_name
            )

    if position_like_actuators:

        raise RuntimeError(
            "\nOSC TEST ABORTED.\n"
            "The P-Rob3 arm appears to use MuJoCo "
            "<position> actuators:\n"
            f"{position_like_actuators}\n\n"
            "robosuite OSC_POSE produces joint torques. "
            "Replace ONLY the six arm actuators in "
            "PRob3.xml with <motor> actuators before "
            "running this test.\n"
            "Do NOT change the P-Grip actuator."
        )

    print(
        "Arm actuator check: OK for OSC test."
    )


def pick_and_place_task(
    config: dict,
    task_index: int
) -> bool:

    env = create_randomized_env(
        config
    )

    try:

        obj = env.objects[0]

        obj_name = obj.root_body

        obj_id = (
            env.sim.model.body_name2id(
                obj_name
            )
        )

        pos = (
            env.sim.data.body_xpos[
                obj_id
            ].copy()
        )

        quat = (
            env.sim.data.body_xquat[
                obj_id
            ].copy()
        )

        print("\n" + "=" * 60)

        print(
            f"OSC TEST {task_index + 1}"
        )

        print(
            "Initial object position:",
            np.round(pos, 5)
        )

        print(
            "Initial object quaternion:",
            np.round(quat, 5)
        )

        print("=" * 60)

        success = pick_and_place_action(
            env,
            obj,
            pos,
            quat,
            task_index
        )

        return success

    finally:

        env.close()


def pick_and_place_action(
    env: "SingleArmEnv",
    obj,
    pos: np.ndarray,
    quat: np.ndarray,
    task_index: int
) -> bool:

    obj_id = env.sim.model.body_name2id(
        obj.root_body
    )

    movements_ctrl = OSCMovementController(
        env,
        tracked_body_name=obj.root_body
    )

    total_start = time.perf_counter()

    # -------------------------------------------------------------
    # 1. MOVE TO GRASP POSE
    # -------------------------------------------------------------
    print("\n[1] Moving to grasp pose with OSC_POSE...")

    done = movements_ctrl.move_robot_to_position(
        pos,
        quat
    )

    if not done:

        print(
            "OSC failed while reaching "
            "the grasp pose."
        )

        return False

    # -------------------------------------------------------------
    # 2. CLOSE GRIPPER
    # -------------------------------------------------------------
    print("\n[2] Closing gripper...")

    movements_ctrl.toggle_grab()

    print(
        "Object position after grasp:",
        np.round(
            env.sim.data.body_xpos[
                obj_id
            ],
            5
        )
    )

    # -------------------------------------------------------------
    # 3. ELEVATE
    # -------------------------------------------------------------
    print("\n[3] Elevating object with OSC_POSE...")

    elevated_pos = movements_ctrl.elevate_obj(
        pos,
        quat
    )

    if elevated_pos is None:

        print(
            "OSC failed during elevation."
        )

        return False

    print(
        "Object position after elevation:",
        np.round(
            env.sim.data.body_xpos[
                obj_id
            ],
            5
        )
    )

    # -------------------------------------------------------------
    # 4. TRANSPORT TO TABLE CENTER
    # -------------------------------------------------------------
    print(
        "\n[4] Moving horizontally toward "
        "the table center with OSC_POSE..."
    )

    centered_pos = movements_ctrl.move_to_center(
        elevated_pos,
        quat
    )

    if centered_pos is None:

        print(
            "OSC failed while moving "
            "toward the table center."
        )

        return False

    print(
        "Object position after transport:",
        np.round(
            env.sim.data.body_xpos[
                obj_id
            ],
            5
        )
    )

    # -------------------------------------------------------------
    # 5. RELEASE
    # -------------------------------------------------------------
    print("\n[5] Releasing object...")

    movements_ctrl.toggle_grab()

    # Allow the object to settle.
    movements_ctrl.hold(
        steps=100
    )

    final_object_pos = (
        env.sim.data.body_xpos[
            obj_id
        ].copy()
    )

    elapsed = (
        time.perf_counter()
        - total_start
    )

    print("\n" + "=" * 60)

    print(
        "Final object position:",
        np.round(
            final_object_pos,
            5
        )
    )

    print(
        f"Total OSC task time: "
        f"{elapsed:.3f} s"
    )

    print("=" * 60)

    # For this first experiment just check XY center accuracy.
    center_error = np.linalg.norm(
        final_object_pos[:2]
        - np.array([0.0, 0.0])
    )

    print(
        f"Final XY center error: "
        f"{center_error * 1000:.2f} mm"
    )

    return center_error < 0.03


if __name__ == "__main__":
    main()