"""
find_home_pose_v2.py

Standalone P-Rob3 home-pose calibration utility.

This file is intentionally separate from the thesis control code:
- it does not use Mink;
- it does not modify movements.py;
- it does not modify settings.py;
- it only reads the current robosuite / MuJoCo model and searches for
  a good 6-joint starting pose.

Place it next to:
    main.py
    settings.py
    custom_env.py
    custom_robot.py

Run:
    python find_home_pose_v2.py
"""

import math

import mujoco
import numpy as np
import robosuite as suite
from robosuite import load_controller_config
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation as R

# Importing these modules registers the custom robot and environment.
import custom_env  # noqa: F401
import custom_robot  # noqa: F401
import settings


# ============================================================================
# TASK GEOMETRY
# ============================================================================

ROBOT_BASE_XY = np.array([-0.5, 0.0], dtype=float)

R_MIN = 0.35
R_MAX = 0.55

TABLE_X_MIN = -0.4
TABLE_X_MAX = +0.4
TABLE_Y_MIN = -0.4
TABLE_Y_MAX = +0.4

# The useful annulus is clipped by the table.
# Its actual spatial centre is therefore NOT x=-0.05.
# Around x=-0.15 is a much better central point for this task.
HOME_X_CANDIDATES = [-0.18, -0.15, -0.12]
HOME_Y = 0.0

# Do not assume a single Z beforehand.
# We scan several realistic heights automatically.
HOME_Z_CANDIDATES = [0.28, 0.30, 0.32, 0.34, 0.35]

# Representative approach height for checking workspace reachability.
# Your object centres are around the table top, and the current grasp logic
# adds roughly 10 cm to the EEF target, so ~0.32 m is a useful test height.
WORKSPACE_TEST_Z = 0.32

# Same top-down world orientation used by your OSC grasp controller.
TOP_DOWN_ROT = np.array(
    [
        [1.0,  0.0,  0.0],
        [0.0, -1.0,  0.0],
        [0.0,  0.0, -1.0],
    ],
    dtype=float,
)


# ============================================================================
# SEARCH PARAMETERS
# ============================================================================

RNG = np.random.default_rng(42)

# Random initializations per Cartesian home target.
N_RANDOM_SEEDS = 35

# Joint-limit safety inset used only by the calibration optimizer.
JOINT_LIMIT_INSET_FRACTION = 0.015

# Home-target acceptance.
HOME_POS_TOL = 0.006               # 6 mm
HOME_ROT_TOL_DEG = 2.0

# Workspace-target acceptance.
WORKSPACE_POS_TOL = 0.008          # 8 mm
WORKSPACE_ROT_TOL_DEG = 3.0

MAX_NFEV_HOME = 350
MAX_NFEV_WORKSPACE = 250

# Representative points of the spawning annulus.
TEST_RADII = [0.35, 0.45, 0.55]
TEST_ANGLES_DEG = [
    -80, -65, -50, -35, -20, 0,
    20, 35, 50, 65, 80,
]


# ============================================================================
# ENVIRONMENT HELPERS
# ============================================================================

def create_calibration_env():
    controller_config = load_controller_config(
        default_controller="OSC_POSE"
    )

    # Keep the same OSC overrides used by the project when possible.
    if hasattr(settings, "osc_config"):
        controller_config.update(settings.osc_config)

    env = suite.make(
        env_name=settings.env_name,
        robots=settings.robot,
        gripper_types=settings.gripper_types,
        controller_configs=controller_config,
        has_renderer=True,
        has_offscreen_renderer=False,
        use_camera_obs=False,
        control_freq=20,
        horizon=1000,
        hard_reset=False,
        reward_shaping=False,
    )

    env.reset()

    return env


def get_native_model(env):
    model = env.sim.model
    return model._model if hasattr(model, "_model") else model


def get_native_data(env):
    data = env.sim.data
    return data._data if hasattr(data, "_data") else data


def get_arm_info(env):
    model = get_native_model(env)

    joint_names = [
        "robot0_joint1",
        "robot0_joint2",
        "robot0_joint3",
        "robot0_joint4",
        "robot0_joint5",
        "robot0_joint6",
    ]

    qpos_indices = []
    dof_indices = []
    lower = []
    upper = []

    for joint_name in joint_names:
        joint_id = mujoco.mj_name2id(
            model,
            mujoco.mjtObj.mjOBJ_JOINT,
            joint_name,
        )

        if joint_id == -1:
            raise RuntimeError(
                f"Joint '{joint_name}' was not found."
            )

        qpos_indices.append(
            model.jnt_qposadr[joint_id]
        )

        dof_indices.append(
            model.jnt_dofadr[joint_id]
        )

        lower.append(
            model.jnt_range[joint_id, 0]
        )

        upper.append(
            model.jnt_range[joint_id, 1]
        )

    return (
        np.asarray(qpos_indices, dtype=int),
        np.asarray(dof_indices, dtype=int),
        np.asarray(lower, dtype=float),
        np.asarray(upper, dtype=float),
    )


def set_arm_qpos(env, q_arm, qpos_indices):
    env.sim.data.qpos[qpos_indices] = np.asarray(
        q_arm,
        dtype=float,
    )
    env.sim.forward()


def get_eef_pose(env, site_id):
    pos = np.asarray(
        env.sim.data.site_xpos[site_id],
        dtype=float,
    ).copy()

    rot = np.asarray(
        env.sim.data.site_xmat[site_id],
        dtype=float,
    ).reshape(3, 3).copy()

    return pos, rot


# ============================================================================
# IK
# ============================================================================

def rotation_error_vector(current_rot, target_rot):
    return R.from_matrix(
        target_rot @ current_rot.T
    ).as_rotvec()


def pose_residual(
    q_arm,
    env,
    qpos_indices,
    site_id,
    target_pos,
    target_rot,
):
    set_arm_qpos(
        env,
        q_arm,
        qpos_indices,
    )

    current_pos, current_rot = get_eef_pose(
        env,
        site_id,
    )

    position_error = current_pos - target_pos

    orientation_error = rotation_error_vector(
        current_rot,
        target_rot,
    )

    # Give position a little more importance.
    return np.concatenate(
        [
            5.0 * position_error,
            orientation_error,
        ]
    )


def solve_pose(
    env,
    seed,
    qpos_indices,
    site_id,
    lower_bounds,
    upper_bounds,
    target_pos,
    target_rot,
    max_nfev,
):
    seed = np.clip(
        np.asarray(seed, dtype=float),
        lower_bounds,
        upper_bounds,
    )

    result = least_squares(
        pose_residual,
        x0=seed,
        bounds=(
            lower_bounds,
            upper_bounds,
        ),
        args=(
            env,
            qpos_indices,
            site_id,
            target_pos,
            target_rot,
        ),
        method="trf",
        max_nfev=max_nfev,
        xtol=1e-9,
        ftol=1e-9,
        gtol=1e-9,
    )

    q = result.x.copy()

    set_arm_qpos(
        env,
        q,
        qpos_indices,
    )

    actual_pos, actual_rot = get_eef_pose(
        env,
        site_id,
    )

    pos_error = np.linalg.norm(
        actual_pos - target_pos
    )

    rot_error = np.linalg.norm(
        rotation_error_vector(
            actual_rot,
            target_rot,
        )
    )

    return q, pos_error, rot_error, result.success


# ============================================================================
# KINEMATIC QUALITY
# ============================================================================

def minimum_normalized_joint_margin(
    q,
    lower,
    upper,
):
    joint_range = upper - lower

    lower_margin = (
        q - lower
    ) / joint_range

    upper_margin = (
        upper - q
    ) / joint_range

    return float(
        np.min(
            np.minimum(
                lower_margin,
                upper_margin,
            )
        )
    )


def jacobian_metrics(
    env,
    site_id,
    dof_indices,
):
    model = get_native_model(env)
    data = get_native_data(env)

    jacp = np.zeros(
        (3, model.nv),
        dtype=float,
    )

    jacr = np.zeros(
        (3, model.nv),
        dtype=float,
    )

    mujoco.mj_jacSite(
        model,
        data,
        jacp,
        jacr,
        site_id,
    )

    J = np.vstack(
        [
            jacp[:, dof_indices],
            jacr[:, dof_indices],
        ]
    )

    singular_values = np.linalg.svd(
        J,
        compute_uv=False,
    )

    sigma_min = float(
        np.min(singular_values)
    )

    sigma_max = float(
        np.max(singular_values)
    )

    condition_number = (
        float("inf")
        if sigma_min < 1e-9
        else sigma_max / sigma_min
    )

    return sigma_min, condition_number


def kinematic_score(
    joint_margin,
    sigma_min,
    condition_number,
):
    if np.isfinite(condition_number):
        conditioning_bonus = (
            1.0 / (1.0 + condition_number)
        )
    else:
        conditioning_bonus = 0.0

    return (
        1.0 * joint_margin
        + 0.20 * sigma_min
        + 0.10 * conditioning_bonus
    )


# ============================================================================
# WORKSPACE
# ============================================================================

def is_on_table(x, y):
    return (
        TABLE_X_MIN <= x <= TABLE_X_MAX
        and TABLE_Y_MIN <= y <= TABLE_Y_MAX
    )


def is_outside_drop_zone(x, y):
    drop = np.asarray(
        settings.drop_point,
        dtype=float,
    )

    return (
        np.linalg.norm(
            np.array([x, y], dtype=float) - drop
        )
        >= 0.08
    )


def build_workspace_points():
    points = []

    for radius in TEST_RADII:
        for angle_deg in TEST_ANGLES_DEG:
            theta = math.radians(
                angle_deg
            )

            x = (
                ROBOT_BASE_XY[0]
                + radius * math.cos(theta)
            )

            y = (
                ROBOT_BASE_XY[1]
                + radius * math.sin(theta)
            )

            if not is_on_table(x, y):
                continue

            if not is_outside_drop_zone(x, y):
                continue

            points.append(
                np.array(
                    [x, y, WORKSPACE_TEST_Z],
                    dtype=float,
                )
            )

    # Also verify the release point.
    points.append(
        np.array(
            [
                settings.drop_point[0],
                settings.drop_point[1],
                WORKSPACE_TEST_Z,
            ],
            dtype=float,
        )
    )

    return points


def evaluate_workspace(
    env,
    q_home,
    qpos_indices,
    site_id,
    lower_bounds,
    upper_bounds,
):
    test_points = build_workspace_points()

    successes = 0
    failures = []

    for target_pos in test_points:
        (
            _,
            pos_error,
            rot_error,
            solver_success,
        ) = solve_pose(
            env=env,
            seed=q_home,
            qpos_indices=qpos_indices,
            site_id=site_id,
            lower_bounds=lower_bounds,
            upper_bounds=upper_bounds,
            target_pos=target_pos,
            target_rot=TOP_DOWN_ROT,
            max_nfev=MAX_NFEV_WORKSPACE,
        )

        valid = (
            solver_success
            and pos_error <= WORKSPACE_POS_TOL
            and rot_error <= math.radians(
                WORKSPACE_ROT_TOL_DEG
            )
        )

        if valid:
            successes += 1
        else:
            failures.append(
                (
                    target_pos.copy(),
                    pos_error,
                    math.degrees(rot_error),
                )
            )

    return successes, len(test_points), failures


# ============================================================================
# HOME SEARCH
# ============================================================================

def generate_seeds(
    current_q,
    lower_bounds,
    upper_bounds,
):
    seeds = [
        np.clip(
            current_q,
            lower_bounds,
            upper_bounds,
        ),
        (lower_bounds + upper_bounds) / 2.0,
    ]

    # A slightly more bent manual seed close to the family of your old pose.
    manual_seed = np.array(
        [
            0.0,
            -0.60,
            1.10,
            0.0,
            1.07,
            0.0,
        ],
        dtype=float,
    )

    seeds.append(
        np.clip(
            manual_seed,
            lower_bounds,
            upper_bounds,
        )
    )

    for _ in range(N_RANDOM_SEEDS):
        seeds.append(
            RNG.uniform(
                low=lower_bounds,
                high=upper_bounds,
            )
        )

    return seeds


def search_home_candidates(env):
    (
        qpos_indices,
        dof_indices,
        lower,
        upper,
    ) = get_arm_info(env)

    site_id = int(
        env.robots[0].eef_site_id
    )

    joint_range = upper - lower

    inset = (
        JOINT_LIMIT_INSET_FRACTION
        * joint_range
    )

    lower_bounds = lower + inset
    upper_bounds = upper - inset

    current_q = np.asarray(
        env.sim.data.qpos[qpos_indices],
        dtype=float,
    ).copy()

    current_pos, current_rot = get_eef_pose(
        env,
        site_id,
    )

    print()
    print("Current robot configuration:")
    print(
        np.array2string(
            current_q,
            precision=6,
            separator=", ",
        )
    )

    print(
        "Current EEF position:",
        np.array2string(
            current_pos,
            precision=6,
        )
    )

    current_to_top_down_deg = math.degrees(
        np.linalg.norm(
            rotation_error_vector(
                current_rot,
                TOP_DOWN_ROT,
            )
        )
    )

    print(
        f"Current EEF orientation distance from top-down: "
        f"{current_to_top_down_deg:.2f} deg"
    )

    seeds = generate_seeds(
        current_q,
        lower_bounds,
        upper_bounds,
    )

    candidates = []

    # Keep track of the closest failed result too.
    best_failed = None

    print()
    print("Scanning Cartesian home targets...")
    print()

    for x in HOME_X_CANDIDATES:
        for z in HOME_Z_CANDIDATES:
            target_pos = np.array(
                [x, HOME_Y, z],
                dtype=float,
            )

            target_solutions = 0

            for seed in seeds:
                (
                    q,
                    pos_error,
                    rot_error,
                    solver_success,
                ) = solve_pose(
                    env=env,
                    seed=seed,
                    qpos_indices=qpos_indices,
                    site_id=site_id,
                    lower_bounds=lower_bounds,
                    upper_bounds=upper_bounds,
                    target_pos=target_pos,
                    target_rot=TOP_DOWN_ROT,
                    max_nfev=MAX_NFEV_HOME,
                )

                combined_error = (
                    pos_error
                    + 0.05 * rot_error
                )

                if (
                    best_failed is None
                    or combined_error
                    < best_failed["combined_error"]
                ):
                    best_failed = {
                        "target": target_pos.copy(),
                        "q": q.copy(),
                        "pos_error": pos_error,
                        "rot_error": rot_error,
                        "combined_error": combined_error,
                    }

                valid = (
                    solver_success
                    and pos_error <= HOME_POS_TOL
                    and rot_error <= math.radians(
                        HOME_ROT_TOL_DEG
                    )
                )

                if not valid:
                    continue

                duplicate = any(
                    np.linalg.norm(
                        q - c["q"]
                    ) < 0.04
                    for c in candidates
                    if np.linalg.norm(
                        c["target"] - target_pos
                    ) < 1e-9
                )

                if duplicate:
                    continue

                set_arm_qpos(
                    env,
                    q,
                    qpos_indices,
                )

                joint_margin = (
                    minimum_normalized_joint_margin(
                        q,
                        lower,
                        upper,
                    )
                )

                (
                    sigma_min,
                    condition_number,
                ) = jacobian_metrics(
                    env,
                    site_id,
                    dof_indices,
                )

                score = kinematic_score(
                    joint_margin,
                    sigma_min,
                    condition_number,
                )

                candidates.append(
                    {
                        "target": target_pos.copy(),
                        "q": q.copy(),
                        "pos_error": pos_error,
                        "rot_error": rot_error,
                        "joint_margin": joint_margin,
                        "sigma_min": sigma_min,
                        "condition_number": condition_number,
                        "score": score,
                    }
                )

                target_solutions += 1

            print(
                f"target {target_pos}: "
                f"{target_solutions} valid IK branch(es)"
            )

    return (
        candidates,
        best_failed,
        qpos_indices,
        site_id,
        lower_bounds,
        upper_bounds,
    )


# ============================================================================
# MAIN
# ============================================================================

def main():
    env = create_calibration_env()

    try:
        print("=" * 76)
        print("P-ROB3 HOME POSE CALIBRATION - V2")
        print("=" * 76)

        print(
            f"Radial workspace: "
            f"{R_MIN:.2f} m -> {R_MAX:.2f} m"
        )

        print(
            "Home X candidates:",
            HOME_X_CANDIDATES,
        )

        print(
            "Home Z candidates:",
            HOME_Z_CANDIDATES,
        )

        (
            candidates,
            best_failed,
            qpos_indices,
            site_id,
            lower_bounds,
            upper_bounds,
        ) = search_home_candidates(env)

        if not candidates:
            print()
            print("=" * 76)
            print("NO EXACT TOP-DOWN HOME FOUND")
            print("=" * 76)

            if best_failed is not None:
                print(
                    "Closest Cartesian target:",
                    best_failed["target"],
                )

                print(
                    f"Best position error: "
                    f"{best_failed['pos_error'] * 1000.0:.2f} mm"
                )

                print(
                    f"Best rotation error: "
                    f"{math.degrees(best_failed['rot_error']):.2f} deg"
                )

                print()
                print(
                    "Closest joint configuration found:"
                )

                print(
                    "starting_pose = ["
                    + ", ".join(
                        f"{v:.6f}"
                        for v in best_failed["q"]
                    )
                    + "]"
                )

            print()
            print(
                "This means the problem is no longer just the chosen Z. "
                "The next thing to inspect would be the exact EEF / grip-site "
                "orientation convention."
            )

            return

        # First rank only by kinematic quality, then validate the best subset.
        candidates.sort(
            key=lambda c: c["score"],
            reverse=True,
        )

        candidates_to_validate = candidates[
            :min(15, len(candidates))
        ]

        best = None

        print()
        print("=" * 76)
        print("WORKSPACE VALIDATION")
        print("=" * 76)

        for i, candidate in enumerate(
            candidates_to_validate,
            start=1,
        ):
            (
                reachable,
                total,
                failures,
            ) = evaluate_workspace(
                env=env,
                q_home=candidate["q"],
                qpos_indices=qpos_indices,
                site_id=site_id,
                lower_bounds=lower_bounds,
                upper_bounds=upper_bounds,
            )

            candidate["reachable"] = reachable
            candidate["total"] = total
            candidate["failures"] = failures

            print(
                f"{i:02d}) home={candidate['target']} | "
                f"workspace={reachable}/{total} | "
                f"joint_margin={candidate['joint_margin']:.3f} | "
                f"sigma_min={candidate['sigma_min']:.4f} | "
                f"cond={candidate['condition_number']:.2f}"
            )

            if best is None:
                best = candidate
                continue

            if candidate["reachable"] > best["reachable"]:
                best = candidate
            elif (
                candidate["reachable"] == best["reachable"]
                and candidate["score"] > best["score"]
            ):
                best = candidate

        q_best = best["q"]

        set_arm_qpos(
            env,
            q_best,
            qpos_indices,
        )

        actual_pos, _ = get_eef_pose(
            env,
            site_id,
        )

        print()
        print("=" * 76)
        print("BEST HOME POSE")
        print("=" * 76)

        print(
            "Chosen Cartesian home:",
            best["target"],
        )

        print(
            "Actual EEF position:",
            np.array2string(
                actual_pos,
                precision=6,
            )
        )

        print(
            f"Home position error: "
            f"{best['pos_error'] * 1000.0:.2f} mm"
        )

        print(
            f"Home rotation error: "
            f"{math.degrees(best['rot_error']):.3f} deg"
        )

        print(
            f"Workspace coverage: "
            f"{best['reachable']}/{best['total']}"
        )

        print(
            f"Minimum normalized joint-limit margin: "
            f"{best['joint_margin']:.3f}"
        )

        print(
            f"Jacobian minimum singular value: "
            f"{best['sigma_min']:.5f}"
        )

        print(
            f"Jacobian condition number: "
            f"{best['condition_number']:.2f}"
        )

        print()
        print("COPY THIS INTO settings.py:")
        print()

        print(
            "starting_pose = ["
            + ", ".join(
                f"{value:.6f}"
                for value in q_best
            )
            + "]"
        )

        if best["failures"]:
            print()
            print("Representative workspace targets not reached:")

            for (
                target,
                pos_error,
                rot_error_deg,
            ) in best["failures"]:
                print(
                    f"  {target} | "
                    f"pos={pos_error * 1000.0:.1f} mm | "
                    f"rot={rot_error_deg:.2f} deg"
                )

        print()
        print(
            "The selected configuration is now visible in the MuJoCo window."
        )

        for _ in range(30):
            env.render()

        input(
            "\nPress ENTER to close..."
        )

    finally:
        env.close()


if __name__ == "__main__":
    main()
