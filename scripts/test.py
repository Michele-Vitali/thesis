
import numpy as np
import mujoco
import robosuite

import settings
from custom_osc import install_prob3_custom_osc
from homing import prepare_episode
from hdf5_recorder import NullRecorder
from main import create_randomized_env, pick_and_place_task
from robosuite.controllers import load_controller_config


def main():
    print("MuJoCo:", mujoco.__version__)
    print("Robosuite:", robosuite.__version__)

    # Do not save demonstrations or open the interactive viewer.
    settings.record = False
    settings.render_on_screen = False

    install_prob3_custom_osc()

    joint_config = load_controller_config(
        default_controller="JOINT_POSITION"
    )

    osc_config = load_controller_config(
        default_controller="OSC_POSE"
    )
    osc_config.update(settings.osc_config)

    env = None

    try:
        print("[TEST] Creating environment")
        env = create_randomized_env(joint_config)

        for i in range(3):
            print(f"\n[TEST] Episode {i + 1}")

            obs = prepare_episode(
                env, joint_config, osc_config
            )

            assert env.robots[0].action_dim == 7
            assert env.robots[0].controller.name == "OSC_POSE"

            image = np.asarray(
                obs[settings.dataset_camera_observation_key]
            )

            assert image.shape == (
                settings.dataset_image_height,
                settings.dataset_image_width,
                3,
            )

            print("[OK] Controller and camera")

            error, steps = pick_and_place_task(
                env,
                recorder=NullRecorder(),
                record=False,
            )

            print(
                f"[RESULT] error={error}, steps={steps}"
            )

        print("\n[TEST] Completed without crashes")

    finally:
        if env is not None:
            env.close()


if __name__ == "__main__":
    main()
