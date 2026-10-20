import sys
from pathlib import Path
from typing import Any, Dict

import tensorflow as tf
import tensorflow_datasets as tfds


# ================================================================
# PATHS
# ================================================================

THIS_FILE = Path(__file__).resolve()

THESIS_ROOT = THIS_FILE.parents[2]

RLDS_BUILDER_ROOT = (THESIS_ROOT / "tools" / "rlds_dataset_builder")

RLDS_DATA_ROOT = (THESIS_ROOT / "rlds_data" / "rlds")


# ================================================================
# REGISTER TFDS BUILDER
# ================================================================

# The P-Rob3 TFDS builder lives outside OpenVLA.
#
# Importing the builder module registers the dataset class with TFDS,
# allowing OpenVLA to later call:
#
#     tfds.builder("prob3_pick_place", ...)
#
# normally.

if str(RLDS_BUILDER_ROOT) not in sys.path:
    sys.path.insert(0, str(RLDS_BUILDER_ROOT))


import prob3_pick_place.prob3_pick_place_dataset_builder  # noqa: F401,E402


# ================================================================
# OPENVLA IMPORTS
# ================================================================

from prismatic.vla.datasets.rlds.oxe.configs import (  # noqa: E402
    ActionEncoding,
    OXE_DATASET_CONFIGS,
    StateEncoding,
)

from prismatic.vla.datasets.rlds.oxe.transforms import (  # noqa: E402
    OXE_STANDARDIZATION_TRANSFORMS,
)


DATASET_NAME = "prob3_pick_place"


# ================================================================
# P-ROB3 STANDARDIZATION TRANSFORM
# ================================================================

def prob3_pick_place_dataset_transform(trajectory: Dict[str, Any]) -> Dict[str, Any]:
    """
    Standardization transform for the simulated P-Rob3 dataset.

    The RLDS dataset already stores:

        observation/image
        observation/state
        action
        language_instruction

    The action is already 7D:

        [
            delta_x,
            delta_y,
            delta_z,
            delta_rot_x,
            delta_rot_y,
            delta_rot_z,
            gripper,
        ]

    where:

        gripper = 0.0 -> closed
        gripper = 1.0 -> open

    Therefore no geometric conversion is necessary.
    """

    trajectory["action"] = tf.cast(trajectory["action"], tf.float32)

    return trajectory


# ================================================================
# REGISTRATION
# ================================================================

def register_prob3() -> None:
    """
    Register the P-Rob3 dataset into OpenVLA at runtime.

    This avoids modifying OpenVLA source files.
    """

    OXE_DATASET_CONFIGS[
        DATASET_NAME
    ] = {
        "image_obs_keys": {
            "primary": "image",
            "secondary": None,
            "wrist": None,
        },

        "depth_obs_keys": {
            "primary": None,
            "secondary": None,
            "wrist": None,
        },

        # OpenVLA fine-tuning currently calls the RLDS loader with
        # load_proprio=False, therefore proprioception is not used.
        "state_obs_keys": [
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
        ],

        "state_encoding":
            StateEncoding.NONE,

        # P-Rob3 action:
        #
        # delta XYZ
        # + delta RPY
        # + gripper
        "action_encoding":
            ActionEncoding.EEF_POS,
    }

    OXE_STANDARDIZATION_TRANSFORMS[DATASET_NAME] = prob3_pick_place_dataset_transform


# ================================================================
# TEST
# ================================================================

def main():
    register_prob3()

    print("=" * 60)
    print("P-Rob3 -> OpenVLA registration test")
    print("=" * 60)

    print("Dataset config registered:", DATASET_NAME in OXE_DATASET_CONFIGS)

    print("Dataset transform registered:", DATASET_NAME in OXE_STANDARDIZATION_TRANSFORMS)

    print("RLDS root:", RLDS_DATA_ROOT)

    builder = tfds.builder(DATASET_NAME, data_dir=str(RLDS_DATA_ROOT))

    print()
    print("TFDS dataset:")
    print("Name:", builder.info.name)
    print("Version:", builder.info.version)
    print("Splits:", builder.info.splits)
    print("Data dir:", builder.data_dir)

    print()
    print("Config:")
    print(OXE_DATASET_CONFIGS[DATASET_NAME])

    print()
    print("P-Rob3 registration: OK")

    print("=" * 60)


if __name__ == "__main__":
    main()
