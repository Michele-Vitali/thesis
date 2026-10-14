import glob
from collections.abc import Iterator

import numpy as np
import tensorflow_datasets as tfds


class Prob3PickPlace(tfds.core.GeneratorBasedBuilder):
    """
    RLDS dataset containing simulated P-Rob3
    pick-and-place demonstrations.

    Action:
        [
            delta_x_m,
            delta_y_m,
            delta_z_m,
            delta_rot_x_rad,
            delta_rot_y_rad,
            delta_rot_z_rad,
            gripper_open,
        ]

    Gripper:
        0.0 = closed
        1.0 = open

    State:
        [
            joint_1,
            joint_2,
            joint_3,
            joint_4,
            joint_5,
            joint_6,
            gripper_aperture,
        ]
    """

    VERSION = tfds.core.Version("1.0.0")

    RELEASE_NOTES = {
        "1.0.0": (
            "Initial P-Rob3 simulated "
            "pick-and-place dataset."
        ),
    }

    def _info(self) -> tfds.core.DatasetInfo:

        return (self.dataset_info_from_configs(features=tfds.features.FeaturesDict({
                        "steps": tfds.features.Dataset({
                                "observation": tfds.features.FeaturesDict({
                                            "image": tfds.features.Image(shape=(256, 256, 3), dtype=np.uint8, encoding_format="png"),
                                            "state": tfds.features.Tensor(shape=(7,), dtype=np.float32),
                                        }
                                    ),
                                "action": tfds.features.Tensor(shape=(7,), dtype=np.float32),

                                "discount":
                                    tfds.features.Scalar(
                                        dtype=np.float32,
                                    ),

                                "reward":
                                    tfds.features.Scalar(
                                        dtype=np.float32,
                                    ),

                                "is_first":
                                    tfds.features.Scalar(
                                        dtype=np.bool_,
                                    ),

                                "is_last":
                                    tfds.features.Scalar(
                                        dtype=np.bool_,
                                    ),

                                "is_terminal":
                                    tfds.features.Scalar(
                                        dtype=np.bool_,
                                    ),

                                "language_instruction":
                                    tfds.features.Text(),
                            }
                        ),

                        "episode_metadata":
                            tfds.features.FeaturesDict(
                                {
                                    "file_path":
                                        tfds.features.Text(),
                                }
                            ),
                    }
                )
            )
        )

    def _split_generators(
        self,
        dl_manager: tfds.download.DownloadManager,
    ):
        """
        Define the available dataset splits.

        We intentionally expose only TRAIN.

        OpenVLA automatically uses the final 5% as validation
        when a dedicated val split is absent.
        """

        return {
            "train": self._generate_examples(
                path=(
                    "data/train/"
                    "episode_*.npy"
                )
            ),
        }

    def _generate_examples(
        self,
        path: str,
    ) -> Iterator[
        Tuple[str, Any]
    ]:
        """
        Parse each temporary raw episode and emit
        an RLDS episode.
        """

        episode_paths = sorted(
            glob.glob(path)
        )

        if not episode_paths:

            raise RuntimeError(
                f"No episodes found at: {path}"
            )

        for episode_path in (
            episode_paths
        ):

            data = np.load(
                episode_path,
                allow_pickle=True,
            )

            episode_steps = []

            num_steps = len(data)

            if num_steps == 0:
                continue

            for i, step in enumerate(
                data
            ):

                is_first = (
                    i == 0
                )

                is_last = (
                    i
                    == num_steps - 1
                )

                episode_steps.append(
                    {
                        "observation": {
                            "image":
                                np.asarray(
                                    step["image"],
                                    dtype=np.uint8,
                                ),

                            "state":
                                np.asarray(
                                    step["state"],
                                    dtype=np.float32,
                                ),
                        },

                        "action":
                            np.asarray(
                                step["action"],
                                dtype=np.float32,
                            ),

                        "discount":
                            np.float32(
                                1.0
                            ),

                        # Successful demonstrations:
                        # reward 1 only on final step.
                        "reward":
                            np.float32(
                                1.0
                                if is_last
                                else 0.0
                            ),

                        "is_first":
                            is_first,

                        "is_last":
                            is_last,

                        "is_terminal":
                            is_last,

                        "language_instruction":
                            str(
                                step[
                                    "language_instruction"
                                ]
                            ),
                    }
                )

            sample = {
                "steps": episode_steps,

                "episode_metadata": {
                    "file_path":
                        str(
                            episode_path
                        ),
                },
            }

            yield (
                episode_path,
                sample,
            )