import os

import numpy as np

from robosuite.models.robots.manipulators.manipulator_model import ManipulatorModel
from robosuite.models.robots.robot_model import register_robot
from robosuite.models.grippers.gripper_model import GripperModel
from robosuite.models.grippers import GRIPPER_MAPPING
from robosuite.robots import ROBOT_CLASS_MAPPING
from robosuite.robots.single_arm import SingleArm


class PRob3Gripper(GripperModel):
    """
    P-Rob3 P-Grip.

    The physical gripper has two finger joints, but the real P-Rob3
    exposes a single high-level gripper command. The two physical
    finger joints are therefore coupled in the MuJoCo model.
    """

    def __init__(self, idn=0):
        path = os.path.join(
            os.path.dirname(__file__),
            "../assets/robots/PRob3/PGripper.xml",
        )
        super().__init__(fname=path, idn=idn)

    @property
    def dof(self):
        """
        Number of high-level gripper control dimensions.

        The P-Rob3 exposes one gripper command even though the
        MuJoCo model contains two physical finger joints.
        """
        return 1

    @property
    def init_qpos(self):
        """
        Initial positions of the physical finger joints.
        """
        return np.array([0.0, 0.0])

    @property
    def speed(self):
        return 0.20

    @property
    def _important_sites(self):
        return {
            "grip_site": "grip_site",
            "grip_cylinder": "grip_site_cylinder",
            "ee": "ee",
            "ee_x": "ee_x",
            "ee_y": "ee_y",
            "ee_z": "ee_z",
        }

    @property
    def _important_geoms(self):
        return {
            "left_finger": [
                "left_finger_col_0",
                "left_finger_col_1",
                "left_finger_col_2",
                "left_finger_col_3",
                "left_finger_col_4",
                "left_finger_col_5",
                "left_finger_col_6",
            ],
            "right_finger": [
                "right_finger_col_0",
                "right_finger_col_1",
                "right_finger_col_2",
                "right_finger_col_3",
                "right_finger_col_4",
                "right_finger_col_5",
                "right_finger_col_6",
            ],
            "left_fingerpad": [
                "left_finger_col_0",
                "left_finger_col_1",
                "left_finger_col_2",
                "left_finger_col_3",
                "left_finger_col_4",
                "left_finger_col_5",
                "left_finger_col_6",
            ],
            "right_fingerpad": [
                "right_finger_col_0",
                "right_finger_col_1",
                "right_finger_col_2",
                "right_finger_col_3",
                "right_finger_col_4",
                "right_finger_col_5",
                "right_finger_col_6",
            ],
        }

    def format_action(self, action):

        assert len(action) == 1
        self.current_action = np.clip(
            self.current_action * 1.0 + self.speed * np.sign(action), -1.0, 1.0
        )
        return self.current_action

GRIPPER_MAPPING["PRob3Gripper"] = PRob3Gripper


@register_robot
class PRob3(ManipulatorModel):
    """
    F&P Robotics P-Rob3 robot model for robosuite.
    """

    def __init__(self, idn=0):
        path = os.path.join(
            os.path.dirname(__file__),
            "../assets/robots/PRob3/PRob3.xml",
        )
        super().__init__(fname=path, idn=idn)

    @property
    def name(self):
        return "PRob3"

    @property
    def arm_type(self):
        return "single"

    @property
    def default_controller_config(self):
        return {"right": "default_ur5e"}

    @property
    def base_xpos_offset(self):
        return {
            "bins": (-0.5, -0.1, 0),
            "empty": (-0.6, 0, 0),
            "table": lambda table_length: (
                -0.16 - table_length / 2,
                0,
                0,
            ),
        }

    @property
    def top_offset(self):
        return np.array((0, 0, 1.0))

    @property
    def _horizontal_radius(self):
        return 0.5

    @property
    def default_gripper(self):
        return {"right": "PRob3Gripper"}

    @property
    def default_mount(self):
        return None

    @property
    def init_qpos(self):
        return np.array(
            [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
        )


ROBOT_CLASS_MAPPING["PRob3"] = SingleArm