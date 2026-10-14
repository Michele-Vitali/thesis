import os

import numpy as np
from robosuite.models.grippers import GRIPPER_MAPPING
from robosuite.models.grippers.gripper_model import GripperModel
from robosuite.models.robots.manipulators.manipulator_model import ManipulatorModel
from robosuite.models.robots.robot_model import register_robot
from robosuite.robots import ROBOT_CLASS_MAPPING
from robosuite.robots.single_arm import SingleArm


class PRob3Gripper(GripperModel):
    """
    P-Rob3 P-Grip.

    The physical gripper has two finger joints, but the real P-Rob3 exposes a single high-level gripper command. The two physical finger joints are therefore coupled in the MuJoCo model.
    """

    def __init__(self, idn=0):
        """Initializes the PRob3Gripper.

        Args:
            idn (int, optional): The id of the gripper. Defaults to 0.
        """

        # Define and use the xml which defines the gripper.
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

        return np.array([0.523599, 0.523599]) # Gripper initially fully open

    @property
    def speed(self):
        """
        The speed at which the fingers of the gripper move
        """
        return 0.01

    def format_action(self, action: np.ndarray):
        """Formats the passed gripper action in the convention:
            - +1 = Fully open
            - -1 = Fully closed
            - 0 = Hold the current position
            
        Args:
            action (np.ndarray): The passed gripper action

        Returns:
            np.ndarray: the formatted action.
        """

        # Ensure the action is an np array and flatten it.
        action = np.asarray(action, dtype=float).reshape(-1)

        # If an action longer than 1 was passed we have an error!
        assert len(action) == 1

        # Clip the action value between -1.0 and 1.0
        self.current_action = np.clip(self.current_action + self.speed * np.sign(action), -1.0, 1.0)

        return self.current_action

GRIPPER_MAPPING["PRob3Gripper"] = PRob3Gripper


@register_robot
class PRob3(ManipulatorModel):
    """
    F&P Robotics P-Rob3 robot model for robosuite.
    """

    def __init__(self, idn=0):
        """Initialized the PRob3.

        Args:
            idn (int, optional): The id of the robot. Defaults to 0.
        """

        # Define and use the xml which defines the robot.
        path = os.path.join(
            os.path.dirname(__file__),
            "../assets/robots/PRob3/PRob3.xml",
        )
        super().__init__(fname=path, idn=idn)

    @property
    def name(self):
        """
        The name of the robot
        """
        return "PRob3"

    @property
    def arm_type(self):
        """
        The type of the robot's arm.
        """
        return "single"

    @property
    def default_controller_config(self):
        """
        The default controller configuration of the robot.
        """
        return {"right": "default_ur5e"}

    @property
    def base_xpos_offset(self):
        """
        The initial position of the robot's base.
        """
        return {
            "bins": (-0.5, -0.1, 0),
            "empty": (-0.6, 0, 0),
            "table": lambda table_length: (
                -0.16 - table_length / 2,
                0,
                0,
            ),
        }

    # Physical properties of the robot...

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
        # We start with the arm completely extended!
        return np.zeros(6)

class PRob3SingleArm(SingleArm):

    def reset(self, deterministic=False):
        super().reset(deterministic)

        if self.has_gripper:
            # P-Grip starts completely open
            self.gripper.current_action = np.array([1.0])
    


ROBOT_CLASS_MAPPING["PRob3"] = PRob3SingleArm