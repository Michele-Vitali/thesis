import os

from robosuite.models.robots.manipulators import UR5e
from robosuite.models.robots.robot_model import register_robot
from robosuite.robots import ROBOT_CLASS_MAPPING
from robosuite.robots.single_arm import SingleArm


# Register the XML file
@register_robot
class CustomUR5e(UR5e):

    @property
    def name(self):
        return "CustomUR5e"

    @property
    def _init_xml(self):
        # Specify the new custom filepath
        return os.path.join(os.path.dirname(__file__), "../assets/robots/custom_ur5e/robot.xml")

    @property
    def default_gripper(self):
        # Let's use the original UR5e gripper
        return "Robotiq85Gripper"

    # If the gripper is misplaced due to XML modifications
    """
    @property
    def gripper_mount_pos_offset(self):
        return {"right": [0.0, 0.0, 0.0]}
    """

ROBOT_CLASS_MAPPING["CustomUR5e"] = SingleArm