import os
from robosuite.models.robots.manipulators import UR5e
from robosuite.models.robots.robot_model import register_robot
from robosuite.robots import register_robot_class

# Register the XML file
@register_robot
# Register the logic
@register_robot_class("FixedBaseRobot")
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
        return {"right": "Robotiq85Gripper"}

    # If the gripper is misplaced due to XML modifications
    """
    @property
    def gripper_mount_pos_offset(self):
        return {"right": [0.0, 0.0, 0.0]}
    """