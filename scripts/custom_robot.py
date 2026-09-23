import os

import numpy as np
from robosuite.models.robots.manipulators import UR5e
from robosuite.models.robots.robot_model import register_robot
from robosuite.robots import ROBOT_CLASS_MAPPING
from robosuite.robots.single_arm import SingleArm


# Register the XML file
@register_robot
class PRob3(UR5e):

    def __init__(self, idn=0):
            # Specify the new custom filepath
            path = os.path.join(os.path.dirname(__file__), "../assets/robots/P-Rob3/P-Rob3.xml")
            super(UR5e, self).__init__(fname=path, idn=idn)

    @property
    def name(self):
        return "PRob3"

    @property
    def init_qpos(self):
        return np.array([0.0, -1.0, 1.0, -1.57, 1.57, 0.0, 0.0, 0.0])
        
ROBOT_CLASS_MAPPING["PRob3"] = SingleArm