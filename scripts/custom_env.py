import random

import numpy as np
from robosuite.models.arenas import TableArena
import robosuite.models.objects as objects
from robosuite.models.tasks import ManipulationTask
from robosuite.environments.base import register_env
from robosuite.environments.manipulation.single_arm_env import SingleArmEnv
import settings

# UniformRandomSampler places objects randomly within a given range
# We can also give a sampler per object with the SequentialCompositeSampler (future?)
from robosuite.utils.placement_samplers import UniformRandomSampler, SequentialCompositeSampler

@register_env
class CustomTask(SingleArmEnv):

    def __init__(self, **kwargs):
        self.available_objects = [
            objects.BoxObject,
            objects.BallObject,
            objects.CylinderObject,
            #objects.CapsuleObject,
            #objects.CanObject,
            #objects.MilkObject,
            #objects.CerealObject
        ]
        self.gripper_state = -1.0 # Start with the gripper open
        super().__init__(**kwargs)

    def _load_model(self):
        super()._load_model()

        # Create the environment

        # 1. Table
        self.mujoco_arena = TableArena(
            table_full_size=(0.8, 0.8, 0.05),
            table_offset=(0, 0, 0.8)
        )

        self.mujoco_arena.set_origin([0.16, 0, 0])
        self.robots[0].robot_model.set_base_xpos([-0.5, 0, 0])

        # 2. Spawn and place some objects...
        self.objects = []
        self.objects.append(self._create_objects())

        # 3. Sampler for placing objects
        self.sampler, samplers_names = self._define_sampler()
        random.shuffle(samplers_names)

        for id, obj in enumerate(self.objects):
            target_sampler = samplers_names[id % len(samplers_names)]
            self.sampler.add_objects_to_sampler(sampler_name=target_sampler, mujoco_objects=obj)

        # 4. Robot + Arena + Object
        self.model = ManipulationTask(
            mujoco_arena=self.mujoco_arena,
            mujoco_robots=[self.robots[0].robot_model],
            mujoco_objects=self.objects,
            #placement_initializer=self.sampler
        )

    def _define_sampler(self):
        # Define the spawnable surface without the central square
        # which is the area of release
        X_MIN, X_MAX = -0.35, +0.05
        Y_MIN, Y_MAX = -0.2, +0.2

        CX_MIN, CX_MAX = -0.05, +0.05
        CY_MIN, CY_MAX = -0.05, +0.05

        sampler = SequentialCompositeSampler(name="Sampler")

        # Now we define the 4 separate spawnable areas (left, right, bottom, up)
        # 1. Left
        sampler.append_sampler(UniformRandomSampler(
            name="LeftSampler", mujoco_objects=None, x_range=[X_MIN, X_MAX], y_range=[Y_MIN, CY_MIN], rotation=[-np.pi, np.pi], 
            reference_pos=self.mujoco_arena.table_top_abs,ensure_object_boundary_in_range=True, ensure_valid_placement=True
        ))

        # 2. Right
        sampler.append_sampler(UniformRandomSampler(
            name="RightSampler", mujoco_objects=None, x_range=[X_MIN, X_MAX], y_range=[CY_MAX, Y_MAX], rotation=[-np.pi, np.pi],
            reference_pos=self.mujoco_arena.table_top_abs,ensure_object_boundary_in_range=True, ensure_valid_placement=True
        ))

        # 3. Up
        sampler.append_sampler(UniformRandomSampler(
            name="UpSampler", mujoco_objects=None, x_range=[CX_MAX, X_MAX], y_range=[CY_MIN, CY_MAX], rotation=[-np.pi, np.pi],
            reference_pos=self.mujoco_arena.table_top_abs,ensure_object_boundary_in_range=True, ensure_valid_placement=True
        ))

        # 4. Bottom
        sampler.append_sampler(UniformRandomSampler(
            name="BottomSampler", mujoco_objects=None, x_range=[X_MIN, CX_MIN], y_range=[CY_MIN, CY_MAX], rotation=[-np.pi, np.pi],
            reference_pos=self.mujoco_arena.table_top_abs,ensure_object_boundary_in_range=True, ensure_valid_placement=True
        ))

        return sampler, ["LeftSampler", "RightSampler", "UpSampler", "BottomSampler"]

    def _create_objects(self):
        chosen_objects = random.sample(self.available_objects, settings.n_objects)
        self.objects = []

        for i, class_obj in enumerate(chosen_objects):
            name = f"RNG_Object_{i}"
            kwargs = {
                "name": name,
                "size_min": [0.01, 0.01, 0.01],
                "size_max": [0.035, 0.035, 0.035]
            }

            obj_instance = class_obj(**kwargs)
            self.objects.append(obj_instance)

    def _reset_internal(self):
        super()._reset_internal()

        # For every reset we re-compute a random positioning for objects
        self._place_objects()

    def reward(self, action):
        # In the future for RL we will use this as the reward function (logic)
        return 0.0

    def _check_success(self):
        # In the future for RL this will be our function for binary success checking...
        return False

    def _place_objects(self):
        new_pos = self.sampler.sample()
        """
        sample() returns a dict shaped like this:
            - Key, the object name (e.g. "Cube")
            - Value, a numpy tuple of 8 dimensions that represents the tridimensional placement
                     of the object in the space and its instance:
                        - The first 3 (x, y, z) are the absolute cartesian coordinates in space
                        - The next 4 (qw, qx, qy, qz) is the quaternion which defines the random rotation of the object
                        - The last 1 is the object instance 
        """
        for obj_pos, obj_quat, obj in new_pos.values():
            # Update the object position by changing its "free joint"'s state.
            self.sim.data.set_joint_qpos(obj.joints[0], np.concatenate([np.array(obj_pos), np.array(obj_quat)]))
