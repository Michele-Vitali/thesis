import random

import numpy as np
import settings
from robosuite.environments.base import register_env
from robosuite.environments.manipulation.single_arm_env import SingleArmEnv
from robosuite.models import objects
from robosuite.models.arenas import TableArena
from robosuite.models.tasks import ManipulationTask

# UniformRandomSampler places objects randomly within a given range
# We can also give a sampler per object with the SequentialCompositeSampler (future?)
from robosuite.utils.placement_samplers import (
    SequentialCompositeSampler,
    UniformRandomSampler,
)


@register_env
class CustomTask(SingleArmEnv):

    def __init__(self, reward_shaping: bool = False, **kwargs) -> None:
        """
        Initializes the custom environment inherited by the SingleArmEnv environment provided by robosuite.

        Args:
            reward_shaping (bool): Whether to use reward shaping.
            **kwargs: Additional keyword arguments for the environment.
        """
        self.reward_shaping = reward_shaping
        self.available_objects = [
            objects.BoxObject,
            objects.BoxObject,
            objects.BoxObject,
            #objects.BallObject,
            #objects.CylinderObject,
            #objects.CapsuleObject,
            #objects.CanObject,
            #objects.MilkObject,
            #objects.CerealObject
        ]
        self.gripper_state = -1.0 # Start with the gripper open
        self.sampler = None
        super().__init__(**kwargs)

    def _load_model(self) -> None:
        """
        Creates the actual environment and initializes all objects contained in it.
        """
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
        self._create_objects()

        # 3. Sampler for placing objects
        samplers_names = self._define_sampler()
        random.SystemRandom().shuffle(samplers_names) # Randomize the order of the samplers for each reset

        for id, obj in enumerate(self.objects):
            target_sampler = samplers_names[id % len(samplers_names)]
            self.sampler.add_objects_to_sampler(sampler_name=target_sampler, mujoco_objects=obj)

        # 4. Robot + Arena + Object
        self.model = ManipulationTask(
            mujoco_arena=self.mujoco_arena,
            mujoco_robots=[self.robots[0].robot_model],
            mujoco_objects=self.objects,
        )

    def _define_sampler(self) -> list[str]:
        """
        Defines the sampler used for deciding the placement of newly spawned objects.
        In particular a SequentialCompositeSample has been used to combine multiple UniformRandomSampler(s)
        for defining a particular spawning area.

        Returns:
            sampler, samplers_names[] (SequentialCompositeSampler, np.ndarray): The created SequentialCompositeSampler
                and the names of the samplers used for defining the spawning area. 
        """
        # Define the spawnable surface without the central square
        # which is the area of release
        X_MIN, X_MAX = -0.35, +0.05
        Y_MIN, Y_MAX = -0.2, +0.2

        CX_MIN, CX_MAX = -0.05, +0.05
        CY_MIN, CY_MAX = -0.05, +0.05

        self.sampler = SequentialCompositeSampler(name="Sampler")

        # Now we define the 4 separate spawnable areas (left, right, bottom, up)
        # 1. Left
        self.sampler.append_sampler(UniformRandomSampler(
            name="LeftSampler", mujoco_objects=None, x_range=[X_MIN, X_MAX], y_range=[Y_MIN, CY_MIN], rotation=[-np.pi, np.pi], 
            reference_pos=self.mujoco_arena.table_top_abs,ensure_object_boundary_in_range=True, ensure_valid_placement=True
        ))

        # 2. Right
        self.sampler.append_sampler(UniformRandomSampler(
            name="RightSampler", mujoco_objects=None, x_range=[X_MIN, X_MAX], y_range=[CY_MAX, Y_MAX], rotation=[-np.pi, np.pi],
            reference_pos=self.mujoco_arena.table_top_abs,ensure_object_boundary_in_range=True, ensure_valid_placement=True
        ))

        # 3. Up
        self.sampler.append_sampler(UniformRandomSampler(
            name="UpSampler", mujoco_objects=None, x_range=[CX_MAX, X_MAX], y_range=[CY_MIN, CY_MAX], rotation=[-np.pi, np.pi],
            reference_pos=self.mujoco_arena.table_top_abs,ensure_object_boundary_in_range=True, ensure_valid_placement=True
        ))

        # 4. Bottom
        self.sampler.append_sampler(UniformRandomSampler(
            name="BottomSampler", mujoco_objects=None, x_range=[X_MIN, CX_MIN], y_range=[CY_MIN, CY_MAX], rotation=[-np.pi, np.pi],
            reference_pos=self.mujoco_arena.table_top_abs,ensure_object_boundary_in_range=True, ensure_valid_placement=True
        ))

        return ["LeftSampler", "RightSampler", "UpSampler", "BottomSampler"]

    def _create_objects(self) -> None:
        """
        This function randomly picks n_objects (defined in settings.py) by the list of possible objects defined.
        in the environment, then create them one by one and adds them to the objects list of the environment.
        """
        chosen_objects = random.SystemRandom().sample(self.available_objects, settings.n_objects)
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
        """
        Resets the internal simulation state of the environment.
        """
        super()._reset_internal()

        # For every reset we re-create the object and compute a random positioning
        self._create_objects()
        self._place_objects()

    def reward(self, action):

        reward = 0.0

        # If the simulation has succeded return 1!
        if self._check_success():
            return 1.0

        # If there should not be any reward policy or there is no object, return 0
        if not self.reward_shaping or not self.objects:
            return 0.0

        # Compute how far is the object from the central drop point
        obj = self.objects[0]
        obj_id = self.sim.model.body_name2id(obj.root_body)
        obj_pos = self.sim.data.body_xpos[obj_id]

        # Introduce favouring of smoother movements
        # Get eef position
        eef_pos = self.sim.data.site_xpos[self.robots[0].eef_site_id]

        # 1. Bonus for closeness to objects
        dist_eef_obj = np.linalg.norm(eef_pos - obj_pos)
        # We use tanh to normalize the value between [0, 1] scaled by importance
        reward += settings.closeness * (1.0 - np.tanh(10.0 * dist_eef_obj))

        # 2. Bonus for grasping and lifting the object
        is_grasped = self._check_grasp(gripper=self.robots[0].gripper, object_geoms=obj)
        if is_grasped:
            reward += settings.grasp

            # Bonus for elevating the cube (future: reward linear to a range of ideal z-values)
            table_z = self.mujoco_arena.table_top_abs[2]
            # We favour higher z when distant from the drop point, otherwise we favour lower z
            dist_xy = np.linalg.norm(obj_pos[:2] - np.array([0.0, 0.0]))
            drop_zone_factor = np.clip(dist_xy / 0.10, 0.0, 1.0)

            # Define an ideal z height
            target_z = table_z + (settings.z_target * drop_zone_factor)

            # Compute how off we are from that ideal z
            delta_z = abs(obj_pos[2] - target_z)

            # Bonus for stability near the ideal z
            reward += settings.lift * (1.0 - np.tanh(15.0 * delta_z))

            # Bonus for centered placement
            reward += settings.placing * (1.0 - np.tanh(10.0 * dist_xy))

        # Penalty for sudden movements
        action_penalty = settings.sudden_movements_penalty * np.linalg.norm(action)
        reward -= action_penalty

        # Clip the reward between 0.0 and 0.95
        return float(np.clip(reward, 0.0, 0.95))

    def _check_success(self, xy_tolerance : float = 0.03, velocity_tolerance : float = 0.01) -> bool:

        # If there are no object then we should not even evaluate the success
        if not self.objects:
            return False

        # Compute the distance from the drop point
        obj = self.objects[0]
        obj_id = self.sim.model.body_name2id(obj.root_body)
        obj_pos = self.sim.data.body_xpos[obj_id].copy()

        dist_xy = np.linalg.norm(obj_pos[:2] - np.array([0.0, 0.0]))
        # If the distance is greater than the tolerance we failed
        if dist_xy > xy_tolerance:
            return False

        # Compute the object velocity
        obj_vel = self.sim.data.get_body_xvelp(obj.root_body)
        # If it is "fast enough" (let's say it is moving or rolling), we failed
        if np.linalg.norm(obj_vel) > velocity_tolerance:
            return False

        # If the gripper is open (-1.0) it means our simulation returned to neutral correctly and we return True
        return self.gripper_state <= 0

    def _place_objects(self):
        """
        This function positions the objects in the environment by using the defined env.sampler.
        """
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
