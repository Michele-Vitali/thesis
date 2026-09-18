import numpy as np
from robosuite.models.arenas import TableArena
from robosuite.models.objects import BoxObject
from robosuite.models.tasks import ManipulationTask
from robosuite.environments.base import register_env
from robosuite.environments.manipulation.manipulation_env import ManipulationEnv

# UniformRandomSampler places objects randomly within a given range
# We can also give a sampler per object with the SequentialCompositeSampler (future?)
from robosuite.utils.placement_samplers import UniformRandomSampler

@register_env
class CustomTask(ManipulationEnv):

    def __init__(self, **kwargs):
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

        # 2. A red box
        cube = BoxObject(
            name="Cube",
            size=[0.02, 0.02, 0.02],
            rgba=[1, 0, 0, 1]
        )

        self.objects = [cube]

        # 3. Sampler for placing objects
        self.placement_initializer = UniformRandomSampler(
            name="ObjectSampler",
            mujoco_objects=self.objects,
            x_range=[-0.4, 0.3], # As +0.4 was sometimes unfeasible for grabbing!
            y_range=[-0.4, 0],
            rotation=[-np.pi, np.pi], # Spawn the cube with a random rotation between -180° and +180°
            reference_pos=self.mujoco_arena.table_top_abs,  # Tells the placer to choose x and y relative to the table
            ensure_object_boundary_in_range=True,
            ensure_valid_placement=True
        )

        # 4. Robot + Arena + Object
        self.model = ManipulationTask(
            mujoco_arena=self.mujoco_arena,
            mujoco_robots=[self.robots[0].robot_model],
            mujoco_objects=self.objects
        )

    def _reset_internal(self):
        super()._reset_internal()

        # For every reset we re-compute a random positioning for objects
        # and we reset the model timesteps to 0
        self._place_objects()

    def reward(self, action):
        # In the future for RL we will use this as the reward function (logic)
        return 0.0

    def _check_success(self):
        # In the future for RL this will be our function for binary success checking...
        return False

    def _place_objects(self):
        new_pos = self.placement_initializer.sample()
        """
        sample() returns a dict shaped like this:
            - Key, the object instance (e.g. "Cube")
            - Value, a numpy tuple of 7 dimensions that represents the tridimensional placement
                     of the object in the space:
                        - The first 3 (x, y, z) are the absolute cartesian coordinates in space
                        - The last 4 (qx, qy, qx, qw) is the quaternion which defines the random rotation of the object
        """
        for obj_pos, obj_quat, obj in new_pos.values():
            # Update the object position by changing its "free joint"'s state.
            self.sim.data.set_joint_qpos(obj.joints[0], np.concatenate([np.array(obj_pos), np.array(obj_quat)]))
