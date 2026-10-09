import numpy as np
import settings
from robosuite.utils.placement_samplers import UniformRandomSampler


class RadialRandomSampler(UniformRandomSampler):
    def __init__(self, name, mujoco_objects=None, radial_range=settings.radial_range, angle_range=(-np.pi, np.pi),
        table_bounds=None, forbidden_pos=None, forbidden_radius=0.0, rotation=None, rotation_axis="z",
        ensure_object_boundary_in_range=True, ensure_valid_placement=True, reference_pos=(0, 0, 0), z_offset=0.0):

        self.radial_range = radial_range
        self.angle_range = angle_range

        self.table_bounds = table_bounds

        self.forbidden_center = None if forbidden_pos is None else np.asarray(forbidden_pos, dtype=float)

        self.forbidden_radius = float(forbidden_radius)
        self._cached_xy = None

        super().__init__(name=name, mujoco_objects=mujoco_objects, x_range=(0.0, 0.0), y_range=(0.0, 0.0), rotation=rotation,
            rotation_axis=rotation_axis, ensure_object_boundary_in_range=ensure_object_boundary_in_range, 
            ensure_valid_placement=ensure_valid_placement, reference_pos=reference_pos, z_offset=z_offset)

    def _sample_xy(self, object_horizontal_radius):
        """
        Samples a valid XY point from the annular workspace.

        Sampling is uniform with respect to AREA, not radius.

        Returns:
            np.ndarray:
                [x_relative, y_relative]

                Coordinates are relative to reference_pos because the parent
                UniformRandomSampler will add reference_pos afterwards.
        """

        r_min, r_max = self.radial_range
        theta_min, theta_max = self.angle_range

        if r_min < 0.0:
            raise ValueError("radial_range minimum cannot be negative.")

        if r_max <= r_min:
            raise ValueError(
                "radial_range must satisfy r_max > r_min."
            )

        for _ in range(5000):
            # Sampling r^2 uniformly gives a uniform distribution by area.
            radius = np.sqrt(np.random.uniform(low=r_min ** 2, high=r_max ** 2))
            theta = np.random.uniform(low=theta_min, high=theta_max)

            # Position relative to the robot / reference position.
            x_relative = radius * np.cos(theta)
            y_relative = radius * np.sin(theta)

            # Convert to world coordinates for validity checks.
            x_world = x_relative + self.reference_pos[0]
            y_world = y_relative + self.reference_pos[1]

            if self.table_bounds is not None:
                x_min, x_max, y_min, y_max = self.table_bounds

                # If requested, keep the ENTIRE object inside the table, not only its center.
                margin = object_horizontal_radius if self.ensure_object_boundary_in_range else 0.0

                if not (x_min + margin <= x_world <= x_max - margin and y_min + margin <= y_world <= y_max - margin):
                    continue

            if self.forbidden_center is not None and self.forbidden_radius > 0.0:
                point = np.array([x_world, y_world], dtype=float)

                distance_from_forbidden_center = np.linalg.norm(point - self.forbidden_center)

                # Also take the object's physical radius into account so
                # that the object itself cannot enter the forbidden zone.
                minimum_distance = self.forbidden_radius + object_horizontal_radius

                if distance_from_forbidden_center < minimum_distance:
                    continue

            return np.array([x_relative, y_relative], dtype=float)

        raise ValueError("RadialRandomSampler could not find a valid XY position after 5000 attempts.")

    def _sample_x(self, object_horizontal_radius):
        """
        Generates an XY pair and returns X.

        Y is cached because UniformRandomSampler.sample() requests the two
        coordinates separately.
        """

        self._cached_xy = self._sample_xy(object_horizontal_radius)

        return self._cached_xy[0]

    def _sample_y(self, object_horizontal_radius):
        """
        Returns the Y coordinate generated together with the previous X.
        """

        if self._cached_xy is None:
            self._cached_xy = self._sample_xy(object_horizontal_radius)

        y = self._cached_xy[1]

        self._cached_xy = None

        return y
