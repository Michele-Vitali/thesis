# P-Rob3 robosuite integration

This bundle adapts the Webots-derived P-Rob3 MJCF to robosuite 1.4.1.

## Important architecture

robosuite treats the manipulator arm and gripper as two separate MJCF models. The original P-Rob3 model contained the P-Grip inside the robot XML, which caused the `ManipulatorModel` end-effector lookup to fail and also caused the 8 P-Rob3 joints to be interpreted as arm joints.

The bundle therefore contains:

- `assets/robots/PRob3/PRob3.xml`: six-DOF P-Rob3 arm only.
- `assets/robots/PRob3/PGripper.xml`: P-Grip model merged at `right_hand` by robosuite.
- `assets/robots/PRob3/PRob3_standalone.xml`: original full standalone MJCF kept as a reference.
- `scripts/custom_robot.py`: custom robot + custom 1-DOF robosuite gripper registration.
- `scripts/settings.py`: uses `PRob3Gripper` instead of `None`.

## Important physics fix

The six arm actuators are MuJoCo `motor` actuators, not `position` actuators. robosuite's OSC controller writes torques directly to `sim.data.ctrl`, so using position actuators would interpret those values as position targets rather than joint torques.

The gripper retains position actuators because robosuite's gripper interface maps the normalized [-1, 1] gripper command to the physical joint position ranges.

## Run

Keep the directory structure intact and run from `scripts/`:

```bash
python main.py
```

The expected action dimensionality is now 7:

- 6 arm OSC-Pose dimensions
- 1 P-Grip command

The custom gripper maps that one command to the two physical P-Grip joint actuators.
