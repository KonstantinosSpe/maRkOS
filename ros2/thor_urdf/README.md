# thor_urdf

Robot description of the [Thor](https://github.com/AngelLM/Thor) arm for RViz, taken from the
[Thor-ROS](https://github.com/AngelLM/Thor-ROS) project (`ws_thor/src/thor_urdf`). It is included here, with its
original license files, so the repository builds and visualises without a second checkout. See
[`NOTICE.md`](../../NOTICE.md) for the licensing of the package and of the CAD-derived meshes.

## What was changed for this project

Everything else is unmodified upstream content.

| File | Change |
|---|---|
| `urdf/thor.urdf.xacro` | Joints renamed to the axis names the firmware and bridge use (`E`, `D`, `A`, `BC`); the redundant joints 2 and 6 made fixed; the Gazebo include removed |
| `launch/follow_palm.launch.py` | Added: RViz plus `robot_state_publisher` without the slider GUI, for the hand-following demo (`vision/experiments/follow_palm_ros.py`) |
| `rviz/display.rviz` | A marker display for `/bottle_target` (where the vision pipeline places the gripper) and a closer default camera |

## Use

```bash
ros2 launch thor_urdf display.launch.py          # RViz with a joint slider GUI
ros2 launch thor_urdf follow_palm.launch.py      # RViz only; joint states come from elsewhere (bridge or demo)
```

With the bridge running (`ros2 run markos thor_bridge`), RViz shows the real arm's confirmed position.
