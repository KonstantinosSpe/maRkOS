# ROS 2 packages

This folder is the `src/` of a colcon workspace.

| Package | Build type | Purpose |
|---|---|---|
| [`markos`](markos) | `ament_python` | Bridge to the Arduino firmware, reach planner, firmware simulator |
| [`thor_urdf`](thor_urdf) | `ament_cmake` | Robot description of the Thor arm (from Thor-ROS, adapted) |

Build from the repository root inside WSL/Ubuntu with ROS 2 sourced:

```bash
scripts/wsl/build_ros.sh          # colcon build into ~/markos_ws, with --symlink-install
source ~/markos_ws/install/setup.bash
```

The workspace is built outside the repository on purpose, so `build/`, `install/` and `log/` never end up in the
working tree. `--symlink-install` means the installed `markos` package points at these sources, so an edit takes
effect without rebuilding.
