# Testing

```bash
pip install -e ".[dev,headless]"
pytest                    # everything: about 190 tests, roughly a minute
pytest vision/tests -k diameter          # a slice
pytest -m ros             # only the tests that need a real ROS 2 (sourced, with the workspace built)
ruff check .              # lint
```

Nothing here needs a camera, an arm or a GPU, and ROS 2 is optional.

## What is tested, and how

| Where | What |
|---|---|
| `ros2/markos/test` | The axis controller against the firmware **simulator**: rate cap, limits, blocked-direction memory, lost acknowledgements, flipped axes. The reach planner: what is reachable, which pose, the mirrored limits |
| `vision/tests` | The camera pipeline against **synthetic cameras with known truth** (below), and the decision logic of the bottle finder against scripted outlines |
| `teleop/tests` | The pure modules of the desktop app: hand-landmark geometry, the two-link IK checked by forward kinematics, the safety configuration |

### Synthetic cameras

The camera-facing code cannot be checked against the real arm in CI, so the tests build a pinhole camera with radial lens bend and pixel
noise, put bottles at known desk positions, render what it would see, and ask the pipeline where they are:

* `test_desk_calibration`: accuracy over the workspace, leave-one-out, coverage advice, a mis-measured point, a mirrored measuring frame.
* `test_star_lock`, `test_level_camera`, `test_calibration_with_moved_laptop`: the laptop moved to six places, including a small change of
  lid angle, and the refusals (stars swapped, laptop 9 cm higher, picture shaking).
* `test_bottle_diameter`: a bottle's near edge versus its axis, with an independent brute-force ground truth for the foot's rim.
* `test_desk_markers`: printed ArUco markers rendered under perspective, hidden, moved.
* `test_pose_probe`: the aim probe's fit, at poses it was not fitted on.

### The real programs, with fakes at the edges

`test_hover_*` and `test_calibrate_desk_tool` run `hover.run_camera` and `calibrate_desk.main` themselves, with a fake camera, a stand-in for
the YOLO model that returns scripted outlines, and a fake ROS link (`vision/tests/harness_hover.py`). They check the arm is sent to the right
place, is not sent before `g`, and is not sent when the camera has moved.

### ROS

The ROS modules are stubbed (`vision/tests/ros_stubs.py`) when they cannot be imported, so the hover logic is still tested in CI. The one
test that needs the real thing, `test_bridge_control`, starts the bridge in mock mode **on its own ROS domain** (77) and checks arming and
homing through the window's control object, so it can never reach a real arm. Run it with the workspace sourced:

```bash
scripts/wsl/build_ros.sh && source scripts/wsl/env.sh && pytest -m ros
```

## Continuous integration

`.github/workflows/ci.yml` runs `ruff` and `pytest` on every push and pull request with headless OpenCV, and shell-checks the launcher scripts.
The tests point `MARKOS_DATA_DIR` at a temporary folder before anything is imported, so no test can touch a real calibration.
