# Hardware notes

The arm is [Thor](https://github.com/AngelLM/Thor), an open-source 3D-printed robotic arm, driven here by an Arduino Mega running the
firmware in [`firmware/thor_teleop_firmware`](../firmware/thor_teleop_firmware/thor_teleop_firmware.ino).

## The kinematic chain

Walking from the base up to the gripper:

```
base_link (the surface the robot is bolted to)
  └─ E   base twist                     vertical axis, +E swings the gripper anticlockwise seen from above
       └─ D   shoulder hinge            362 mm above base_link
            └─ A   upper twist          never driven: no home sensor, loses steps under load
                 └─ B/C  elbow hinge    194 mm from D   (two motors, C mirrors B)
                      └─ gripper        hangs at right angles to the forearm
                                        grasp point 19 mm along and 184 mm across the forearm from the elbow axle
```

D and B/C are the two hinges the planar inverse kinematics solves for; E and A are twists that turn the whole assembly. D is the
hinge nearest the base, so it takes the angle from vertical, and B/C the angle of the forearm relative to the upper arm. Which joint is
"first" is a statement about the physical chain, not a labelling choice.

## Measured on the real arm

All of these live in [`ros2/markos/markos/hw_config.py`](../ros2/markos/markos/hw_config.py).

| Quantity | Value | How it was found |
|---|---|---|
| Direction of each axis against the model | `E +1, D -1, B +1` | `direction_test`: each axis moved a few degrees and the observer said which way it went. **D runs backwards** relative to the model |
| D steps per degree | 3700 / 216 | Commanded 45 and 75 degrees showed the arm at 54 and 90; after the correction 75 showed 74.8. So 3700 steps are about 216 degrees, not 180 |
| B steps per degree | 2.09 | Commanded -30 degrees showed about -39 in a photo, so B moves about 1.3x what it was asked. Photo estimate, roughly 5 to 10 % uncertain |
| E steps per degree | 3.87 | 60 degrees commanded turned the base about 36 for 167 steps and about 71 for 270. Good to about 5 % |
| Step limits | B −168..8, D −1698..1498, E ±460 | The firmware's soft limits pulled in by a hair; E held to ±460 steps so the base's cables do not wind up |
| Gripper geometry | 19 mm along, 184 mm across | Photos of the arm tipped 75 degrees; agrees with a 185 mm tape measurement |
| Gripper servo angles | closed 60, open 105 | Hand-tested on the MG996R; the firmware clamps `G<n>` to the same range |

The desktop app's `teleop/hw_config.py` holds an older set of scales (D `3700/180`, E `1004/360`, B `490/180`) from the first
calibration; the ROS side carries the re-measured ones.

### What the arm can and cannot reach

![reach envelope](images/reach-envelope.png)

The real joint ranges are much smaller than a textbook two-link arm's: B has only about 62 degrees of travel, so most of the workspace
that a generic IK happily answers for is not reachable. The consequences that shaped the project:

* The arm **cannot reach down to the surface it stands on.** With the robot on the desk, the lowest grasp point is about 15 cm above
  `base_link`, reached 18 to 22 cm out; it rises to 22 cm at 34 cm out and nothing below 38 cm is reachable at all. A bottle must therefore be tall enough, or stand on a riser, for its cap to fall in that band.
* An exact 5 cm hover above a 22 cm bottle needs the bottle **26 to 36 cm from the base axis**. Closer than about 24 to 26 cm is a
  dead zone where the planner hovers higher instead (`hover_plan` reports it as *degraded*).
* E's travel of about ±119 degrees decides which pose is available: leaning toward a point needs its azimuth within ±119 degrees of E = 0,
  leaning away (base turned around) needs it more than 61 degrees from E = 0, so both are open only between 61 and 119 degrees.

Regenerate the figure with `python scripts/plot_reach_envelope.py`.

## Firmware protocol

Plain text lines at 115200 baud; every command is acknowledged with a line starting `[fw]`. Moves **block** until they finish.

| Command | Effect |
|---|---|
| `MOVE <axis><steps> ...` | Move one or more axes by signed step counts (streaming use). Acknowledged with `[fw] MOVE <axis> done, moved <n>`, where `n` is the step count actually completed and, for a flipped axis, has the opposite sign |
| `JOG <axis>+` / `JOG <axis>-` | A small bounded nudge, checked against the limit switch every pulse |
| `HOME <axis>` / `HOME?` | Drive B, D or E to its homing beacon and zero its counter (`HOME?` only reports the sensors) |
| `SWEEP E+` / `SWEEP E-` | Run E a full lap in one forced direction, to measure steps per turn |
| `CAL <axis>` / `CAL STOP` | Record the net steps moved while jogging, to measure steps per degree |
| `G<n>` | Set the gripper servo to an absolute angle |
| `JOGSTEP <n>`, `SPEED <vmin> <vmax>` | Tune the nudge size and the ramp speed live, without reflashing |

D has no working hardware limit switch, so its moves are clamped in software. The firmware has no notion of "armed": all of that caution
lives on the Python side, in `hw_config` and the bridge.

[`firmware/pin_scanner`](../firmware/pin_scanner/pin_scanner.ino) is a diagnostic sketch used once to find which Mega pin each homing
sensor was wired to. Do not leave the arm connected to anything that can move while it runs.

## The scene the camera sees

* A laptop webcam at 640x480 about a metre from the arm, at about 24 cm height and looking level or slightly up.
* Two red star stickers on the end face of the robot's box, 7.9 cm apart, about 3.5 cm below its top edge. Their template images are
  made once with `markos_vision.apps.select_stars`.
* A bottle standing on the desk to the right of the box (only the enrolled 22 cm bottle with a 5.5 cm foot has been used so far).
