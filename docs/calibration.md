# Calibration

Four independent things are calibrated. Each has its own tool and its own saved file under `data/`.

| What | Tool | Saved in |
|---|---|---|
| Which way each axis turns; steps per degree | `direction_test`, and the step-accuracy tools in `teleop/tools` | `ros2/markos/markos/hw_config.py` |
| Where a pixel on the desk is | `calibrate_desk` (a bottle and a tape measure) | `data/calibration/desk_calibration.json` |
| Where the laptop is, if it is moved | `calibrate_desk`, keys `s`, `h` (the two red stars and a ruler) | the same file |
| Where the gripper *really* goes | `pose_probe` (the bottle as a ruler) | `data/calibration/aim_fit.json`, applied to `pick_config.json` |

## 1. Directions and scales

`python -m markos_vision.apps.direction_test` (bridge running and homed) moves E, D and B a few degrees, one at a time, and asks what the arm
did in terms of the room ("the gripper end swung toward you or away", "the arm tipped to the right or left"). Wrong answers become sign
flips to put in `hw_config.SIGN`. Steps per degree were measured against photographs of the arm and the homing beacons; the raw datasets
and the tools that produced them are in `teleop/measurements` and `teleop/tools`.

## 2. The desk (tape calibration)

The tool prints the measuring frame when it starts. In short: measure each bottle position from the **right-hand corner of the robot box's
end face, as seen from the laptop** (`u` to the right of the corner, `v` toward the laptop from the end face, both in cm, to the *centre* of
the bottle's foot). The robot's axis is built in at `(-9.0, -21.9)`.

1. `Calibrate desk.bat` (close the hover window first: both use the camera). Do not move the laptop or the robot during the session.
2. Stand **the same bottle** on 8 to 11 spots spread over where it will be picked up, both in bearing and distance from the base. For the 22 cm bottle the
   exact 5 cm hover works 26 to 36 cm from the axis ([hardware.md](hardware.md)).
3. At each spot wait until the foot dot is cyan and the window says `STILL`, press `c`, type `u v`.
4. `t` lists the points with their **fit error** and **leave-one-out error** (the honest number: the mapping refitted without that point). A
   point far above the rest was mis-measured; `u` undoes the last point. Four points fit exactly and prove nothing, so use at least six.
5. `d` sets the diameter of the calibration bottle's foot (see below).

On the author's setup: 11 points, worst fit error 5.9 mm, leave-one-out median 5.8 mm and worst 13.3 mm.

### Why the diameter matters

The camera sees the lowest point of the bottle's round foot, its *near edge*, while the tape marks the *centre*. Every point in the
calibration therefore maps a near edge to a centre, which is fine for that bottle from that camera position. Two things break it: another
bottle with a different width, and a laptop that has been moved (the near edge is now on a different side). With the diameter known, the
tool fits the camera with the foot modelled and corrects both. Enrol bottles with their foot diameter (`a` in the finder or hover
window); one without a diameter is taken to be the size of the calibration bottle.

## 3. Moving the laptop: the stars and the camera height

After at least six spots, with both red stars in view, press `s`. It asks once for the two stars' positions (how far left of the top-right
corner of the box end face, and how far down from its top edge, to the millimetre) and fits the camera. Measure the height of the camera
lens above the desk with a ruler and press `h` to give it: the picture alone cannot tell a higher camera from a farther one.

From then on the hover window works out where the laptop has been put from the stars. Keep the laptop **at the same height and lid angle**
(a small tilt change is tolerated), with both stars visible for about two seconds while it locks. It refuses, and shows why, when the stars
do not fit the calibration, when the answer would need an implausible move, or while the picture is shaking. If points are added or the
diameter changes, press `s` again so the camera is refitted (the tool tells you when it is stale).

Printed ArUco markers (`python -m markos_vision.geometry.desk_markers` writes the pages, key `m` learns them) are an alternative that needs
no star measurements.

## 4. The aim probe

The planner assumes the arm's centre from the CAD drawing, a turn scale of 1 and the drawing's link lengths. If any is off, the gripper
lands elsewhere. `Calibrate aim.bat` moves the arm to six poses about 40 cm above the desk; at each one you put the bottle under the
gripper's fingers and the camera (already calibrated with the tape) says where it is. It fits the arm's true centre, the turn offset and
scale and the reach error, and prints what to change. `Calibrate aim.bat --apply` writes the result to `pick_config.json` and moves the
axis in the desk calibration (with `*.before_aim_fit.json` backups). It is checked in simulation to correct 5 to 13 cm misses to about 2 mm; it
has not yet been run on the arm.

## Tuning parameters (`pick_config.json`)

| Key | Meaning |
|---|---|
| `e_sign` | +1 if a bottle further right in the picture needs E to increase |
| `e_zero_deg` | planner azimuth of the direction from the robot toward the camera |
| `e_gain` | E commanded per degree the base should turn: above 1 if it falls short |
| `radius_offset_mm` | added to the measured distance from the base axis |
| `star_to_axis_cm` | how much farther the base axis is from the camera than the stars (rough star fallback only) |
| `hover_clearance_mm` | how far above the middle of the cap to hover |
| `base_height_cm` | how far the robot's base sits above the surface the bottle stands on |
