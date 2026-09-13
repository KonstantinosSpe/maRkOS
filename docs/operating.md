# Operating the arm

**The arm is a machine that can hurt someone and break itself.** It has no position feedback and no enclosure. Keep your hand near the
power switch whenever anything that can move is running, and keep the area around the arm clear.

## A normal session

1. Power the arm and plug in the Arduino and the camera.
2. **`Start Thor.bat`**: attaches the two USB devices to WSL (and keeps them attached if they drop out), waits until the serial port answers
   steadily, asks you to confirm the arm is clear, then starts the bridge, which **homes the arm** (B, then D, then E; D can take up to
   40 seconds). When homing is complete it opens the hover window. *Nothing is armed at this point.*
3. Put the bottle in view. The window shows the outline, the recognised type and, once it has held still, the joint angles it would send.
4. **`Arm Thor.bat`** lets the bridge accept commands. Then press **`g`** in the hover window: the arm moves over the bottle and follows it.
5. **`Disarm Thor.bat`** (or `g` again) stops it at once; the motors keep holding. **`Stop Thor.bat`** disarms and closes everything.

Arming and `g` are separate on purpose. The window refuses `p` (arm) while `g` is on, because arming with `g` on would move the arm at once.

## Hover window keys

| Key | Action |
|---|---|
| `g` | go / stop sending commands |
| `p` | arm / disarm the bridge from the window (the status line at the top shows the state) |
| `h`, `h` | home the arm: press twice within 3 seconds; it disarms and moves B, D, E |
| `w` / `s` | hover height up / down in 33 mm steps (six presses are about 20 cm) |
| `,` / `.` | hover height in 5 mm steps (never below 30 mm above the middle of the cap) |
| `[` `]` / `{` `}` | turn the base zero by 1 / 10 degrees |
| `<` / `>` | base-turn gain (steps of 0.02) |
| `-` / `=` | radius offset (5 mm steps) |
| `e` | flip the base direction |
| `;` / `'` | raise / lower the riser the bottle stands on, in 0.5 cm steps |
| `a` | enrol the bottle in view; `x` mark a false alarm; `t` list the enrolled types |
| `q` | quit (this also disarms the bridge) |

The tuning keys are saved to `data/calibration/pick_config.json` and read again the next time the window starts.

## What blocks the arm

The hover window lists every reason it is not sending, on the line starting `waiting:` in the overlay (and every couple of seconds
in `data/logs/hover.log`):

* `no bottle in view`, or `bottle not enrolled (a)`: the detection is not an enrolled type;
* `bottle still moving`: the position has not held steady for about eight frames;
* `camera moved: redo calibrate_desk.py`, or a star-lock message: the laptop is not where the calibration was made and the stars have
  not (yet) placed it;
* `nowhere reachable`: not even a nearby spot can be reached;
* `press g`, and `no bridge listening on /joint_command`.

Even when the window is sending, the **bridge ignores commands unless it is armed**, and refuses to be armed until homing is done.

## When something goes wrong

| Symptom | What to check |
|---|---|
| `Start Thor.bat` cannot see a device | The USB cable and the arm's power; `Check Thor devices.bat` repeats just the device check |
| "has never been shared" | The one-time `usbipd bind` in an administrator PowerShell ([setup.md](setup.md)) |
| The bridge window says `Could not open ...` | The Arduino is not attached in WSL (`ls /dev/ttyUSB*`), or another program owns the port |
| The bridge reports a **FAULT** and disarms | A lost acknowledgement, or a direction that disagrees with `hw_config`. Re-home (`Home Thor.bat`) before arming again |
| Homing does not finish | A sensor or cable; the bridge gives up after 75 seconds per axis. Look at the bridge window |
| The overlay says `ROUGH STAR ESTIMATE` | There is no desk calibration yet: run `Calibrate desk.bat` ([calibration.md](calibration.md)) |
| Positions are off after moving the laptop | Keep it at the same height and lid angle as at calibration, with both stars in view for about two seconds |

The bridge log is written to `data/logs/bridge.log` and the hover window logs the reason it is or is not sending to
`data/logs/hover.log`.
