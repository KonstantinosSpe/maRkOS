# markos (ROS 2 package)

The hardware layer of the project: everything between "a joint target in radians" and "a line of text on the
Arduino's serial port", plus the geometry that decides which targets the arm can reach at all.

| Module | What it does | Needs ROS? |
|---|---|---|
| `bridge_node.py` | The `thor_bridge` node: homes the arm, publishes `/joint_states`, executes `/joint_command` while armed | yes |
| `axis_controller.py` | Decides which `MOVE` line goes out next: rate cap, limits, blocked-direction memory, lost-ack handling | no |
| `serial_link.py` | The one serial connection to the firmware, read in a background thread | no (pyserial) |
| `sim_firmware.py` | A stand-in Arduino speaking the same protocol, used by `mock:=true` and by the tests | no |
| `hw_config.py` | Measured steps per degree, step limits, axis signs, gripper geometry: the single source of truth | no |
| `reach.py` | Can the gripper be put at this point, and with which joint angles? Solves the planar 2-link problem and rejects anything outside the real joint ranges | no |

Keeping the decision logic (`axis_controller`, `reach`) free of `rclpy` is deliberate: it runs in a plain `pytest`
against `sim_firmware`, so the safety behaviour is checked before anything is pointed at the real arm.

## Interface of `thor_bridge`

| Kind | Name | Type | Notes |
|---|---|---|---|
| Publishes | `/joint_states` | `sensor_msgs/JointState` | Only axes whose position is known (homed). Undriven joints are reported at 0 so the URDF stays complete |
| Subscribes | `/joint_command` | `sensor_msgs/JointState` | Radians, joint names `E`, `D`, `BC`. Ignored unless armed |
| Subscribes | `/gripper_command` | `std_msgs/Float32` | Gripper openness. Ignored unless armed |
| Service | `/home` | `std_srvs/Trigger` | Disarms, clears a latched fault and re-homes B, D, E |
| Parameter | `port` | string | Serial device, default `/dev/ttyACM0` |
| Parameter | `mock` | bool | Talk to `sim_firmware` instead of hardware |
| Parameter | `armed` | bool | **Starts `false`.** The arm only moves while this is true |
| Parameter | `sign_E`, `sign_D`, `sign_BC` | int | Direction of each axis against the model (defaults from `hw_config.SIGN`) |

## Safety behaviour

* The node **never moves the arm on its own**: `armed` defaults to `false`, and it refuses to be armed while
  homing is running, before every axis is homed, or while a fault is latched.
* Any fault (a lost acknowledgement, a firmware direction that disagrees with `hw_config`) disarms the node and
  keeps it refused until the arm has been re-homed.
* Targets outside the mechanical range are clamped, and the step limits mirror the firmware's own soft limits.
* Axis A has no home sensor and loses steps under load, so it is never driven from here.

```bash
ros2 run markos thor_bridge --ros-args -p port:=/dev/ttyUSB0      # real arm: homes B, D, E at start
ros2 run markos thor_bridge --ros-args -p mock:=true              # no hardware
ros2 param set /thor_bridge armed true                            # let it move
ros2 param set /thor_bridge armed false                           # stop at once
ros2 service call /home std_srvs/srv/Trigger                      # re-home after a fault
```

## Tests

```bash
pytest ros2/markos/test        # from the repository root, no ROS installation needed
```
