# Third-party notices

The original code in this repository is released under the [MIT License](LICENSE). The following material comes from
other projects and stays under its own terms.

## Redistributed in this repository

### `ros2/thor_urdf` — Thor-ROS robot description

* Source: [AngelLM/Thor-ROS](https://github.com/AngelLM/Thor-ROS), package `ws_thor/src/thor_urdf`
  (package maintainer in `package.xml`: JorgePRamos).
* The package declares **Apache-2.0** (`ros2/thor_urdf/LICENSE`, `package.xml`); the Thor-ROS repository as a whole is
  published under **CC BY-SA 4.0**. Both notices are kept as published; consult the upstream repository for the
  authoritative terms.
* The meshes (`ros2/thor_urdf/meshes/*.dae`) are derived from the CAD of the Thor arm
  ([AngelLM/Thor](https://github.com/AngelLM/Thor)) and are included unmodified.
* Modifications made for this project are listed in [`ros2/thor_urdf/README.md`](ros2/thor_urdf/README.md).

## Used, not redistributed

These are fetched or installed by the person who sets the project up (see [`docs/setup.md`](docs/setup.md)).

| Component | License | Used for |
|---|---|---|
| [Thor](https://github.com/AngelLM/Thor) robot arm design | see upstream | The hardware this project drives |
| [Ultralytics YOLO11](https://github.com/ultralytics/ultralytics) (`yolo11n-seg.pt`) | AGPL-3.0 | Bottle segmentation. The weights are downloaded, not committed |
| [MediaPipe](https://github.com/google-ai-edge/mediapipe) | Apache-2.0 | Hand landmarks for gesture control |
| [OpenCV](https://opencv.org) | Apache-2.0 | Image processing, camera access, ArUco markers |
| [NumPy](https://numpy.org) | BSD-3-Clause | Numerics |
| [pySerial](https://github.com/pyserial/pyserial) | BSD-3-Clause | Serial link to the Arduino |
| [ROS 2](https://www.ros.org) | Apache-2.0 | Bridge node, launch files, RViz |
| Arduino `Servo` library | LGPL-2.1 | Gripper servo in the firmware |
