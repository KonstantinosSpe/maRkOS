"""Shared test setup.

* The pipeline reads and writes calibration, bottle types and logs under one data directory (``markos_vision.config``).
  Point it at a throw-away folder before anything imports the package, so no test can touch a real calibration.
* ROS 2 is optional. Without it the ROS modules are stubbed (see ``ros_stubs``) and tests that need the real thing are skipped.
"""

import os
import tempfile

import pytest
import ros_stubs

os.environ["MARKOS_DATA_DIR"] = tempfile.mkdtemp(prefix="markos-test-data-")
ROS_AVAILABLE = ros_stubs.install_if_missing()


def pytest_collection_modifyitems(config, items):
    if ROS_AVAILABLE:
        return
    skip = pytest.mark.skip(reason="needs a ROS 2 installation (rclpy)")
    for item in items:
        if "ros" in item.keywords:
            item.add_marker(skip)
