"""Stand-ins for the ROS 2 Python modules, so the hover window's logic can be tested where ROS is not installed (CI).

``markos_vision.apps.hover`` imports rclpy and the message types at module level, but the tests replace its ROS link with a
fake, so importing is all that is needed. Where a real ROS 2 is installed nothing is stubbed and ``ROS_AVAILABLE`` is true.
"""

import sys
import types

_STUBBED = (
    "rclpy",
    "geometry_msgs", "geometry_msgs.msg",
    "rcl_interfaces", "rcl_interfaces.msg", "rcl_interfaces.srv",
    "sensor_msgs", "sensor_msgs.msg",
    "std_msgs", "std_msgs.msg",
    "std_srvs", "std_srvs.srv",
    "visualization_msgs", "visualization_msgs.msg",
)


class _StubModule(types.ModuleType):
    """A module in which any attribute you ask for is an empty class."""

    def __getattr__(self, name):
        if name.startswith("__"):
            raise AttributeError(name)
        value = type(name, (), {})
        setattr(self, name, value)
        return value


def install_if_missing():
    """Stub the ROS modules unless the real ones can be imported. Returns True when real ROS is available."""
    try:
        import rclpy  # noqa: F401
    except ImportError:
        for name in _STUBBED:
            sys.modules[name] = _StubModule(name)
        return False
    return True
