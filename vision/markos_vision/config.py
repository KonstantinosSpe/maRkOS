"""Where the vision pipeline keeps its files.

Everything the pipeline learns or records on one machine -- the desk calibration, the enrolled bottles, the star
templates, model weights and logs -- lives under a single data directory that is kept out of version control. It
defaults to ``<repository>/data`` and can be moved by setting the ``MARKOS_DATA_DIR`` environment variable.

All paths are plain strings, so callers can build file names next to them (``path.replace(".json", ".bak.json")``).
"""

import os
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = os.environ.get("MARKOS_DATA_DIR") or str(REPO_ROOT / "data")


def data_path(*parts):
    """A path under the data directory."""
    return str(Path(DATA_DIR, *parts))


def ensure_parent(path):
    """Create the folder a file is about to be written into, and return the path."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    return path


# Calibration: what ties the camera's view to the arm's
DESK_CALIBRATION = data_path("calibration", "desk_calibration.json")
PICK_CONFIG = data_path("calibration", "pick_config.json")
AIM_PROBE_DATA = data_path("calibration", "aim_probe.json")
AIM_FIT = data_path("calibration", "aim_fit.json")
DIRECTION_RESULT = data_path("calibration", "direction_result.json")

# The two red stars on the robot's base, used to place a laptop that has been moved
STAR_TEMPLATES = data_path("stars", "star_templates.npz")
STAR_FOCAL_SAMPLES = data_path("stars", "star_focal_samples.npz")
STAR_REFERENCE_CONTOUR = data_path("stars", "star_reference_contour.npy")

# Enrolled bottle types (name, height, cap height, appearance signatures)
BOTTLE_TYPES = data_path("bottles", "bottle_types.json")

# Model weights, fetched by scripts/download_models.sh
YOLO_WEIGHTS = data_path("models", "yolo11n-seg.pt")
HAND_LANDMARKER = data_path("models", "hand_landmarker.task")
EFFICIENTDET_MODEL = data_path("models", "efficientdet_lite0.tflite")   # only the superseded first tracker uses it
SCENE_REFERENCE = data_path("bottles", "scene_reference.npz")           # likewise

# Logs and debugging images
LOG_DIR = data_path("logs")
HOVER_LOG = data_path("logs", "hover.log")
PROBE_LOG = data_path("logs", "probe.log")
BOTTLE_REASONS_LOG = data_path("logs", "reasons.log")
BOTTLE_DEBUG_DIR = data_path("diagnostics", "bottles")
STAR_DEBUG_DIR = data_path("diagnostics", "stars")
ITEM_DEBUG_DIR = data_path("diagnostics", "items")

# Printable desk markers and the item-teaching crops
MARKER_SHEETS = data_path("markers", "desk_markers_page")
ITEM_CROPS = data_path("items", "crops")
