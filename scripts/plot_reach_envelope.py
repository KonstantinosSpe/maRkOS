"""Draws where the arm's gripper can be, from the reach planner itself: docs/images/reach-envelope.png.

For every point in the vertical plane through the base axis (signed radius: positive when the arm leans toward the point,
negative when the base has turned around and the arm leans away; height above base_link) it asks ``markos.reach.solve_plane``
whether the arm's real joint ranges can hold the gripper there. Needs matplotlib; everything else comes from the repository.

Run: python scripts/plot_reach_envelope.py [output.png]
"""

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "ros2" / "markos"))
from markos import hw_config as hw  # noqa: E402
from markos import reach  # noqa: E402

RADII = np.arange(-450, 451, 5)
HEIGHTS = np.arange(0, 701, 5)
BOTTLE_CM, CAP_CM = 22.0, 1.8  # the enrolled bottle


def reachable(sign):
    grid = np.zeros((len(HEIGHTS), len(RADII)), bool)
    for j, r in enumerate(RADII):
        if (r > 0) != (sign > 0) and r != 0:
            continue
        for i, h in enumerate(HEIGHTS):
            grid[i, j] = reach.solve_plane(float(r), float(h)) is not None
    return grid


def main(out):
    fig, ax = plt.subplots(figsize=(8.2, 5.6), dpi=150)
    toward, away = reachable(+1), reachable(-1)
    ax.contourf(RADII, HEIGHTS, toward.astype(int), levels=[0.5, 1.5], colors=["#2f7fbf"], alpha=0.55)
    ax.contourf(RADII, HEIGHTS, away.astype(int), levels=[0.5, 1.5], colors=["#d98a2b"], alpha=0.55)
    ax.axvline(0, color="#555", lw=0.8)

    cap_mm = (BOTTLE_CM - CAP_CM / 2) * 10
    hover_mm = cap_mm + 50
    ax.plot([300, 300], [0, BOTTLE_CM * 10], color="#3a7d44", lw=5, solid_capstyle="round")
    ax.plot([300], [hover_mm], marker="v", color="#c0392b", ms=9)
    ax.annotate("bottle 22 cm at 300 mm", (300, 10), textcoords="offset points", xytext=(8, 4), fontsize=8, color="#3a7d44")
    ax.annotate("gripper hover\n5 cm above the cap", (300, hover_mm), textcoords="offset points", xytext=(10, 8),
                fontsize=8, color="#c0392b")
    ax.axhline(hw.BASE_TO_SHOULDER, color="#888", lw=0.7, ls=":")
    ax.annotate("shoulder (D axis)", (-440, hw.BASE_TO_SHOULDER), textcoords="offset points", xytext=(0, 4), fontsize=8, color="#666")

    ax.set_xlabel("signed distance from the base axis, mm  (negative: the base turns around and the arm leans away)")
    ax.set_ylabel("height above the base, mm")
    ax.set_title("Where the gripper can be: Thor with the real joint ranges", fontsize=11)
    ax.set_xlim(-450, 450)
    ax.set_ylim(0, 700)
    ax.set_aspect("equal")
    ax.grid(alpha=0.25)
    handles = [plt.Rectangle((0, 0), 1, 1, color="#2f7fbf", alpha=0.55), plt.Rectangle((0, 0), 1, 1, color="#d98a2b", alpha=0.55)]
    ax.legend(handles, ["arm leans toward the point", "base turned around, arm leans away"], loc="upper left", fontsize=8)
    fig.tight_layout()
    fig.savefig(out)
    print("wrote", out)


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else REPO / "docs" / "images" / "reach-envelope.png")
