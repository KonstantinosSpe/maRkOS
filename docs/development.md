# Development

## Environment

```bash
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev,headless]"
pre-commit install          # optional: runs ruff before each commit (pip install pre-commit)
```

`pip install -e .` makes two import packages available from the sources: `markos` (from `ros2/markos/markos`, the same code colcon builds as a
ROS package) and `markos_vision` (from `vision/markos_vision`). The desktop app in `teleop/` is a folder of flat modules run as a script.

## Conventions

* **No absolute paths in code.** Everything a machine learns or records goes under one data directory and is reached through
  `markos_vision.config`; the shell scripts find the repository from their own location.
* **Calibration data is not source.** `data/` is git-ignored except for its README and examples. Model weights are downloaded, not committed.
* **Measured constants live in one place** (`ros2/markos/markos/hw_config.py`) with how they were measured, next to the number.
* **Lint with `ruff`** (`ruff check .`, configured in `pyproject.toml`). Shell scripts pass `shellcheck`.
* **Tests for behaviour, with known truth.** A geometric change comes with a simulation that checks the answer, not just that the code runs.
* **Line endings** are normalised by `.gitattributes`: LF everywhere except the Windows scripts (`.bat`, `.ps1`), which stay CRLF.

## Git

* Work on a branch, keep commits small and single-purpose, and write them as *type(scope): what*: `feat`, `fix`, `refactor`, `test`,
  `docs`, `style`, `chore`, `build`, `ci`. The body says *why*, not a list of files.
* `master` stays working: tests pass and the launchers start.

## Adding a bottle type

Press `a` with the bottle outlined in the finder or the hover window and answer the prompts (name, real height, cap height, foot
diameter). Enrol the same name a few times while turning the bottle so the label does not have to face the camera. Types are stored in
`data/bottles/bottle_types.json`. Calibrate the desk with one bottle and enrol others with their diameter.

## Adding a launcher

Put the logic in `scripts/wsl/<name>.sh` starting with `source "$(dirname "${BASH_SOURCE[0]}")/env.sh"`, and a thin `scripts/windows/<Name>.bat`
that calls `wsl -d %THOR_DISTRO% --cd "%THOR_REPO%" -- bash scripts/wsl/<name>.sh` after `call "%~dp0_common.bat"`.
