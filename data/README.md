# data/

Everything the vision pipeline learns or records on one machine. **This folder is git-ignored** (except this file and `examples/`):
calibration describes one desk and one laptop, and model weights are downloaded, not committed. The location can be moved with the
`MARKOS_DATA_DIR` environment variable.

```
data/
  calibration/
    desk_calibration.json   the tape calibration, the fitted camera, the star positions, the calibration bottle's diameter
    pick_config.json        the live tuning (hover height, base-turn zero and gain, radius offset); see examples/
    aim_probe.json          raw measurements of the aim probe
    aim_fit.json            its fitted correction (applied with `aim_probe.sh --apply`)
    direction_result.json   the answers from the direction test
  bottles/
    bottle_types.json       enrolled bottle types: name, height, cap height, foot diameter, colour signatures
  stars/
    star_templates.npz      image patches of the two red stars (markos_vision.apps.select_stars)
    star_focal_samples.npz  focal-length samples from the same tool
  models/
    yolo11n-seg.pt          fetched by scripts/wsl/download_models.sh
  logs/
    bridge.log  hover.log  probe.log  reasons.log
  diagnostics/              debugging images the tools save on request
```

The tools create the folders they write to. The hover window needs the star templates and focal-length samples (the one-time
`select_stars` setup) and an enrolled bottle; without a desk calibration it falls back to a rough star-based estimate, labelled
`ROUGH STAR ESTIMATE`, and asks for one. See [docs/setup.md](../docs/setup.md) and [docs/calibration.md](../docs/calibration.md).
