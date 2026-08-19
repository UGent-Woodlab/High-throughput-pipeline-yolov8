# CLAUDE.md

Guidance for Claude Code (and other AI assistants) working in this repository.

## What this repository is

**YoloAnatomy** — a high-throughput quantitative wood anatomy (QWA) pipeline. It
trains Ultralytics YOLO segmentation models on wood microscopy images and applies
them to very large (gigapixel-scale) images of disc surfaces and increment cores,
exporting binary masks, per-object measurements, and vessel groups.

Published as: Van den Bulcke et al. 2025, *Plant Methods*
(https://doi.org/10.1186/s13007-025-01330-7). Licensed AGPL-3.0. Maintained by
UGent-Woodlab. Because the code accompanies a publication, **reproducibility and
readable, well-commented settings matter more than clever abstractions.**

## Repository layout

```
cropimages.py        Step 1: crop full-size images into 640x640 YOLO training tiles
YoloTrain.py         Step 2: train an Ultralytics YOLO segmentation model
YoloAntomicalSeg.py  Step 3: tiled inference, mask export, measurements, vessel groups
QWAexample.Rmd       Step 4: R example assigning features to tree rings (external tools)
environment.yml      Frozen conda environment (Windows, CUDA 11.8, python 3.10)
OLD/                 Legacy notebooks kept for reference — do not develop here
README.md            User-facing documentation; keep in sync with the scripts
Fig.JPG              Figure used by README.md
```

Note the filename typo `YoloAntomicalSeg.py` ("Antomical"). It is referenced in
the README and in user workflows — **do not rename it** unless asked.

## How these scripts are meant to be used

All three Python scripts are **standalone, no-CLI scripts**. There is no package,
no `__init__.py`, no argparse, and no shared helper module. A user opens the file,
edits the `USER SETTINGS` block at the top, and runs `python <script>.py`.

Follow that model when making changes:

- New behavior is configured by an `ALL_CAPS` module-level constant in the
  `USER SETTINGS` block, with a comment explaining *what it does and when to
  change it* — not just what it is.
- Do **not** introduce CLI arguments, config files, or a shared import module
  unless explicitly asked. Users copy single scripts between machines and
  projects, so self-containment is a feature here.
- Duplicating a helper across `cropimages.py` and `YoloAntomicalSeg.py` is
  acceptable and deliberate. When you duplicate, say so in a comment ("keep
  these settings identical to ...") so the two stay aligned.
- `PROJECT_ROOT` and the other paths are the authors' Windows paths
  (`r"D:\Users\labo\..."`). They are examples that users overwrite; leaving them
  in place is normal, and there is no need to sanitize them.
- Functions are `snake_case`, comments and docstrings are plain English and
  explanatory. Match the surrounding, comment-heavy style; this code is read by
  wood scientists, not only by software engineers.

## Key domain invariants

Breaking any of these silently corrupts scientific results, so treat them as
hard constraints:

- **Preprocessing is computed on the FULL image, never per tile or per crop.**
  Non-8-bit input (16-bit / float microscopy data) is dynamic-range stretched
  once on the whole image before it is cut into tiles or crops. Per-tile
  stretching would put neighboring tiles on different intensity scales and
  create visible seams in merged masks. See `stretch_to_uint8()` /
  `stretch_channels_to_uint8()` in both scripts.
- **`cropimages.py` and `YoloAntomicalSeg.py` must preprocess identically.**
  The `STRETCH_*` settings exist in both files and are meant to hold the same
  values, so training images look like inference images.
- **8-bit input stays untouched by default** (`STRETCH_ALSO_8BIT_IMAGES = False`),
  so results on existing 8-bit datasets do not change.
- **Never use `PIL.Image.convert("RGB")` on non-8-bit data.** It silently
  destroys 16-bit/float contrast. Load through numpy and convert explicitly.
- **Masks are single-channel `uint8` with values 0 or 255 only.** Downstream
  measurement, overlap resolution, and cell-wall arithmetic assume this.
- **Tile ownership is exclusive.** In `apply_tile_outputs_to_full()`, each
  full-image pixel is written by exactly one tile, and the filled mask and the
  border mask for that pixel come from the same tile. Do not change this to
  blending or OR-ing: instance borders would stop matching their filled masks.
- **Output masks keep the input filename**, in a per-feature folder
  (`masks/<feature>/<originalname>`), so masks can be matched to source images.
- **`PIXEL_SIZE_M` drives every physical measurement** (areas, diameters,
  vessel-group distances). Keep pixel units and metric units clearly separated
  and named (`_px` vs `_m` / `_um` suffixes, as in the existing code).
- **`run_config.txt` is the reproducibility record.** When you add a setting that
  affects results, also write it in `save_run_config()` in `YoloAntomicalSeg.py`.

## Verifying changes

There is no test suite, and no CI. The heavy dependencies (torch, ultralytics,
opencv, pyometiff) and the GPU are usually **not** available in a dev container,
and no image data or model weights are committed. So:

1. Always at least byte-compile what you touched:
   `python -m py_compile cropimages.py YoloTrain.py YoloAntomicalSeg.py`
2. `cropimages.py` only needs `numpy` + `pillow`. It can be exercised for real:
   generate synthetic 16-bit/float TIFFs in a temp folder, point
   `RAW_IMAGES_FOLDER` / `CROPS_OUTPUT_FOLDER` at temp dirs by assigning to the
   module constants after `import cropimages`, and call `main()`.
3. For `YoloAntomicalSeg.py`, the image-loading and preprocessing half can be
   tested by stubbing the heavy imports (`sys.modules["torch"] = ...` etc.)
   before importing the module, then calling `load_image_as_rgb_uint8()`.
   The YOLO inference and measurement stages cannot be verified without weights
   and a GPU — say so plainly instead of implying they were tested.
4. Cover the awkward image cases when touching preprocessing: uint16, float with
   `NaN`, single-channel, channel-first `(C, H, W)`, extra leading dimensions,
   and flat/constant images (must not crash or produce `NaN` casts).

## Gotchas

- `OVERLAP_PERCENT` is a fraction (0.5 = 50%), not a percentage.
- `IOU_NMS` is ignored from YOLOv26 onwards; the comment in the file says so.
- `FEATURES[...]["class_id"]` must match the class indices in the trained model's
  `data.yaml`. The defaults in the file are not universal.
- The cell-wall mask (`OUTPUT_CELLWALL_MASK`) is derived arithmetic, not a model
  output; it inherits every error of the masks it subtracts.
- `Image.MAX_IMAGE_PIXELS = None` disables Pillow's decompression-bomb guard.
  It is intentional for trusted gigapixel images — keep the warning comment.
- `environment.yml` is a fully pinned Windows/CUDA export. Do not hand-edit
  builds into it; it is regenerated with `conda env export`.
- README.md documents each script's feature list. When you change user-visible
  behavior or add a setting, update the matching README section in the same
  commit.
