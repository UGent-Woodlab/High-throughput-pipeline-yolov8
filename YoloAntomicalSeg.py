"""

What this code can do per image:
1) Segment multiple anatomical features with YOLO segmentation:
   - vessels, rays, fibers, parenchyma 
2) Merge tiled results into full-resolution binary masks (0 or 255)
3) Optionally resolve overlaps between masks using a priority order
4) Optionally compute a "cell wall" mask:
      cellwall = clip(255 - fibers - vessels - rays - parenchyma, 0..255)
   (parenchyma is subtracted only if it is available)
5) Optionally split the full feature masks into individual objects for measuring.
   The measurement stage uses YOLO instance borders:
   - filled masks are accumulated from all enabled YOLO passes
   - border masks are accumulated from the largest available field of view
     (large-FOV borders if large_fov=True, otherwise standard-FOV borders)
   - borders are used to split merged masks into object cores
   - border pixels are assigned back to the nearest object before measuring
   - optional hole filling can be applied to the reconstructed final objects
6) Optionally cluster measured vessels into vessel groups.
   - vessels are connected into the same group when their closest mask pixels
     are within a configurable distance threshold
   - grouping is transitive: if A is close to B and B is close to C, then
     A, B, and C are all in one group even if A and C are not directly close
7) Export:
   - masks to logically named folders (same base filename as input)
   - optional overlay images
   - measurement visuals per feature
   - optional vessel-group color images
   - one combined CSV + separate CSV per feature

OME-TIFF support:
- If enabled and you open .ome.tif/.ome.tiff, it can read via pyometiff.
- You can select which “page/channel” to use (see OME_READ_INDEX below).





Conda environment setup: 
# optional
conda update -n base -c defaults conda

conda create -n AIAnatomyEnv python=3.10 -y
conda activate AIAnatomyEnv

# GPU pytorch
conda install -y -c pytorch -c nvidia pytorch torchvision pytorch-cuda=11.8

# scientific + imaging
conda install -y -c conda-forge numpy pandas scipy scikit-image matplotlib opencv pillow

# jupyter kernel (optional)
conda install -y -c conda-forge ipykernel jupyterlab

# pip-only
python -m pip install pyometiff ultralytics

python -m ipykernel install --user --name AIAnatomyEnv --display-name "AIAnatomyEnv"

"""

import os
import sys
import math
import cv2
import numpy as np
import pandas as pd
import torch
from PIL import Image
from ultralytics import YOLO

# OME-TIFF reader (optional, controlled by USE_OME_TIFF_READER)
from pyometiff import OMETIFFReader

# Measurement stage imports (only used if RUN_MEASUREMENT_STAGE=True)
from scipy import ndimage as ndi
from skimage import measure, morphology
from skimage.segmentation import find_boundaries
from skimage.io import imsave
from matplotlib import colormaps


# =============================================================================
#                               USER SETTINGS
# =============================================================================

# -----------------------------
# Input / output
# -----------------------------
INPUT_FOLDER = r"D:\Users\labo\Lverschuren\William\Raw images"
OUTPUT_ROOT  = r"D:\Users\labo\Lverschuren\William\Segmented"
MODEL_WEIGHTS = r"D:\Users\labo\Lverschuren\William\ModelYOLOv11\weights\best.pt"




# -----------------------------
# Tiling for YOLO segmentation
# -----------------------------
SUB_IMAGE_SIZE   = 640 # size of the moving window for YOLO segmentation; adjust based on your model, 640 default
OVERLAP_PERCENT  = 0.5 # fraction of tile size to overlap 
IOU_NMS          = 0.5 # IOU threshold for NMS within each tile; # Higher = allow more overlapping detections to remain. # Lower = suppress overlapping detections more aggressively. 
PRINT_EVERY_TILE = 25 # how often to print progress during tiling

# Save optional overlay images (can be large)
SAVE_OVERLAY_IMAGES = True

# Resolve overlaps between features (recommended). Higher priority wins; lower priority loses pixels where higher exists
RESOLVE_OVERLAPS = True
FEATURE_PRIORITY = ["fibers", "vessels", "rays", "parenchyma"]

# Optional cell wall / fiber-wall mask output (the leftover area after subtracting all segmented features). This can be useful for some analyses, but it is not a direct model output and may contain errors.
OUTPUT_CELLWALL_MASK = False

# Feature configuration 
# - enabled: segment it or not
# - class_id: YOLO class index corresponding to this feature
# - conf: confidence threshold
# - large_fov: do a second pass with 2x tile size for context
FEATURES = {
    "vessels": {
        "enabled": True,
        "class_id": 0,
        "conf": 0.2,
        "large_fov": False,
    },
    "rays": {
        "enabled": False,
        "class_id": 1,
        "conf": 0.2,
        "large_fov": False,
    },
    "fibers": {
        "enabled": False,
        "class_id": 0,
        "conf": 0.2,
        "large_fov": False,
    },
    # Future-proof extra class
    "parenchyma": {
        "enabled": False,
        "class_id": 3,
        "conf": 0.2,
        "large_fov": False,
    },
}





# -----------------------------
# Measurement: measure the segmented features (optional)
# -----------------------------
RUN_MEASUREMENT_STAGE = True

# Pixel size: width of one pixel (meters)
PIXEL_SIZE_M = 2.25e-6

# Which features to measure using YOLO-border instance reconstruction?
MEASURE_FEATURES = {
    "vessels": True,
    "rays": False, 
    "fibers": False,
    "parenchyma": False,
}

# The current YOLO-border measurement method uses min_size_px to remove tiny
# fragments before object labeling/measurement.
DEFAULT_MEAS_PARAMS = {"min_size_px": 30}
MEAS_PARAMS = {
    "vessels":     {"min_size_px": 30},
    "rays":        {"min_size_px": 30},
    "fibers":      {"min_size_px": 30},
    "parenchyma":  {"min_size_px": 30},
}

# Save the accumulated full-image border masks used for object measurement.
SAVE_INSTANCE_BORDER_MASKS = True

# Save an RGB image where each measured object has a random color.
SAVE_MEASURED_OBJECT_IMAGES = True

# Remove perfectly horizontal or vertical border-mask sections longer than this
# many pixels before using the border mask for object segmentation.
REMOVE_LONG_STRAIGHT_BORDER_SECTIONS = True
LONG_STRAIGHT_BORDER_SECTION_MAX_PX = 200

# Fill holes inside final reconstructed objects before measuring them.
FILL_HOLES_BEFORE_MEASUREMENT = {
    "vessels": True,
    "rays": False,
    "fibers": True,
    "parenchyma": True,
}





# -----------------------------
# Vessel grouping: Vessels are assigned to groups (optional)
# -----------------------------
CALCULATE_VESSEL_GROUPS = True

# Vessels are assigned to the same group when the closest points of their final measured masks are at most this many micrometers apart.
# This value should be the 95th percentile of the minimum edge-to-edge distance between distinct vessels in your dataset, plus a small margin of 1 or 2 pixels for segmentation error. 
VESSEL_GROUP_DISTANCE_UM = 25.0     # in micrometers

# Save an RGB image where each vessel group has a random color.
SAVE_VESSEL_GROUP_IMAGES = True






# -----------------------------
# OME-TIFF handling (optional)
# -----------------------------
USE_OME_TIFF_READER = False

# Many OME-TIFF readers return arrays with extra dimensions (Z/C/T).
# This option tells the loader which index to select if multiple pages exist.
# Typical choices: 0 for first page/channel.
OME_READ_INDEX = 0

# Pillow safety limit: if you only process trusted images, you can set to None.
# (If you process untrusted images, do NOT disable this protection.)
PIL_MAX_IMAGE_PIXELS = None
Image.MAX_IMAGE_PIXELS = PIL_MAX_IMAGE_PIXELS








# =============================================================================
#                          OUTPUT FOLDER STRUCTURE
# =============================================================================
# OUTPUT_ROOT/
#   masks/<feature>/<originalname>.<ext>
#   masks/cellwallmask/<originalname>.<ext>             (optional)
#   borders/<feature>/<originalname>.<ext>              (optional, yolo_borders method)
#   overlays/<originalname>.<ext>                        (optional)
#   measurements/
#       all_features_properties.csv                      (combined)
#       vessels_properties.csv                           (per feature)
#       rays_properties.csv
#       fibers_properties.csv
#       parenchyma_properties.csv
#       <feature>_segmented/<originalname>__<feature>_segmented.tif
#       vessel_groups/<originalname>__vessel_groups.tif  (optional)


def ensure_dirs():
    """Create only the output directories that can actually be used."""
    os.makedirs(OUTPUT_ROOT, exist_ok=True)
    os.makedirs(os.path.join(OUTPUT_ROOT, "masks"), exist_ok=True)

    if SAVE_OVERLAY_IMAGES:
        os.makedirs(os.path.join(OUTPUT_ROOT, "overlays"), exist_ok=True)

    if RUN_MEASUREMENT_STAGE:
        os.makedirs(os.path.join(OUTPUT_ROOT, "measurements"), exist_ok=True)

    if RUN_MEASUREMENT_STAGE and SAVE_INSTANCE_BORDER_MASKS:
        os.makedirs(os.path.join(OUTPUT_ROOT, "borders"), exist_ok=True)

    # Only create feature-specific folders for enabled segmentation features.
    for feat, cfg in FEATURES.items():
        if not cfg.get("enabled", False):
            continue

        os.makedirs(os.path.join(OUTPUT_ROOT, "masks", feat), exist_ok=True)

        if RUN_MEASUREMENT_STAGE and SAVE_INSTANCE_BORDER_MASKS and MEASURE_FEATURES.get(feat, False):
            os.makedirs(os.path.join(OUTPUT_ROOT, "borders", feat), exist_ok=True)

        # Only create measurement folders for features that are both:
        # 1) segmented
        # 2) selected for measurement
        if RUN_MEASUREMENT_STAGE and SAVE_MEASURED_OBJECT_IMAGES and MEASURE_FEATURES.get(feat, False):
            os.makedirs(
                os.path.join(OUTPUT_ROOT, "measurements", f"{feat}_segmented"),
                exist_ok=True
            )

    # Vessel-group images are saved separately from the measured-object
    # visuals because they encode group identity, not object identity.
    if RUN_MEASUREMENT_STAGE and CALCULATE_VESSEL_GROUPS and SAVE_VESSEL_GROUP_IMAGES:
        os.makedirs(os.path.join(OUTPUT_ROOT, "measurements", "vessel_groups"), exist_ok=True)

    # Only create cellwall folder if cellwall output is enabled.
    if OUTPUT_CELLWALL_MASK:
        os.makedirs(os.path.join(OUTPUT_ROOT, "masks", "cellwallmask"), exist_ok=True)


# =============================================================================
#                             IMAGE LOADING
# =============================================================================

# -----------------------------
# Image conversion / normalization
# -----------------------------
# YOLO models normally expect 8-bit RGB images. However, microscopy / special
# TIFF data can be grayscale, uint16, float, channel-first, multi-page, etc.
# The old script passed PIL crops more directly to YOLO. To avoid silently
# damaging special data types with img.convert("RGB"), this script now loads the
# image as an array first and converts to RGB uint8 in a controlled way.

# Normalization method for non-uint8 images:
# - "percentile": robust contrast stretch; usually best when exposure varies
# - "minmax": stretch actual min/max of each image
# - "fixed_16bit": map 0..65535 to 0..255; useful for true 16-bit images
NORMALIZATION_MODE = "percentile"
PERCENTILE_LOW = 0.5
PERCENTILE_HIGH = 99.5

# Optional inversion. Leave False unless your model was trained on inverted data.
INVERT_IMAGE = False

# Optional channel selection for multi-channel images.
# None = use first 3 channels if available, or replicate a single channel.
# 0, 1, 2, ... = use that one channel and replicate it to RGB.
INPUT_CHANNEL = None


def normalize_to_uint8(arr: np.ndarray) -> np.ndarray:
    """
    Convert an image array to uint8 safely.

    Why this is needed:
    - astype(np.uint8) is unsafe for uint16 / float microscopy images because it
      can wrap, clip, or destroy contrast.
    - PIL.convert("RGB") can also hide important datatype/channel changes.

    This function keeps uint8 unchanged, and deliberately scales other data types.
    """
    arr = np.asarray(arr)

    if arr.dtype == np.uint8:
        out = arr.copy()
    else:
        arr_float = arr.astype(np.float32)
        finite = np.isfinite(arr_float)

        if not np.any(finite):
            out = np.zeros(arr_float.shape, dtype=np.uint8)
        else:
            if NORMALIZATION_MODE == "fixed_16bit":
                # Best when input is true 16-bit data with meaningful 0..65535 range.
                out_float = np.clip(arr_float / 65535.0, 0, 1)

            elif NORMALIZATION_MODE == "minmax":
                lo = float(np.min(arr_float[finite]))
                hi = float(np.max(arr_float[finite]))
                if hi <= lo:
                    out_float = np.zeros(arr_float.shape, dtype=np.float32)
                else:
                    out_float = np.clip((arr_float - lo) / (hi - lo), 0, 1)

            elif NORMALIZATION_MODE == "percentile":
                # Robust to outlier bright/dark pixels.
                lo, hi = np.percentile(arr_float[finite], [PERCENTILE_LOW, PERCENTILE_HIGH])
                if hi <= lo:
                    lo = float(np.min(arr_float[finite]))
                    hi = float(np.max(arr_float[finite]))

                if hi <= lo:
                    out_float = np.zeros(arr_float.shape, dtype=np.float32)
                else:
                    out_float = np.clip((arr_float - lo) / (hi - lo), 0, 1)

            else:
                raise ValueError(
                    f"Unknown NORMALIZATION_MODE: {NORMALIZATION_MODE}. "
                    "Use 'percentile', 'minmax', or 'fixed_16bit'."
                )

            out = (out_float * 255).round().astype(np.uint8)

    if INVERT_IMAGE:
        out = 255 - out

    return out


def array_to_rgb_uint8(arr: np.ndarray) -> np.ndarray:
    """
    Convert special image arrays to RGB uint8 for YOLO.

    Supported common cases:
    - 2D grayscale: (H, W)
    - HWC color/multichannel: (H, W, C)
    - CHW color/multichannel: (C, H, W)
    - arrays with extra OME-style dimensions after indexing

    Output is always: (H, W, 3), dtype uint8, RGB order.
    """
    arr = np.asarray(arr)

    # Remove singleton dimensions, e.g. (1, H, W) or (H, W, 1).
    arr = np.squeeze(arr)

    # OME-TIFF readers can return many dimensions (T/Z/C/Y/X etc.).
    # Without full axis metadata handling, the safest general fallback is to use
    # OME_READ_INDEX repeatedly until the data looks like 2D or 3D image data.
    # If your OME files have a known layout, customize this selection here.
    while arr.ndim > 3:
        arr = arr[OME_READ_INDEX]
        arr = np.squeeze(arr)

    # 2D grayscale: normalize once and replicate to RGB.
    if arr.ndim == 2:
        arr8 = normalize_to_uint8(arr)
        return np.stack([arr8, arr8, arr8], axis=-1)

    if arr.ndim != 3:
        raise ValueError(f"Unsupported image shape after squeezing/indexing: {arr.shape}")

    # Detect channel-first arrays: (C, H, W).
    # This is common for OME or scientific arrays.
    if arr.shape[0] in (1, 2, 3, 4) and arr.shape[-1] not in (1, 2, 3, 4):
        arr = np.moveaxis(arr, 0, -1)

    # Now expect channel-last: (H, W, C).
    if arr.shape[-1] == 1:
        arr8 = normalize_to_uint8(arr[..., 0])
        return np.stack([arr8, arr8, arr8], axis=-1)

    if arr.shape[-1] >= 2:
        if INPUT_CHANNEL is not None:
            # Use one explicitly selected channel and replicate it to RGB.
            if INPUT_CHANNEL < 0 or INPUT_CHANNEL >= arr.shape[-1]:
                raise ValueError(f"INPUT_CHANNEL={INPUT_CHANNEL} outside available channels: {arr.shape[-1]}")
            arr8 = normalize_to_uint8(arr[..., INPUT_CHANNEL])
            return np.stack([arr8, arr8, arr8], axis=-1)

        if arr.shape[-1] >= 3:
            # Use the first three channels as RGB. If your data is BGR or has a
            # different channel meaning, reorder/select channels here.
            return normalize_to_uint8(arr[..., :3])

        # Two-channel fallback: use channel 0 as grayscale.
        arr8 = normalize_to_uint8(arr[..., 0])
        return np.stack([arr8, arr8, arr8], axis=-1)

    raise ValueError(f"Unsupported channel layout: {arr.shape}")


def load_image_as_rgb_uint8(path: str) -> np.ndarray:
    """
    Load an input image and return an RGB uint8 numpy array.

    - If USE_OME_TIFF_READER is True and the file looks like OME-TIFF, try pyometiff.
    - Otherwise fall back to Pillow.
    - Special data types are converted deliberately using normalize_to_uint8().

    OME handling note:
    - OMETIFFReader.read() can return arrays with extra dimensions.
      We attempt to select OME_READ_INDEX along the first axis if needed.
      If you know the exact axis order of your OME files, customize the indexing
      in array_to_rgb_uint8().
    """
    lower = path.lower()
    is_ome = lower.endswith(".ome.tif") or lower.endswith(".ome.tiff")

    if USE_OME_TIFF_READER and is_ome:
        try:
            reader = OMETIFFReader(path)
            arr, metadata, xml_metadata = reader.read()
            return array_to_rgb_uint8(arr)

        except Exception as e:
            print(f"  WARNING: OME-TIFF read failed for {os.path.basename(path)}; falling back to Pillow. ({e})")

    # Default: Pillow. We do NOT use img.convert("RGB") here, because that can
    # silently alter special TIFF / microscopy data. Instead, convert explicitly.
    img = Image.open(path)
    arr = np.asarray(img)
    return array_to_rgb_uint8(arr)


def crop_rgb_uint8(img_rgb: np.ndarray, box):
    """
    Crop an RGB uint8 numpy image.

    box = (x1, y1, x2, y2)
    This replaces PIL crop + convert("RGB") so every tile sent to YOLO comes
    from the same controlled preprocessing path.
    """
    x1, y1, x2, y2 = box
    return img_rgb[y1:y2, x1:x2, :]


# =============================================================================
#                           TILING HELPERS
# =============================================================================

def compute_tile_starts(length: int, tile: int, stride: int):
    """
    Create tile start positions that fully cover the image dimension.

    Important edge cases:
    - If length < tile, we still return [0] and crop at (0,0,tile,tile).
    - We always include the last tile aligned to the far edge.
    """
    if length <= tile:
        return [0]

    starts = list(range(0, length - tile + 1, stride))
    last = length - tile
    if starts[-1] != last:
        starts.append(last)

    # Remove duplicates while preserving order
    out = []
    seen = set()
    for s in starts:
        if s not in seen:
            out.append(s)
            seen.add(s)
    return out


def make_valid_region_mask(tile_size: int, overlap_percent: float,
                           x: int, y: int, img_w: int, img_h: int):
    """
    To reduce duplicate detections between overlapping tiles, we only “trust”
    detections in the central region of each tile. If the tile lies on the image
    border, we extend validity to that border region.

    This matches your intended behavior from earlier versions.
    """
    margin = math.ceil(tile_size * overlap_percent / 2)

    valid = np.zeros((tile_size, tile_size), dtype=bool)
    ymin, ymax = margin, tile_size - margin
    xmin, xmax = margin, tile_size - margin
    valid[ymin:ymax, xmin:xmax] = True

    # If the tile touches a global border, allow border detections too
    if y == 0:
        valid[:margin, :] = True
    if y + tile_size >= img_h:
        valid[-margin:, :] = True
    if x == 0:
        valid[:, :margin] = True
    if x + tile_size >= img_w:
        valid[:, -margin:] = True

    return valid


def binary_mask_to_border(mask_bool: np.ndarray) -> np.ndarray:
    """
    Convert one binary YOLO instance mask into a one-pixel inner border mask.

    Why this helper exists:
    - During the initial YOLO segmentation, the model predicts individual object
      masks inside each tile.
    - For measurement, we keep a second mask containing only the one-pixel
      border of each YOLO-predicted instance.
    - These border pixels are accumulated over the full image using the same
      overlap-margin validity logic as the filled masks.

    Later, the accumulated full-image border map is used to split the merged
    full mask back into individual objects for measurement.

    Returns
    -------
    np.ndarray
        Binary uint8 border mask with values {0,255}.
    """
    mask_bool = np.asarray(mask_bool, dtype=bool)

    if not np.any(mask_bool):
        return np.zeros(mask_bool.shape, dtype=np.uint8)

    # Always use a one-pixel inner boundary. Keeping this fixed avoids another
    # tuning parameter and preserves the original YOLO instance shape as closely
    # as possible.
    border = find_boundaries(mask_bool, mode="inner")

    return (border.astype(np.uint8) * 255)


# =============================================================================
#                           YOLO MASKING
# =============================================================================

def yolo_predict_class_mask_and_border(yolo_model: YOLO,
                                       tile_rgb_np: np.ndarray,
                                       class_id: int,
                                       conf: float,
                                       iou: float):
    """
    Run YOLOv8 segmentation on one RGB tile and return two outputs for the
    requested class_id:

    1) a filled binary mask of all selected instances in that tile
    2) a binary mask containing only the border pixels of those same instances

    Why keep both outputs?
    - The filled mask is still the main segmentation output.
    - The border-only mask preserves instance separation information from YOLO.
      When accumulated across all tiles, it can later be used to split the full
      merged mask into individual objects without using a distance-transform split.

    Returns
    -------
    (tile_fill_mask, tile_border_mask)
        Both are uint8 arrays with values {0,255} and the same height/width as
        the tile.
    """

    tile_rgb_np = np.ascontiguousarray(tile_rgb_np, dtype=np.uint8)
    tile_pil = Image.fromarray(tile_rgb_np, mode="RGB")

    try:
        results = yolo_model.predict(
            tile_pil,
            save=False,
            save_txt=False,
            save_conf=False,
            conf=conf,
            iou=iou,
            verbose=False,
            retina_masks=True,
            max_det=1000
        )

        # No results or no masks => empty outputs
        if not results or len(results) == 0 or results[0].masks is None:
            empty = np.zeros(tile_rgb_np.shape[:2], dtype=np.uint8)
            return empty, empty.copy()

        masks = results[0].masks.data           # torch: (n, h, w)
        boxes = results[0].boxes.data           # torch: (n, 6) [x1,y1,x2,y2,conf,cls]
        clss  = boxes[:, 5].to(torch.int64)

        idxs = (clss == int(class_id)).nonzero(as_tuple=True)[0]
        if idxs.numel() == 0:
            empty = np.zeros(tile_rgb_np.shape[:2], dtype=np.uint8)
            return empty, empty.copy()

        selected = masks[idxs]                  # (k, h, w)

        # Filled tile mask = logical OR of all selected instances.
        combined_fill = selected.any(dim=0)
        out_fill = (combined_fill.to(torch.uint8) * 255).cpu().numpy().astype(np.uint8)

        # Border tile mask = logical OR of the borders of each selected instance.
        combined_border = np.zeros(tile_rgb_np.shape[:2], dtype=bool)
        for inst_mask_t in selected:
            inst_mask = inst_mask_t.detach().cpu().numpy() > 0
            if not np.any(inst_mask):
                continue
            inst_border = binary_mask_to_border(inst_mask) > 0
            combined_border |= inst_border

        out_border = (combined_border.astype(np.uint8) * 255)

        return out_fill, out_border.astype(np.uint8)

    except Exception as e:
        print("YOLO prediction error:", repr(e))
        empty = np.zeros(tile_rgb_np.shape[:2], dtype=np.uint8)
        return empty, empty.copy()




def apply_tile_outputs_to_full(full_mask: np.ndarray,
                               full_border_mask: np.ndarray,
                               assigned_pixels: np.ndarray,
                               tile_mask: np.ndarray,
                               tile_border: np.ndarray,
                               x: int, y: int,
                               tile_size: int,
                               img_w: int, img_h: int,
                               overlap_percent: float):
    """
    Merge one tile into the full masks with strict tile ownership.

    Each full-image pixel is written by only one tile. The filled mask and the
    border mask use the same write region, so object-support pixels and
    object-border pixels come from the same YOLO view of that image location.
    """
    roi_h = min(tile_size, img_h - y)
    roi_w = min(tile_size, img_w - x)

    tile_mask = tile_mask[:roi_h, :roi_w]
    tile_border = tile_border[:roi_h, :roi_w]
    valid = make_valid_region_mask(tile_size, overlap_percent, x, y, img_w, img_h)
    valid = valid[:roi_h, :roi_w]

    assigned_roi = assigned_pixels[y:y + roi_h, x:x + roi_w]
    write = valid & (~assigned_roi)

    mask_roi = full_mask[y:y + roi_h, x:x + roi_w]
    border_roi = full_border_mask[y:y + roi_h, x:x + roi_w]

    mask_roi[write] = tile_mask[write]
    border_roi[write] = tile_border[write]
    assigned_roi[write] = True

    full_mask[y:y + roi_h, x:x + roi_w] = mask_roi
    full_border_mask[y:y + roi_h, x:x + roi_w] = border_roi
    assigned_pixels[y:y + roi_h, x:x + roi_w] = assigned_roi





def segment_feature_full_image(img_rgb: np.ndarray,
                               yolo_model: YOLO,
                               feature_name: str,
                               class_id: int,
                               conf: float,
                               large_fov: bool):
    """
    Segment one feature across the full image by tiling.

    This function returns two full-image masks:

    1) full_mask
       The normal filled binary segmentation mask.

    2) full_border_mask
       The object-border mask used later to split the merged full mask into
       individual objects for measurement.

    Tile ownership policy
    ---------------------
    Each full-image pixel is written by only one tile within the selected field
    of view. The filled mask and border mask are written from the same tile for
    each pixel. This prevents overlapping tiles from both contributing slightly
    different YOLO edges to the final masks.

    Field-of-view policy
    --------------------
    If large_fov=True, the final filled mask and final border mask both come
    from the large-FOV pass. If large_fov=False, both come from the standard-FOV
    pass. This keeps the filled mask and measurement border mask synchronized.

    Returns
    -------
    (full_mask, full_border_mask)
        Both are uint8 arrays with values {0,255}.
    """
    img_h, img_w = img_rgb.shape[:2]

    def run_tiled_pass(tile_size: int, pass_label: str):
        pass_mask = np.zeros((img_h, img_w), dtype=np.uint8)
        pass_border_mask = np.zeros((img_h, img_w), dtype=np.uint8)
        assigned_pixels = np.zeros((img_h, img_w), dtype=bool)

        overlap_px = int(tile_size * OVERLAP_PERCENT)
        stride = max(1, tile_size - overlap_px)

        xs = compute_tile_starts(img_w, tile_size, stride)
        ys = compute_tile_starts(img_h, tile_size, stride)

        total_tiles = len(xs) * len(ys)
        done = 0

        for yy in ys:
            for xx in xs:
                box = (xx, yy, xx + tile_size, yy + tile_size)
                tile_rgb = crop_rgb_uint8(img_rgb, box)

                tile_mask, tile_border = yolo_predict_class_mask_and_border(
                    yolo_model=yolo_model,
                    tile_rgb_np=tile_rgb,
                    class_id=class_id,
                    conf=conf,
                    iou=IOU_NMS,
                )

                apply_tile_outputs_to_full(
                    full_mask=pass_mask,
                    full_border_mask=pass_border_mask,
                    assigned_pixels=assigned_pixels,
                    tile_mask=tile_mask,
                    tile_border=tile_border,
                    x=xx,
                    y=yy,
                    tile_size=tile_size,
                    img_w=img_w,
                    img_h=img_h,
                    overlap_percent=OVERLAP_PERCENT,
                )

                done += 1
                if done % PRINT_EVERY_TILE == 0:
                    sys.stdout.write(f"\r  {pass_label}: tiles {done}/{total_tiles}")
                    sys.stdout.flush()

        sys.stdout.write(f"\r  {pass_label}: tiles {total_tiles}/{total_tiles}\n")
        sys.stdout.flush()

        return pass_mask, pass_border_mask

    # -----------------------------
    # Pass 1: standard-FOV tiles
    # -----------------------------
    standard_mask, standard_border_mask = run_tiled_pass(
        tile_size=SUB_IMAGE_SIZE,
        pass_label=feature_name,
    )

    # -----------------------------
    # Pass 2: large-FOV tiles
    # -----------------------------
    if large_fov:
        large_mask, large_border_mask = run_tiled_pass(
            tile_size=SUB_IMAGE_SIZE * 2,
            pass_label=f"{feature_name} (large FOV)",
        )
        return large_mask, large_border_mask

    return standard_mask, standard_border_mask


# =============================================================================
#                       OVERLAP RESOLUTION
# =============================================================================

def resolve_overlaps(masks: dict, priority: list):
    """
    Higher priority keeps pixels; lower priority loses pixels where higher exists.
    """
    occupied = np.zeros_like(next(iter(masks.values())), dtype=np.uint8)

    for feat in priority:
        if feat not in masks:
            continue

        m = masks[feat].copy()
        m[occupied > 0] = 0
        masks[feat] = m

        occupied = np.maximum(occupied, m)

    return masks


# =============================================================================
#               CELL WALL / FIBER-WALL MASK
# =============================================================================

def compute_cellwall_mask(fiber_mask, vessel_mask, ray_mask, parenchyma_mask=None):
    """
    Fiber-wall mask logic:
      cellwall = clip(255 - fibers - vessels - rays - parenchyma, 0..255)

    All inputs are expected to be uint8 masks containing {0,255}.
    """
    fiber = fiber_mask.astype(np.int16)
    vessel = vessel_mask.astype(np.int16)
    ray = ray_mask.astype(np.int16)

    result = 255 - fiber - vessel - ray

    if parenchyma_mask is not None:
        result = result - parenchyma_mask.astype(np.int16)

    return np.clip(result, 0, 255).astype(np.uint8)


# =============================================================================
#               MEASUREMENT STAGE (WATERSHED + EDGE CORRECTION)
# =============================================================================

def boundary_and_edge_stats(region_mask_crop, bbox, image_shape):
    """
    Compute boundary pixels on the crop, and assess contact with full-image borders.

    Returns:
    - num_total boundary pixels
    - num_edge boundary pixels touching ANY border
    - num_inside
    - perimeter_ratio = num_edge/num_inside (inf if inside==0)
    - chosen_edge = edge with most boundary pixels ("top","bottom","left","right")
    - edge_rc = global coords of boundary pixels on chosen edge
    - non_edge_rc = global coords of boundary pixels NOT on any edge
    """
    H, W = image_shape
    minr, minc, maxr, maxc = bbox

    padded = np.pad(region_mask_crop, 1, mode="constant", constant_values=0)
    eroded = morphology.binary_erosion(padded)
    boundary_crop = (padded ^ eroded)[1:-1, 1:-1]

    coords = np.argwhere(boundary_crop)
    num_total = int(coords.shape[0])
    if num_total == 0:
        return 0, 0, 0, float("inf"), None, None, None

    global_r = coords[:, 0] + minr
    global_c = coords[:, 1] + minc
    global_rc = np.column_stack([global_r, global_c])

    top = global_r == 0
    bottom = global_r == H - 1
    left = global_c == 0
    right = global_c == W - 1

    any_edge = top | bottom | left | right
    num_edge = int(np.count_nonzero(any_edge))
    num_inside = num_total - num_edge
    perimeter_ratio = float("inf") if num_inside == 0 else num_edge / num_inside

    if num_edge == 0:
        return num_total, 0, num_inside, perimeter_ratio, None, None, global_rc

    counts = {
        "top": int(np.count_nonzero(top)),
        "bottom": int(np.count_nonzero(bottom)),
        "left": int(np.count_nonzero(left)),
        "right": int(np.count_nonzero(right)),
    }
    chosen_edge = max(counts, key=counts.get)

    if chosen_edge == "top":
        edge_rc = global_rc[top]
    elif chosen_edge == "bottom":
        edge_rc = global_rc[bottom]
    elif chosen_edge == "left":
        edge_rc = global_rc[left]
    else:
        edge_rc = global_rc[right]

    non_edge_rc = global_rc[~any_edge]
    if non_edge_rc.size == 0:
        non_edge_rc = None

    return num_total, num_edge, num_inside, perimeter_ratio, chosen_edge, edge_rc, non_edge_rc


def remove_long_straight_border_sections(border_mask_255: np.ndarray) -> np.ndarray:
    """Remove long perfectly horizontal or vertical runs from a border mask."""
    if not REMOVE_LONG_STRAIGHT_BORDER_SECTIONS:
        return border_mask_255

    max_len = int(LONG_STRAIGHT_BORDER_SECTION_MAX_PX)
    if max_len < 1:
        return border_mask_255

    border = np.asarray(border_mask_255) > 0
    out = border.copy()

    # Horizontal runs.
    for r in range(border.shape[0]):
        row = border[r, :]
        padded = np.r_[False, row, False]
        changes = np.flatnonzero(padded[1:] != padded[:-1])
        for start, end in zip(changes[::2], changes[1::2]):
            if end - start > max_len:
                out[r, start:end] = False

    # Vertical runs.
    for c in range(border.shape[1]):
        col = border[:, c]
        padded = np.r_[False, col, False]
        changes = np.flatnonzero(padded[1:] != padded[:-1])
        for start, end in zip(changes[::2], changes[1::2]):
            if end - start > max_len:
                out[start:end, c] = False

    return (out.astype(np.uint8) * 255)


def label_instances_from_yolo_borders(binary_mask_255: np.ndarray,
                                     border_mask_255: np.ndarray):
    """
    Reconstruct full-image individual object labels from:
    1) the merged filled full-image mask
    2) the accumulated full-image border mask built from the original YOLO
       instance masks

    Core idea
    ---------
    If we remove the border pixels from the filled mask, the remaining pixels
    become object "cores". Connected-component labeling of those cores gives one
    marker per object in many cases. We then assign the removed border pixels
    back to the nearest labeled core so that the final labels again cover the
    original full mask.

    Important measurement detail
    ----------------------------
    The labels are not measured on the border-removed cores. The inner border
    pixels are restored before measurement, so the measured area is not reduced
    by the one-pixel border subtraction. This is not an outward dilation beyond
    the original YOLO mask; it simply gives the removed mask pixels back to the
    nearest object.

    The object separation information comes directly from YOLO's own per-instance masks.

    Returns
    -------
    labels : np.ndarray of int32
        Labeled instance image with 0 as background.
    """
    mask = (binary_mask_255 > 0)
    border_mask_255 = remove_long_straight_border_sections(border_mask_255)
    border = (border_mask_255 > 0) & mask

    if not np.any(mask):
        return np.zeros(mask.shape, dtype=np.int32)

    # Remove borders to expose one or more object cores.
    core_mask = mask & (~border)

    # If borders consumed everything (can happen for very small objects or very
    # thick borders), fall back to simple connected components on the full mask.
    if not np.any(core_mask):
        return measure.label(mask, connectivity=2).astype(np.int32)

    # Connected components on the border-removed mask give the initial object
    # cores. These cores act as the seeds/identity for each object.
    labels = measure.label(core_mask, connectivity=2).astype(np.int32)

    # Assign border pixels back to the nearest labeled core. This reconstructs a
    # full object label image that again covers the original mask before any
    # size measurements or vessel-group distances are calculated.
    unlabeled_inside_mask = mask & (labels == 0)
    if np.any(unlabeled_inside_mask) and np.any(labels > 0):
        # labels == 0 is True for unlabeled pixels. Distance transform then
        # points each unlabeled pixel to the nearest labeled pixel (a zero in the
        # boolean array below).
        _, indices = ndi.distance_transform_edt(labels == 0, return_indices=True)
        nearest_r = indices[0][unlabeled_inside_mask]
        nearest_c = indices[1][unlabeled_inside_mask]
        nearest_labels = labels[nearest_r, nearest_c]
        labels[unlabeled_inside_mask] = nearest_labels

    # Keep labels only inside the original mask.
    labels[~mask] = 0
    return labels.astype(np.int32)


def fill_holes_per_instance_label(labels: np.ndarray) -> np.ndarray:
    """
    Fill internal background holes inside each reconstructed object label.

    Why fill per label instead of filling the whole binary feature mask?
    ------------------------------------------------------------------
    Filling the whole vessel mask at once can be dangerous when several vessels
    form a ring-like cluster: the empty space inside the cluster could be filled
    even though it is not part of any vessel. By filling holes separately inside
    each individual label, only holes that are enclosed by one object are added
    to that same object.

    Existing non-background labels are never overwritten. If a rare case occurs
    where one labeled object lies inside another object's hole, the inner object
    keeps its own label.
    """
    labels = np.asarray(labels)

    if labels.max() == 0:
        return labels.astype(np.int32, copy=True)

    filled_labels = labels.astype(np.int32, copy=True)

    for region in measure.regionprops(labels):
        lab = int(region.label)
        minr, minc, maxr, maxc = region.bbox

        crop_labels = labels[minr:maxr, minc:maxc]
        object_crop = (crop_labels == lab)
        filled_object_crop = ndi.binary_fill_holes(object_crop)

        # Only add pixels that are currently background. This avoids overwriting
        # another object label in unusual nested-object cases.
        out_crop = filled_labels[minr:maxr, minc:maxc]
        pixels_to_add = filled_object_crop & (out_crop == 0)
        out_crop[pixels_to_add] = lab
        filled_labels[minr:maxr, minc:maxc] = out_crop

    return filled_labels.astype(np.int32)


def should_fill_holes_for_feature(feature_name: str) -> bool:
    """Return whether reconstructed objects of this feature should be hole-filled."""
    return bool(FILL_HOLES_BEFORE_MEASUREMENT.get(feature_name, False))



def micrometers_to_pixels(distance_um: float) -> float:
    """
    Convert a physical distance in micrometers to pixels.

    The script currently uses one scalar pixel size (PIXEL_SIZE_M), so this
    assumes square pixels. For the default PIXEL_SIZE_M = 2.25e-6, a 10 um
    grouping distance corresponds to about 4.44 pixels.
    """
    if distance_um < 0:
        raise ValueError("VESSEL_GROUP_DISTANCE_UM must be zero or positive.")

    return (float(distance_um) * 1e-6) / float(PIXEL_SIZE_M)


def calculate_vessel_group_assignments(labels: np.ndarray,
                                        measured_instance_labels: list,
                                        max_distance_um: float):
    """
    Assign measured vessel instances to proximity-based vessel groups.

    Group definition
    ----------------
    Each measured vessel instance is a node in a graph. Two vessels receive an
    edge when their closest mask pixels are no farther apart than
    max_distance_um. Vessel groups are then the connected components of this
    graph. This makes grouping transitive: a large chain/cluster of vessels is
    one group even when two vessels at opposite ends are not directly close.

    Distance calculation
    --------------------
    The distance is measured on the final labeled vessel mask in pixel units and
    converted from micrometers using PIXEL_SIZE_M. For each vessel, only a local
    bounding box expanded by the threshold is inspected, which avoids building a
    large all-vs-all distance matrix and keeps the method practical for large
    images or large vessel groups.

    Parameters
    ----------
    labels : np.ndarray
        Labeled instance image with 0 as background.
    measured_instance_labels : list
        Original label values that survived the measurement filters.
    max_distance_um : float
        Maximum closest-point distance, in micrometers, for two vessels to be
        directly linked into the same group.

    Returns
    -------
    group_by_instance_label : dict
        Maps the original label value in `labels` to a compact 1-based vessel
        group id. Only this group number is exported to the measurement CSV.
    """
    measured_labels = sorted({int(v) for v in measured_instance_labels if int(v) > 0})
    max_distance_px = micrometers_to_pixels(max_distance_um)

    if len(measured_labels) == 0:
        return {}

    # Disjoint-set / union-find structure for connected components of the
    # vessel-proximity graph. This is what makes the grouping transitive.
    parent = {lab: lab for lab in measured_labels}

    def find(lab):
        while parent[lab] != lab:
            parent[lab] = parent[parent[lab]]
            lab = parent[lab]
        return lab

    def union(a, b):
        root_a = find(a)
        root_b = find(b)
        if root_a == root_b:
            return
        # Deterministic root choice: smaller original instance label wins.
        if root_a < root_b:
            parent[root_b] = root_a
        else:
            parent[root_a] = root_b

    measured_set = set(measured_labels)
    props_by_label = {int(region.label): region.bbox for region in measure.regionprops(labels)}
    # Add one pixel because edge/point distance is approximated from pixel-center
    # distances by subtracting one pixel below. The expanded crop must therefore
    # include candidate vessels whose centers are just outside max_distance_px.
    search_radius_px = int(math.ceil(max_distance_px + 1.0))
    H, W = labels.shape

    for lab in measured_labels:
        bbox = props_by_label.get(lab)
        if bbox is None:
            continue

        minr, minc, maxr, maxc = bbox
        r0 = max(0, minr - search_radius_px)
        c0 = max(0, minc - search_radius_px)
        r1 = min(H, maxr + search_radius_px)
        c1 = min(W, maxc + search_radius_px)

        crop_labels = labels[r0:r1, c0:c1]
        object_crop = (crop_labels == lab)
        if not np.any(object_crop):
            continue

        # Distance to the current vessel mask, in pixels. The distance transform
        # returns pixel-center distances. For grouping, we want the distance
        # between the closest mask edges/points, so we subtract approximately one
        # pixel and clip at zero. This makes, for example, a 4-pixel empty gap
        # count as about 4 px instead of 5 px between pixel centers.
        center_distance_to_object_px = ndi.distance_transform_edt(~object_crop)
        distance_to_object_px = np.maximum(0.0, center_distance_to_object_px - 1.0)
        nearby_label_values = np.unique(crop_labels[distance_to_object_px <= max_distance_px])

        for other in nearby_label_values:
            other = int(other)
            if other == lab or other == 0 or other not in measured_set:
                continue
            union(lab, other)

    # Convert connected components to compact 1-based group numbers. The order
    # is deterministic: groups are sorted by the smallest original instance label
    # they contain, which usually follows top-to-bottom/left-to-right labeling.
    root_to_members = {}
    for lab in measured_labels:
        root = find(lab)
        root_to_members.setdefault(root, []).append(lab)

    ordered_roots = sorted(root_to_members, key=lambda root: min(root_to_members[root]))
    group_by_instance_label = {}

    for group_id, root in enumerate(ordered_roots, start=1):
        members = sorted(root_to_members[root])
        for lab in members:
            group_by_instance_label[lab] = group_id

    return group_by_instance_label


def make_vessel_group_rgb(labels: np.ndarray,
                          group_by_instance_label: dict):
    """
    Create an RGB visualization in which each vessel group has a random color.

    Background and unmeasured vessels are black. This is intentional: the image
    matches the measurement table, where only vessels that passed the measurement
    filters receive a vessel_group value.
    """
    out = np.zeros((*labels.shape, 3), dtype=np.uint8)

    if not group_by_instance_label:
        return out

    rng = np.random.default_rng()
    group_ids = sorted(set(group_by_instance_label.values()))

    # Avoid very dark colors so small groups remain visible on a black background.
    group_color = {
        group_id: rng.integers(40, 256, size=3, dtype=np.uint8)
        for group_id in group_ids
    }

    for instance_label, group_id in group_by_instance_label.items():
        out[labels == int(instance_label)] = group_color[int(group_id)]

    return out


def make_measured_objects_rgb(labels: np.ndarray,
                              measured_instance_labels: list):
    """Create an RGB visualization in which each measured object has a random color."""
    out = np.zeros((*labels.shape, 3), dtype=np.uint8)

    if not measured_instance_labels:
        return out

    rng = np.random.default_rng()

    for instance_label in measured_instance_labels:
        color = rng.integers(40, 256, size=3, dtype=np.uint8)
        out[labels == int(instance_label)] = color

    return out


def measure_from_instance_labels(labels: np.ndarray,
                                 feature_name: str,
                                 image_id: str,
                                 filename: str):
    """
    Measure individual objects from a precomputed labeled-instance image.

    This helper contains the measurement logic shared by different instance
    segmentation approaches. In the current script it is mainly used by the
    "yolo_borders" method after the label image has been reconstructed from the
    accumulated border map.

    Hole filling, when enabled for a feature, happens here after YOLO-border
    reconstruction and before area, diameter, centroid, and vessel-group
    distances are calculated. This means vessels are measured from the final
    border-restored and hole-filled label mask.

    For vessels, this function can also calculate vessel groups after the normal
    measurement filters have been applied. Grouping after filtering keeps the
    vessel_group column and the vessel-group image synchronized with the objects
    that are actually exported in the measurements.

    Returns
    -------
    (rgb, rows, vessel_group_rgb)
        rgb is the existing measurement visualization. rows is the
        list of measurement dictionaries. vessel_group_rgb is an RGB group image
        for vessels when grouping is enabled; otherwise it is None.
    """
    p = MEAS_PARAMS.get(feature_name, DEFAULT_MEAS_PARAMS)
    min_size_px = int(p["min_size_px"])

    if labels.max() == 0:
        empty_rgb = np.zeros((*labels.shape, 3), dtype=np.uint8)
        return empty_rgb, [], None

    if should_fill_holes_for_feature(feature_name):
        labels = fill_holes_per_instance_label(labels)

    props = measure.regionprops(labels)
    H, W = labels.shape
    area_map = np.zeros_like(labels, dtype=np.float32)

    image_width_m = W * PIXEL_SIZE_M
    image_height_m = H * PIXEL_SIZE_M

    rows = []
    measured_instance_labels = []
    row_instance_labels = []
    label_counter = 1

    for region in props:
        region_crop = region.image
        bbox = region.bbox

        (num_total, num_edge, num_inside, perimeter_ratio,
         chosen_edge, edge_rc, non_edge_rc) = boundary_and_edge_stats(
            region_crop, bbox, (H, W)
        )

        corrected_diameter_px = None

        if num_edge > 0 and edge_rc is not None and non_edge_rc is not None:
            if chosen_edge in ("top", "bottom"):
                chord_px = int(edge_rc[:, 1].max() - edge_rc[:, 1].min())
                edge_row = 0 if chosen_edge == "top" else (H - 1)
                sagitta_px = int(np.max(np.abs(non_edge_rc[:, 0] - edge_row)))
                if chord_px > 0 and sagitta_px > 0:
                    corrected_diameter_px = (chord_px ** 2) / (4 * sagitta_px) + sagitta_px
            else:
                chord_px = int(edge_rc[:, 0].max() - edge_rc[:, 0].min())
                edge_col = 0 if chosen_edge == "left" else (W - 1)
                sagitta_px = int(np.max(np.abs(non_edge_rc[:, 1] - edge_col)))
                if chord_px > 0 and sagitta_px > 0:
                    corrected_diameter_px = (chord_px ** 2) / (4 * sagitta_px) + sagitta_px

        keep = (num_edge == 0 or perimeter_ratio < (2 / np.pi)) and (region.area >= min_size_px)
        if not keep:
            continue

        if corrected_diameter_px is not None:
            diameter_px = float(corrected_diameter_px)
            area_px2 = float(np.pi * (diameter_px / 2) ** 2)
        else:
            diameter_px = float(region.equivalent_diameter)
            area_px2 = float(region.area)

        d4_px4 = float(diameter_px ** 4)
        diameter_m = diameter_px * PIXEL_SIZE_M
        area_m2 = area_px2 * (PIXEL_SIZE_M ** 2)
        d4_m4 = d4_px4 * (PIXEL_SIZE_M ** 4)

        cy_px, cx_px = region.centroid
        cx_m = float(cx_px) * PIXEL_SIZE_M
        cy_m = float(cy_px) * PIXEL_SIZE_M

        minr, minc, maxr, maxc = bbox
        area_crop = area_map[minr:maxr, minc:maxc]
        area_crop[region_crop] = area_px2
        area_map[minr:maxr, minc:maxc] = area_crop

        source_instance_label = int(region.label)
        measured_instance_labels.append(source_instance_label)
        row_instance_labels.append(source_instance_label)

        rows.append({
            "image": image_id,
            "filename": filename,
            "feature": feature_name,
            "label": label_counter,
            "centroid_x_px": float(cx_px),
            "centroid_y_px": float(cy_px),
            "centroid_x_m": cx_m,
            "centroid_y_m": cy_m,
            "diameter_px": diameter_px,
            "area_px2": area_px2,
            "d4_px4": d4_px4,
            "diameter_m": diameter_m,
            "area_m2": area_m2,
            "d4_m4": d4_m4,
            "image_width_px": int(W),
            "image_height_px": int(H),
            "image_width_m": float(image_width_m),
            "image_height_m": float(image_height_m),
            "perimeter_total_px": int(num_total),
            "perimeter_edge_px": int(num_edge),
            "perimeter_ratio": float(perimeter_ratio),
            "edge_used_for_correction": chosen_edge if chosen_edge else "",
            "meas_min_size_px": min_size_px,
            "pixel_size_m": PIXEL_SIZE_M,
            "instance_method": "yolo_borders_largest_available_fov",
        })

        label_counter += 1

    vessel_group_rgb = None

    # Add only the vessel-group number to the CSV. Other grouping metadata, such
    # as the threshold or group size, remains in run_config.txt or can be inferred
    # from the vessel_group column if needed.
    if feature_name == "vessels" and CALCULATE_VESSEL_GROUPS:
        group_by_instance_label = calculate_vessel_group_assignments(
            labels=labels,
            measured_instance_labels=measured_instance_labels,
            max_distance_um=VESSEL_GROUP_DISTANCE_UM,
        )

        for row, source_label in zip(rows, row_instance_labels):
            group_id = group_by_instance_label.get(int(source_label), 0)
            row["vessel_group"] = int(group_id) if group_id > 0 else ""

        vessel_group_rgb = make_vessel_group_rgb(
            labels=labels,
            group_by_instance_label=group_by_instance_label,
        )

    else:
        for row in rows:
            row["vessel_group"] = ""

    rgb = make_measured_objects_rgb(labels, measured_instance_labels)

    return rgb, rows, vessel_group_rgb


def segment_and_measure(binary_mask_255: np.ndarray,
                        border_mask_255: np.ndarray,
                        feature_name: str,
                        image_id: str,
                        filename: str):
    """
    Measure individual objects using the border information extracted from the
    original YOLO instance masks.

    Workflow
    --------
    1) Use the full merged class mask as the object support.
    2) Use the accumulated YOLO border mask to cut the full mask into object
       cores.
    3) Label those cores.
    4) Assign removed border pixels back to the nearest labeled core.
    5) Optionally fill holes inside the final object labels.
    6) Measure the final object labels.

    This method is often a good choice when the original YOLO instance masks are
    more trustworthy than a later generic post-processing split.
    """
    if border_mask_255 is None:
        raise ValueError("border_mask_255 is required for YOLO-border measurement.")

    labels = label_instances_from_yolo_borders(binary_mask_255, border_mask_255)
    return measure_from_instance_labels(labels, feature_name, image_id, filename)



# =============================================================================
#                                SAVING
# =============================================================================

def save_mask(mask: np.ndarray, out_path: str):
    """Save a single-channel mask (0/255)."""
    cv2.imwrite(out_path, mask)


def save_overlay(img_rgb: np.ndarray, masks: dict, out_path: str):
    """
    Save a visualization overlay of masks on the original image.
    Colors (RGB):
      fibers: red
      vessels: green
      rays: blue
      parenchyma: cyan
    """
    img = img_rgb.astype(np.uint8, copy=False)
    overlay = img.copy()

    def blend(mask, rgb_color, alpha=0.3):
        m = (mask > 0).astype(np.float32)[..., None]
        color = np.array(rgb_color, dtype=np.float32)[None, None, :]
        return np.clip(overlay * (1 - alpha * m) + color * (alpha * m), 0, 255).astype(np.uint8)

    if "fibers" in masks:
        overlay[:] = blend(masks["fibers"], (255, 0, 0))
    if "vessels" in masks:
        overlay[:] = blend(masks["vessels"], (0, 255, 0))
    if "rays" in masks:
        overlay[:] = blend(masks["rays"], (0, 0, 255))
    if "parenchyma" in masks:
        overlay[:] = blend(masks["parenchyma"], (0, 255, 255))

    # cv2 expects BGR for color images
    cv2.imwrite(out_path, cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR))



#=============================================================================
#                              CONFIG SAVE
#=============================================================================

def save_run_config():
    """Save the most important run settings for reproducibility."""
    config_path = os.path.join(OUTPUT_ROOT, "run_config.txt")

    with open(config_path, "w", encoding="utf-8") as f:
        f.write("YOLO segmentation run configuration\n")
        f.write("=" * 40 + "\n\n")

        f.write(f"INPUT_FOLDER = {INPUT_FOLDER}\n")
        f.write(f"OUTPUT_ROOT = {OUTPUT_ROOT}\n")
        f.write(f"MODEL_WEIGHTS = {MODEL_WEIGHTS}\n\n")

        f.write(f"SUB_IMAGE_SIZE = {SUB_IMAGE_SIZE}\n")
        f.write(f"OVERLAP_PERCENT = {OVERLAP_PERCENT}\n")
        f.write(f"IOU_NMS = {IOU_NMS}\n")
        f.write(f"RESOLVE_OVERLAPS = {RESOLVE_OVERLAPS}\n")
        f.write(f"FEATURE_PRIORITY = {FEATURE_PRIORITY}\n\n")

        f.write("FEATURES:\n")
        for feat, cfg in FEATURES.items():
            f.write(f"  {feat}: {cfg}\n")

        f.write("\nMEASUREMENT:\n")
        f.write(f"RUN_MEASUREMENT_STAGE = {RUN_MEASUREMENT_STAGE}\n")
        f.write(f"MEASURE_FEATURES = {MEASURE_FEATURES}\n")
        f.write(f"MEAS_PARAMS = {MEAS_PARAMS}\n")
        f.write(f"PIXEL_SIZE_M = {PIXEL_SIZE_M}\n")
        f.write(f"SAVE_INSTANCE_BORDER_MASKS = {SAVE_INSTANCE_BORDER_MASKS}\n")
        f.write(f"SAVE_MEASURED_OBJECT_IMAGES = {SAVE_MEASURED_OBJECT_IMAGES}\n")
        f.write(f"REMOVE_LONG_STRAIGHT_BORDER_SECTIONS = {REMOVE_LONG_STRAIGHT_BORDER_SECTIONS}\n")
        f.write(f"LONG_STRAIGHT_BORDER_SECTION_MAX_PX = {LONG_STRAIGHT_BORDER_SECTION_MAX_PX}\n")
        f.write(f"FILL_HOLES_BEFORE_MEASUREMENT = {FILL_HOLES_BEFORE_MEASUREMENT}\n")
        f.write("INSTANCE_BORDER_POLICY = one-pixel borders restored before measurement; tile-exclusive ownership\n")

        f.write("\nVESSEL GROUPING:\n")
        f.write(f"CALCULATE_VESSEL_GROUPS = {CALCULATE_VESSEL_GROUPS}\n")
        f.write(f"VESSEL_GROUP_DISTANCE_UM = {VESSEL_GROUP_DISTANCE_UM}\n")
        f.write(f"SAVE_VESSEL_GROUP_IMAGES = {SAVE_VESSEL_GROUP_IMAGES}\n")


# =============================================================================
#                                 MAIN
# =============================================================================

def main():
    ensure_dirs()
    save_run_config()

    # Load YOLO model once (fastest way)
    model = YOLO(MODEL_WEIGHTS)
    print("\nLoaded model:")
    print(f"  path: {MODEL_WEIGHTS}")
    print(f"  classes: {model.names}")
    print("  measurement border source: largest available FOV")

    # Collect image files
    exts = (".jpg", ".png", ".jpeg", ".tif", ".tiff", ".ome.tif", ".ome.tiff")
    files = [f for f in os.listdir(INPUT_FOLDER) if f.lower().endswith(exts)]
    total = len(files)

    if total == 0:
        print("No images found in input folder.")
        return

    # Measurement rows:
    # - combined list (for all_features_properties.csv)
    # - per-feature dict (for separate feature CSVs)
    combined_rows = []
    per_feature_rows = {feat: [] for feat in FEATURES.keys()}

    for i, filename in enumerate(files, start=1):
        print(f"\nProcessing {filename} ({i}/{total})")

        in_path = os.path.join(INPUT_FOLDER, filename)
        image_id = os.path.splitext(filename)[0]

        # Load image (OME-aware if enabled)
        # This returns a controlled RGB uint8 array so YOLO sees consistent tiles
        # without blindly using PIL.convert("RGB") on special data types.
        img_rgb = load_image_as_rgb_uint8(in_path)
        img_h, img_w = img_rgb.shape[:2]

        # -----------------------------
        # 1) YOLO segmentation per feature
        # -----------------------------
        masks = {}
        instance_border_masks = {}

        for feat_name, cfg in FEATURES.items():
            if not cfg["enabled"]:
                continue

            print(f" Segmenting: {feat_name}")
            mask, border_mask = segment_feature_full_image(
                img_rgb=img_rgb,
                yolo_model=model,
                feature_name=feat_name,
                class_id=cfg["class_id"],
                conf=cfg["conf"],
                large_fov=cfg["large_fov"],
            )
            masks[feat_name] = mask
            instance_border_masks[feat_name] = border_mask

            # Save the filled mask immediately (same original filename, feature folder)
            out_mask_path = os.path.join(OUTPUT_ROOT, "masks", feat_name, filename)
            save_mask(mask, out_mask_path)

            # Optionally save the accumulated object-border mask used by the
            # yolo_borders measurement method.
            if RUN_MEASUREMENT_STAGE and SAVE_INSTANCE_BORDER_MASKS and MEASURE_FEATURES.get(feat_name, False):
                out_border_path = os.path.join(OUTPUT_ROOT, "borders", feat_name, filename)
                save_mask(border_mask, out_border_path)

        # -----------------------------
        # 2) Resolve overlaps (optional)
        # -----------------------------
        if RESOLVE_OVERLAPS and masks:
            masks = resolve_overlaps(masks, FEATURE_PRIORITY)

            # Re-save masks after overlap resolution
            for feat_name, m in masks.items():
                out_mask_path = os.path.join(OUTPUT_ROOT, "masks", feat_name, filename)
                save_mask(m, out_mask_path)

        # -----------------------------
        # 3) Cell wall / fiber-wall mask (optional)
        # -----------------------------
        if OUTPUT_CELLWALL_MASK:
            fiber = masks.get("fibers", np.zeros((img_h, img_w), dtype=np.uint8))
            vessel = masks.get("vessels", np.zeros((img_h, img_w), dtype=np.uint8))
            ray = masks.get("rays", np.zeros((img_h, img_w), dtype=np.uint8))

            # If parenchyma is segmented, subtract it too
            par = masks.get("parenchyma", None)

            cellwall = compute_cellwall_mask(fiber, vessel, ray, parenchyma_mask=par)
            out_cellwall = os.path.join(OUTPUT_ROOT, "masks", "cellwallmask", filename)
            save_mask(cellwall, out_cellwall)

        # -----------------------------
        # 4) Overlay visualization (optional)
        # -----------------------------
        if SAVE_OVERLAY_IMAGES and masks:
            out_overlay = os.path.join(OUTPUT_ROOT, "overlays", filename)
            save_overlay(img_rgb, masks, out_overlay)

        # -----------------------------
        # 5) Measurement stage (optional)
        # -----------------------------
        if RUN_MEASUREMENT_STAGE:
            for feat_name, do_measure in MEASURE_FEATURES.items():
                if not do_measure:
                    continue
                if feat_name not in masks:
                    continue

                print(f" Measuring: {feat_name}")
                seg_rgb, rows, vessel_group_rgb = segment_and_measure(
                    binary_mask_255=masks[feat_name],
                    border_mask_255=instance_border_masks.get(feat_name),
                    feature_name=feat_name,
                    image_id=image_id,
                    filename=filename,
                )

                # Add to combined and per-feature tables
                combined_rows.extend(rows)
                per_feature_rows[feat_name].extend(rows)

                # Save measurement visualization
                if SAVE_MEASURED_OBJECT_IMAGES:
                    out_seg = os.path.join(
                        OUTPUT_ROOT, "measurements", f"{feat_name}_segmented",
                        f"{image_id}__{feat_name}_segmented.tif"
                    )
                    imsave(out_seg, seg_rgb)

                # Save vessel-group visualization when available. The output is
                # an RGB image with one random color per vessel group.
                if (
                    feat_name == "vessels"
                    and SAVE_VESSEL_GROUP_IMAGES
                    and vessel_group_rgb is not None
                ):
                    out_group = os.path.join(
                        OUTPUT_ROOT, "measurements", "vessel_groups",
                        f"{image_id}__vessel_groups.tif"
                    )
                    imsave(out_group, vessel_group_rgb)

    # -----------------------------
    # CSV exports
    # -----------------------------
    if RUN_MEASUREMENT_STAGE:
        # Combined CSV
        combined_df = pd.DataFrame(combined_rows)
        combined_csv = os.path.join(OUTPUT_ROOT, "measurements", "all_features_properties.csv")
        combined_df.to_csv(combined_csv, index=False)
        print(f"\nSaved combined measurements CSV: {combined_csv}")

        # Separate CSV per feature
        for feat_name, rows in per_feature_rows.items():
            if not rows:
                continue
            df = pd.DataFrame(rows)
            out_csv = os.path.join(OUTPUT_ROOT, "measurements", f"{feat_name}_properties.csv")
            df.to_csv(out_csv, index=False)
            print(f"Saved {feat_name} measurements CSV: {out_csv}")

    print("\nDone.")


if __name__ == "__main__":
    main()
