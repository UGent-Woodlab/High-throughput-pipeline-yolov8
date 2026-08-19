"""
Crop large microscopy images into YOLO-sized training images.

What this script does
---------------------
1) Reads all images from RAW_IMAGES_FOLDER.
2) Stretches the dynamic range of non-8-bit images (16-bit or float microscopy
   data) over the FULL image before cropping, so the anatomy is clearly visible
   in every crop and all crops of one image share the same intensity scale.
3) Converts them to RGB so the output is consistent for YOLO training.
4) Creates fixed-size crops, normally 640 x 640 pixels.
5) Saves each crop as PNG by default.
6) Includes the crop coordinates in the output filename.
7) If an image is smaller than the crop size, it can be placed on a black
   640 x 640 canvas so every saved crop has the same size.
8) Optionally copies one reproducibly random crop per original image into a
   selection folder for quick visual inspection or annotation sampling.

Recommended project folder structure
------------------------------------
PROJECT_ROOT/
  raw_images/              original full-size images
  cropped/                 output crops from this script
    selection/             one selected crop per source image
  trainingdata/            YOLO dataset exported from Roboflow or prepared locally
  training_runs/           YOLO training runs
  models/                  copied/final model weights
  segmentation_output/     output from the segmentation script

"""

import os
import random
import shutil
from pathlib import Path

import numpy as np
from PIL import Image


# =============================================================================
#                               USER SETTINGS
# =============================================================================

# -----------------------------
# Project folders
# -----------------------------
# Main project folder. The other paths are built from this where possible so the
# scripts follow one consistent folder convention.
PROJECT_ROOT = r"D:\Users\labo\Lverschuren\BlancaHoutskoolVaten"

# Folder containing the full-size input images.
RAW_IMAGES_FOLDER = os.path.join(PROJECT_ROOT, "x140 Raw images batch 2")

# Folder where all crop images will be saved.
CROPS_OUTPUT_FOLDER = os.path.join(PROJECT_ROOT, "croppedtest")

# Subfolder inside CROPS_OUTPUT_FOLDER where one crop per input image is copied.
SELECTION_SUBFOLDER = "selection"




# -----------------------------
# Crop settings
# -----------------------------
# YOLO models are commonly trained at 640 x 640 pixels. Keeping training crops
# the same size as the training image size avoids extra resizing surprises.
CROP_SIZE = (640, 640)  # (width, height), in pixels

# Default crop format. PNG is lossless and avoids JPG compression artifacts in
# microscopy images. Use "jpg" only if file size is more important than exact
# pixel values.
OUTPUT_FORMAT = "png"  # "png", "jpg", or "tif"

# JPEG quality is only used when OUTPUT_FORMAT = "jpg".
JPEG_QUALITY = 95

# If True, images smaller than CROP_SIZE are centered on a black canvas. This
# ensures every output crop has exactly CROP_SIZE, which is convenient for YOLO.
PAD_SMALL_IMAGES = True
PADDING_COLOR = (0, 0, 0)  # black RGB padding

# If an image is between 640 and 1279 pixels in each direction, one centered crop is made.
USE_SINGLE_CENTER_CROP_WHEN_ONLY_ONE_TILE = True

# Add one reproducibly random crop per original image to a selection folder.
SAVE_RANDOM_SELECTION_PER_IMAGE = True
RANDOM_SEED = 12345

# If False, the script stops before overwriting an existing crop file. If True,
# rerunning the script replaces crops with the same filename.
OVERWRITE_EXISTING_FILES = True

# Supported input image types. Matching is case-insensitive.
IMAGE_EXTENSIONS = (".jpg", ".jpeg", ".png", ".gif", ".tif", ".tiff", ".bmp")


# -----------------------------
# Dynamic range stretching
# -----------------------------
# Microscopy images are often 16-bit or float, and then normally use only a
# small part of their theoretical value range. Converting such an image to 8-bit
# without rescaling gives dark, low-contrast crops in which the anatomy is hard
# to see and hard to annotate.
#
# When STRETCH_DYNAMIC_RANGE is True, every non-8-bit image is contrast
# stretched first and cropped afterwards. The stretch is always computed on the
# FULL image, never per crop, so every crop of one image stays on exactly the
# same intensity scale and identical structures keep an identical brightness.
#
# The stretch maps the STRETCH_PERCENTILE_LOW..STRETCH_PERCENTILE_HIGH
# percentiles of the image to 0..255 and clips everything outside that window.
# Using percentiles instead of the true min/max makes the stretch robust against
# a few extreme outlier pixels, such as dust, scratches, or hot pixels.
#
# Keep these settings identical to the ones in YoloAntomicalSeg.py, so the
# images the model is trained on look like the images it must segment later.
STRETCH_DYNAMIC_RANGE = True

# Lower and upper percentile of the image histogram that become 0 and 255.
# The default 0.5 and 99.5 stretch the middle 99% of the dynamic range, which
# removes only the most extreme outlier pixels and keeps the image close to the
# original tonality.
# Use a narrower window, for example 2.5 and 97.5, for clearly more contrast:
# that maps the middle 95% to 0..255 and clips the darkest and brightest 2.5%
# of the pixels to pure black and pure white.
STRETCH_PERCENTILE_LOW = 0.5
STRETCH_PERCENTILE_HIGH = 99.5

# Also stretch images that are already 8-bit. Off by default, because normal
# 8-bit images usually already use their full range, and stretching them would
# change the appearance of existing training data.
STRETCH_ALSO_8BIT_IMAGES = False

# True  = one shared low/high value for all channels, which keeps the color
#         balance of the original image.
# False = stretch every channel separately. This gives more contrast, but it can
#         shift colors when the channels have different intensity ranges.
STRETCH_CHANNELS_JOINTLY = True

# Print the value range that was mapped to 0..255 for each image.
PRINT_STRETCH_RANGE = True

# Scaling used for non-8-bit images when STRETCH_DYNAMIC_RANGE = False:
# - "minmax":      map the real min/max of the image to 0..255
# - "fixed_16bit": map 0..65535 to 0..255, for true 16-bit data
FALLBACK_NORMALIZATION_MODE = "minmax"

# PIL image modes that already contain 8 bits per channel. Every other mode
# ("I", "I;16", "F", ...) is treated as non-8-bit input.
EIGHT_BIT_PIL_MODES = {
    "1", "L", "LA", "P", "PA", "RGB", "RGBA", "RGBX", "CMYK", "YCbCr", "LAB", "HSV",
}


# =============================================================================
#                           VALIDATION AND HELPERS
# =============================================================================

def validate_settings():
    """Check common setup problems before starting the full crop run."""
    if not os.path.isdir(RAW_IMAGES_FOLDER):
        raise FileNotFoundError(
            "RAW_IMAGES_FOLDER does not exist. Check this path:\n"
            f"  {RAW_IMAGES_FOLDER}"
        )

    crop_w, crop_h = CROP_SIZE
    if crop_w <= 0 or crop_h <= 0:
        raise ValueError("CROP_SIZE must contain positive width and height values.")

    fmt = OUTPUT_FORMAT.lower().strip().lstrip(".")
    if fmt not in {"png", "jpg", "jpeg", "tif", "tiff"}:
        raise ValueError("OUTPUT_FORMAT must be 'png', 'jpg', or 'tif'.")

    if not isinstance(PADDING_COLOR, tuple) or len(PADDING_COLOR) != 3:
        raise ValueError("PADDING_COLOR must be an RGB tuple, for example (0, 0, 0).")

    if not 0.0 <= STRETCH_PERCENTILE_LOW < STRETCH_PERCENTILE_HIGH <= 100.0:
        raise ValueError(
            "Dynamic range percentiles must satisfy "
            "0 <= STRETCH_PERCENTILE_LOW < STRETCH_PERCENTILE_HIGH <= 100. Got: "
            f"{STRETCH_PERCENTILE_LOW} and {STRETCH_PERCENTILE_HIGH}."
        )

    if FALLBACK_NORMALIZATION_MODE not in {"minmax", "fixed_16bit"}:
        raise ValueError("FALLBACK_NORMALIZATION_MODE must be 'minmax' or 'fixed_16bit'.")


def ensure_dirs():
    """Create output folders used by this script."""
    os.makedirs(CROPS_OUTPUT_FOLDER, exist_ok=True)

    if SAVE_RANDOM_SELECTION_PER_IMAGE:
        os.makedirs(os.path.join(CROPS_OUTPUT_FOLDER, SELECTION_SUBFOLDER), exist_ok=True)


def normalized_output_extension():
    """Return the filename extension corresponding to OUTPUT_FORMAT."""
    fmt = OUTPUT_FORMAT.lower().strip().lstrip(".")
    if fmt == "jpeg":
        fmt = "jpg"
    if fmt == "tiff":
        fmt = "tif"
    return fmt


def list_image_files(folder):
    """Return image filenames sorted by name for reproducible processing order."""
    return sorted(
        f for f in os.listdir(folder)
        if f.lower().endswith(IMAGE_EXTENSIONS)
    )


def safe_stem(filename):
    """
    Return the filename without extension.

    The stem is used as the first part of each crop filename. Spaces are kept so
    you can still recognize the original image, but path separators are not
    possible because we use only os.path.basename output from os.listdir().
    """
    return Path(filename).stem


# =============================================================================
#                    DYNAMIC RANGE STRETCHING AND IMAGE LOADING
# =============================================================================

def stretch_to_uint8(arr, label=""):
    """
    Rescale one image array to uint8 using a dynamic range stretch.

    The array can be 2D (one channel) or 3D with the channels last. All values
    in the array are used together to find the low and high value, so pass the
    full image here, not a single crop.

    Values at or below the low percentile become 0, values at or above the high
    percentile become 255, and everything in between is scaled linearly.
    """
    arr_float = np.asarray(arr, dtype=np.float32)
    finite = np.isfinite(arr_float)

    # Completely empty or all-NaN input: return a black image instead of failing.
    if not np.any(finite):
        return np.zeros(arr_float.shape, dtype=np.uint8)

    values = arr_float[finite]

    if STRETCH_DYNAMIC_RANGE:
        lo, hi = np.percentile(values, [STRETCH_PERCENTILE_LOW, STRETCH_PERCENTILE_HIGH])
        lo, hi = float(lo), float(hi)
        range_label = f"percentile {STRETCH_PERCENTILE_LOW}-{STRETCH_PERCENTILE_HIGH}"

        # Low-contrast images can give an empty percentile window. Fall back to
        # the real min/max so such an image is still visible instead of flat.
        if hi <= lo:
            lo, hi = float(np.min(values)), float(np.max(values))
            range_label += " (empty window, using min-max)"

    elif FALLBACK_NORMALIZATION_MODE == "fixed_16bit":
        lo, hi = 0.0, 65535.0
        range_label = "fixed 16-bit"

    else:
        lo, hi = float(np.min(values)), float(np.max(values))
        range_label = "min-max"

    # A single-value image has no range to stretch.
    if hi <= lo:
        return np.zeros(arr_float.shape, dtype=np.uint8)

    if PRINT_STRETCH_RANGE:
        print(f"  dynamic range{label}: {range_label}, {lo:.6g} .. {hi:.6g} -> 0 .. 255")

    scaled = np.clip((arr_float - lo) / (hi - lo), 0.0, 1.0)

    # NaN/inf pixels cannot be scaled. Set them to 0 so the uint8 cast stays
    # defined instead of producing undefined values.
    scaled = np.where(finite, scaled, 0.0)

    return (scaled * 255.0).round().astype(np.uint8)


def stretch_channels_to_uint8(arr_hwc):
    """Stretch an (H, W, C) array to uint8, jointly or one channel at a time."""
    if STRETCH_CHANNELS_JOINTLY:
        return stretch_to_uint8(arr_hwc)

    channels = [
        stretch_to_uint8(arr_hwc[..., c], label=f" (channel {c})")
        for c in range(arr_hwc.shape[-1])
    ]
    return np.stack(channels, axis=-1)


def array_to_rgb_uint8(arr):
    """
    Convert a raw image array to an (H, W, 3) uint8 RGB array.

    Handled layouts:
    - 2D grayscale (H, W)
    - channel-last (H, W, C)
    - channel-first (C, H, W), which is common in scientific TIFF files
    - arrays with extra leading dimensions, for example (Z, H, W): the first
      plane is used

    The dynamic range stretch is applied here, on the full image.
    """
    arr = np.squeeze(np.asarray(arr))

    # Multi-page / Z-stack data: keep taking the first plane until 2D or 3D.
    while arr.ndim > 3:
        arr = np.squeeze(arr[0])

    if arr.ndim == 2:
        gray = stretch_to_uint8(arr)
        return np.stack([gray, gray, gray], axis=-1)

    if arr.ndim != 3:
        raise ValueError(f"Unsupported image shape after squeezing: {arr.shape}")

    if arr.shape[0] in (1, 2, 3, 4) and arr.shape[-1] not in (1, 2, 3, 4):
        arr = np.moveaxis(arr, 0, -1)

    # One or two channels: use the first channel and replicate it to RGB.
    if arr.shape[-1] < 3:
        gray = stretch_to_uint8(arr[..., 0])
        return np.stack([gray, gray, gray], axis=-1)

    # Three or more channels: use the first three as RGB. Reorder or select
    # channels here if your data is BGR or has a different channel meaning.
    return stretch_channels_to_uint8(arr[..., :3])


def load_image_as_rgb(image_path):
    """
    Open one input image and return it as a PIL RGB image.

    Already 8-bit images keep the plain PIL conversion, so existing behavior for
    JPG/PNG/8-bit TIFF does not change. Non-8-bit images go through numpy, where
    the dynamic range of the full image is stretched before any crop is taken.
    """
    img = Image.open(image_path)

    if img.mode in EIGHT_BIT_PIL_MODES:
        rgb = img if img.mode == "RGB" else img.convert("RGB")
        if not (STRETCH_DYNAMIC_RANGE and STRETCH_ALSO_8BIT_IMAGES):
            return rgb
        return Image.fromarray(stretch_channels_to_uint8(np.asarray(rgb)))

    return Image.fromarray(array_to_rgb_uint8(np.asarray(img)))


def crop_filename(original_filename, left, upper, right, lower, padded=False):
    """
    Build a crop filename that contains the crop coordinates in the original image.

    Coordinates are source-image coordinates:
    - x = column coordinate
    - y = row coordinate
    - x2/y2 = exclusive right/bottom crop limits, matching PIL crop boxes

    Example:
      sample__x000000_y000640_x2000640_y2001280.png

    If padded=True, the crop image may contain black padding outside these source
    coordinates, but the filename still records the real part taken from the
    original image.
    """
    ext = normalized_output_extension()
    pad_tag = "__padded" if padded else ""
    return (
        f"{safe_stem(original_filename)}"
        f"__x{left:06d}_y{upper:06d}_x2{right:06d}_y2{lower:06d}"
        f"{pad_tag}.{ext}"
    )


def save_crop_image(crop_img, out_path):
    """Save one crop using the selected output format."""
    ext = normalized_output_extension()

    if ext == "jpg":
        crop_img.save(out_path, quality=JPEG_QUALITY)
    else:
        crop_img.save(out_path)


def make_fixed_size_crop(img, left, upper, right, lower):
    """
    Crop a source region and return an image of exactly CROP_SIZE.

    For normal full-size crops, this simply returns img.crop(...). For small
    images, or any region smaller than CROP_SIZE, the crop is centered on a black
    canvas when PAD_SMALL_IMAGES=True.
    """
    crop_w, crop_h = CROP_SIZE
    source_crop = img.crop((left, upper, right, lower))

    if source_crop.size == CROP_SIZE:
        return source_crop, False

    if not PAD_SMALL_IMAGES:
        return source_crop, False

    canvas = Image.new("RGB", CROP_SIZE, PADDING_COLOR)
    paste_x = (crop_w - source_crop.size[0]) // 2
    paste_y = (crop_h - source_crop.size[1]) // 2
    canvas.paste(source_crop, (paste_x, paste_y))
    return canvas, True


def create_crop_boxes(width, height):
    """
    Create crop boxes for one image.

    This preserves the behavior of the original script:
    - very small images produce one centered crop, now padded to 640 x 640
    - images with exactly one full 640 x 640 tile in each direction produce one
      centered crop
    - larger images are split into non-overlapping 640 x 640 tiles

    Returned boxes use PIL format: (left, upper, right, lower), where right and
    lower are exclusive coordinates.
    """
    crop_w, crop_h = CROP_SIZE
    num_x = width // crop_w
    num_y = height // crop_h

    # Small images, or one-tile images, get one centered crop.
    if (
        num_x == 0
        or num_y == 0
        or (USE_SINGLE_CENTER_CROP_WHEN_ONLY_ONE_TILE and num_x == 1 and num_y == 1)
    ):
        actual_w = min(crop_w, width)
        actual_h = min(crop_h, height)
        left = (width - actual_w) // 2
        upper = (height - actual_h) // 2
        return [(left, upper, left + actual_w, upper + actual_h)]

    # Larger images are cropped into a regular non-overlapping grid.
    boxes = []
    for ix in range(num_x):
        for iy in range(num_y):
            left = ix * crop_w
            upper = iy * crop_h
            boxes.append((left, upper, left + crop_w, upper + crop_h))

    return boxes


# =============================================================================
#                                  MAIN LOGIC
# =============================================================================

def process_one_image(image_filename, rng):
    """
    Crop one image and return the list of crop paths saved for that image."""
    image_path = os.path.join(RAW_IMAGES_FOLDER, image_filename)
    saved_crop_paths = []

    try:
        # Load as RGB. Non-8-bit images are dynamic range stretched over the
        # full image here, so every crop below uses the same intensity scale.
        img = load_image_as_rgb(image_path)

        width, height = img.size
        crop_boxes = create_crop_boxes(width, height)

        for left, upper, right, lower in crop_boxes:
            crop_img, padded = make_fixed_size_crop(img, left, upper, right, lower)
            out_name = crop_filename(image_filename, left, upper, right, lower, padded=padded)
            out_path = os.path.join(CROPS_OUTPUT_FOLDER, out_name)

            if os.path.exists(out_path) and not OVERWRITE_EXISTING_FILES:
                raise FileExistsError(
                    "Output crop already exists and OVERWRITE_EXISTING_FILES=False:\n"
                    f"  {out_path}"
                )

            save_crop_image(crop_img, out_path)
            saved_crop_paths.append(out_path)

    except Exception as e:
        print(f"WARNING: Could not process {image_filename}: {e}")
        return []

    # Copy one crop to the selection folder, using the seeded random generator so
    # the same source image gets the same selected crop each time the script runs.
    if SAVE_RANDOM_SELECTION_PER_IMAGE and saved_crop_paths:
        chosen = rng.choice(saved_crop_paths)
        selection_folder = os.path.join(CROPS_OUTPUT_FOLDER, SELECTION_SUBFOLDER)
        dest_path = os.path.join(selection_folder, os.path.basename(chosen))
        shutil.copy2(chosen, dest_path)

    return saved_crop_paths


def main():
    validate_settings()
    ensure_dirs()

    rng = random.Random(RANDOM_SEED)
    image_files = list_image_files(RAW_IMAGES_FOLDER)

    print("\nCrop images configuration")
    print("=" * 40)
    print(f"Input folder:  {RAW_IMAGES_FOLDER}")
    print(f"Output folder: {CROPS_OUTPUT_FOLDER}")
    print(f"Crop size:     {CROP_SIZE[0]} x {CROP_SIZE[1]} px")
    print(f"Output format: {normalized_output_extension()}")
    print(f"Padding:       {PAD_SMALL_IMAGES} with color {PADDING_COLOR}")
    if STRETCH_DYNAMIC_RANGE:
        print(
            f"Range stretch: {STRETCH_PERCENTILE_LOW}% - {STRETCH_PERCENTILE_HIGH}% "
            f"of the full image (8-bit images included: {STRETCH_ALSO_8BIT_IMAGES})"
        )
    else:
        print(f"Range stretch: off, non-8-bit scaling = {FALLBACK_NORMALIZATION_MODE}")
    print(f"Random seed:   {RANDOM_SEED}")
    print(f"Images found:  {len(image_files)}")

    if not image_files:
        print("No supported image files found. Nothing to crop.")
        return

    total_crops = 0
    images_with_no_crops = 0

    for index, image_filename in enumerate(image_files, start=1):
        print(f"\nProcessing {image_filename} ({index}/{len(image_files)})")
        saved = process_one_image(image_filename, rng)
        total_crops += len(saved)
        if not saved:
            images_with_no_crops += 1
        print(f"  saved crops: {len(saved)}")

    print("\nDone.")
    print(f"Total crops saved: {total_crops}")
    print(f"Images with no crops saved: {images_with_no_crops}")


if __name__ == "__main__":
    main()
