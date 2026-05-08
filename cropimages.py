"""
Crop large microscopy images into YOLO-sized training images.

What this script does
---------------------
1) Reads all images from RAW_IMAGES_FOLDER.
2) Converts them to RGB so the output is consistent for YOLO training.
3) Creates fixed-size crops, normally 640 x 640 pixels.
4) Saves each crop as PNG by default.
5) Includes the crop coordinates in the output filename.
6) If an image is smaller than the crop size, it can be placed on a black
   640 x 640 canvas so every saved crop has the same size.
7) Optionally copies one reproducibly random crop per original image into a
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
        with Image.open(image_path) as img:
            # Convert to RGB so PNG/JPG output is consistent. This is also what
            # YOLO normally expects for training images.
            if img.mode != "RGB":
                img = img.convert("RGB")

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
