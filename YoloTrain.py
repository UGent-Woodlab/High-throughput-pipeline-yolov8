"""
Train a YOLO segmentation model with documented settings.

Recommended project folder structure
------------------------------------
PROJECT_ROOT/
  raw_images/              original full-size images
  cropped/                 crops used for annotation or training
  trainingdata/            YOLO dataset folder containing data.yaml
  training_runs/           output folder for YOLO training runs
  models/                  copied/final model weights
  segmentation_output/     output from the segmentation script
"""

import os
import random
from pathlib import Path

import numpy as np
import torch
from ultralytics import YOLO, settings


# =============================================================================
#                               USER SETTINGS
# =============================================================================

# -----------------------------
# Project folders
# -----------------------------
PROJECT_ROOT = r"D:\Users\labo\Lverschuren\BlancaHoutskoolVaten"

# Dataset folder and data.yaml file. The data.yaml file tells YOLO where the
# train/val images and labels are, and which class names are present.
DATASET_FOLDER = os.path.join(PROJECT_ROOT, "TrainingdataV1YOLOV26")
DATA_YAML = os.path.join(DATASET_FOLDER, "data.yaml")

# Where all training runs should be saved. This is the important setting for
# your issue where training output was not saved in the folder you expected.
# Ultralytics uses project=RUNS_ROOT and name=RUN_NAME to build the final run
# folder: RUNS_ROOT/RUN_NAME.
RUNS_ROOT = os.path.join(PROJECT_ROOT, "training_runs")
RUN_NAME = "ModelV1"

# Optional folder for final copied/exported model weights if you later want to
# keep a clean model archive separate from the full training run folder.
MODELS_FOLDER = os.path.join(PROJECT_ROOT, "models")


# -----------------------------
# Model and training mode
# -----------------------------
# This can be either:
# - a model name that Ultralytics can find/download, such as "yolo26l-seg.pt"
# - a local path to a .pt model file
PRETRAINED_MODEL = "yolo26l-seg.pt"

# Set this to True if PRETRAINED_MODEL is a local file and you want the script to
# stop immediately when that file is missing. Leave False for official model
# names that Ultralytics may download automatically.
REQUIRE_PRETRAINED_MODEL_EXISTS = False

# Optional class filter. Use None to train all classes in data.yaml.
# Example: CLASSES_TO_TRAIN = [2]
CLASSES_TO_TRAIN = None




# -----------------------------
# Training settings
# -----------------------------
AUGMENT = True       # Whether to apply data augmentation during training. Usually recommended for better generalization. 
PATIENCE = 0         # EarlyStopping after this many epochs without improvement; 0 disables EarlyStopping.
BATCH_SIZE = 8       # Number of images per training batch. Adjust based on GPU memory; common values are 8, 16, or 32.
EPOCHS = 300         # Total number of training epochs. More epochs can improve performance but take longer; monitor validation metrics to avoid overfitting.
IMAGE_SIZE = 640     # Input image size for training. Common values are 640 or 1280; larger sizes can improve accuracy but require more GPU memory and time.
CACHE_IMAGES = True  # Whether to cache images in RAM for faster training. Set to True if you have enough RAM; otherwise, set to False to read from disk each epoch.
DEVICE = 0           # 0 = first CUDA GPU, "cpu" = CPU training
PLOTS = True         # Save training curves, validation plots, and prediction examples.

# If False and RUNS_ROOT/RUN_NAME already exists, Ultralytics may create a new
# numbered folder instead of overwriting. If True, it can reuse the same run name.
EXIST_OK = False

# If DEVICE requests a GPU but CUDA is unavailable, stop with a clear error.
# Set to True only if you want the script to fall back to CPU automatically.
ALLOW_CPU_FALLBACK = False



# -----------------------------
# Augmentation settings
# -----------------------------
# These are kept from your original script, with explanations preserved and
# slightly expanded for wood-anatomy segmentation.

HSV_H = 0.03
# Hue variation. This changes color tone. For wood microscopy, small hue shifts
# can help if staining, lighting, or camera white balance varies between images.

HSV_S = 0.7
# Saturation variation. This changes color intensity. It can improve robustness
# when images differ in contrast or staining strength, but too much may make
# anatomy look unrealistic. Keep an eye on validation examples.

HSV_V = 0.3
# Brightness/value variation. Useful when exposure or illumination varies across
# microscopy sessions.

DEGREES = 30
# Random rotation range in degrees. Wood anatomical structures usually do not
# have a single fixed orientation in the crop, so rotation is generally safe and
# useful. If orientation itself becomes meaningful for a future task, reduce this.

TRANSLATE = 0.1
# Random horizontal/vertical shift as a fraction of image size. This helps the
# model learn objects that are partially visible near crop borders.

SCALE = 0.2
# Random scaling. This simulates modest changes in magnification or object size.
# Be careful with very strong scaling if your pixel size is highly standardized.

SHEAR = 0
# Shear transformation in degrees. Left at 0 because strong shear can distort
# vessel/fiber/ray shape in ways that are less biologically realistic.

PERSPECTIVE = 0
# Perspective transformation. Left at 0 for microscopy images because the sample
# plane is normally flat and perspective distortion is usually not realistic.

FLIPUD = 0.5
# Vertical flip probability. Usually safe for wood anatomy because the crop does
# not have an absolute up/down direction for segmentation.

FLIPLR = 0.5
# Horizontal flip probability. Usually safe for the same reason as vertical flips.

MOSAIC = 1
# Mosaic combines four training images. It can improve robustness and increase
# context variation, but it may also create artificial seams. If validation masks
# look unstable at crop borders, try lowering this later.

MIXUP = 0
# MixUp blends two images and labels. Left at 0 because blended microscopy images
# may create unrealistic anatomy and ambiguous segmentation boundaries.

COPY_PASTE = 0
# Copy-paste augmentation copies objects between images. Left at 0 because pasted
# vessels/fibers may not match surrounding tissue context unless carefully tuned.





# -----------------------------
# Reproducibility settings
# -----------------------------
# A fixed seed makes data shuffling and augmentation choices more repeatable.
# Perfect bit-for-bit reproducibility can still depend on GPU, CUDA, PyTorch,
# and Ultralytics versions, but these settings make runs much easier to compare.
SEED = 12345
DETERMINISTIC = True








# =============================================================================
#                           VALIDATION AND SETUP
# =============================================================================

def require_file(path, description):
    """Raise a clear error if a required file is missing."""
    if not os.path.isfile(path):
        raise FileNotFoundError(
            f"Missing {description}. Check this path:\n  {path}"
        )


def require_folder(path, description):
    """Raise a clear error if a required folder is missing."""
    if not os.path.isdir(path):
        raise FileNotFoundError(
            f"Missing {description}. Check this path:\n  {path}"
        )


def looks_like_local_model_path(model_path):
    """
    Decide whether PRETRAINED_MODEL probably points to a local file.

    Official model names can look like filenames, for example yolo26l-seg.pt.
    Therefore, by default we only require existence when the setting above asks
    for it. This helper lets us print a useful warning for obvious local paths.
    """
    return os.path.isabs(model_path) or ("/" in model_path) or ("\\" in model_path)


def validate_settings():
    """Check common setup problems before training starts."""
    require_folder(DATASET_FOLDER, "dataset folder")
    require_file(DATA_YAML, "YOLO data.yaml file")

    if REQUIRE_PRETRAINED_MODEL_EXISTS:
        require_file(PRETRAINED_MODEL, "pretrained model file")
    elif looks_like_local_model_path(PRETRAINED_MODEL) and not os.path.isfile(PRETRAINED_MODEL):
        raise FileNotFoundError(
            "PRETRAINED_MODEL looks like a local path, but it does not exist.\n"
            f"  {PRETRAINED_MODEL}\n"
            "Use an existing .pt file, or set PRETRAINED_MODEL to an official model name."
        )

    os.makedirs(RUNS_ROOT, exist_ok=True)
    os.makedirs(MODELS_FOLDER, exist_ok=True)

    if CLASSES_TO_TRAIN is not None:
        if not isinstance(CLASSES_TO_TRAIN, (list, tuple)):
            raise ValueError("CLASSES_TO_TRAIN must be None or a list/tuple of class IDs, for example [0, 2].")
        if not all(isinstance(c, int) and c >= 0 for c in CLASSES_TO_TRAIN):
            raise ValueError("Every class ID in CLASSES_TO_TRAIN must be a non-negative integer.")

    if isinstance(DEVICE, int) or (isinstance(DEVICE, str) and DEVICE.isdigit()):
        if not torch.cuda.is_available():
            if ALLOW_CPU_FALLBACK:
                print("WARNING: CUDA is not available. Training will fall back to CPU.")
            else:
                raise RuntimeError(
                    "DEVICE requests GPU training, but PyTorch does not see CUDA.\n"
                    "Check your PyTorch/CUDA installation, set DEVICE='cpu', or set ALLOW_CPU_FALLBACK=True."
                )


def configure_ultralytics_paths():
    """
    Update Ultralytics global settings and also pass project/name to train().

    The train() arguments project=RUNS_ROOT and name=RUN_NAME are the most
    important part for controlling where the run is saved. The settings update is
    kept as an extra convenience so other Ultralytics operations use the same
    folder convention where possible.
    """
    settings.update({"datasets_dir": DATASET_FOLDER})
    settings.update({"runs_dir": RUNS_ROOT})
    settings.update({"weights_dir": MODELS_FOLDER})


def set_reproducibility():
    """Set seeds and deterministic flags for more reproducible training runs."""
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(SEED)

    if DETERMINISTIC:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def build_train_kwargs():
    """Build the keyword arguments passed to model.train()."""
    train_kwargs = dict(
        data=DATA_YAML,
        augment=AUGMENT,
        patience=PATIENCE,
        batch=BATCH_SIZE,
        epochs=EPOCHS,
        imgsz=IMAGE_SIZE,
        cache=CACHE_IMAGES,
        device=DEVICE,
        plots=PLOTS,
        project=RUNS_ROOT,
        name=RUN_NAME,
        exist_ok=EXIST_OK,
        seed=SEED,
        deterministic=DETERMINISTIC,
        hsv_h=HSV_H,
        hsv_s=HSV_S,
        hsv_v=HSV_V,
        degrees=DEGREES,
        translate=TRANSLATE,
        scale=SCALE,
        shear=SHEAR,
        perspective=PERSPECTIVE,
        flipud=FLIPUD,
        fliplr=FLIPLR,
        mosaic=MOSAIC,
        mixup=MIXUP,
        copy_paste=COPY_PASTE,
    )

    if CLASSES_TO_TRAIN is not None:
        train_kwargs["classes"] = list(CLASSES_TO_TRAIN)

    if ALLOW_CPU_FALLBACK and not torch.cuda.is_available():
        train_kwargs["device"] = "cpu"

    return train_kwargs


def print_run_summary():
    """Print the key settings before training starts."""
    expected_run_folder = os.path.join(RUNS_ROOT, RUN_NAME)

    print("\nYOLO training configuration")
    print("=" * 40)
    print(f"Project root:       {PROJECT_ROOT}")
    print(f"Dataset folder:     {DATASET_FOLDER}")
    print(f"data.yaml:          {DATA_YAML}")
    print(f"Pretrained model:   {PRETRAINED_MODEL}")
    print(f"Runs root:          {RUNS_ROOT}")
    print(f"Run name:           {RUN_NAME}")
    print(f"Expected run folder:{expected_run_folder}")
    print(f"Device:             {DEVICE}")
    print(f"Epochs:             {EPOCHS}")
    print(f"Batch size:         {BATCH_SIZE}")
    print(f"Image size:         {IMAGE_SIZE}")
    print(f"Seed:               {SEED}")
    print(f"Deterministic:      {DETERMINISTIC}")


# =============================================================================
#                                  MAIN
# =============================================================================

def main():
    os.chdir(PROJECT_ROOT)
  
    validate_settings()
    configure_ultralytics_paths()
    set_reproducibility()
    print_run_summary()

    model = YOLO(PRETRAINED_MODEL)
    results = model.train(**build_train_kwargs())

    # Ultralytics results usually expose save_dir. If the API changes, this still
    # fails safely by printing the expected folder from our settings.
    save_dir = getattr(results, "save_dir", None)
    print("\nTraining finished.")
    if save_dir is not None:
        print(f"Ultralytics reported save_dir: {save_dir}")
    print(f"Configured runs root: {RUNS_ROOT}")
    print(f"Configured run name:  {RUN_NAME}")


if __name__ == "__main__":
    main()
