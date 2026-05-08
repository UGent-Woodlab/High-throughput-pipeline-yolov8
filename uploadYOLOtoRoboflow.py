"""
Upload trained YOLO model weights to Roboflow.

Recommended project folder structure
------------------------------------
PROJECT_ROOT/
  raw_images/              original full-size images
  cropped/                 crops used for annotation or training
  trainingdata/            YOLO dataset folder containing data.yaml
  training_runs/           output folder for YOLO training runs
    ModelV1YOLOV26/
      weights/
        best.pt
        last.pt
  models/                  copied/final model weights
  segmentation_output/     output from the segmentation script
"""

import os
import sys

from roboflow import Roboflow


# =============================================================================
#                               USER SETTINGS
# =============================================================================

# -----------------------------
# Project folders and weights
# -----------------------------
PROJECT_ROOT = r"D:\Users\labo\Lverschuren\BlancaHoutskoolVaten"

# This should match RUNS_ROOT and RUN_NAME in YoloTrain_structured.py.
TRAINING_RUNS_ROOT = os.path.join(PROJECT_ROOT, "training_runs")
TRAINING_RUN_NAME = "ModelV1YOLOV26"

# Folder containing best.pt or last.pt.
WEIGHTS_FOLDER = os.path.join(TRAINING_RUNS_ROOT, TRAINING_RUN_NAME, "weights")
WEIGHTS_FILENAME = "best.pt"


# -----------------------------
# Roboflow project settings
# -----------------------------
WORKSPACE_NAME = "ugent-woodlab"
PROJECT_NAME = "charcoal-vessels-yangambi"
PROJECT_VERSION = 1

# Model type passed to Roboflow's deploy call. Keep this matching the model type
# expected by your Roboflow project/version.
MODEL_TYPE = "yolov26"


# -----------------------------
# API key settings
# -----------------------------
# Do not write your Roboflow API key directly in this script. Set it once in your
# environment instead.
#
# Windows PowerShell, current session only:
#   $env:ROBOFLOW_API_KEY="your_key_here"
#
# Windows Command Prompt, current session only:
#   set ROBOFLOW_API_KEY=your_key_here
#
# Windows permanent user environment variable:
#   setx ROBOFLOW_API_KEY "your_key_here"
ROBOFLOW_API_KEY_ENV_NAME = "ROBOFLOW_API_KEY"


# =============================================================================
#                           VALIDATION AND HELPERS
# =============================================================================

def get_roboflow_api_key():
    """Read the Roboflow API key from the environment."""
    api_key = os.environ.get(ROBOFLOW_API_KEY_ENV_NAME, "").strip()

    if not api_key:
        raise RuntimeError(
            f"Missing Roboflow API key. Set the environment variable {ROBOFLOW_API_KEY_ENV_NAME}.\n"
            "Example in Windows PowerShell for the current session:\n"
            f"  $env:{ROBOFLOW_API_KEY_ENV_NAME}=\"your_key_here\""
        )

    return api_key


def validate_settings():
    """Check common upload problems before contacting Roboflow."""
    if not isinstance(PROJECT_VERSION, int) or PROJECT_VERSION <= 0:
        raise ValueError("PROJECT_VERSION must be a positive integer, for example 1.")

    if not os.path.isdir(WEIGHTS_FOLDER):
        raise FileNotFoundError(
            "WEIGHTS_FOLDER does not exist. Check this path:\n"
            f"  {WEIGHTS_FOLDER}\n"
            "This should normally be the 'weights' folder inside your YOLO training run."
        )

    weights_path = os.path.join(WEIGHTS_FOLDER, WEIGHTS_FILENAME)
    if not os.path.isfile(weights_path):
        raise FileNotFoundError(
            "The selected weights file does not exist. Check this path:\n"
            f"  {weights_path}\n"
            "Use WEIGHTS_FILENAME='best.pt' for the best validation model, or 'last.pt' for the final epoch model."
        )

    if not WORKSPACE_NAME:
        raise ValueError("WORKSPACE_NAME cannot be empty.")
    if not PROJECT_NAME:
        raise ValueError("PROJECT_NAME cannot be empty.")
    if not MODEL_TYPE:
        raise ValueError("MODEL_TYPE cannot be empty.")


def print_upload_summary():
    """Print exactly what will be uploaded."""
    print("\nRoboflow upload configuration")
    print("=" * 40)
    print(f"Workspace:       {WORKSPACE_NAME}")
    print(f"Project:         {PROJECT_NAME}")
    print(f"Version:         {PROJECT_VERSION}")
    print(f"Model type:      {MODEL_TYPE}")
    print(f"Weights folder:  {WEIGHTS_FOLDER}")
    print(f"Weights file:    {WEIGHTS_FILENAME}")
    print(f"API key source:  environment variable {ROBOFLOW_API_KEY_ENV_NAME}")


# =============================================================================
#                                  MAIN
# =============================================================================

def main():
    try:
        validate_settings()
        api_key = get_roboflow_api_key()
        print_upload_summary()

        rf = Roboflow(api_key=api_key)
        project = rf.workspace(WORKSPACE_NAME).project(PROJECT_NAME)
        version = project.version(PROJECT_VERSION)

        upload_response = version.deploy(
            MODEL_TYPE,
            WEIGHTS_FOLDER,
            filename=WEIGHTS_FILENAME,
        )

        print("\nUpload finished.")
        print("Upload response:")
        print(upload_response)

    except Exception as e:
        print("\nUpload failed.")
        print(str(e))
        sys.exit(1)


if __name__ == "__main__":
    main()
