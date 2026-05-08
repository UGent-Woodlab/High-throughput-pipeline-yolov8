<p align="center">
  <img src="./Fig.JPG" width="100%" alt="YoloAnatomy logo">
</p>

<p align="center">
  <h1 align="center">YoloAnatomy</h1>
</p>

<p align="center">
  <strong>High-throughput YOLO-based wood anatomy segmentation and quantification</strong>
</p>

<p align="center">
  <a href="https://doi.org/10.5281/zenodo.14637854">
    <img src="https://zenodo.org/badge/DOI/10.5281/zenodo.14637854.svg" alt="DOI">
  </a>
</p>

<p align="center">
  <img src="https://img.shields.io/badge/Python-3776AB.svg?style=default&logo=Python&logoColor=white" alt="Python">
  <img src="https://img.shields.io/badge/Ultralytics-YOLO-blue" alt="Ultralytics YOLO">
  <img src="https://img.shields.io/badge/NumPy-013243.svg?style=default&logo=NumPy&logoColor=white" alt="NumPy">
  <img src="https://img.shields.io/badge/OpenCV-5C3EE8.svg?style=default&logo=OpenCV&logoColor=white" alt="OpenCV">
  <img src="https://img.shields.io/badge/scikit--image-F7931E.svg?style=default&logo=scikitlearn&logoColor=white" alt="scikit-image">
</p>

---

## Authors

[Verschuren, Louis <img src="https://info.orcid.org/wp-content/uploads/2019/11/orcid_16x16.png" alt="ORCID logo">](https://orcid.org/0000-0002-3102-4588)[^aut][^cre][^UG-WL];  
[Wyffels, Francis <img src="https://info.orcid.org/wp-content/uploads/2019/11/orcid_16x16.png" alt="ORCID logo">](https://orcid.org/0000-0002-5491-8349)[^aut][^AI-RO];  
[Van den Bulcke, Jan <img src="https://info.orcid.org/wp-content/uploads/2019/11/orcid_16x16.png" alt="ORCID logo">](https://orcid.org/0000-0003-2939-5408)[^aut][^UG-WL]

[^aut]: author  
[^cre]: contact person  
[^UG-WL]: UGent-Woodlab  
[^AI-RO]: AI and Robotics Lab, IDLab-AIRO  

---

## Overview

**YoloAnatomy** is a Python-based pipeline for training YOLO segmentation models and applying them to large microscopy or gigapixel-scale wood anatomy images.

The pipeline supports:

- cropping large images into YOLO-sized training tiles;
- training an Ultralytics YOLO segmentation model;
- optionally uploading trained weights to Roboflow;
- tiled full-image segmentation using a sliding-window approach;
- exporting binary masks for anatomical features such as vessels, rays, fibers, and parenchyma;
- optional object-level measurements;
- optional vessel-group detection.

The workflow is designed for high-throughput quantitative wood anatomy, including full disc surfaces, increment cores, and other large microscopy image datasets.

---

## Workflow

### 1. Crop images

Script:

```text
cropimages.py
```

This script reads all supported images from a raw image folder and crops them into fixed-size YOLO training tiles, normally **640 × 640 px**.

Main features:

- reads common image formats, including `.jpg`, `.png`, `.tif`, `.tiff`, and `.bmp`;
- converts images to RGB for consistent YOLO training input;
- saves crops as PNG, JPG, or TIFF;
- includes crop coordinates in each output filename;
- pads images smaller than the crop size onto a black canvas;
- optionally saves one reproducibly random crop per source image into a `selection/` folder.

---

### 2. Train a YOLO segmentation model

Script:

```text
YoloTrain.py
```

This script trains an Ultralytics YOLO segmentation model using a YOLO-format dataset.

The dataset folder must contain a valid `data.yaml` file that defines the training and validation image paths and the class names.

Main features:

- configurable project folders;
- configurable pretrained model or local `.pt` file;
- optional class filtering;
- documented augmentation settings;
- reproducible training using a fixed random seed;
- explicit control over the training output folder;
- CUDA/GPU validation before training;
- saved plots and training diagnostics when enabled.


---

### 3. Upload trained weights to Roboflow

Script:

```text
uploadYOLOtoRoboflow.py
```

This optional script uploads trained YOLO weights to a Roboflow project version.

Main features:

- reads the Roboflow API key from an environment variable;
- validates that the selected `best.pt` or `last.pt` file exists;
- uploads the model to a selected Roboflow workspace, project, and version.

---

### 4. Segment anatomical features and export masks

Script:

```text
YoloAntomicalSeg.py
```

This is the main analysis script. It applies a trained YOLO segmentation model to all images in an input folder using a tiled sliding-window approach.

Main features:

- segments large images tile by tile;
- supports overlapping tiles;
- supports multiple anatomical feature classes:
  - vessels;
  - rays;
  - fibers;
  - parenchyma;
- exports full-resolution binary masks;
- optionally saves overlay images;
- optionally resolves overlaps between feature masks using a priority order;
- optionally calculates a cell-wall mask;
- optionally reconstructs individual objects from YOLO instance borders;
- optionally measures anatomical objects;
- optionally groups vessels based on distance;
- supports controlled RGB conversion for grayscale, TIFF, OME-TIFF, uint16, float, and multichannel images.


---

## Installation

Create a conda environment:

```bash
conda create -n AIAnatomyEnv python=3.10 -y
conda activate AIAnatomyEnv
```

Install GPU-enabled PyTorch:

```bash
conda install -y -c pytorch -c nvidia pytorch torchvision pytorch-cuda=11.8
```

Install scientific and imaging dependencies:

```bash
conda install -y -c conda-forge numpy pandas scipy scikit-image matplotlib opencv pillow
```

Install pip-only dependencies:

```bash
python -m pip install ultralytics pyometiff roboflow
```

Optional Jupyter kernel:

```bash
conda install -y -c conda-forge ipykernel jupyterlab
python -m ipykernel install --user --name AIAnatomyEnv --display-name "AIAnatomyEnv"
```

---

## Main dependencies

The scripts use:

- Python
- Ultralytics YOLO
- PyTorch
- NumPy
- pandas
- OpenCV
- Pillow
- SciPy
- scikit-image
- matplotlib
- pyometiff
- Roboflow

---

## Repository structure

Recommended project folder structure:

```text
PROJECT_ROOT/
  raw_images/              original full-size images
  cropped/                 YOLO-sized image crops
    selection/             optional random crop selection for inspection
  trainingdata/            YOLO dataset folder containing data.yaml
  training_runs/           output folder for YOLO training runs
  models/                  copied or final model weights
  segmentation_output/     segmentation masks, overlays, and measurements
```

---

## Data and trained models

A trained network example and accompanying training data are available on Zenodo.

The software for image acquisition with the Gigapixel Woodbot can be found here:

https://github.com/UGent-Woodlab/Gigapixel-Woodbot

The trained YOLO model and training data can be found here:

https://doi.org/10.5281/zenodo.14604996

The increment core images can be found here:

https://doi.org/10.5281/zenodo.14627909

The disk images can be found here:

https://doi.org/10.6019/S-BIAD1574

When using the software, please also cite the relevant Zenodo DOI for the software release:

- analysis software: https://doi.org/10.5281/zenodo.14637855
- imaging software: https://doi.org/10.5281/zenodo.14637832

---

## Cite our work

The full pipeline is described in:

https://doi.org/10.1186/s13007-025-01330-7

Please cite:

```tex
@Article{VandenBulcke2025,
  author={{Van den Bulcke}, Jan and Verschuren, Louis and De Blaere, Ruben and Vansuyt, Simon and Dekegeleer, Maxime and Kibleur, Pierre and Pieters, Olivier and De Mil, Tom and Hubau, Wannes and Beeckman, Hans and Van Acker, Joris and Wyffels, Francis},
  title={Enabling high-throughput quantitative wood anatomy through a dedicated pipeline},
  journal={Plant Methods},
  year={2025},
  month={Feb},
  day={04},
  volume={21},
  number={1},
  pages={11},
  issn={1746-4811},
  doi={10.1186/s13007-025-01330-7}
}

@software{software2025,
  author={{Van den Bulcke}, Jan and Verschuren, Louis and Wyffels, Francis},
  title={UGent-Woodlab/High-throughput-pipeline-yolov8},
  month={jan},
  year={2025},
  publisher={Zenodo},
  doi={10.5281/zenodo.14637854}
}
```

---

## License

This software is licensed under the GNU Affero General Public License v3.0.

See the license text here:

https://choosealicense.com/licenses/agpl-3.0/
