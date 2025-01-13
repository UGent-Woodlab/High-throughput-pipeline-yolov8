<p align="center">
  <img src="./Fig.jpg" width="100%" alt="logo">
</p>
<p align="center">
    <h1 align="center">YoloAnatomy</h1>
</p>


[Verschuren, Louis![ORCID logo](https://info.orcid.org/wp-content/uploads/2019/11/orcid_16x16.png)](https://orcid.org/0000-0002-3102-4588)[^aut][^cre][^UG-WL];
[Wyffels, Francis![ORCID logo](https://info.orcid.org/wp-content/uploads/2019/11/orcid_16x16.png)](https://orcid.org/0000-0002-5491-8349)[^aut][^AI-RO];
[Van den Bulcke, Jan![ORCID logo](https://info.orcid.org/wp-content/uploads/2019/11/orcid_16x16.png)](https://orcid.org/0000-0003-2939-5408)[^aut][^UG-WL]

[^aut]: author
[^cre]: contact person
[^UG-WL]: UGent-Woodlab
[^AI-RO]: AI and Robotics Lab, IDLab-AIRO


<p align="left">
   This is the repository for a Python routine which is used to train a YOLOv8 deep learning segmentation model and subsequently uses it to do fully automated analysis of gigapixel sized images, here exemplified for the quantification of vessels and rays on full disc surfaces and increment cores. 
</p>


<p align="center">
	<!-- local repository, no metadata badges. --></p>
<p align="center">
		<em>Built with the tools and technologies:</em>
</p>
<p align="center">
	<img src="https://img.shields.io/badge/Python-3776AB.svg?style=default&logo=Python&logoColor=white" alt="Python">
	<img src="https://img.shields.io/badge/NumPy-013243.svg?style=default&logo=NumPy&logoColor=white" alt="NumPy">
	<img src="https://img.shields.io/badge/yolo-8-blue" alt="YOLOv8">	
</p>
<br>

#####  Table of Contents

- [ Crop images](#crop-images-crop-imagesipynb)
- [ Model training](#model-training-yolo8-vessel-detector-trainipynb)
- [ Image analysis](#image-analysis-sliding-window-yolov8-maskipynb)
- [ Getting Started](#getting-started)
- [ Cite our work](#cite-our-work)
- [ License](#license)

---

##  Crop images: crop-images.ipynb
This will take all images from a specified folder and crops them to 640 x 640 images which can be used for annotation. 640 x 640 is the standard image size for YOLOv8. 

---

## Model training: yolo8-vessel-detector-train.ipynb
This will train a YOLOv8 segentation model from an annotated training dataset. Training data can be created using [Roboflow](https://roboflow.com/). Depening on the datset size and application, different augmentation parameters can be chosen.

---

## Image analysis: sliding window Yolov8 mask.ipynb
This segments all images in a specified folder using a trained YOLOv8 model. It creates binary masks for each of the classes in the model. This code can also count individual detections of objects, like individual vessels, and removes double detections. It uses a sliding window approach with a user defined overlap percentage. 

---

## Getting started

Before running the notebooks, ensure that you have the following dependencies installed:
- from ultralytics import YOLO
- os
- from PIL import Image
- numpy
- torch
- cv2
- from pyometiff import OMETIFFReader
- sys
- from torchvision.ops import nms
- math

A trained network example and accompanying training data are available on Zenodo (see 'Cite our work').

---

## Cite our work

You can find the paper where the entire pipeline is described [here](TO DO), or cite our work with the following bibtex snippet:

```tex
TODO
```

The software for image acquisition with the Gigapixel Woodbot can be found [here](https://github.com/UGent-Woodlab/Gigapixel-Woodbot), the trained YOLOv8 model and training data can be found [here](https://doi.org/10.5281/zenodo.14604996), the increment core images can be found [here](https://doi.org/10.5281/zenodo.14627909) and the disk images [here](https://doi.org/10.6019/S-BIAD1574).

When using any of the software, also cite the proper Zenodo DOI ([here for analysis](https://doi.org/10.5281/zenodo.14637855) and [here for imaging](https://doi.org/10.5281/zenodo.14637832)) related to the releases of the software.

---

##  License

This software is protected under the [GNU AGPLv3](https://choosealicense.com/licenses/agpl-3.0/) license. 

---
