# Python Blood Vessel Topology Analysis

![Example](docs/readme_imgs/video.gif)

Routines for identifying and transforming 2D and 3D blood vessel images into graphs. The graphs represent the **topology** of the blood vessels, that is, bifurcations and terminations are represented as nodes and two nodes are connected if there is a blood vessel segment between them.

Some functions are provided for measuring blood vessel density, number of bifurcation points and tortuosity, but other metrics can be implemented. The created graphs are objects from the [Networkx](https://networkx.org/) libray.

### 3D Blood Vessel Image

The library works for 2D and 3D blood vessel images but the focus of the library lies on 3D confocal microscopy images, such as this one:

<img src="docs/readme_imgs/original.png" width="600" />

### Segmentation

File [segmentation.py](pyvesto/segmentation.py) contains the segmentation routines, aimed at classifying pixels into two categories: blood vessel or background. The image below is a sum projection of a 3D binary image.

<img src="docs/readme_imgs/binary.png" width="600" />

### Medial Lines

File [skeleton.py](pyvesto/skeleton.py) contains a skeletonization function implemented in C and interfaced using ctypes for calculating the medial lines of the blood vessels. This function was compiled for Linux.

<img src="docs/readme_imgs/skeleton.png" width="600" />

### Blood Vessel Reconstruction

Having the binary image and the medial lines, a model of the blood vessels surface can be generated:

<img src="docs/readme_imgs/reconstructed.png" width="600" />

### Graph Generation and Adjustment

Files inside the [graph](pyvesto/graph.py) folder are responsible for creating the graph and removing some artifacts such as small branches generated from the skeleton calculation.

<img src="docs/readme_imgs/graph.png" width="600" />

### Measurements

Functions inside [measure.py](pyvesto/measure.py) implement some basic blood vessel measurmeents.

### Whole Pipeline

The notebook [blood_vessel_pipeline.ipynb](notebooks/blood_vessel_pipeline.ipynb) contains an example pipeline for applying all the functionalities. 

### Dependencies (version)
* Python (3.7.4)
* scipy (1.4.1)
* numpy (1.19.2)
* networkx (2.4)
* matplotlib (3.3.4)
* igraph (0.7.1) - optional

**Warning, the skeletonization functions only work on Linux.**