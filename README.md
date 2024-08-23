# Gaussian Mixture Model on Signed distance Persistence Homology

<!-- Outline -->
## Outline

This repository contains data and code for reproducing all results and figures from [``A Topological Gaussian Mixture Model for Bone Marrow
Morphology in Leukaemia"]().

We integrate Signed Distance Persistence Homology (SDPH) from [Song et al.](https://github.com/annasongmaths/SDPH) with stage-dependent Gaussian Mixture Model (GMM) to infer patterns and changes in morphologies of bone marrow vessels with Leukemia.

<!-- Structure of the repository -->
## Structure of the repository
This repository contains the following:
- `.ipynb`: notebook to reproduce the models and predictions 
- `Figures.ipynb`: notebook to reproduce all figures and results
- Folder `GMM` contains
- Folder `Data_loading` contains Python script to load data from pickle files and the notebook `How_to_load_PHloc.ipynb` provides an example for generating the Gaussian kernel approximation heatmaps.

The SDPH diagrams for the knee can be found [here](https://drive.google.com/file/d/14v3P8qcZBDP8Z1BZtfFc9K9scKrBinVQ/view?usp=sharing) and for the long region [here](https://drive.google.com/file/d/14v3P8qcZBDP8Z1BZtfFc9K9scKrBinVQ/view?usp=sharing). For the local analysis, the location data can be found [here](https://drive.google.com/file/d/1xQJ1nZtzw0xLi8JUDsgsI0mWN-Gt-g3f/view?usp=drive_link).

<!-- Package installation -->


## Package installation
packages
- Gaussian kernel densities were computed using either keops (GPU, fast) or scikit-... (CPU, slow). Using keops is state of the art for more than tens of thousands of points in the diagrams, which is recommended to use


