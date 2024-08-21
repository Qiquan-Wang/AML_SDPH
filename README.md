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

The data can be found at [link]()

<!-- Package installation -->


## Package installation
packages
- Gaussian kernel densities were computed using either keops (GPU, fast) or scikit-... (CPU, slow). Using keops is state of the art for more than tens of thousands of points in the diagrams, which is recommended to use


