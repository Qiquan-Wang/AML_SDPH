# Gaussian Mixture Model on Signed distance Persistence Homology

<!-- Outline -->
## Outline

This repository contains data and code for reproducing all results and figures from [``A Topological Gaussian Mixture Model for Bone Marrow
Morphology in Leukaemia"]().

We integrate Signed Distance Persistence Homology (SDPH) from [Song et al.](https://github.com/annasongmaths/SDPH) with stage-dependent Gaussian Mixture Model (GMM) to infer patterns and changes in morphologies of bone marrow vessels with Leukemia.

<!-- Structure of the repository -->
## Structure of the repository
This repository contains the following:
- Folder `Data_loading` contains Python script to load data from pickle files and the notebook `How_to_load_PHloc.ipynb` provides an example for generating the heatmaps from Gaussian kernel approximations.
- Folder `GMM` contains notebooks to reproduce the Phase 0, I, and II models for the different quadrants, labelled as `<<quadrant>>_GMM.ipynb`. `Distance table.ipynb` contains code to reproduce quantitative evaluation of models.
- Folder `Global Analysis` contains notebook for global analysis via hierarchical clustering on SDPH diagrams.
- Folder `Local Analysis` contains notebooks for local analyses using different clustering methods on features extracted from SDPH diagrams, namely K-Means, GMM and CLARA, with varying number of clusters.
- Utilities file `utils_load_PHloc.py` to load PH diagrams, critical points and critical sizes.

The SDPH diagrams for the knee can be found [here](https://drive.google.com/file/d/14v3P8qcZBDP8Z1BZtfFc9K9scKrBinVQ/view?usp=sharing) and for the long region [here](https://drive.google.com/file/d/14v3P8qcZBDP8Z1BZtfFc9K9scKrBinVQ/view?usp=sharing). For the local analysis, the location data can be found [here](https://drive.google.com/file/d/1xQJ1nZtzw0xLi8JUDsgsI0mWN-Gt-g3f/view?usp=drive_link).

<!-- Package installation -->
## Package installation

This implementation requires the following packages, including standard packages such as:
- [numpy](https://numpy.org/)
- [pandas](https://pandas.pydata.org/)
- [sys](https://docs.python.org/3/library/sys.html)
- [pickle](https://docs.python.org/3/library/pickle.html)
- [math](https://docs.python.org/3/library/math.html)
- [time](https://docs.python.org/3/library/time.html)
- [matplotlib](https://matplotlib.org/)

For the computation of SDPH:
- [giotto-tda](https://giotto-ai.github.io/gtda-docs/0.5.1/library.html)
- [GUDHI](https://gudhi.inria.fr/)
- [CubicalRipser](https://github.com/shizuo-kaji/CubicalRipser_3dim)

For the computation of Gaussian kernel density approximations:
- [scikit-learn](https://scikit-learn.org/stable/)
- [KeOps](https://www.kernel-operations.io/keops/index.html)
- [torch](https://pypi.org/project/torch/)

The combination of KeOps and torch allows for GPU acceleration and is recommended for efficiency.

For local analysis:
- [scikit-learn](https://scikit-learn.org/stable/)
- [scikit-learn-extra](https://scikit-learn-extra.readthedocs.io/en/stable/)

For global analysis:
- [BioPython](https://biopython.org/)

For GMM:
- [pomegranate](https://pomegranate.readthedocs.io/en/latest/)


