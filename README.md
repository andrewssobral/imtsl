Last update: **05/11/2014**

IMTSL
-----
Incremental and Multi-feature Tensor Subspace Learning applied for Background Modeling and Subtraction

<p align="center"><img src="https://sites.google.com/site/ihosvd/_/rsrc/1405352965295/ihosvd.png" /></p>

```
Matlab Tensor Tools
https://github.com/andrewssobral/mtt

Matrix and Tensor Tools for Computer Vision 
http://www.slideshare.net/andrewssobral/matrix-and-tensor-tools-for-computer-vision
```

Highlights
----------
* Proposes an incremental low-rank HoSVD (iHOSVD) for background modeling and subtraction.
* A unified tensor model to represent the features extracted from the streaming video data.

Citation
---------
If you use this code for your publications, please cite it as:
```
@inproceedings{asobral2014,
    author       = "Sobral, A. and Baker, C. G. and Bouwmans, T. and Zahzah, E.",
    title        = "Incremental and Multi-feature Tensor Subspace Learning applied for  Background Modeling and Subtraction",
    booktitle    = "International Conference on Image Analysis and Recognition (ICIAR'14)",
    year         = "2014",
    month        = "October",
    publisher    = "Lecture Notes in Computer Science (Springer LNCS)",
    url          = "https://sites.google.com/site/ihosvd/"
}
```

Abstract
--------
Background subtraction (BS) is the art of separating moving objects from their background. The Background Modeling (BM) is one of the main steps of the BS process. Several subspace learning (SL) algorithms based on matrix and tensor tools have been used to perform the BM of the scenes. However, several SL algorithms work on a batch process increasing memory consumption when data size is very large. Moreover, these algorithms are not suitable for streaming data when the full size of the data is unknown. In this work, we propose an incremental tensor subspace learning that uses only a small part of the entire data and updates the low-rank model incrementally when new data arrive. In addition, the multi-feature model allows us to build a robust low-rank background model of the scene. Experimental results shows that the proposed method achieves interesting results for background subtraction task.

IncrementalHoSVD
----------------
<p align="left"><img src="https://sites.google.com/site/ihosvd/incrementalHOSVD.png?width=600" /></p>

Source code
-----------
```
dataset - dataset folder
libs - libraries/toolboxes
output - output folder
IMTSL.m - proposed algorithm
compute_similarity3.m - similarity measure of the current feature and it low-rank representation (for gradient magnitude)
compute_similarity2.m - "" (all other features)
displog.m - display message with time
perform_feature_extraction_rgb.m - perform feature extraction (by default, 8 features are extracted)
showResults.m - display results (input frame, low-rank, sparse and outliers)
```

License
-------
The source code is available only for academic/research purposes (non-commercial).

Problems or Questions
---------------------
If you have any problems or questions, please contact the author: Andrews Sobral (andrewssobral@gmail.com)

References
----------
[1] Baker, C.G.; Gallivan, K.A.; Van Dooren, P. Low-rank incremental methods for computing dominant singular subspaces, 2012.
