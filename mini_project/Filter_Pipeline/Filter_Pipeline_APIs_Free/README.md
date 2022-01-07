[![N|Solid](https://miro.medium.com/max/1408/1*2Y3467WK-pqjsUxf_oJGAA.png)](https://nodesource.com/products/nsolid)
# 3D perception using PointNet
**Kushagra Shukla, Greeswar R S**
**EYSIP Intern 2020-21**
## Overview of Script
1.
Filter_Pipeline_APIs_Free.cpp script don't uses any APIs to cluster out the ROI objects from a 2.5D noisy scan which is taken from a depth camera. The algorithm and code has been implemented from scratch.

**Following describes the flow and steps involved**
- 1.Pass-Through Filter and Downsampling
- 2.RANSAC Plane Detection
- 3.Clustering the ROI objects by DBSCAN method
2.
Filter_Pipeline_APIs_Free_optimized_beta_version.cpp is a modifed version of the above script with z and y plane removal combined. This script contains an algorithm for the same (plane_check) which works with certain point clouds and does'nt with some. One might work on this futher to get a more optimized pipeline.(Use the above script instead of this)
