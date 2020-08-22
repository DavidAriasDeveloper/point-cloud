# 3D point cloud projection through stereoscopic images with OpenCV and CUDA

Currently, machines require more visual information than was previously required. While a few centuries ago the biggest challenge was to permanently fix an image by means of light, nowadays we are looking for the best way to capture the information of an entire object and project it correctly at high speeds. Some techniques such as stereoscopy, photogrammetry and point cloud have partially solved this problem, but when working with so much information, they are not optimal enough to be processed in a conventional CPU. The present investigation, has like objective realize a demonstration of the mapping of objects in a space 3d through the stereoscopy, and later realize a comparison of the speed of the processing in CPU vs GPU

![App result: 3D Point Cloud image generated](https://github.com/dvariaz/point-cloud/blob/master/images/AppResult.png?raw=true)

## Implementation

This project was implemented in C++ and Python to take advantage of OpenCV version on these languages.

![App result: 3D Point Cloud image generated](https://github.com/dvariaz/point-cloud/blob/images/AppArchitecture.jpg?raw=true)

[Watch the paper](https://github.com/dvariaz/point-cloud/blob/master/paper.pdf?raw=true)
