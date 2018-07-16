We followed the SVO paper and implemented the monocular SLAM.

Configurations like dataset directory and intrinstics parameters are set in the /data/default.yaml file. The reading sequence is specified in the rgb.txt in the dataset and the estimated trajectory will be recorded in the estimation.txt.

The main.cpp reads the image frames and the process of the frames is in the viso.cpp.
