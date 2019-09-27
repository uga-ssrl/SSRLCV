# ~ Tegra SfM ~

## Create the following directories if they do not exist
bin, data, obj, out, src, util
(util, data, and src should be in repository)

## File Naming Convention
If a file has CUDA in it of any sort -> .cu and its header -> .cuh.
All other files can be .cpp and .h

## Dependencies
* libpng
* g++
* gcc
* nvcc
* CUDA 9.0
* python 2.7.15

## Compilation

If you are running this for the first time you need to make a `bin` folder in the root of this repo.

`make` and `make clean`, per the uz

For sm detection related to gpu architecture you must make twice, once 
to compile the detection program, and once to actually compile the program. 
After doing this once you will not have to compile the sm detector again. 

## Documentation
* Generate Doxygen by executing `doxygen doc/doxygen/Doxyfile` from within the projects root directory
* index.html will be available in doc/doxygen/documentation/html and will allow traversal of the documentation

## Running
### Full Pipeline

Currently, the fully pipeline can be run from the `main.sh` script located in the root folder.

### SIFT

information on SIFT can be learned here: [Anatomy of SIFT](http://gitlab.smallsat.uga.edu/Caleb/anatomy-of-sift/blob/master/Anatomy%20of%20SIFT.pdf), this isn't Lowe's original thing but it explains it way better.

##### CPU
two SIFT executables are generated in the `bin` directory during the make, `sift_cli` and `match_cli`. The first of witch produces SIFT keypoints with descriptors and the latter of which matches those keypoints. There is a python script in the `util/io` folder that converts the raw match output into our format (specified below)

These executables return raw output, which is meant to be piped to a file.

`./bin/sift_cli path/to/image.png > keypoints.kp` would generate a keypoint file

`./bin/match_cli keypoints1.kp keypoints2.kp > raw_matches.txt` would generate a raw matches file

`python /util/io/raw_matches_to_matches.py` converts the `raw_matches.txt` file in root to a `matches.txt` file to be used by reprojection

##### GPU
not yet implemented

### reprojection
The executables for `.cpp` and `.cu` programs are placed in the `bin` folder with the extension `.x`

To run the reprojection from the root folder of this repo: `./bin/reprojection.x path/to/cameras.txt path/to/matches.txt`

Additionally, a `1` or a `0` can be placed at the end of the statement. A `0` makes it so that the program runs on the CPU with a CPU implementation. Be careful, this is super slow.

## Source

Source files for the nominal program are located in the `src` folder. Some additional programs are located in the `util` folder.

## File Formats

### Matches
Feature matches, for sift, are stored in a `.txt` file. The first line of the file is the number of matches.

> matches (int)

The file type is basically a `.csv` of the following format:

> image 1 (string),image 2 (string),image 1 x value (float),image 1 y value (float),image 2 x value (float),image 2 y value (float),r (0-255),g (0-255),b (0-255)

### Cameras
Camera location and pointing data is stored in a in a `.txt` file. The file includes a location in `(x,y,z)` as well as a unit vector `(u_x,u_y,u_z)` to represent the orientation of the camera. The file type is basically a `.csv` of the following format:

> image number (int), camera x (float), camera y (float), camera z (float), camera unit x (float), camera unit y (float), camera unit z (float)

## TX1 information

* Username: ubuntu
* Password: ubuntu
* IP: 128.192.19.163

## TX2 information

* Username: nvidia
* Password: nivida
* IP: 172.28.143.74
