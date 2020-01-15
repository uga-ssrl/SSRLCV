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
* CUDA 10.0

## Compilation

When making you should use the SM of your arch, you do this by setting the `SM` variable. I also reccomend doing a multicore make with the `-j` flag. See below, where `#` are digits of integers:


```
make -j# SM=## 
```

| Device        | Reccomended           | 
| ------------- |:-------------:|
| TX2i          | `make -j3 SM=52` | 
| Jetson Nano      | `make -j3 SM=53`   
|  Ubuntu 16.04 with GTX 1070     | `make -j9 SM=61` |

You can also clean out the repo, to just have the standard files again, with

```
make clean
```

## Documentation
* Generate Doxygen by executing `doxygen doc/doxygen/Doxyfile` from within the projects root directory
* index.html will be available in doc/doxygen/documentation/html and will allow traversal of the documentation

## Running
### Full Pipeline

There is no full pipeline main setup so far. There are separate pipelines compiled in the `/bin` directory. To learn about the popeline, 
you can find information on SIFT can be learned here: [Anatomy of SIFT](http://gitlab.smallsat.uga.edu/Caleb/anatomy-of-sift/blob/master/Anatomy%20of%20SIFT.pdf), this 
sn't Lowe's original thing but it explains it pretty well. You should also see the latex doc that has been made, [located here](https://gitlab.smallsat.uga.edu/payload_software/Tegra-SFM/blob/master/doc/paper/main.pdf) - this is 
known as the [Algorithm Theoretical Basis Document](https://gitlab.smallsat.uga.edu/payload_software/Tegra-SFM/blob/master/doc/paper/main.pdf).

## Source

Source files for the nominal program are located in the `src` folder. Some additional programs are located in the `util` folder. 
Dependinces for the source file are list here, but dependencies for the util files may vary. 

## File Formats

### Binary Cameras

### Binary Matches

### ASCII Matches
Feature matches, for sift, are stored in a `.txt` file. The first line of the file is the number of matches.

> matches (int)

The file type is basically a `.csv` of the following format:

> image 1 (string),image 2 (string),image 1 x value (float),image 1 y value (float),image 2 x value (float),image 2 y value (float),r (0-255),g (0-255),b (0-255)

### ASCII Cameras
Camera location and pointing data is stored in a in a `.txt` file. The file includes a location in `(x,y,z)` as well as a unit vector `(u_x,u_y,u_z)` to represent the orientation of the camera. The file type is basically a `.csv` of the following format:

> image number (int), camera x (float), camera y (float), camera z (float), camera unit x (float), camera unit y (float), camera unit z (float)

### PLY File