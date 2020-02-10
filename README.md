```
  _______                               _____ ______ __  __ 
 |__   __|                             / ____|  ____|  \/  |
    | | ___  __ _ _ __ __ _   ______  | (___ | |__  | \  / |
    | |/ _ \/ _` | '__/ _` | |______|  \___ \|  __| | |\/| |
    | |  __/ (_| | | | (_| |           ____) | |    | |  | |
    |_|\___|\__, |_|  \__,_|          |_____/|_|    |_|  |_|
             __/ |                                          
            |___/                                           
```

# Tegra - Structure from Motion 

## Create the following directories if they do not exist
bin, data, obj, out, src, util
(util, data, and src should be in repository)

## File Naming Convention
If a file has CUDA in it of any sort -> .cu and its header -> .cuh.
All other files can be .cpp and .h

## Dependencies
* libpng
* libtiff-dev
* g++
* gcc
* nvcc
* CUDA 10.0

## Compilation

When making you should use the SM of your arch, you do this by setting the `SM` variable. I also recommend doing a multicore make with the `-j` flag. See below, where `#` are digits of integers:


```
make sfm -j# SM=##
```

| Device                               | Recommended          |
|:------------------------------------:|:--------------------:|
| TX2i                                 | `make sfm -j3 SM=52` |
| Jetson Nano                          | `make sfm -j3 SM=53` |   
| Ubuntu 16.04 with GTX 1060/1070      | `make sfm -j5 SM=61` |

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
Dependences for the source file are list here, but dependencies for the util files may vary.

## Camera Parameters

The image rotation encodes which way the camera was facing as a [rotation of axes](https://en.wikipedia.org/wiki/Rotation_of_axes) around the individual x, y, and z axes in R3. This, along with a physical position in R3, should be passed in by the ADCS. All other parameters should be known. The focal length is usually on the order of mm and the dpix is usually on the order of nm.

| Data type       | Variable Name     |  SI unit        | Description                                |
|:---------------:|:-----------------:|:---------------:|:------------------------------------------:|
| `float3`        | `cam_pos`         | Meters          |  The x,y,z camera position                 |
| `float3`        | `cam_rot`         | Radians         |  The x,y,z camera rotation                 |  
| `float2`        | `fov`             | Radians         |  The x and y field of view                 |
| `float`         | `foc`             | Meters          |  The camera's focal length                 |
| `float2`        | `dpix`            | Meters          |  The physical dimensions of a pixel well   |
| `long long int` | `timeStamp`       | UNIX timestamp  |  A UNIX timestap from the time of imaging  |
| `uint2`         | `size`            | Pixels          |  The x and y pixel size of the image       |

## File Formats

### ASCII Camera Parameters - `.csv` ASCII encoded file

The ASCII encoded files that contain camera parameters should be included in the same directory as the images you wish to run a reconstruction on. It is required that the file be named `params.csv`. The file consists of the `Image.camera` struct parameters  (mentioned above for ease) in order. The format is as follows:


```
filename,x position,y position, z position, x rotation, y rotation, z rotation, x field of view, y field of view, camera focal length, x pixel well size, y pixel well size, UNIX timestamp, x pixel count, y pixel count
```

the files should be listed in a numerical order, each file should have a new line.

and example of this is:

```
ev01.png,781.417,0.0,4436.30,0.0,0.1745329252,0.0,0.19933754453,0.19933754453,0.16,0.4,0.4,1580766557,1024,1024
ev02.png,0.0,0.0,4500.0,0.0,0.0,0.0,0.19933754453,0.19933754453,0.16,0.4,0.4,1580766557,1024,1024
```

### Binary Camera Parameters - `.bcp` file type

This is the binary version of the ascii format.

### Image File Formats - `.png` , `.tiff` , `.jpg`

TODO information about image support limitations here

### Point Clouds - `.ply` stanford PLY format

TODO information about ply support and limitations here

### Match Files - unknown

TODO Match file support here

Check out the [contributors guide](CONTRIB.md) for imformation on contributions

# TODO
* ensure that thrust functions are usi ng GPU
* more documentations































<!-- yeet -->
