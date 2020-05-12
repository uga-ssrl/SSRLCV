```

_______________________________________________________________________________________________________________
 _____/\\\\\\\\\\\_______/\\\\\\\\\\\______/\\\\\\\\\______/\\\____________________/\\\\\\\\\__/\\\________/\\\_
  ___/\\\/////////\\\___/\\\/////////\\\__/\\\///////\\\___\/\\\_________________/\\\////////__\/\\\_______\/\\\_
   __\//\\\______\///___\//\\\______\///__\/\\\_____\/\\\___\/\\\_______________/\\\/___________\//\\\______/\\\__
    ___\////\\\___________\////\\\_________\/\\\\\\\\\\\/____\/\\\______________/\\\______________\//\\\____/\\\___
     ______\////\\\___________\////\\\______\/\\\//////\\\____\/\\\_____________\/\\\_______________\//\\\__/\\\____
      _________\////\\\___________\////\\\___\/\\\____\//\\\___\/\\\_____________\//\\\_______________\//\\\/\\\_____
       __/\\\______\//\\\___/\\\______\//\\\__\/\\\_____\//\\\__\/\\\______________\///\\\______________\//\\\\\______
        _\///\\\\\\\\\\\/___\///\\\\\\\\\\\/___\/\\\______\//\\\_\/\\\\\\\\\\\\\\\____\////\\\\\\\\\______\//\\\_______
         ___\///////////_______\///////////_____\///________\///__\///////////////________\/////////________\///________
          _______________________________________________________________________________________________________________

```

# UGA SSRL Computer Vision

[University of Georgia Small Satellite Research Laboratory](smallsat.uga.edu) Computer Vision, SSRLCV, is a computer vision software library written in C++ and the Nvidia CUDA programming language for Nvidia GPU SoCs in space environments. The software will be used onboard the MOCI satellite with our modified TX2i, but is also compatible with Ubuntu 16.04, Ubuntu 18.04, and Linux for Tegra. SSRLCV can also run on the TX2 and the Jetson Nano. The software currently includes SIFT feature detection, SIFT feature generation, SIFT feature matching, point cloud filtering, 2 view triangulation, N view triangulation, and 2 view bundle adjustment. SSRLCV is capable of generating point clouds with 15 - 100 meter accuracy from a 400 km circular orbit and a 6 meter GSD; the results are documented in [this thesis research](http://piepieninja.github.io/research-papers/thesis.pdf) and several updates are expected in the near future.

You can begin reading documentation on [the SSRLCV github wiki](https://github.com/uga-ssrl/SSRLCV/wiki), and view code documentation at [data.calebadams.space/doxygen/documentation/html/](data.calebadams.space/doxygen/documentation/html/examples.html). It is also recommended to clone the following repositories:

* Sample Data: **Highly Recommended** to use, this is maintained as [SSRLCV-Sample-Data](https://gitlab.smallsat.uga.edu/payload_software/ssrlcv-sample-data) on gitlab, and [mirrored on github](https://github.com/uga-ssrl/SSRLCV-Sample-Data)
* Utilities: maintained as [SSRLCV-Utilities](https://gitlab.smallsat.uga.edu/payload_software/ssrlcv-utilities) on gitlab, and [mirrored on github](https://github.com/uga-ssrl/SSRLCV-Util)

Check out the [contributors guide](CONTRIB.md) if you would like to help further develop SSRLCV

## Dependencies

  * libpng-dev
  * libtiff-dev
  * g++
  * gcc
  * nvcc
  * CUDA 10.0

## Compilation

When making you should use the SM of your arch, you do this by setting the `SM` variable. I also recommend doing a multicore make with the `-j` flag. See below, where `#` are digits of integers:

All executables can be generated by simply using `make -j# SM=##`, additionally neither the `-j` or the `SM` variable are necessary. However, if these are not used then compilation will take much, much longer.

```
make sfm -j# SM=##
```

| Device                               | Recommended          | SM |
|:------------------------------------:|:--------------------:|:--:|
| Jetson Nano                          | `make sfm -j4 SM=53` | 53 |
| TX1                                  | `make sfm -j2 SM=53` | 53 |
| TX2 / TX2i                           | `make sfm -j6 SM=62` | 62 |
| Jetson Xavier                        | `make sfm -j6 SM=72` | 72 |
| Ubuntu 16.04+ with GTX 1060/1070     | `make sfm -j8 SM=61` | 61 |

You can clean back to source files only with:

```
make clean
```

## Usage

Simply use the command `./sfm -d /path/to/images -s path/to/seed.png`

| Flag              | Command Line Argument          | Details                      |
|:-----------------:|:------------------------------:|:----------------------------:|
| -i or --image     | `<path/to/single/image>`       | absolute or relative         |
| -d or --directory | `<path/to/directory/of/images>`| absolute or relative         |
| -s or --seed      | `<path/to/seed/image>`         | absolute or relative         |
| -np or --noparams |             N/A                | signify no use of params.csv |


### Output

SSRLCV currently produces `.ply` files in the `out` folder. A future release will allow for better control of output files and allow

## Camera Parameters

The image rotation encodes which way the camera was facing as a [rotation of axes](https://en.wikipedia.org/wiki/Rotation_of_axes) around the individual x, y, and z axes in R3. This, along with a physical position in R3, should be passed in by the ADCS. All other parameters should be known. The focal length is usually on the order of mm and the dpix is usually on the order of nm.

| Data type       | Variable Name     |  SI unit        | Description                                |
|:---------------:|:-----------------:|:---------------:|:------------------------------------------:|
| `float3`        | `cam_pos`         | Kilometers      |  The x,y,z camera position                 |
| `float3`        | `cam_rot`         | Radians         |  The x,y,z camera rotation                 |
| `float2`        | `fov`             | Radians         |  The x and y field of view                 |
| `float`         | `foc`             | Meters          |  The camera's focal length                 |
| `float2`        | `dpix`            | Meters          |  The physical dimensions of a pixel well   |
| `long long int` | `timeStamp`       | UNIX timestamp  |  A UNIX timestamp from the time of imaging  |
| `uint2`         | `size`            | Pixels          |  The x and y pixel size of the image       |

## File Formats

### The SSRLCV Logger

SSRLCV includes a logger that produces a comma segmented, `.csv` encoded, log file at `out/ssrlcv.log`



### ASCII Camera Parameters - `.csv` ASCII encoded file

The ASCII encoded files that contain camera parameters should be included in the same directory as the images you wish to run a reconstruction on. It is required that the file be named `params.csv`. The file consists of the `Image.camera` struct parameters  (mentioned above for ease) in order. The format is as follows:


```
filename,x position,y position, z position, x rotation, y rotation, z rotation, x field of view, y field of view, camera focal length, x pixel well size, y pixel well size, UNIX timestamp, x pixel count, y pixel count
```

the files should be listed in a numerical order, each camera should be on one line and end with a `,`

and example of this is:

```
ev01.png,781.417,0.0,4436.30,0.0,0.1745329252,0.0,0.19933754453,0.19933754453,0.16,0.4,0.4,1580766557,1024,1024,
ev02.png,0.0,0.0,4500.0,0.0,0.0,0.0,0.19933754453,0.19933754453,0.16,0.4,0.4,1580766557,1024,1024,
```

Examples of such parameters can be found at [data.calebadams.space/CalebAdams-Tests-Used-In-Thesis/](http://104.236.14.11/CalebAdams-Tests-Used-In-Thesis/)

### Binary Camera Parameters - `.bcp` file type

Binary camera parameters are not currently defined but will be in a later release

## Documentation

### Online Documentation

Documentation on the use of SSRLCV can be found at:

* The SSRLCV wiki is located at [https://github.com/uga-ssrl/SSRLCV/wiki](https://github.com/uga-ssrl/SSRLCV/wiki)
* Code documentation is located at [data.calebadams.space/doxygen/documentation/html/](data.calebadams.space/doxygen/documentation/html)

### SSRLCV Utilities

The SSRLCV has various utilities for testing, IO, and data visualization. These can be found at the [SSRLCV utilities gitlab](https://gitlab.smallsat.uga.edu/payload_software/ssrlcv-utilities) repository.

These additional software packages are beneficial:
* [MeshLab](http://www.meshlab.net/) - Critical for viewing the results of SSRLCV.
* [CloudCompare](https://cloudcompare.org/) - Useful for comparing ground truth models, the ICP algorithm within CC is great for this.

### Manual Generation

Generate Doxygen by executing `doxygen doc/doxygen/Doxyfile` from within the projects root directory. An `index.html` file will be available in `doc/doxygen/documentation/html`, you can start there when exploring documentation locally.

# Citations

Upon usage please cite one or more of the following:

[High Performance Computation with Small Satellites and Small Satellite Swarms for 3D Reconstruction](http://piepieninja.github.io/research-papers/thesis.pdf)

```
@mastersthesis{CalebAdamsMSThesis,
  author={Caleb Ashmore Adams},
  title={High Performance Computation with Small Satellites and Small Satellite Swarms for 3D Reconstruction},
  school={The University of Georgia},
  url={http://piepieninja.github.io/research-papers/thesis.pdf},
  year=2020,
  month=may
}
```

[Towards an Integrated GPU Accelerated SoC as a Flight Computer for Small Satellites](https://ieeexplore.ieee.org/document/8741765)

```
@inproceedings{TowardsAdams2019,
  doi = {10.1109/aero.2019.8741765},
  url = {https://doi.org/10.1109/aero.2019.8741765},
  year = {2019},
  month = mar,
  publisher = {{IEEE}},
  author = {Caleb Adams and Allen Spain and Jackson Parker and Matthew Hevert and James Roach and David Cotten},
  title = {Towards an Integrated {GPU} Accelerated {SoC} as a Flight Computer for Small Satellites},
  booktitle = {2019 {IEEE} Aerospace Conference}
}
```

[A Near Real Time Space Based Computer Vision System for Accurate Terrain Mapping](https://digitalcommons.usu.edu/cgi/viewcontent.cgi?article=4216&context=smallsat)

```
@inproceedings{CVAdams2018,
  title={A Near Real Time Space Based Computer Vision System for Accurate Terrain Mapping},
  author={Adams, Caleb},
  journal={32nd Annual AIAA/USU Conference on Small Satellites},
  year={2018},
  publisher={AIAA}
}
```












<!-- yeet -->
