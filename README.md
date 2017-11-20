# Tegra SfM

## Compilation

If you are running this for the first time you need to make a `bin` folder in the root of this repo.

`make` and `make clean`, per the uz

## Running

The executables for `.cpp` programs are placed in the `bin` folder with the extension `.x`

To run the reprojection from the root folder of this repo: `./bin/reprojection.x path/to/cameras.txt path/to/matches.txt 0/1`

where '0' represents a reprojection constrained to a plane and '1' represents a projection constrained to a line.

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
