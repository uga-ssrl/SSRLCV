# Tegra SfM

## Compilation

TBD

## Running

TBD

## Source

Source files for the nominal program are located in the `src` folder. Some additional programs are located in the `util` folder. 

## File Formats

### Matches
Feature matches, for sift, are stored in a `.txt` file. The first line of the file is the number of matches.

> matches (int)

The file type is basically a `.csv` of the following format:

> image 1 (string),image 2 (string),image 1 x value (float),image 1 y value (float),image 2 x value (float),image 2 y value (float),r (0-255),g (0-255),b (0-255)

## TX1 information

Username: ubuntu

Password: ubuntu

IP: 128.192.19.163
