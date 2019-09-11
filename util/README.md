# Tegra-SFM Utilities

**NOTHING CRITICAL TO THE PIPELINE SHOULD BE STORD IN THIS DIRECTORY.**

python version 2.7.15 is required

This folder is used for storing unit tests, examples (such as prototyping math or features of the pipeline), and testing individual inputs and outputs of segments of the pipeline.

* `CI` -> contains unit test scripts
* `examples` -> contains potentially useful examples
* `io` -> contains scripts and programs to help generate portions of the pipeline

# Use in testing Accuracy of the reprojection stage

you **MUST** be on the utilities branch. type: `git checkout utilities`

these scripts, for testing this accuracy, are located in the `io` folder.

**THIS PART SHOULD AREADY BE DONE** and only needs to be done on new installs.

## MacOS
... also this part only works on mac right now

you **MUST** be on the utilities branch. type: `git checkout utilities`

you need to have brew installed. run: `ruby -e "$(curl -fsSL https://raw.github.com/mxcl/homebrew/go)"` if you don't

> Add this line to your `~/.bashrc` or `~/.zshrc` file:
>
> `export PATH=/usr/local/bin:$PATH`
>
> `export PATH=/usr/local/share/python:$PATH`
>
> `brew install python`
>
> `. ~/.bashrc` or `. ~/.zshrc`
>

now if you type `python -V` it should return at least Python 2.7.15

if you don't see at least that version number, try `brew remove python@2 --ignore-dependencies` then `brew install python@2` and then type `. ~/.bashrc`

> first you need to make sure pip is installed:
>
> `curl https://bootstrap.pypa.io/get-pip.py -o get-pip.py`
>
> `sudo python get-pip.py`
>
> `sudo pip install numpy`
>
> `sudo pip install plyfile`
>

### Steps for MOPS when prepping for Cloud Compare

you should do these steps both with and without the gaussian randomness turned on.

#### getting data for comparison
* ssh into the tx2: `ssh nvidia@tx2` the password is nvidia
* then `git checkout utilities` and then run `git pull`
* run `make clean` in the root folder
* ON YOUR MAC:  scp the ply from the mac to the tx2: `scp path/to/the/file.ply nvidia@tx2:~/Development/Tegra-SFM`
  * the file should now be on the tx2
* run the `ply_to_matches.py` script on the ply files
  * this will output a `matches.txt` and a `cameras.txt` file
* in the root of the folder run `make -j4`
* run the gpu reprojection program with `./bin/reprojection.x path/to/cameras.txt path/to/matches.txt`
  * this will output another ply file
* ON YOUR MAC: copy the ply file back to ur mac: `scp nvidia@tx2:~/Development/Tegra-SFM/output.ply ~/Downloads`
  * this will go to your downloads folder
* you now have the 2 ply's you need to compare
