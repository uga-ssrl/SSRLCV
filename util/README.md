# Tegra-SFM Utilities

**NOTHING CRITICAL TO THE PIPELINE SHOULD BE STORD IN THIS DIRECTORY.**

This folder is used for storing unit tests, examples (such as prototyping math or features of the pipeline), and testing individual inputs and outputs of segments of the pipeline.

* `CI` -> contains unit test scripts
* `examples` -> contains potentially useful examples
* `io` -> contains scripts and programs to help generate portions of the pipeline

# Use in testing Accuracy of the reprojection stage

these scripts, for testing this accuracy, are located in the `io` folder.

**THIS PART SHOULD AREADY BE DONE** and only needs to be done on new installs.

## MacOS
... also this part only works on mac right now

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

you **MUST** be on the utilities branch. type: `git checkout utilities`
