import sys, os, pyopencl, time, urllib2
from math import sin, cos, tan, sqrt, floor, radians
from PIL import Image
from sympy.solvers import solve
from sympy import *
import logging
logger = logging.getLogger("sift")
import silx.image.sift as sift
import numpy
import scipy.misc
import pylab
import gc

print '================================'
print '=             SIFT             ='
print '================================'

print'THIS IS ONLY KNOW TO WORK ON MACOS!!!!'


# Get the args
directory = ''

# Do sift

# write the keypoints to a file
