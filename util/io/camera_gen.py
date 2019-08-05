import sys, os, time
from math import sin, cos, tan, sqrt, floor, radians
from sympy.solvers import solve
from sympy import *
import numpy
import scipy.misc
#import pylab
import gc

def pan():
    d_theta = radians(45.0)
    foc     = 0.035 #= 0.035  # focal length of system
    fov     = radians(45.0)
    res     = 1024  # resolution of image in one d
    r       = 2.0   # meters
    d_pix = (foc*tan(fov/2))/(res/2)
    step = 0.02
    start = -1.0 #meters
l
        cameras.append(camera)

    f = open('carl_pan_cameras.txt', 'w')
    for c in cameras:
        f.write(str(c[0]) + ',' + str(c[1]) + ',' + str(c[2]) + ',' + str(c[3]) + ',' + str(c[4]) + ',' + str(c[5]) + ',' + str(c[6]) + '\n')

def rot():
    d_theta = radians(10.0) # the step size per rotation, TODO make this more clear
    foc     = 0.035 # focal length of system
    fov     = radians(3.4)
    res     = 2000 #1024  # resolution of image in one d
    r       = 400.0  #2.0   # meters
    d_pix = (foc*tan(fov/2))/(res/2)

    cameras = []
    for dt in range (1,2): # should be the the number of steps u want to take TODO make this more clear
        u_x = float(sin((dt-1) * d_theta))
        u_y = float(cos((dt-1) * d_theta))
        u_z = 0.0
        # there were issues w floats being v inaccurate at small values:
        if (abs(u_x) < 0.0001):
            u_x = 0.0
        if (abs(u_y) < 0.0001):
            u_y = 0.0
        x = r * u_x
        y = r * u_y
        z = r * u_z
        camera = [dt, x, y, z, u_x, u_y, u_z]
        cameras.append(camera)

    f = open('cameras.txt', 'w')
    for c in cameras:
        f.write(str(c[0]) + ',' + str(c[1]) + ',' + str(c[2]) + ',' + str(c[3]) + ',' + str(c[4]) + ',' + str(c[5]) + ',' + str(c[6]) + '\n')

###############
# entry point #
###############

print '>> This is just a test util to make an example camera_image match'

rot()
