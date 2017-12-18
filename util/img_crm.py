import sys, os, time
from math import sin, cos, tan, sqrt, floor, radians
from sympy.solvers import solve
from sympy import *
import numpy
import scipy.misc
import pylab
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

    cameras = []
    for x in range (1,101):
        camera = [x, start + x*step, 0.5, 0.0, 1.0, 0.0, 0.0]
        cameras.append(camera)

    f = open('carl_pan_cameras.txt', 'w')
    for c in cameras:
        f.write(str(c[0]) + ',' + str(c[1]) + ',' + str(c[2]) + ',' + str(c[3]) + ',' + str(c[4]) + ',' + str(c[5]) + ',' + str(c[6]) + '\n')

def rot():
    d_theta = radians(360.0/200.0)
    foc     = 0.035 # focal length of system
    fov     = radians(49.135)
    res     = 1024  # resolution of image in one d
    r       = 50.0  #2.0   # meters
    d_pix = (foc*tan(fov/2))/(res/2)

    cameras = []
    for dt in range (1,201):
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


def everest_rot():
    #d_theta = radians(40.0/200.0)
    #foc     = 0.18288 # focal length of system
    #fov     = radians(10.0)
    #res     = 4208  # resolution of image in one d
    #r       = 401000.0   # meters
    #d_pix = (foc*tan(fov/2))/(res/2)

    d_theta = radians(40.0/20.0)
    foc     = 0.18288 # focal length of system
    fov     = radians(10.0)
    res     = 4208  # resolution of image in one d
    r       = 401.0   # kmeters
    d_pix = (foc*tan(fov/2))/(res/2)

    cameras = []
    for dt in range (-19,21):
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
#everest_rot()
