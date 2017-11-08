import sys, os, time
from math import sin, cos, tan, sqrt, floor, radians
from sympy.solvers import solve
from sympy import *
import numpy
import scipy.misc
import pylab
import gc

print '>> This is just a test util to make an example camera_image match'

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
    #print camera
    cameras.append(camera)
    #print 'test'
    #print step

f = open('carl_pan_cameras.txt', 'w')
for c in cameras:
    f.write(str(c[0]) + ',' + str(c[1]) + ',' + str(c[2]) + ',' + str(c[3]) + ',' + str(c[4]) + ',' + str(c[5]) + ',' + str(c[5]) + '\n')
