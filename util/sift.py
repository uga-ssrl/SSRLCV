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

#########
verbose = False


#########

print '================================'
print '=             SIFT             ='
print '================================'

print'THIS IS ONLY KNOWN TO WORK ON MACOS!!!!'


# Get the args
if len(sys.argv) < 2:
    print 'DIRECTORY EXPECTED'
    print 'USAGE:'
    print 'sift.py /path/to/images/      # does sift on the images in that DIR'
    print 'sift.py /path/to/images/ -v   # does sift, but is verbose'

directory = sys.argv[1]

if len(sys.argv) == 3 and sys.argv[2] == '-v':
    verbose = True

# load the images

print 'Loading Images ...'
try:
    files = []
    for filename in os.listdir(directory):
        if filename.endswith(".jpg") or filename.endswith(".png"):
            rgb = scipy.misc.imread(directory + filename)
            if rgb.ndim != 2:
                bw = 0.299 * rgb[:, :, 0] + 0.587 * rgb[:, :, 1] + 0.114 * rgb[:, :, 2]
            else:
                bw = rgb
            files.append([filename, rgb, bw])
            continue
        else:
            continue
    if verbose:
        print 'loaded ' + str(len(files)) + ' files'
except Exception:
    print 'ERROR: could not load files from: ' + directory
    print 'USAGE:'
    print 'sift.py /path/to/images/      # does sift on the images in that DIR'
    print 'sift.py /path/to/images/ -v   # does sift, but is verbose'


# run sift
print 'Running SIFT ...'
sift_kp = []
for n in range(0, len(files)):
    ocl    = sift.SiftPlan(template=files[n][1]) # get rgb
    kp_ocl = numpy.empty(0)
    kp_ocl = ocl.keypoints(files[n][1])
    kp_ocl.sort(order=["scale", "angle", "x", "y"])
    sift_kp.append(kp_ocl)
    if verbose:
        print str(float(n)/float(len(files)) * 100.0) + ' %' # give a progress update
if verbose:
    print '100.0 %'
print 'Features Extracted ...'

# match the images
# http://www.silx.org/doc/silx/dev/modules/image/sift.html
matches = []
counter = 0.0
total_matches = 0
for n in range(0, len(files)):
    mp = sift.MatchPlan()
    for m in range(0, len(files)):
        if (m != n):  # just the same frame
            match = mp(sift_kp[n], sift_kp[m])
            if verbose:
                print str(float(counter)/float(len(files)**2 - 100.0) * 100.0) + ' %\t| between image: ' + str(n) + ' and ' + str(m) + ', matched ' + str(match.shape[0]) + ' keypoints'
            total_matches += match.shape[0]
            counter = counter + 1.0
            img = Image.open(directory+files[n][0])
            pix = img.load()
            #print match[0]
            #print match[0][0]
            if match.shape[0] > 0:
                for x in range(0,match.shape[0]):
                    pix_x = int(match[x][0][0])
                    pix_y = int(match[x][0][1])
                    matches.append([files[n][0],files[m][0], match, pix[pix_x,pix_y]])
if verbose or debug:
    print '100.0%'
print 'Total Matches: ' + str(total_matches)

# write the keypoints to a file

f = open('matches.txt', 'w')
f.write(str(total_matches) + '\n')
for m in matches:
    f.write(str(m[0])+','+str(m[1])+','+str(m[2][0][0][0])+','+str(m[2][0][0][1])+','+str(m[2][0][1][0])+','+str(m[2][0][1][1])+','+str(m[3][0])+','+str(m[3][1])+','+str(m[3][2])+'\n')
    # print m[0]       # image 1
    # print m[1]       # image 2
    # print m[2][0][0][0] # image 1 points
    # print m[2][0][0][1] # image 2 points
    # print m[2][0][1][0] # image 1 points
    # print m[2][0][1][1] # image 2 points
    # print m[3][0]    # image rgb
    # print m[3][1]
    # print m[3][2]
