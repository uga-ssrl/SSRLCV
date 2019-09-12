import csv
import math
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from matplotlib.patches import ConnectionPatch
from PIL import Image

#match_data_src = "everest2000/brute_unfiltered_matches.txt"
#match_data_src = "everest254/anatomy_sift_matches.txt"
match_data_src = "everest254/subpixel_matches.txt"
match_data = []
disparity_raw = []

def euclid_dist((x0,y0),(x1,y1)):
    return math.sqrt((x0-x1)**2 + (y0-y1)**2)

count = 0
with open(match_data_src, "rb") as f:
    reader = csv.reader(f, delimiter=",")
    for i, line in enumerate(reader):
        match_data.append(line)
        if '.' in str(line):
            print line
            count += 1
print count


for match in match_data:
    left  = (float(match[0]),float(match[1]))
    right = (float(match[2]),float(match[3]))
    disparity_raw.append(euclid_dist(left,right))

# here we assume matches are no farther than lim pixels away
# I AM ASSUMING A SQUARE PICTURE
side     = int(math.sqrt(len(match_data)))
pix      = []
lim      = 25
mul      = math.floor(255/lim) # the miltplier per pixel
iterator = 0

print "data_len: " + str(len(match_data))
print "side: " + str(side)

for x in range(0,side):
    temp_row = []
    for y in range(0,side):
        if disparity_raw[iterator] > lim:
            temp_row.append(0)
            iterator += 1
            continue
        temp_row.append(int(disparity_raw[iterator]*mul))
        iterator += 1
    pix.append(temp_row)

img = Image.new( 'RGB', (side,side), "black") # create a new black image
pixels = img.load() # create the pixel map

for x in range(0,side):
    for y in range(0,side):
        pixels[x,y] = (pix[x][y],pix[x][y],pix[x][y])

img.show()
img.save("disparity.png")
