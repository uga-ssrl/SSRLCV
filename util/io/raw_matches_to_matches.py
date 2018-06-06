# this is the bash script to run
# ./bin/sift_cli images/everest01.png > p01.kp; ./bin/sift_cli images/everest02.png > p02.kp; ./bin/match_cli p01.kp p02.kp > matches_raw.txt

import sys
import math
from PIL import Image

# globals

raw_matches = []
image1 = sys.argv[1] # path to the first image
debug = False

# methods

def get_color(x,y):
    im = Image.open(str(image1)) # Can be many different formats.
    pix = im.load()
    value = pix[x,y]  # Set the RGBA Value of the image (tuple)
    return value

#
###
##### entry POINT
###
#

print "loading color data from: " + str(image1) + " ..."

with open('matches_raw.txt') as f:
    for line in f:
        items = line.split( )
        raw_matches.append([items[0],items[1],items[4],items[5]])

if (debug):
    print raw_matches

match_file = open('matches.txt', 'w')
match_file.write("%s\n" % str(len(raw_matches)-1))

for item in raw_matches:
    if (debug):
        print '============='
        print item[0]
    x = math.floor(float(item[0]))
    y = math.floor(float(item[1]))
    if (debug):
        print (x,y)
    rgb = get_color(x,y);
    if (debug):
        print rgb
    write_string = "0001.jpg,0002.jpg," + str(item[0]) + "," + str(item[1]) + "," + str(item[2]) + "," + str(item[3]) +","+ str(rgb[0]) +","+ str(rgb[1])+"," + str(rgb[2])+ "\n"
    match_file.write(write_string)

print 'generated clean matches file ...'
