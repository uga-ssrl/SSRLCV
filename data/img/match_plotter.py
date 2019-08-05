import csv
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from matplotlib.patches import ConnectionPatch


print '========================='
print '=  Matches  Visualizer  ='
print '========================='
print '\n^C to quit...'


img_left  = mpimg.imread('everest2000/img/ev01.png')
img_right = mpimg.imread('everest2000/img/ev02.png')

match_data_src = "everest2000/brute_unfiltered_matches.txt"

match_data = []

with open(match_data_src, "rb") as f:
    reader = csv.reader(f, delimiter=",")
    for i, line in enumerate(reader):
        match_data.append(line)

fig = plt.figure()
a1 = fig.add_subplot(1, 2, 1)
imgplot = plt.imshow(img_left)
a1.set_title('Image 1')

a2 = fig.add_subplot(1, 2, 2)
imgplot = plt.imshow(img_right)
imgplot.set_clim(0.0, 1.0)
a2.set_title('Image 2')

print 'total matches: ' + str(len(match_data))

# just get a few matches to show
s_size = 15
range = 5000
sample = np.random.randint(range, size=s_size)
for m_i in sample:
    left  = (int(match_data[m_i][0]), int(match_data[m_i][1]))
    right = (int(match_data[m_i][2]), int(match_data[m_i][3]))
    # print 'left:'
    # print left
    # print 'right:'
    # print right
    con = ConnectionPatch(xyA=left, xyB=right, coordsA="data", coordsB="data", axesA=a2, axesB=a1, color="red")
    a2.add_artist(con)
    a1.plot( left[0], left[1],'ro',markersize=10)
    a2.plot(right[0],right[1],'ro',markersize=10)

plt.show()
