import csv
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from PIL import Image
from matplotlib.patches import ConnectionPatch


print '========================='
print '=  Vector Shower        ='
print '========================='
print '\n^C to quit...'



def get_i(rgb):
    return (rgb[0]/4 + rgb[1]/2 + rgb[0]/4)


img = "everest254/ev_18x18.png"
#img = "everest254/simple_gradient.png"
#img = "everest254/simple_gradient2.png"
gradients = []
guy = Image.open(img) 
pix = guy.load()

for y in range (1,17):
    temp = []
    for x in range(1,17):
        x_grad = get_i(pix[x + 1,y]) - get_i(pix[x - 1,y])
        y_grad = get_i(pix[x,y + 1]) - get_i(pix[x,y - 1])
        unit = (float(x_grad)/255.0,float(y_grad)/255.0)
        temp.append(unit)
    gradients.append(temp)
print gradients

ax = plt.axes()
im = plt.imread(img)
implot = plt.imshow(im)

print gradients[15][15]

for x in range(0,16):
    for y in range(0,16):
        ax.arrow(y+1, x+1, gradients[x][y][0], gradients[x][y][1], head_width=0.06, head_length=0.20, fc='k', ec='k',color='g')

plt.show()
