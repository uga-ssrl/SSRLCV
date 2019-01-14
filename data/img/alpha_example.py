import math
import numpy
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

print 'Example alpha matrix'

def get_z(x,y,r):
    return ((math.sqrt(r**2 - x**2 - y**2) + r))
    #return (initial - r)
    
# ===== ENTRY ===== #

size   = 20
radius = 6000
alpha  = []
space  = 1
step   = float(space)/float(size/2)
curr_x = -1 * space/2
curr_y = -1 * space/2

print 'step:' + str(step)
print 'starting x: ' + str(curr_x)
print 'starting y: ' + str(curr_y)

for y in range(-1,1,0.25):
    temp = []
    for x in range(-1,1,0.25):
        temp.append(get_z(x,y,radius))
    alpha.append(temp)

print alpha

# Set up grid and test data
hf = plt.figure()
ha = hf.add_subplot(111, projection='3d')

x = range(0,10)
y = range(0,10)

data = numpy.array(alpha)

X, Y = numpy.meshgrid(x, y)  # `plot_surface` expects `x` and `y` data to be 2D
ha.plot_surface(X, Y, data)

plt.show()
