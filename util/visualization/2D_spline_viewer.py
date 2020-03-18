import numpy as np
import matplotlib.pyplot as plt
from scipy import interpolate

points = [(1.0,8.00),(2.00,9.2),(3.00,4.0),(4.00,2.6),(5.00,2.9),(6.00,7.5),(7.0,8.6)]
data = np.array(points)

tck,u = interpolate.splprep(data.transpose(), s=0)
unew = np.arange(0, 1.01, 0.01)
out = interpolate.splev(unew, tck)

plt.figure()
plt.plot(out[0], out[1], color='green')
plt.plot(data[:,0], data[:,1], 'ob')
plt.show()
