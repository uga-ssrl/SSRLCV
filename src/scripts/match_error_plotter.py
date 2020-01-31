import csv
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from matplotlib.patches import ConnectionPatch

print '========================='
print '=   Error   Visualizer  ='
print '========================='
print '\n^C to quit...'

#match_error_src = "everest254/brute_unfiltered_distances.txt"
match_error_src = "everest254/everest254_matches.txt"

error_data_raw = []
error_data = []

with open(match_error_src, "rb") as f:
    reader = csv.reader(f, delimiter=",")
    for row in reader:
        error_data_raw = row

for x in error_data_raw:
    if x == '' or x == "":
        continue
    error_data.append(int(x))


fig = plt.figure()
ax = fig.add_subplot(111)

ax.set_xlabel('best-fit matching error')
ax.set_ylabel('error occurace')

plt.hist(error_data, bins=100)
plt.show()
