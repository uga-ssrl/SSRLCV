#!/usr/bin/python3

# imports
import sys
import csv
import os
import numpy as np
import matplotlib.mlab as mlab
import matplotlib.pyplot as plt

#
# Plots a graph of the log data of SSRLCV given an input log
#

# get the user inputs
if (len(sys.argv) < 4):
    print("NOT ENOUGH ARGUMENTS")
    print("USAGE:")
    print("\npython3 bundle_adjustment_tests_viewer.py file/path/to/ba/tests/ badRunCut OptimalVal\n")
    print("\t <> directoryPath  -- a directory path")
    print("\t <> badRunCut      -- A value representing a bad run")
    print("\t <> OptimalVal     -- A value representing the optimal value")
    exit(-1)

directory  = sys.argv[1]
badRunCut  = int(sys.argv[2])
OptimalVal = int(sys.argv[3])

BA_bois = []
BA_lens = []


print('Finding ...')
for filename in os.listdir(directory):
    if filename.endswith("Errors.csv") and not (filename.endswith("wErrors.csv")):
        print(os.path.join(directory, filename))
        fileguy = os.path.join(directory, filename)
        # read in the data
        with open(fileguy) as csvfile:
            readCSV = csv.reader(csvfile, delimiter=',')
            entry = []
            lens  = 0
            for i in readCSV:
                for j in i:
                    if (len(j) > 0):
                        entry.append(float(j))
                        lens += 1
            BA_bois.append(entry)
            BA_lens.append(lens)
        continue

axes = plt.gca()
for b in BA_bois:
    t = []
    for i in range(0,len(b)):
        t.append(i)
    if (b[-1] < badRunCut):
        plt.plot(t, b, '#0165fc')
    else:
        plt.plot(t, b, '#929591')

axes.set_yscale('log') # set to true if log scale desired


plt.axhline(y=(2 * OptimalVal), linewidth=4, color='#ffb07c', label='Optima')

plt.xlabel('iterations')
plt.ylabel('linear error')

plt.legend(loc="upper right")

plt.show()





#
