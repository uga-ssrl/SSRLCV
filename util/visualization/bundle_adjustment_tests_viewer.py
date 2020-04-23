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
    if (b[-1] > badRunCut or len(b) < 5):
        plt.plot(t, b, color='#929591')
        plt.plot(0,b[0], marker='o', color='#929591')
    else:
        plt.plot(t, b, color='#0165fc')
        plt.plot(0,b[0], marker='o', color='#0165fc')


print('================================================')
average_start = 0
average_good_start = 0
avg_denom = 0
for b in BA_bois:
    average_start += b[0]
    if (b[-1] < badRunCut):
        average_good_start += b[0]
        avg_denom += 1
print('\tAverage Start: ' + str(average_start / len(BA_bois)))
print('\tAverage Good Start: ' + str(average_good_start / avg_denom))

average_start = 0
average_good_start = 0
avg_denom = 0
for b in BA_bois:
    average_start += b[-1]
    if (b[-1] < badRunCut):
        average_good_start += b[-1]
        avg_denom += 1
print('\tAverage Endind Error: ' + str(average_start / len(BA_bois)))
print('\tAverage Good Ending Error: ' + str(average_good_start / avg_denom))

average_start = 0
average_good_start = 0
avg_denom = 0
for b in BA_bois:
    average_start += len(b)
    if (b[-1] < badRunCut):
        average_good_start += len(b)
        avg_denom += 1
print('\tIterations: ' + str(average_start / len(BA_bois)))
print('\tAverage Good Iterations: ' + str(average_good_start / avg_denom))

axes.set_yscale('log') # set to true if log scale desired

plt.axhline(y=(2 * OptimalVal), linewidth=4, color='#ffb07c', label='Optima')

plt.xlabel('iterations')
plt.ylabel('linear error')

plt.legend(loc="upper right")

plt.show()





#
