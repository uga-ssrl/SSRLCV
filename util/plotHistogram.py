#!/usr/bin/python3

# imports
import sys
import csv
import numpy as np
import matplotlib.mlab as mlab
import matplotlib.pyplot as plt


#
# Plots a histogram of given inputs
#

# get the user inputs
if (len(sys.argv) < 4):
    print("NOT ENOUGH ARGUMENTS")
    print("USAGE:")
    print("\npython3 plotHistogram.py file/path/to/file.csv bin# xlabel ylabel\n")
    print("\t <> file.csv -- a path to a csv file that you want to make a histogram of")
    print("\t <> bin#     -- an integer representing the number of bars you want in the histogram")
    print("\t <> xlabel   -- a string to label the x axis")
    print("\t <> ylabel   -- a string to label the y axis")
    exit(-1)


# load in the csv file
x = []
with open(str(sys.argv[1])) as csvfile:
    readCSV = csv.reader(csvfile, delimiter=',')
    for i in readCSV:
        for j in i:
            if (len(j) > 0):
                x.append(float(j))

# print(x)

# set bin num
num_bins = int(sys.argv[2])

# the labels
plt.xlabel(str(sys.argv[3]))
plt.ylabel(str(sys.argv[4]))

n, bins, patches = plt.hist(x, num_bins, facecolor='blue', alpha=0.5)
plt.show()
