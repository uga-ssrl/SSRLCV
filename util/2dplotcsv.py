#!/usr/bin/python3

# imports
import sys
import csv
import numpy as np
import matplotlib.mlab as mlab
import matplotlib.pyplot as plt

#
# Plots a graph of given csv inputs
#

# get the user inputs
if (len(sys.argv) < 3):
    print("NOT ENOUGH ARGUMENTS")
    print("USAGE:")
    print("\npython3 2dplotcsv.py file/path/to/file.csv bin# xlabel ylabel\n")
    print("\t <> file.csv -- a path to a csv file that you want to make a histogram of")
    print("\t <> xlabel   -- a string to label the x axis")
    print("\t <> ylabel   -- a string to label the y axis")
    exit(-1)

x=[]
y=[]
boi = 0;

with open(str(sys.argv[1])) as csvfile:
    readCSV = csv.reader(csvfile, delimiter=',')
    for i in readCSV:
        for j in i:
            if (len(j) > 0):
                # print(j)
                x.append(float(boi))
                y.append(float(j))
                boi += 1.0;


plt.plot(x,y, marker='o')

# the labels
plt.xlabel(str(sys.argv[2]))
plt.ylabel(str(sys.argv[3]))

plt.show()
