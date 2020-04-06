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
    print("\npython3 plot-x-y.py file/path/to/file.csv xlabel ylabel\n")
    print("\t <> file.csv -- a path to a csv file that you want to graph")
    print("\t <> xlabel   -- a string to label the x axis")
    print("\t <> ylabel   -- a string to label the y axis")
    exit(-1)

x=[]
y=[]

raw_file = open(str(sys.argv[1]),'r',newline='')
raw_data = csv.reader(raw_file);

for point in raw_data:
    x.append(float(point[0]))
    y.append(float(point[1]))

plt.plot(x,y, marker='o')

# the labels
plt.xlabel(str(sys.argv[2]))
plt.ylabel(str(sys.argv[3]))

plt.show()
