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
    print("\npython3 plot-sensitivity.py file/path/to/partialFilename xlabel ylabel\n")
    print("\t <> partialFilename -- a path with a PARTIAL filename, which does not inclde the _Delta######.csv part")
    print("\t <> xlabel          -- a string to label the x axis")
    print("\t <> ylabel          -- a string to label the y axis")
    exit(-1)

# Open the linear files

raw_file_XL = open(str(sys.argv[1]) + '_DeltaXLinear.csv' ,'r', newline='')
raw_data_XL = csv.reader(raw_file_XL);

raw_file_YL = open(str(sys.argv[1]) + '_DeltaYLinear.csv' ,'r', newline='')
raw_data_YL = csv.reader(raw_file_YL);

raw_file_ZL = open(str(sys.argv[1]) + '_DeltaZLinear.csv' ,'r', newline='')
raw_data_ZL = csv.reader(raw_file_ZL);

# Open the angular files

raw_file_XA = open(str(sys.argv[1]) + '_DeltaXAngular.csv' ,'r', newline='')
raw_data_XA = csv.reader(raw_file_XA);

raw_file_YA = open(str(sys.argv[1]) + '_DeltaYAngular.csv' ,'r', newline='')
raw_data_YA = csv.reader(raw_file_YA);

raw_file_ZA = open(str(sys.argv[1]) + '_DeltaZAngular.csv' ,'r', newline='')
raw_data_ZA = csv.reader(raw_file_ZA);

# prep the linear file data

x_XL = []
y_XL = []
x_YL = []
y_YL = []
x_ZL = []
y_ZL = []

for point in raw_data_XL:
    x_XL.append(float(point[0]));
    y_XL.append(float(point[1]));

for point in raw_data_YL:
    x_YL.append(float(point[0]));
    y_YL.append(float(point[1]));

for point in raw_data_ZL:
    x_ZL.append(float(point[0]));
    y_ZL.append(float(point[1]));

# prep the angular file data

x_XA = []
y_XA = []
x_YA = []
y_YA = []
x_ZA = []
y_ZA = []

for point in raw_data_XA:
    x_XA.append(float(point[0]));
    y_XA.append(float(point[1]));

for point in raw_data_YA:
    x_YA.append(float(point[0]));
    y_YA.append(float(point[1]));

for point in raw_data_ZA:
    x_ZA.append(float(point[0]));
    y_ZA.append(float(point[1]));

fig, axs = plt.subplots(2,3)

axs[0, 0].plot(x_XL, y_XL)
axs[0, 0].set_title('Linear X Sensitivity')
axs[0, 1].plot(x_YL, y_YL)
axs[0, 1].set_title('Linear Y Sensitivity')
axs[0, 2].plot(x_ZL, y_ZL)
axs[0, 2].set_title('Linear Z Sensitivity')
axs[1, 0].plot(x_XA, y_XA)
axs[1, 0].set_title('Angular X Sensitivity')
axs[1, 1].plot(x_YA, y_YA)
axs[1, 1].set_title('Angular Y Sensitivity')
axs[1, 2].plot(x_ZA, y_ZA)
axs[1, 2].set_title('Angular Z Sensitivity')

for ax in axs.flat:
    ax.set(xlabel=str(sys.argv[2]), ylabel=str(sys.argv[3]))

#
# plt.plot(x,y, marker='o')
#
# # the labels
# plt.xlabel(str(sys.argv[2]))
# plt.ylabel(str(sys.argv[3]))
#
plt.show()
