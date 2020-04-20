#!/usr/bin/python3

# imports
import sys
import csv
import numpy as np
import matplotlib.mlab as mlab
import matplotlib.pyplot as plt

#
# Plots a graph of the log data of SSRLCV given an input log
#

# get the user inputs
if (len(sys.argv) < 5):
    print("NOT ENOUGH ARGUMENTS")
    print("USAGE:")
    print("\npython3 logGrapher.py file/path/to/log.csv enableStates enableVoltage enableCurrent enablePower \n")
    print("\t <> file.csv       -- a path to the log file")
    print("\t <> enableStates   -- Graph the state transitions of the pipeline")
    print("\t <> enableVoltage  -- Graph the voltage of the system over time")
    print("\t <> enableCurrent  -- Graph the voltage of the system over time")
    print("\t <> enablePower    -- Graph the wattage of the system over time")
    # print("\t <> enableMemory   -- Graph the RAM of the system over time") # TODO add this boi
    exit(-1)

# enable the params to log
enableStates  = int(sys.argv[2])
enableVoltage = int(sys.argv[3])
enableCurrent = int(sys.argv[4])
enablePower   = int(sys.argv[5])

# the raw data to log
states   = []
voltages = []
currents = []
powers   = []
memories = []

# read teh data
with open(str(sys.argv[1])) as csvfile:
    readCSV = csv.reader(csvfile, delimiter=',')
    for i in readCSV:
        for j in i:
            

# x=[]
# y=[]
# boi = 0;
#
# with open(str(sys.argv[1])) as csvfile:
#     readCSV = csv.reader(csvfile, delimiter=',')
#     for i in readCSV:
#         for j in i:
#             if (len(j) > 0):
#                 # print(j)
#                 x.append(float(boi))
#                 y.append(float(j))
#                 boi += 1.0;
#
#
# plt.plot(x,y, marker='o')
#
# # the labels
# plt.xlabel(str(sys.argv[2]))
# plt.ylabel(str(sys.argv[3]))
#
# plt.show()
