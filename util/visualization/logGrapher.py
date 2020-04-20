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
if (len(sys.argv) < 6):
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

# the starttime
start_time = 0

# state numbers (for easy parsing)
io          = 0
seed        = 1
features    = 2
matching    = 3
triangulate = 4
filter      = 5
bundleboi   = 6

# the raw data to log
states   = []
voltages = []
currents = []
powers   = []
memories = []

# read the data
with open(str(sys.argv[1])) as csvfile:
    readCSV = csv.reader(csvfile, delimiter=',')
    for i in readCSV:
        entry = []
        local_time = int(i[0])
        if (i[1] == 'start'): # set start time
            start_time = local_time
        elif (i[1] == 'state' and enableStates > 0):
            # -------------------- #
            # Load State Data Here #
            # -------------------- #
            print("boi")
            if (i[2] == 'reading images'): # loading images
                entry.append(local_time - start_time)
                entry.append(io)
                states.append(entry)
            elif (i[2] == 'done reading images'): # done loding images
                entry.append(local_time - start_time)
                entry.append(io)
                states.append(entry)
            elif (i[2] == 'matching images'): # done loding images
                entry.append(local_time - start_time)
                entry.append(matching)
                states.append(entry)
            elif (i[2] == 'done matching images'): # done loding images
                entry.append(local_time - start_time)
                entry.append(matching)
                states.append(entry)
            elif (i[2] == 'generating features'): # done loding images
                entry.append(local_time - start_time)
                entry.append(features)
                states.append(entry)
            elif (i[2] == 'done generating features'): # done loding images
                entry.append(local_time - start_time)
                entry.append(features)
                states.append(entry) # generating seed matches
            elif (i[2] == 'generating seed matches'): # done loding images
                entry.append(local_time - start_time)
                entry.append(seed)
                states.append(entry)
            elif (i[2] == 'done generating seed matches'):
                entry.append(local_time - start_time)
                entry.append(seed)
                states.append(entry)
            elif (i[2] == 'triangulation'):
                entry.append(local_time - start_time)
                entry.append(triangulate)
                states.append(entry)
            elif (i[2] == 'end triangulation'):
                entry.append(local_time - start_time)
                entry.append(triangulate)
                states.append(entry)
            elif (i[2] == 'filter'):
                entry.append(local_time - start_time)
                entry.append(filter)
                states.append(entry)
            elif (i[2] == 'end filter'):
                entry.append(local_time - start_time)
                entry.append(filter)
                states.append(entry)
        elif (i[1] == 'PWR_SYS_GPU' and enablePower > 0):
            # -------------------- #
            # Load Power Data Here #
            # -------------------- #
            # for details on power info see: https://docs.nvidia.com/jetson/archives/l4t-archived/l4t-3231/index.html#page/Tegra%2520Linux%2520Driver%2520Package%2520Development%2520Guide%2Fpower_management_tx2_32.html%23wwpID0E0OF0HA
            #      0           1         2          3     4     5    6     7          8
            # 1587359573412,PWR_SYS_GPU,195,PWR_SYS_SOC,586,PWR_IN,3249,PWR_SYS_CPU,1075,
            entry.append(local_time - start_time) # time
            entry.append(float(i[2])) # GPU
            entry.append(float(i[4])) # SOC
            entry.append(float(i[6])) # IN
            entry.append(float(i[8])) # CPU
            powers.append(entry)
        elif (i[1] == 'VDD_SYS_GPU' and enableVoltage > 0):
            # ---------------------- #
            # Load Voltage Data Here #
            # ---------------------- #
            # TODO implement
            print("do voltage")
        elif (i[1] == 'CRR_SYS_GPU' and enableCurrent > 0):
            # ---------------------- #
            # Load Current Data Here #
            # ---------------------- #
            # TODO implement
            print("do current")

# ~~~
#
# Graph the desired output of the log
#
# ~~~

plt.xlabel("time in milliseconds")

# TODO make label dynamic
plt.ylabel("milliwatts")

# plot the power
t   = []
gpu = []
soc = []
cpu = []
# a but waistful, but I'm just graphing things give me a break
for p in powers:
    t.append(p[0])
    gpu.append(p[1])
    soc.append(p[2])
    cpu.append(p[4])

# draw the state lines:
for s in states:
    label_name = ''
    if (s[1] == io):
        label_name = "file io"
    elif (s[1] == seed):
        label_name = "seed image"
    elif (s[1] == features):
        label_name = "feature generation"
    elif (s[1] == matching):
        label_name = "feature matching"
    elif (s[1] == triangulate):
        label_name = "triangulation"
    elif (s[1] == filter):
        label_name = "filtering"
    elif (s[1] == bundleboi):
        label_name = "bundle adjustment"
    plt.axvline(x=s[0], linewidth=4, color='y', label=label_name)


plt.plot(t, gpu, 'r', label="GPU power usage")
plt.plot(t, soc, 'g', label="SoC power usage")
plt.plot(t, cpu, 'b', label="CPU power usage")

plt.legend(loc="upper right")

plt.show()











#
