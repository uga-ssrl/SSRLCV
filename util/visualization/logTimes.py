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
if (len(sys.argv) < 2):
    print("NOT ENOUGH ARGUMENTS")
    print("USAGE:")
    print("\npython3 logTimes.py file/path/to/log.csv \n")
    print("\t <> file.csv       -- a path to the log file")
    exit(-1)

# the starttime
start_time = 0
endin_time = 0
# the log stamps
seed_start = 0
seed_end   = 0
features_start = 0
features_end   = 0
matching_start = 0
matching_end   = 0
triangulate_start = 0
triangulate_end   = 0
filter_start = 0
filter_end   = 0
bundle_start = 0
bundle_end   = 0


# read the data
with open(str(sys.argv[1])) as csvfile:
    readCSV = csv.reader(csvfile, delimiter=',')
    for i in readCSV:
        entry = []
        local_time = int(i[0])
        if (i[1] == 'state'):
            # -------------------- #
            # Load State Data Here #
            # -------------------- #
            if (i[2] == 'SEED'): # loading images
                if (seed_start == 0):
                    seed_start = local_time
                else:
                    seed_end = local_time
            if (i[2] == 'FEATURES'):
                if (features_start == 0):
                    features_start = local_time
                else:
                    features_end = local_time
            if (i[2] == 'MATCHING'):
                if (matching_start == 0):
                    matching_start = local_time
                else:
                    matching_end = local_time
            if (i[2] == 'TRIANGULATE'):
                if (triangulate_start == 0):
                    triangulate_start = local_time
                else:
                    triangulate_end = local_time
            if (i[2] == 'FILTER'):
                if (filter_start == 0):
                    filter_start = local_time
                else:
                    filter_end = local_time
            if (i[2] == 'BA'):
                if (bundle_start == 0):
                    bundle_start = local_time
                else:
                    bundle_end = local_time
            if (i[2] == 'start'): # set start time
                start_time = local_time
            if (i[2] == 'end'):
                endin_time = local_time

#
# Print off the results
#

print('Total Runtime: ' + str(endin_time - start_time) + ' ms, ' + str((endin_time - start_time)/1000.0) + ' s' )
print('\t  Seed Image Time: ' + str(seed_end - seed_start) + ' ms, ' + str((seed_end - seed_start)/1000.0) + ' s' )
print('\t Feature Gen Time: ' + str(features_end - features_start) + ' ms, ' + str((features_end - features_start)/1000.0) + ' s' )
print('\t    Matching Time: ' + str(matching_end - matching_start) + ' ms, ' + str((matching_end - matching_start)/1000.0) + ' s' )
print('\t Triangulate Time: ' + str(triangulate_end - triangulate_start) + ' ms, ' + str((triangulate_end - triangulate_start)/1000.0) + ' s' )
print('\t   Filtering Time: ' + str(filter_end - filter_start) + ' ms, ' + str((filter_end - filter_start)/1000.0) + ' s' )
print('\t  Bundle Adj Time: ' + str(bundle_end - bundle_start) + ' ms, ' + str((bundle_end - bundle_start)/1000.0) + ' s' )








#
