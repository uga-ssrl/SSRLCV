#!/usr/bin/python3

# imports
import sys
import csv
import numpy as np

#
# Does a sum of either side of a cutoff
#

# get the user inputs
if (len(sys.argv) < 2):
    print("NOT ENOUGH ARGUMENTS")
    print("USAGE:")
    print("\npython3 sumcut.py file/path/to/file.csv cutoff \n")
    print("\t <> file.csv -- a path to a csv file that you want to make a histogram of")
    print("\t <> cutoff number -- a number which represend the cutoff value to sum either side of")
    exit(-1)

x=[]
y=[]
boi = 0;

cutoff = float(sys.argv[2])

print("cutoff set as: " + str(cutoff))

left  = 0;
right = 0;

with open(str(sys.argv[1])) as csvfile:
    readCSV = csv.reader(csvfile, delimiter=',')
    for i in readCSV:
        for j in i:
            if (len(j) > 0):
                if (float(j) > cutoff):
                    right += float(j)
                else:
                    left += float(j)
                # print(j)


print("left: " + str(left) + "\t right: " + str(right))
if (left > right):
    print("The left side dominates")
else:
    print("The right side dominates")
