import sys
from math import sin, cos, tan, pi, radians
from numpy.polynomial import Polynomial as P

#
# Generates accurate orbital coordinates for cameras given a scan
#

# get the user inputs
if (len(sys.argv) < 3):
    print("NOT ENOUGH ARGUMENTS")
    print("USAGE:")
    print("\npython3 scan_orbit_camera_gen.py altitidue stepAngle numSteps\n")
    print("\t <> altitidue    -- the altitidue of the circular orbit in km")
    print("\t <> scanDistance -- the distance on earth to scan, in km")
    print("\t <> numImages    -- the number of images locations to produce during the scan")
    exit(-1)

# I derived this elsewhere
earth = 6537.0
alt   = float(sys.argv[1])
arc   = float(sys.argv[2])
num   = int(sys.argv[2])

theta = arc / earth # in radians

params = []
blender_values = []


# img_num = 1
# for step in range(0,int((steps+1)/2)):
#     if (step):
#
#         # x^2 term
#         A = (1 + tan(theta)**2)
#
#         # x term
#         B = (2 * -earth * tan(theta))
#
#         # constant
#         C = (earth**2 - (alt + earth)**2)
#
#         # the polynomial
#         p = P([C,B,A])
#         x = 0
#         if (p.roots()[0] > 0):
#             x = p.roots()[0]
#         else:
#             x = p.roots()[1]
#         y = tan(theta) * x
#         ## now, because of how blender renders things after this I have to do some coordinate adjustments
#         param_str  = str(img_num) + '.png,' + str(x) + ',0.0,' + str(-1.0 * y) + ',0.0,' + str((theta + pi/2)) + ',0.0,'
#         params.append(param_str)
#         param_str  = str(img_num+1) + '.png,' + str(-1.0 * x) + ',0.0,' + str(-1.0 * y) + ',0.0,' + str(-1.0 * (theta + pi/2)) + ',0.0,'
#         params.append(param_str)
#         y *= -1000
#         x *= -1000
#         blender_str = str(-1.0 * x) + ',0.0,' + str(y) + ', y rotate: ' + str( b_theta) + ' deg'
#         blender_values.append(blender_str)
#         blender_str = str(x) + ',0.0,' + str(y) + ', y rotate: ' + str(-1.0 * b_theta) + ' deg'
#         blender_values.append(blender_str)
#         # print(str(step+1) + '.png,' + str(x) + ',0.0,' + str(y) + ',0.0,' + str(-step * (theta + pi/2)) + ',0.0,')
#         # print('theta: ' + str(theta) + '\t step: ' + str(step) + '\t step * theta: ' + str(step * theta))
#         b_theta += float(sys.argv[2])
#         theta = radians(b_theta - 90.0)
#         img_num += 2
#     else:
#         param_str = str(img_num) + '.png,0.0,0.0,' + str(alt*1000) + ',0.0,0.0,0.0,'
#         params.append(param_str)
#         # print('1.png,0.0,0.0,' + str(alt*1000) + ',0.0,0.0,0.0,')
#         blender_values.append('0.0,0.0,0.0, y rotate: 0.0')
#         img_num += 1
#
# print('\t ---- Copy to params.csv ----')
# if (not twoview):
#     for value in params:
#         print(value)
# else:
#     print(params[0])
#     print(params[1])
#
#
# print('\t ----   Copy to Blender  ----')
# if (not twoview):
#     for value in blender_values:
#         print(value)
# else:
#     print(blender_values[0])
#     print(blender_values[1])








# done
