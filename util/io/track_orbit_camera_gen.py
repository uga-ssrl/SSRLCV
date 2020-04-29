import sys
from math import sin, cos, tan, pi, radians, atan, sqrt
from numpy.polynomial import Polynomial as P

#
# Generates accurate orbital coordinates for cameras given angles
#

# get the user inputs
if (len(sys.argv) < 4):
    print("NOT ENOUGH ARGUMENTS")
    print("USAGE:")
    print("\npython3 track_orbit_camera_gen.py altitidue stepAngle numSteps\n")
    print("\t <> altitidue -- the altitidue of the circular orbit in km")
    print("\t <> stepAngle -- the off angle from the point track normal")
    print("\t <> numSteps  -- the numbers of angle steps to take ")
    exit(-1)

earth = 6537
# gravitational constant
G     = 6.67430 * (10**(-11))
# mass of earth in kg
M     = 5.97237 * (10**(24))
mu    = 3.986004 * (10**(14))
alt   = float(sys.argv[1])
theta = radians(float(sys.argv[2]) - 90.0)
b_theta = float(sys.argv[2])
steps = int(sys.argv[3])

params = []
blender_values = []

twoview = False

if (theta > pi/2):
    print("theta is too big, use a smaller angle")
    exit(-1)

# set theta for the slew
if ((not (steps%2)) and (steps != 2) ):
    print("please only enter an odd number of steps when doing more than 2 views")
    exit(-1)

if (steps == 2):
    steps += 1
    twoview = True

img_num = 1
for step in range(0,int((steps+1)/2)):
    if (step):

        # x^2 term
        A = (1 + tan(theta)**2)

        # x term
        B = (2 * -earth * tan(theta))

        # constant
        C = (earth**2 - (alt + earth)**2)

        # the polynomial
        p = P([C,B,A])
        x = 0
        if (p.roots()[0] > 0):
            x = p.roots()[0]
        else:
            x = p.roots()[1]
        y = tan(theta) * x
        orbit_angle = (pi / 2.0) - atan( (y + earth) / (x))
        print('      angle (rad): ' + str( orbit_angle ))
        orbit_arc   = (earth + alt) * orbit_angle
        print('      arclen (km): ' + str( orbit_arc ))
        orbit_speed = sqrt( (mu) / ((earth + alt) * 1000 ) )
        print('  velocity (km/s): ' + str(orbit_speed / 1000))
        slew_time   = orbit_arc / orbit_speed
        print('    slew time (s): ' + str(slew_time))
        slew_rate   = (float(sys.argv[2]) * (pi/180) )/ slew_time
        print('Tracking Slews:')
        print('\tslew rate (rad/s): ' + str(slew_rate))
        print('\tslew rate (deg/s): ' + str(slew_rate * (180/pi)))
        print('Nadir Slews:')
        pnt_rate   = orbit_angle / slew_time
        print('\tslew rate (rad/s): ' + str(pnt_rate))
        print('\tslew rate (deg/s): ' + str(pnt_rate * (180/pi)))
        ## now, because of how blender renders things after this I have to do some coordinate adjustments
        param_str  = str(img_num) + '.png,' + str(x) + ',0.0,' + str(-1.0 * y) + ',0.0,' + str((theta + pi/2)) + ',0.0,'
        params.append(param_str)
        param_str  = str(img_num+1) + '.png,' + str(-1.0 * x) + ',0.0,' + str(-1.0 * y) + ',0.0,' + str(-1.0 * (theta + pi/2)) + ',0.0,'
        params.append(param_str)
        y *= -1000
        x *= -1000
        blender_str = str(-1.0 * x) + ',0.0,' + str(y) + ', y rotate: ' + str( b_theta) + ' deg'
        blender_values.append(blender_str)
        blender_str = str(x) + ',0.0,' + str(y) + ', y rotate: ' + str(-1.0 * b_theta) + ' deg'
        blender_values.append(blender_str)
        # print(str(step+1) + '.png,' + str(x) + ',0.0,' + str(y) + ',0.0,' + str(-step * (theta + pi/2)) + ',0.0,')
        # print('theta: ' + str(theta) + '\t step: ' + str(step) + '\t step * theta: ' + str(step * theta))
        b_theta += float(sys.argv[2])
        theta = radians(b_theta - 90.0)
        img_num += 2
    else:
        param_str = str(img_num) + '.png,0.0,0.0,' + str(alt*1000) + ',0.0,0.0,0.0,'
        params.append(param_str)
        # print('1.png,0.0,0.0,' + str(alt*1000) + ',0.0,0.0,0.0,')
        blender_values.append('0.0,0.0,0.0, y rotate: 0.0')
        img_num += 1

print('\t ---- Copy to params.csv ----')
if (not twoview):
    for value in params:
        print(value)
else:
    print(params[0])
    print(params[1])


print('\t ----   Copy to Blender  ----')
if (not twoview):
    for value in blender_values:
        print(value)
else:
    print(blender_values[0])
    print(blender_values[1])








# done
