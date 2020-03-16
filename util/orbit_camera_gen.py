import sys
from math import sin, cos, tan, pi, radians
from numpy.polynomial import Polynomial as P

#
# Generates accurate orbital coordinates for cameras given angles
#

# get the user inputs
if (len(sys.argv) < 3):
    print("NOT ENOUGH ARGUMENTS")
    print("USAGE:")
    print("\npython3 orbit_camera_gen.py altitidue stepAngle numSteps\n")
    print("\t <> altitidue -- the altitidue of orbit in km")
    print("\t <> stepAngle -- the off angle from the point track normal")
    print("\t <> numSteps  -- the numbers of angle steps to take ")
    exit(-1)

# I derived this elsewhere
earth = 6537
alt   = float(sys.argv[1])
theta = radians(float(sys.argv[2]) - 90.0)
steps = int(sys.argv[3])

if (theta > pi/2):
    print("theta is too big, use a smaller angle")
    exit(-1)

for step in range(0,steps):
    if (step):
        # x^2 term
        A = (1 + tan(step * theta)**2)

        # x term
        B = (2 * -earth * tan(step * theta))

        # constant
        C = (earth**2 - (alt + earth)**2)

        # the polynomial
        p = P([C,B,A])
        x = 0
        if (p.roots()[0] > 0):
            x = p.roots()[0]
        else:
            x = p.roots()[1]
        y = tan(step * theta) * x
        ## now, because of how blender renders things after this I have to do some coordinate adjustments
        y *= -1000
        x *= -1000
        print(str(step+1) + '.png,' + str(x) + ',0.0,' + str(y) + ',0.0,' + str(-step * (theta + pi/2)) + ',0.0,')
    else:
        print('1.png,0.0,0.0,' + str(alt*1000) + ',0.0,0.0,0.0,')

#
