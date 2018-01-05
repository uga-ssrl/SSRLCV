import sys, os
from math import sin, cos, tan, sqrt, floor, radians

## flags and shit
verbose = True

# sample cube points
cube_points = [[ -1.0,  1.0, -1.0], # 0 A
               [  1.0,  1.0, -1.0], # 1 B
               [  1.0, -1.0, -1.0], # 2 C
               [ -1.0, -1.0, -1.0], # 3 D
               [ -1.0,  1.0,  1.0], # 4 E
               [  1.0,  1.0,  1.0], # 5 F
               [  1.0, -1.0,  1.0], # 6 G
               [ -1.0, -1.0,  1.0]] # 7 H

origin = [0.0,0.0,0.0]
foc    = 0.035 # in meters
fov    = 0.8575553107; # 49.1343 degrees
alt    = 5.0 # in meters
res    = 1024 # pixels
dpix   = (foc*tan(fov/2))/(res/2);

print '      CONFIGURATION:'
print '<========================>'
print 'origin: ' + str(origin)
print 'foc: ' + str(foc)
print 'fov: ' + str(fov)
print 'alt: ' + str(alt)
print 'res: ' + str(res)
print 'dpix: ' + str(dpix)
print '<========================>'

# we choose to rotate in the x/y-plane!
# based on carl_rot_cameras.txt convention
camera = [0.0,alt,0.0]
plane  = [0.0,alt-foc,0.0]

# the final guy
raw_match_set = []

def rotate_cube_z(r):
    for point in cube_points:
        x = point[0]
        y = point[1]
        z = point[2]
        point[0] = cos(r)*x + -1*sin(r)*y;
        point[1] = sin(r)*x + cos(r)*y;
        point[2] = z

def rotate_plane_z(r):
    x = plane[0]
    y = plane[1]
    z = plane[2]
    plane[0] = cos(r)*x + -1*sin(r)*y;
    plane[1] = sin(r)*x + cos(r)*y;
    plane[2] = z

def rotate_camera_z(r):
    x = camera[0]
    y = camera[1]
    z = camera[2]
    camera[0] = cos(r)*x + -1*sin(r)*y;
    camera[1] = sin(r)*x + cos(r)*y;
    camera[2] = z

def rotate_points_z(x_0,y_0,z_0,r):
    x = cos(r)*x_0 + -1*sin(r)*y_0;
    y = sin(r)*x_0 + cos(r)*y_0;
    z = z_0
    return [x,y,z]
    
    #########################
    ###### Entry Point ######
    #########################

print 'computing projections...'

new_match_set = []
match_count = 0
for point in cube_points:
    # compute vectors for each point pair:
    v_x = point[0] - camera[0]
    v_y = point[1] - camera[1]
    v_z = point[2] - camera[2]
    # compute the parametric variable's intersection with the y-plane
    t = ((alt-foc) - point[1])/v_y
    # compute the points!
    x = v_x * t + point[0]
    y = v_y * t + point[1] # redundant, just for readability
    z = v_z * t + point[2]
    if (verbose):
        print 'projected point: ' + str(point) + '\t --> ' + '[' + str(x) + ',' + str(y) + ',' + str(z) + ']'
    # scale the projected point with dpix
    x = x/dpix + res/2.0
    y = y/dpix + res/2.0
    z = z/dpix + res/2.0
    if (verbose):
        print 'scaled: ' + '[' + str(x) + ',' + str(y) + ',' + str(z) + ']' + ' => ' + str(match_count)
    new_match_set.append([x,y,z,match_count])
    match_count += 1

    
    #print point

raw_match_set.append(new_match_set)

# rotate by 45 degrees
rotate_cube_z(radians(-10))

# do the thing again
new_match_set = []
match_count = 0
for point in cube_points:
    # compute vectors for each point pair:
    v_x = point[0] - camera[0]
    v_y = point[1] - camera[1]
    v_z = point[2] - camera[2]
    # compute the parametric variable's intersection with the y-plane
    t = ((alt-foc) - point[1])/v_y
    # compute the points!
    x = v_x * t + point[0]
    y = v_y * t + point[1] # redundant, just for readability
    z = v_z * t + point[2]
    if (verbose):
        print 'projected point: ' + str(point) + '\t --> ' + '[' + str(x) + ',' + str(y) + ',' + str(z) + ']'
    # scale the projected point with dpix
    x = x/dpix + res/2.0
    y = y/dpix + res/2.0
    z = z/dpix + res/2.0
    if (verbose):
        print 'scaled: ' + '[' + str(x) + ',' + str(y) + ',' + str(z) + ']' + ' => ' + str(match_count)
    new_match_set.append([x,y,z,match_count])
    match_count += 1
    #print point

raw_match_set.append(new_match_set)

#print raw_match_set
f = open('matches.txt', 'w')
f.write(str(len(raw_match_set[0])-1) + '\n')
for i in range(0, len(raw_match_set[0])):
    format_str = '0001.jpg,0002.jpg,'
    format_str += str(raw_match_set[0][i][0]) + ',' + str(raw_match_set[0][i][2]) + ','
    format_str += str(raw_match_set[1][i][0]) + ',' + str(raw_match_set[1][i][2]) + ','
    format_str += '0,0,255\n'
    f.write(format_str)
    if (verbose):
        print format_str


#############################
### cameras
#############################
        
f = open('cameras.txt', 'w')
if verbose:
    print '1,' + str(camera[0]) + ',' + str(camera[1]) + ',' + str(camera[2]) + ',0.0,1.0,0.0\n'
f.write('1,' + str(camera[0]) + ',' + str(camera[1]) + ',' + str(camera[2]) + ',0.0,1.0,0.0\n')


rotate_camera_z(radians(-10))
x_u = abs(cos(radians(-10)))
y_u = abs(sin(radians(-10)))

if verbose:
    print '2,' + str(camera[0]) + ',' + str(camera[1]) + ',' + str(camera[2]) + ',' + str(y_u) + ',' + str(x_u) + ',0.0\n'
f.write('2,' + str(camera[0]) + ',' + str(camera[1]) + ',' + str(camera[2]) + ',' + str(y_u) + ',' + str(x_u) + ',0.0\n')

#if verbose:
#    print '2,' + str(camera[0]) + ',' + str(camera[1]) + ',' + str(camera[2]) + ',0.0,-1.0,0.0\n'
#f.write('2,' + str(camera[0]) + ',' + str(camera[1]) + ',' + str(camera[2]) + ',0.0,-1.0,0.0\n')

