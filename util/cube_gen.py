#!/usr/bin/python

import sys, os
from math import sin, cos, tan, sqrt, floor, radians, pi

## flags and shit
verbose = False

# sample cube points
cube_points = [[ -1.0,  1.0, -1.0], # 0 A
               [  1.0,  1.0, -1.0], # 1 B
               [  1.0, -1.0, -1.0], # 2 C
               [ -1.0, -1.0, -1.0], # 3 D
               [ -1.0,  1.0,  1.0], # 4 E
               [  1.0,  1.0,  1.0], # 5 F
               [  1.0, -1.0,  1.0], # 6 G
               [ -1.0, -1.0,  1.0], # 7 H
               [  0.0,  0.0,  0.0]] # 8 Origin Test Point

# sample "dotted" line
line_points = [[-1.0, -1.0, -1.0], # 0
               [-0.8, -0.8, -0.8],
               [-0.6, -0.6, -0.6], # 2
               [-0.4, -0.4, -0.4],
               [-0.2, -0.2, -0.2], # 4
               [ 0.0,  0.0,  0.0],
               [ 0.2,  0.2,  0.2], # 6
               [ 0.4,  0.4,  0.4],
               [ 0.6,  0.6,  0.6], # 8
               [ 0.8,  0.8,  0.8],
               [ 1.0,  1.0,  1.0]] # 10

single = [[0.0,0.0,0.0]]

origin = [0.0,0.0,0.0]
foc    = 0.25
fov    = radians(15)
alt    = -20.0 # in meters
res    = 1024 # pixels
# cameras[currentKP.parentId].dpix.x = (cameras[currentKP.parentId].foc * tanf(cameras[currentKP.parentId].fov.x / 2.0f)) / (cameras[currentKP.parentId].size.x / 2.0f );
dpix   = (foc*tan(fov/2))/(res/2)

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
camera = [0.0,0.0,alt]
plane  = [0.0,0.0,alt+foc]

# the degrees to rotates
to_rotate = 45;

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

def rotate_camera_x(r):
    x = camera[0]
    y = camera[1]
    z = camera[2]
    camera[0] = x
    camera[1] = cos(r)*y - sin(r)*z
    camera[2] = sin(r)*y + cos(r)*z

def rotate_points_z(x_0,y_0,z_0,r):
    x = cos(r)*x_0 + -1*sin(r)*y_0;
    y = sin(r)*x_0 + cos(r)*y_0;
    z = z_0
    return [x,y,z]

def rotate_points_x(x_0,y_0,z_0,r):
    x = x_0
    y = cos(r)*y_0 - sin(r)*z_0
    z = sin(r)*y_0 + cos(r)*z_0
    return [x,y,z]

    #########################
    ###### Entry Point ######
    #########################

print 'computing projections...'

matches = []

#for point in cube_points:
for point in single:
    if (verbose):
        print 'Camera 1:'
    match = []
    # compute vectors for each point pair:
    v_x = point[0] - camera[0]
    v_y = point[1] - camera[1]
    v_z = point[2] - camera[2]
    # compute the parametric variable's intersection with the y-plane
    t = ((alt+foc) - point[2])/v_z
    # compute the points!
    x = v_x * t + point[0]
    y = v_y * t + point[1] # redundant, just for readability
    z = v_z * t + point[2]
    if (verbose):
        print '\tprojected point: ' + str(point) + '\t --> ' + '[' + str(x) + ',' + str(y) + ',' + str(z) + ']'
    # scale the projected point with dpix
    x = x/dpix + res/2.0
    y = y/dpix + res/2.0
    z = z/dpix + res/2.0
    if (verbose):
        print '\tscaled: ' + '[' + str(x) + ',' + str(y) + ',' + str(z) + ']'
    #
    match.append([x,y,z])

    if (verbose):
        print 'Camera 2:'
    #
    # now rotate the point and do it again
    #print point
    point = rotate_points_x(point[0],point[1],point[2],radians(-to_rotate))
    #print point
    # compute vectors for each point pair:
    v_x = point[0] - camera[0]
    v_y = point[1] - camera[1]
    v_z = point[2] - camera[2]
    # compute the parametric variable's intersection with the y-plane
    t = ((alt+foc) - point[2])/v_z
    # compute the points!
    x = v_x * t + point[0]
    y = v_y * t + point[1] # redundant, just for readability
    z = v_z * t + point[2]
    if (verbose):
        print '\tprojected point: ' + str(point) + '\t --> ' + '[' + str(x) + ',' + str(y) + ',' + str(z) + ']'
    # scale the projected point with dpix
    x = x/dpix + res/2.0
    y = y/dpix + res/2.0
    z = z/dpix + res/2.0
    if (verbose):
        print '\tscaled: ' + '[' + str(x) + ',' + str(y) + ',' + str(z) + ']'
    #
    match.append([x,y,z])
    matches.append(match)
    print ''


print 'Cube Match Attempt: '
for match in matches:
    match_str  = str(match[0][0]) + ',' + str(match[0][1]) + '\t'
    match_str += str(match[1][0]) + ',' + str(match[1][1])
    print match_str

# TODO make it so we can save this as a matches guy
print 'For Copy Paste into Tester: '
match_num = 0
for match in matches:
    print 'matches->host[' + str(match_num) + '].keyPoints[0].parentId = 0;'
    print 'matches->host[' + str(match_num) + '].keyPoints[1].parentId = 1;'
    print 'matches->host[' + str(match_num) + '].keyPoints[0].loc = {' + str(match[0][0]) + ',' + str(match[0][1]) + '};'
    print 'matches->host[' + str(match_num) + '].keyPoints[1].loc = {' + str(match[1][0]) + ',' + str(match[1][1]) + '};'
    match_num += 1
print ''

#print raw_match_set
# print matches
# f = open('matches.txt', 'w')
# f.write(str(len(matches)-1) + '\n')
# for match in matches:
#     print match
#     format_str = '0001.jpg,0002.jpg,'
#     format_str += str(match[0][0]) + ',' + str(match[0][2]) + ','
#     format_str += str(match[1][0]) + ',' + str(match[1][2]) + ','
#     format_str += '150,0,255\n'
#     f.write(format_str)
#     if (verbose):
#         print format_str


#############################
### cameras
#############################

print 'For Copy Past into Tester: \n'
print 'images_vec[0]->id = 0;'
print 'images_vec[0]->camera.size = {' + str(res) + ',' + str(res) + '};'
print 'images_vec[0]->camera.cam_pos = {' + str(camera[0]) + ',' + str(camera[1]) + ',' + str(camera[2]) + '};'
# print 'images_vec[0]->camera.cam_rot = {' + str(radians(180)) + ', 0.0, 0.0};'
print 'images_vec[0]->camera.cam_rot = {0.0, 0.0, 0.0};'
print 'images_vec[0]->camera.fov = {' + str(fov) + ',' + str(fov) + '};'
print 'images_vec[0]->camera.foc = ' + str(foc) + ';'
rotate_camera_x(radians(to_rotate))
# rotate_camera_x(radians(to_rotate))
print 'images_vec[1]->id = 1;'
print 'images_vec[1]->camera.size = {' + str(res) + ',' + str(res) + '};'
print 'images_vec[1]->camera.cam_pos = {' + str(camera[0]) + ',' + str(camera[1]) + ',' + str(camera[2]) + '};'
# print 'images_vec[1]->camera.cam_rot = {' + str(radians(180 + to_rotate)) + ', 0.0, 0.0};'
print 'images_vec[1]->camera.cam_rot = {' + str(radians(to_rotate)) + ', 0.0, 0.0};'
print 'images_vec[1]->camera.fov = {' + str(fov) + ',' + str(fov) + '};'
print 'images_vec[1]->camera.foc = ' + str(foc) + ';'


# f = open('cameras.txt', 'w')
# if verbose:
#     print '1,' + str(camera[0]) + ',' + str(camera[1]) + ',' + str(camera[2]) + ',0.0,1.0,0.0\n'
# f.write('1,' + str(camera[0]) + ',' + str(camera[1]) + ',' + str(camera[2]) + ',0.0,1.0,0.0\n')
#
# rotate_camera_z(radians(-10))
# # needed to be abs for some reason... I'm so good at math
# x_u = abs(cos(radians(-10)))
# y_u = abs(sin(radians(-10)))
#
# if verbose:
#     print '2,' + str(camera[0]) + ',' + str(camera[1]) + ',' + str(camera[2]) + ',' + str(y_u) + ',' + str(x_u) + ',0.0\n'
# f.write('2,' + str(camera[0]) + ',' + str(camera[1]) + ',' + str(camera[2]) + ',' + str(y_u) + ',' + str(x_u) + ',0.0\n')
