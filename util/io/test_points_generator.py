#!/usr/bin/python

import sys, os
from math import sin, cos, tan, sqrt, floor, radians, pi

## flags and shit
verbose = False

# sample cube points
cube_points = [[ -1.0,  1.0, -1.0 ], # 0 A
               [  1.0,  1.0, -1.0 ], # 1 B
               [  1.0, -1.0, -1.0 ], # 2 C
               [ -1.0, -1.0, -1.0 ], # 3 D
               [ -1.0,  1.0,  1.0 ], # 4 E
               [  1.0,  1.0,  1.0 ], # 5 F
               [  1.0, -1.0,  1.0 ], # 6 G
               [ -1.0, -1.0,  1.0 ], # 7 H
               [  0.0,  0.0,  0.0 ]] # 8 Origin Test Point

# sample "dotted" line
xyz_line_points = [[-1.0, -1.0, -1.0 ], # 0
                   [-0.8, -0.8, -0.8 ],
                   [-0.6, -0.6, -0.6 ], # 2
                   [-0.4, -0.4, -0.4 ],
                   [-0.2, -0.2, -0.2 ], # 4
                   [ 0.0,  0.0,  0.0 ],
                   [ 0.2,  0.2,  0.2 ], # 6
                   [ 0.4,  0.4,  0.4 ],
                   [ 0.6,  0.6,  0.6 ], # 8
                   [ 0.8,  0.8,  0.8 ],
                   [ 1.0,  1.0,  1.0 ]] # 10

y_line_points = [[ 0.0, -1.5, 0.0 ], # 0
                [ 0.0, -1.0, 0.0 ],
                [ 0.0, -0.5, 0.0 ], # 2
                [ 0.0,  0.0, 0.0 ],
                [ 0.0,  0.5, 0.0 ], # 4
                [ 0.0,  1.0, 0.0 ],
                [ 0.0,  1.5, 0.0 ]] # 6

three_4_5_points = [[ 0.0, -3.0, 0.0],
                    [ 0.0,  0.0, 0.0],
                    [ 0.0,  3.0, 0.0]]

square_points = [[ 1.0,  1.0,  0.0],
                 [-1.0, -1.0,  0.0],
                 [-1.0,  1.0,  0.0],
                 [ 1.0, -1.0,  0.0]]

single = [[0.0,0.0,0.0]]

origin = [0.0,0.0,0.0]
foc    = 0.16 # in meters
fov    = radians(3.4) #0.19933754453 #radians(10)
alt    = -400 # in meters
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
to_rotate   = 10
num_cameras = 2  # so we can switch the

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

def rotate_camera_x(r):
    x = camera[0]
    y = camera[1]
    z = camera[2]
    camera[0] = x
    camera[1] = cos(r)*y - sin(r)*z
    camera[2] = sin(r)*y + cos(r)*z

def rotate_camera_z(r):
    x = camera[0]
    y = camera[1]
    z = camera[2]
    camera[0] = cos(r)*x + -1*sin(r)*y;
    camera[1] = sin(r)*x + cos(r)*y;
    camera[2] = z

def rotate_points_x(x_0,y_0,z_0,r):
    x = x_0
    y = cos(r)*y_0 - sin(r)*z_0
    z = sin(r)*y_0 + cos(r)*z_0
    return [x,y,z]

def rotate_points_y(x_0,y_0,z_0,r):
    x = cos(r)*x_0 + sin(r)*z_0
    y = y
    z = -sin(r)*x_0 + cos(r)*z_0
    return [x,y,z]

def rotate_points_z(x_0,y_0,z_0,r):
    x = cos(r)*x_0 + -1*sin(r)*y_0;
    y = sin(r)*x_0 + cos(r)*y_0;
    z = z_0
    return [x,y,z]


    #########################
    ###### Entry Point ######
    #########################

print 'computing projections...'

matches = []

#for point in cube_points:
# for point in line_points:
for point in cube_points:
    match = []
    for asdf in range(0,num_cameras):
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
        #
        # now rotate the point and do it again
        #print point
        point = rotate_points_x(point[0],point[1],point[2],radians(to_rotate))
    matches.append(match)



print 'Cube Match Attempt: '
for match in matches:
    match_str  = str(match[0][0]) + ',' + str(match[0][1]) + '\t'
    match_str += str(match[1][0]) + ',' + str(match[1][1])
    print match_str


#############################
### matches
#############################

print 'For Copy Paste into Tester: \n'
if (num_cameras > 2): # N-view case
    print 'ssrlcv::MatchSet matchSet;'
    print 'matchSet.matches   = new ssrlcv::Unity<ssrlcv::MultiMatch>(nullptr,' + str(len(matches)) + ',ssrlcv::cpu);'
    print 'matchSet.keyPoints = new ssrlcv::Unity<ssrlcv::KeyPoint>(nullptr,' + str(len(matches) * num_cameras) + ',ssrlcv::cpu);'
else: # 2-view case
    print 'ssrlcv::Match* matches_host = new ssrlcv::Match[' + str(len(matches)) + '];'
    print 'ssrlcv::Unity<ssrlcv::Match>* matches = new ssrlcv::Unity<ssrlcv::Match>(matches_host, ' + str(len(matches)) + ', ssrlcv::cpu);'

match_num = 0
for match in matches:
    if (num_cameras > 2): # only needed in N-view
        print 'matchSet.matches->host[' + str(match_num) + '] = {' + str(num_cameras) + ',' + str(match_num * num_cameras) + '};'
    for cam in range(0,num_cameras):
        if (num_cameras > 2): # 2-view case
            print 'matchSet.keyPoints->host[' + str(match_num * num_cameras + cam) + '] = {{' + str(cam) + '},{' + str(match[cam][0]) + ',' + str(match[cam][1]) + '}};'
        else: # N-view case
            print 'matches->host[' + str(match_num) + '].keyPoints[' + str(cam) + '].parentId = ' + str(cam) + ';'
            print 'matches->host[' + str(match_num) + '].keyPoints[' + str(cam) +'].loc = {' + str(match[cam][0]) + ',' + str(match[cam][1]) + '};'
    match_num += 1
print ''

#############################
### cameras
#############################

print 'Copy And Paste Cameras: \n'

print 'std::vector<ssrlcv::Image*> images;'

for cam in range(0,num_cameras):
    print 'ssrlcv::Image* image' + str(cam) + ' = new ssrlcv::Image();'
    print 'images.push_back(image' + str(cam) + ');'

for cam in range(0,num_cameras):
    print 'images[' + str(cam) + ']->id = ' + str(cam) + ';'
    print 'images[' + str(cam) + ']->camera.size = {' + str(res) + ',' + str(res) + '};'
    print 'images[' + str(cam) + ']->camera.cam_pos = {' + str('{0:.12f}'.format(camera[0])) + ',' + str('{0:.12f}'.format(camera[1])) + ',' + str('{0:.12f}'.format(camera[2])) + '};'
    print 'images[' + str(cam) + ']->camera.cam_rot = {' + str(cam * radians(to_rotate)) + ', 0.0, 0.0};'
    print 'images[' + str(cam) + ']->camera.fov = {' + str(fov) + ',' + str(fov) + '};'
    print 'images[' + str(cam) + ']->camera.foc = ' + str('{0:.12f}'.format(foc)) + ';'
    rotate_camera_x(radians(to_rotate))






    #
