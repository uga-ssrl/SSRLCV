import sys
import math
import numpy as np
from plyfile import PlyData, PlyElement
# import pymesh
#from pymesh import stl  # Import module
#from pymesh import obj  # Import module
#from pymesh import ply  # Import module
#from osgeo import gdal


def main(inputfile):

    # get the user inputs
    if (len(sys.argv) < 8):
        print("NOT ENOUGH ARGUMENTS")
        print("USAGE:")
        print("\npython3 file.ply modPointCloud.py scale rotate_x rotate_y rotate_z translate_x translate_y translate_z\n")
        print("\t <> file.ply     -- the PLY file to modify")
        print("\t <> scale        -- scales all points in point cloud by this ammount")
        print("\t <> rotate_x     -- in degrees, rotates all points around x axis this ammount")
        print("\t <> rotate_y     -- in degrees, rotates all points around y axis this ammount")
        print("\t <> rotate_z     -- in degrees, rotates all points around z axis this ammount")
        print("\t <> translate_x  -- translates all points on x axis this ammount")
        print("\t <> translate_y  -- translates all points on y axis this ammount")
        print("\t <> translate_z  -- translates all points on z axis this ammount")
        exit(-1)

    print("Reading: " + str(inputfile))
    plydata = PlyData.read(str(inputfile))

    scale = float(sys.argv[2])
    r_x   = float(sys.argv[3]) * (math.pi / 180.0 )
    r_y   = float(sys.argv[4]) * (math.pi / 180.0 )
    r_z   = float(sys.argv[5]) * (math.pi / 180.0 )
    t_x   = float(sys.argv[6])
    t_y   = float(sys.argv[7])
    t_z   = float(sys.argv[8])

    # scale points
    if (scale > 0):
        for point in plydata.elements[0].data:
            point[0] = point[0] * scale
            point[1] = point[1] * scale
            point[2] = point[2] * scale

    # rotate points x
    if ( r_x != 0):
        for point in plydata.elements[0].data:
            point[0] = point[0]
            point[1] = point[1] * math.cos(r_x) + point[2] * math.sin(r_x)
            point[2] = point[1] * -math.sin(r_x) + point[2] * math.cos(r_x)

    # rotate points y
    if ( r_y != 0):
        for point in plydata.elements[0].data:
            point[0] = point[0] * math.cos(r_y) - point[2] * math.sin(r_y)
            point[1] = point[1]
            point[2] = point[0] * math.sin(r_y) + point[2] * math.cos(r_y)

    # rotate points z
    if ( r_z != 0):
        for point in plydata.elements[0].data:
            point[0] = point[0] * math.cos(r_z) + point[1] * math.sin(r_z)
            point[1] = point[0] * -math.sin(r_z) + point[1] * math.cos(r_z)
            point[2] = point[2]

    # translate points
    if ( t_x != 0 or t_y != 0 or t_z != 0 ):
        for point in plydata.elements[0].data:
            point[0] += t_x
            point[1] += t_y
            point[2] += t_z

    PlyData(plydata.elements, text=True).write('modified.ply')


if __name__ == "__main__":
    main(sys.argv[1])
