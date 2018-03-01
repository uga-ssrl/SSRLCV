#include "common_includes.h"
#include "octree.h"

int main(){

  //0. find mins and maxs {minX,minY,minZ} {maxX, maxY, maxZ}
  //1. find keys
  //2. sort keys, points, normals, and device_launch_parameters
  //3. compact the keys

  int numPoints = 10000;//this number is a placeholder
  float3* points = new float3[numPoints];
  float3* normals = new float3[numPoints];

  int depth = 10;//this number is a placeholder
  Octree octree = Octree(points, normals, numPoints, depth);
  octree.findMinMax();
  octree.executeKeyRetrieval();

  return 0;
}
