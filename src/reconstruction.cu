#include "common_includes.h"
#include "octree.cuh"




int main(){

  //0. find mins and maxs {minX,minY,minZ} {maxX, maxY, maxZ}
  //1. find keys
  //2. sort keys, points, normals, and device_launch_parameters
  //3. compact the keys

  int depth = 10;//this number is a placeholder
  Octree octree = Octree("../data/carl.ply", depth);
  //octree.findMinMax();
  //octree.executeKeyRetrieval();

  return 0;
}
