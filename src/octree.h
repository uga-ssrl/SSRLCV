#ifndef OCTREE_H
#define OCTREE_H

#include "common_includes.h"
#include <thrust/sort.h>
#include <thrust/pair.h>
#include <thrust/gather.h>

struct Octree{
  float3* points;
  float3* centers;
  float3* normals;
  int* keys;

  int numPoints;
  float3 min;
  float3 max;
  float width;
  int depth;

  float3* pointsDevice;
  float3* centersDevice;
  float3* normalsDevice;
  int* keysDevice;

  Octree();
  Octree(float3* points, float3* normals, int numPoints, int depth);
  void findMinMax();
  void allocateDeviceVariables();
  void executeKeyRetrieval();

  void cudaFreeMemory();


};

#endif /* OCTREE_H */
