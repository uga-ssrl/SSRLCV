#ifndef OCTREE_H
#define OCTREE_H

#include "common_includes.h"
#include <thrust/sort.h>
#include <thrust/pair.h>
#include <thrust/gather.h>


__global__ void getKeys(float3* points, float3* centers, int* keys, float3 c, float W, int N, int D);


struct Octree{

  float3* points;
  float3* centers;
  float3* normals;
  float3 center;
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
  void parsePLY(std::string pathToFile);
  Octree(std::string pathToFile, int depth);
  void findMinMax();
  void allocateDeviceVariables();
  void executeKeyRetrieval();
  void cudaFreeMemory();

};

#endif /* OCTREE_H */
