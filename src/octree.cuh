#ifndef OCTREE_H
#define OCTREE_H

#include "common_includes.h"
#include <thrust/sort.h>
#include <thrust/pair.h>
#include <thrust/unique.h>
#include <thrust/gather.h>


__global__ void getKeys(float3* points, float3* nodeCenters, int* keys, float3 c, float W, int N, int D);


struct Octree{

  float3* points;
  float3* nodeCenters;
  int* nodeSortedPointIndexes;
  float3* normals;
  float3 center;
  int* keys;

  int numPoints;
  float3 min;
  float3 max;
  float width;
  int depth;

  float3* pointsDevice;
  float3* nodeCentersDevice;
  float3* normalsDevice;
  int* keysDevice;

  Octree();
  ~Octree();
  //parsPLY parse a ply and initializes all variables of an octree
  void parsePLY(std::string pathToFile);
  Octree(std::string pathToFile, int depth);
  void allocateDeviceVariables();
  void copyArraysHostToDevice(bool points, bool nodeCenters, bool normals, bool keys);
  void copyArraysDeviceToHost(bool points, bool nodeCenters, bool normals, bool keys);
  //node array
  void executeKeyRetrieval(dim3 grid, dim3 block);
  void sortByKey();
  void compactData();


  void cudaFreeMemory();


};

#endif /* OCTREE_H */

//TODO complete implementation default constructor and destructor
