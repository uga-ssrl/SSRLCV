#ifndef OCTREE_H
#define OCTREE_H

#include "common_includes.h"
#include <thrust/sort.h>
#include <thrust/pair.h>
#include <thrust/unique.h>
#include <thrust/gather.h>
#include <thrust/scan.h>



__global__ void getNodeKeys(float3* points, float3* nodeCenters,int* nodeKeys,
  float3 c, float W, int N, int D);
__global__ void findAllNodes(int numNodes, int* nodeNumbers, int* nodeKeys);

struct Octree{

  float3* points;
  float3* normals;
  float3 center;

  //nodeArray
  int numNodes;
  float3* nodeCenters;
  int* nodePointIndexes;
  int* nodeKeys;
  int* nodeNumbers;
  int* nodeAddresses;


  int numPoints;
  float3 min;
  float3 max;
  float width;
  int depth;

  float3* pointsDevice;
  float3* normalsDevice;

  float3* nodeCentersDevice;
  int* nodePointIndexesDevice;
  int* nodeKeysDevice;
  int* nodeNumbersDevice;
  int* nodeAddressesDevice;

  Octree();
  ~Octree();
  //parsPLY parse a ply and initializes all variables of an octree
  void parsePLY(std::string pathToFile);
  Octree(std::string pathToFile, int depth);
  void allocateDeviceVariablesNODES(bool afterNodeDetermination);
  void allocateDeviceVariablesGENERAL();
  void copyArraysHostToDeviceNODES(bool transferNodeCenters, bool transferNodeKeys, bool transferNodePointIndexes, bool transferNodeNumbers, bool transferNodeAddresses);
  void copyArraysHostToDeviceGENERAL(bool transferPoints, bool transferNormals);
  void copyArraysDeviceToHostNODES(bool transferNodeCenters, bool transferNodeKeys, bool transferNodePointIndexes, bool transferNodeNumbers, bool transferNodeAddresses);
  void copyArraysDeviceToHostGENERAL(bool transferPoints, bool transferNormals);

  //node array
  void executeKeyRetrieval(dim3 grid, dim3 block);
  void sortByKey();
  void compactData();//also instantiates nodeNumbers and nodeAddresses
  void executeFindAllNodes(dim3 grid, dim3 block);
  void inclusiveScanForNodeAddresses();


  void cudaFreeMemory();


};

#endif /* OCTREE_H */

//TODO complete implementation default constructor and destructor
