#ifndef OCTREE_H
#define OCTREE_H

#include "common_includes.h"
#include <thrust/sort.h>
#include <thrust/pair.h>
#include <thrust/unique.h>
#include <thrust/gather.h>
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>




struct Node{
  int pointIndex;
  float3 center;
  int key;
  int numPoints;

};


__global__ void getNodeKeys(float3* points, float3* nodeCenters,int* nodeKeys, float3 c, float W, int N, int D);
__global__ void findAllNodes(int numUniqueNodes, int* nodeNumbers, Node* uniqueNodes);
__global__ void fill1NodeArray(Node* uniqueNodes, int* nodeAddresses, Node* finalNodeArray, int numUniqueNodes);


struct Octree{

  float3* points;
  float3* normals;
  float3 center;

  int numUniqueNodes;
  int totalNodes;
  float3* nodeCenters;
  int* nodePointIndexes;
  int* nodeKeys;
  Node* uniqueNodeArray;
  Node* finalNodeArray;


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
  Node* finalNodeArrayDevice;

  Octree();
  ~Octree();
  //parsPLY parse a ply and initializes all variables of an octree
  void parsePLY(std::string pathToFile);
  Octree(std::string pathToFile, int depth);

  void copyPointsToDevice();
  void copyPointsToHost();
  void copyNormalsToDevice();
  void copyNormalsToHost();
  void copyNodeCentersToDevice();
  void copyNodeCentersToHost();
  void copyNodeKeysToDevice();
  void copyNodeKeysToHost();
  void copyNodePointIndexesToDevice();
  void copyNodePointIndexesToHost();
  void copyFinalNodeArrayToDevice();
  void copyFinalNodeArrayToHost();

  void executeKeyRetrieval(dim3 grid, dim3 block);
  void sortByKey();
  void compactData();//also instantiates nodeNumbers and nodeAddresses
  void fillInUniqueNodes();
  void executeFindAllNodes(dim3 grid, dim3 block, int numUniqueNodes, int* nodeNumbers, Node* uniqueNodes);


};

#endif /* OCTREE_H */

//TODO complete implementation default constructor and destructor
