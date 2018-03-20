#ifndef OCTREE_H
#define OCTREE_H

#include "common_includes.h"
#include <thrust/sort.h>
#include <thrust/pair.h>
#include <thrust/unique.h>
#include <thrust/gather.h>
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
#include <thrust/scan.h>
#include <thrust/device_ptr.h>

//current max depth is 10

struct Node{
  int pointIndex;
  float3 center;
  int key;
  int numPoints;
  int parent;
  int children[8];
  int neighbors[27];
  int edges[12];
  int vertices[8];
  int faces[6];
};

/*
HELPER METHODS AND CUDA KERNELS
*/
__global__ void getNodeKeys(float3* points, float3* nodeCenters,int* nodeKeys, float3 c, float W, int N, int D);
__global__ void findAllNodes(int numUniqueNodes, int* nodeNumbers, int* uniqueNodeKeys);
int* calculateNodeAddresses(int numUniqueNodes, int* uniqueNodeKeys);
__global__ void fillBlankNodeArray(Node* uniqueNodes, int* nodeAddresses, Node* outputNodeArray, int numUniqueNodes);
__global__ void fillNodeArrayWithUniques(Node* uniqueNodes, int* nodeAddresses, Node* outputNodeArray, int numUniqueNodes);
Node* createParentNodeArray();//not implemented


struct Octree{

  float3* points;
  float3* normals;
  float3 center;

  int numFinestUniqueNodes;
  int totalNodes;

  float3* finestNodeCenters;
  int* finestNodePointIndexes;
  int* finestNodeKeys;

  Node* uniqueNodesAtFinestLevel;

  int numPoints;
  float3 min;
  float3 max;
  float width;
  int depth;

  float3* pointsDevice;
  float3* normalsDevice;

  float3* finestNodeCentersDevice;
  int* finestNodePointIndexesDevice;
  int* finestNodeKeysDevice;

  Node* finalNodeArray;
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

  void copyFinestNodeCentersToDevice();
  void copyFinestNodeCentersToHost();
  void copyFinestNodeKeysToDevice();
  void copyFinestNodeKeysToHost();
  void copyFinestNodePointIndexesToDevice();
  void copyFinestNodePointIndexesToHost();

  void executeKeyRetrieval(dim3 grid, dim3 block);
  void sortByKey();
  void compactData();//also instantiates nodeNumbers and nodeAddresses

  void fillUniqueNodesAtFinestLevel();
  void createFinalNodeArray();

  };

#endif /* OCTREE_H */

//TODO complete implementation default constructor and destructor
