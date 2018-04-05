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
  float3 center;//may not be necessary
  int key;
  int numPoints;
  int parent;
  int children[8];//currently for depth + 1 nodeArray
  int neighbors[27];
  int edges[12];
  int vertices[8];
  int faces[6];
};

/*
HELPER METHODS AND CUDA KERNELS
that do not have return types, they
alter parameters
*/
__global__ void getNodeKeys(float3* points, float3* nodeCenters,int* nodeKeys, float3 c, float W, int N, int D);
__global__ void findAllNodes(int numUniqueNodes, int* nodeNumbers, Node* uniqueNodes);
void calculateNodeAddresses(int numUniqueNodes, Node* uniqueNodes, int* nodeAddressesDevice);
__global__ void fillBlankNodeArray(Node* uniqueNodes, int* nodeAddresses, Node* outputNodeArray, int numUniqueNodes);
__global__ void fillFinestNodeArrayWithUniques(Node* uniqueNodes, int* nodeAddresses, Node* outputNodeArray, int numUniqueNodes, int* pointNodeIndex);
__global__ void fillNodeArrayWithUniques(Node* uniqueNodes, int* nodeAddresses, Node* outputNodeArray, int numUniqueNodes);
__global__ void generateParentalUniqueNodes(Node* uniqueNodes, Node* nodeArrayD, int numNodesAtDepth);
__global__ void computeNeighboringNodes(Node* nodeArray, int numNodes, int* parentLUT, int* childLUT, int* depthIndex, int depth);

struct Octree{

  int* pointNodeIndex;
  float3* points;
  float3* pointsDevice;
  float3* normals;
  float3* normalsDevice;
  float3 center;//where is this used?
  int numPoints;
  float3 min;
  float3 max;
  float width;
  int depth;

  int numFinestUniqueNodes;
  float3* finestNodeCenters;
  int* finestNodePointIndexes;
  int* finestNodeKeys;
  Node* uniqueNodesAtFinestLevel;
  float3* finestNodeCentersDevice;
  int* finestNodePointIndexesDevice;
  int* finestNodeKeysDevice;

  int totalNodes;
  Node* finalNodeArray;
  Node* finalNodeArrayDevice;

  int* depthIndex;
  int* depthIndexDevice;
  int parentLUT[8][27];//indirect pointers
  int* parentLUTDevice;//only need to be copied 1 time (read only)
  int childLUT[8][27];//indirect pointers
  int* childLUTDevice;//only need to be copied 1 time (read only)

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

  void createFinalNodeArray();//also allocates/copies deviceIndices
  void fillLUTs();//aslo allocates the device versions
  void printLUTs();
  void fillNeighborhoods();
  void computeVertexArray();
  void computeEdgeArray();
  void computeFaceArray();

  };

#endif /* OCTREE_H */

//TODO complete implementation default constructor and destructor
