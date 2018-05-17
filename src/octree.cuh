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
#include <thrust/copy.h>

//current max depth is 10

/*
edge = [vertex1, vertex2]
  0 = [0,1]
  1 = [0,2]
  2 = [1,3]
  3 = [2,3]
  //the following ordering was assumed, may need to be revised
  4 = [0,4]
  5 = [1,5]
  6 = [2,6]
  7 = [3,7]
  8 = [4,5]
  9 = [4,6]
  10 = [5,7]
  11 = [6,7]
*/

/*
face = [edge1, edge2, edge3, edge4]
  0 = [0,1,2,3]
  //the following ordering was assumed, may need to be revised
  1 = [0,4,5,8]
  2 = [1,4,6,9]
  3 = [2,5,7,10]
  4 = [3,6,7,11]
  5 = [8,9,10,11]
*/

/*
face = [vertex1, vertex2, vertex3, vertex4]
  0 = [0,1,2,3]
  //the following ordering was assumed, may need to be revised
  1 = [0,1,4,5]
  2 = [0,2,4,6]
  3 = [1,3,5,7]
  4 = [2,3,6,7]
  5 = [4,5,6,7]
*/

struct Vertex{
  float3 coord;
  int nodes[8];
  int depth;

  __device__ __host__ Vertex();

};

struct Edge{
  float3 p1;
  float3 p2;
  int depth;
  int nodes[4];

  __device__ __host__ Edge();

};

struct Face{
  float3 p1;
  float3 p2;
  float3 p3;
  float3 p4;
  int depth;
  int nodes[2];

  __device__ __host__ Face();

};

struct Node{
  int pointIndex;
  float3 center;
  int key;
  int numPoints;
  int depth;

  int parent;
  int children[8];
  int neighbors[27];

  int edges[12];
  int vertices[8];
  int faces[6];

  __device__ __host__ Node();
};

/*
HELPER METHODS AND CUDA KERNELS
that do not have return types, they
alter parameters
*/
__device__ __host__ void printBits(size_t const size, void const * const ptr);
__global__ void getNodeKeys(float3* points, float3* nodeCenters,int* nodeKeys, float3 c, float W, int N, int D);
__global__ void findAllNodes(int numUniqueNodes, int* nodeNumbers, Node* uniqueNodes);

void calculateNodeAddresses(int numUniqueNodes, Node* uniqueNodes, int* nodeAddressesDevice, int* nodeNumbersDevice);
__global__ void fillBlankNodeArray(Node* uniqueNodes, int* nodeNumbers, int* nodeAddresses, Node* outputNodeArray, int numUniqueNodes, int currentDepth);
__global__ void fillFinestNodeArrayWithUniques(Node* uniqueNodes, int* nodeAddresses, Node* outputNodeArray, int numUniqueNodes, int* pointNodeIndex);
__global__ void fillNodeArrayWithUniques(Node* uniqueNodes, int* nodeAddresses, Node* outputNodeArray, Node* childNodeArray ,int numUniqueNodes);
__global__ void generateParentalUniqueNodes(Node* uniqueNodes, Node* nodeArrayD, int numNodesAtDepth, float totalWidth);

__global__ void computeNeighboringNodes(Node* nodeArray, int numNodes, int depthIndex, int* parentLUT, int* childLUT, int* numNeighbors, int childDepthIndex);

__global__ void findVertexOwners(Node* nodeArray, int numNodes, int depthIndex, int* vertexLUT, int* numVertices, int* ownerInidices, int* vertexPlacement);
__global__ void fillUniqueVertexArray(Node* nodeArray, Vertex* vertexArray, int numVertices, int vertexIndex,int depthIndex, int depth, float width, int* vertexLUT, int* ownerInidices, int* vertexPlacement);

__global__ void findEdgeOwners(Node* nodeArray, int numNodes, int depthIndex, int* edgeLUT, int* numEdges, int* ownerInidices, int* edgePlacement);
__global__ void fillUniqueEdgeArray(Node* nodeArray, Edge* edgeArray, int numEdges, int edgeIndex,int depthIndex, int depth, float width, int* edgeLUT, int* ownerInidices, int* edgePlacement);

__global__ void findFaceOwners(Node* nodeArray, int numNodes, int depthIndex, int* faceLUT, int* numFaces, int* ownerInidices, int* facePlacement);
__global__ void fillUniqueFaceArray(Node* nodeArray, Face* faceArray, int numFaces, int faceIndex,int depthIndex, int depth, float width, int* faceLUT, int* ownerInidices, int* facePlacement);

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

  //do not need these
  int numFinestUniqueNodes;
  float3* finestNodeCenters;
  int* finestNodePointIndexes;
  int* finestNodeKeys;
  Node* uniqueNodesAtFinestLevel;
  float3* finestNodeCentersDevice;
  int* finestNodePointIndexesDevice;
  int* finestNodeKeysDevice;

  //final octree structures
  int totalNodes;
  Node* finalNodeArray;
  Node* finalNodeArrayDevice;
  int totalVertices;
  Vertex* vertexArray;
  Vertex* vertexArrayDevice;
  int totalEdges;
  Edge* edgeArray;
  Edge* edgeArrayDevice;
  int totalFaces;
  Face* faceArray;
  Face* faceArrayDevice;

  int* depthIndex;
  int* depthIndexDevice;
  int parentLUT[8][27];//indirect pointers
  int* parentLUTDevice;//only need to be copied 1 time (read only)
  int childLUT[8][27];//indirect pointers
  int* childLUTDevice;//only need to be copied 1 time (read only)
  //these LUTS do not include neighbor 13 (nodes relation to itself)
  int vertexLUT[8][7];//indirect pointers
  int* vertexLUTDevice;//only need to be copied 1 time (read only)
  int edgeLUT[12][3]{
    {1,4,10},
    {3,4,12},
    {4,5,14},
    {4,7,16},
    {9,10,12},
    {10,11,14},
    {12,15,16},
    {14,16,17},
    {10,19,22},
    {12,21,22},
    {14,22,23},
    {16,22,25}
  };//indirect pointers calculated by hand
  int* edgeLUTDevice;//only need to be copied 1 time (read only)
  int faceLUT[6] = {4,10,12,14,16,22};//indirect pointers calculated by hand
  int* faceLUTDevice;//only need to be copied 1 time (read only)

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

  void freePrereqArrays();

  void createFinalNodeArray();//also allocates/copies deviceIndices
  void printLUTs();
  void fillLUTs();//aslo allocates the device versions
  void fillNeighborhoods();
  void computeVertexArray();
  void computeEdgeArray();
  void computeFaceArray();

  };

#endif /* OCTREE_H */

//TODO complete implementation default constructor and destructor
