#ifndef OCTREE_CUH
#define OCTREE_CUH

#include "common_includes.h"
#include <thrust/sort.h>
#include <thrust/pair.h>
#include <thrust/unique.h>
#include <thrust/gather.h>
#include <thrust/iterator/counting_iterator.h>
#include <thrust/device_vector.h>
#include <thrust/scan.h>
#include <thrust/device_ptr.h>
#include <thrust/copy.h>
#include <cublas_v2.h>
#include <cusolverDn.h>

__constant__ int3 coordPlacementIdentity[8] = {
  {-1,-1,-1},
  {-1,-1,1},
  {-1,1,-1},
  {-1,1,1},
  {1,-1,-1},
  {1,-1,1},
  {1,1,-1},
  {1,1,1}
};

__constant__ int2 vertexEdgeIdentity[12] = {
  {0,1},
  {0,2},
  {1,3},
  {2,3},
  {0,4},
  {1,5},
  {2,6},
  {3,7},
  {4,5},
  {4,6},
  {5,7},
  {6,7}
};

__constant__ int4 vertexFaceIdentity[6] = {
  {0,1,2,3},
  {0,1,4,5},
  {0,2,4,6},
  {1,3,5,7},
  {2,3,6,7},
  {4,5,6,7}
};

__constant__ int4 edgeFaceIdentity[6] = {
  {0,1,2,3},
  {0,4,5,8},
  {1,4,6,9},
  {2,5,7,10},
  {3,6,7,11},
  {8,9,10,11}
};

struct Vertex{
  uchar3 color;
  float3 coord;
  int nodes[8];
  int depth;

  __device__ __host__ Vertex();

};

struct Edge{
  uchar3 color;
  int v1;
  int v2;
  int depth;
  int nodes[4];

  __device__ __host__ Edge();

};

struct Face{
  uchar3 color;
  int e1;
  int e2;
  int e3;
  int e4;
  int depth;
  int nodes[2];

  __device__ __host__ Face();

};

struct Node{
  uchar3 color;
  int pointIndex;
  float3 center;
  float width;
  int key;
  int numPoints;
  int depth;
  //TODO check this as it is set in generateParents
  int numFinestChildren;
  int finestChildIndex;

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
__device__ __forceinline__ int floatToOrderedInt(float floatVal);
__device__ __forceinline__ float orderedIntToFloat(int intVal);
//prints the bits of any data type
__device__ __host__ void printBits(size_t const size, void const * const ptr);

//gets the keys of each node in a top down manor
__global__ void getNodeKeys(float3* points, float3* nodeCenters,int* nodeKeys, float3 c, float W, int N, int D);

//following methods are used to fill in the node array in a top down manor
__global__ void findAllNodes(int numUniqueNodes, int* nodeNumbers, Node* uniqueNodes);
void calculateNodeAddresses(dim3 grid, dim3 block,int numUniqueNodes, Node* uniqueNodes, int* nodeAddressesDevice, int* nodeNumbersDevice);
__global__ void fillBlankNodeArray(Node* uniqueNodes, int* nodeNumbers, int* nodeAddresses, Node* outputNodeArray, int numUniqueNodes, int currentDepth, float totalWidth);
__global__ void fillFinestNodeArrayWithUniques(Node* uniqueNodes, int* nodeAddresses, Node* outputNodeArray, int numUniqueNodes, int* pointNodeIndex);
__global__ void fillNodeArrayWithUniques(Node* uniqueNodes, int* nodeAddresses, Node* outputNodeArray, Node* childNodeArray ,int numUniqueNodes);
__global__ void generateParentalUniqueNodes(Node* uniqueNodes, Node* nodeArrayD, int numNodesAtDepth, float totalWidth);
__global__ void computeNeighboringNodes(Node* nodeArray, int numNodes, int depthIndex, int* parentLUT, int* childLUT, int* numNeighbors, int childDepthIndex);

__global__ void findNormalNeighborsAndComputeCMatrix(int numNodesAtDepth, int depthIndex, int maxNeighbors, float maxDistance, Node* nodeArray, float3* points, float* cMatrix, int* neighborIndices, int* numNeighbors);
__global__ void transposeFloatMatrix(int m, int n, float* matrix);
__global__ void setNormal(int currentPoint, float* s, float* vt, float3* normals);
__global__ void checkForAbiguity(int numPoints, int numCameras, float3* normals, float3* points, float3* cameraPositions, bool* ambiguous);
__global__ void reorient(int numPoints, int* numNeighbors, int maxNeighbors, float3* normals, int* neighborIndices, bool* ambiguous, bool* ambiguityExists);

__global__ void findVertexOwners(Node* nodeArray, int numNodes, int depthIndex, int* vertexLUT, int* numVertices, int* ownerInidices, int* vertexPlacement);
__global__ void fillUniqueVertexArray(Node* nodeArray, Vertex* vertexArray, int numVertices, int vertexIndex,int depthIndex, int depth, float width, int* vertexLUT, int* ownerInidices, int* vertexPlacement);
__global__ void findEdgeOwners(Node* nodeArray, int numNodes, int depthIndex, int* edgeLUT, int* numEdges, int* ownerInidices, int* edgePlacement);
__global__ void fillUniqueEdgeArray(Node* nodeArray, Edge* edgeArray, int numEdges, int edgeIndex,int depthIndex, int depth, float width, int* edgeLUT, int* ownerInidices, int* edgePlacement);
__global__ void findFaceOwners(Node* nodeArray, int numNodes, int depthIndex, int* faceLUT, int* numFaces, int* ownerInidices, int* facePlacement);
__global__ void fillUniqueFaceArray(Node* nodeArray, Face* faceArray, int numFaces, int faceIndex,int depthIndex, int depth, float width, int* faceLUT, int* ownerInidices, int* facePlacement);

struct Octree{

  //global variables
  std::string pathToFile;
  bool normalsComputed;
  bool hasColor;
  bool simpleOctree;
  float3* points;
  float3* normals;
  uchar3* colors;
  float3 center;
  int numPoints;
  float3 min;
  float3 max;
  float width;
  int depth;

  //the rest of the variables are allocated in methods where they are first used

  //prerequisite variables - freed in freePrereqArrays()
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
  int* pointNodeIndex;
  int totalVertices;
  Vertex* vertexArray;
  int totalEdges;
  Edge* edgeArray;
  int totalFaces;
  Face* faceArray;

  /*
  THESE ARE DEVICE VARIABLES THAT ARE FREED AND ALLOCATED IN THEIR COPY METHODS
  */
  bool pointNodeDeviceReady;
  int* pointNodeIndexDevice;
  bool vertexArrayDeviceReady;
  Vertex* vertexArrayDevice;
  bool edgeArrayDeviceReady;
  Edge* edgeArrayDevice;
  bool faceArrayDeviceReady;
  Face* faceArrayDevice;
  bool pointsDeviceReady;
  float3* pointsDevice;
  bool normalsDeviceReady;
  float3* normalsDevice;
  bool colorsDeviceReady;
  uchar3* colorsDevice;

  /*
  THESE ARE THE LOOK UP TABLES USED IN NEIGHBORHOOD, VERTEX ARRAY,
  EDGE ARRAY, and FACE ARRAY GENERATION (indirect pointers)
  ***device versions destroyed after being used***
  (only needed once)
  TODO make these constant cuda variables as they are never written to
  TODO just make these flat to start with
  */
  int parentLUT[8][27];
  int* parentLUTDevice;
  int childLUT[8][27];
  int* childLUTDevice;
  int vertexLUT[8][7]{
    {0,1,3,4,9,10,12},
    {1,2,4,5,10,11,14},
    {3,4,6,7,12,15,16},
    {4,5,7,8,14,16,17},
    {9,10,12,18,19,21,22},
    {10,11,14,19,20,22,23},
    {12,15,16,21,22,24,25},
    {14,16,17,22,23,25,26}
  };
  int* vertexLUTDevice;
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
  };
  int* edgeLUTDevice;
  int faceLUT[6] = {4,10,12,14,16,22};
  int* faceLUTDevice;

  Octree();
  ~Octree();

  void parsePLY();
  Octree(std::string pathToFile, int depth);

  /*
  MEMORY OPERATIONS OF GLOBAL OCTREE VARIABLES (deleted when octree is destroyed)
  */
  void init_octree_gpu();
  void copyPointsToDevice();
  void copyPointsToHost();
  void copyNormalsToDevice();
  void copyNormalsToHost();
  void copyColorsToDevice();
  void copyColorsToHost();

  /*
  MEMORY OPERATIONS OF PREREQUISITE VARIABLES
  */
  void copyFinestNodeCentersToDevice();
  void copyFinestNodeCentersToHost();
  void copyFinestNodeKeysToDevice();
  void copyFinestNodeKeysToHost();
  void copyFinestNodePointIndexesToDevice();
  void copyFinestNodePointIndexesToHost();
  void freePrereqArrays();

  /*
  MEMORY OPERATIONS OF COMPUTED OCTREE VARIABLES
  */
  void copyNodesToDevice();
  void copyNodesToHost();

  void copyPointNodeIndexesToDevice();
  void copyPointNodeIndexesToHost();
  void copyVerticesToDevice();
  void copyVerticesToHost();
  void copyEdgesToDevice();
  void copyEdgesToHost();
  void copyFacesToDevice();
  void copyFacesToHost();

  /*
  OCTREE GENERATION PREREQUISITE FUNCTIONS
  */
  void generateKeys();
  void prepareFinestUniquNodes();

  /*
  FILL OCTREE METHODS
  */
  void createFinalNodeArray();//also allocates/copies deviceIndices
  void printLUTs();
  void fillLUTs();//aslo allocates the device versions
  void fillNeighborhoods();
  void computeVertexArray();
  void computeEdgeArray();
  void computeFaceArray();

  void checkForGeneralNodeErrors();

  /*
  NORMAL CALCULATION METHODS
  */
  //TODO implement this part of the pipeline
  //currently using plys that have normals
  void computeNormals(float neighborDistance);

  void writeVertexPLY();
  void writeEdgePLY();
  void writeCenterPLY();
  void writeNormalPLY();

};
#endif /* OCTREE_CUH */
