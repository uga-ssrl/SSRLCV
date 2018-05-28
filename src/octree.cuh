#ifndef OCTREE_CUH
#define OCTREE_CUH

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

//TODO current max depth is 10 -> enable finer if desired

struct Vertex{
  float3 coord;
  int nodes[8];
  int depth;

  __device__ __host__ Vertex();

};

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
struct Edge{
  float3 p1;
  float3 p2;
  int depth;
  int nodes[4];

  __device__ __host__ Edge();

};

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

//prints the bits of any data type
__device__ __host__ void printBits(size_t const size, void const * const ptr);

//gets the keys of each node in a top down manor
__global__ void getNodeKeys(float3* points, float3* nodeCenters,int* nodeKeys, float3 c, float W, int N, int D);

//following methods are used to fill in the node array in a top down manor
__global__ void findAllNodes(int numUniqueNodes, int* nodeNumbers, Node* uniqueNodes);
void calculateNodeAddresses(dim3 grid, dim3 block,int numUniqueNodes, Node* uniqueNodes, int* nodeAddressesDevice, int* nodeNumbersDevice);
__global__ void fillBlankNodeArray(Node* uniqueNodes, int* nodeNumbers, int* nodeAddresses, Node* outputNodeArray, int numUniqueNodes, int currentDepth);
__global__ void fillFinestNodeArrayWithUniques(Node* uniqueNodes, int* nodeAddresses, Node* outputNodeArray, int numUniqueNodes, int* pointNodeIndex);
__global__ void fillNodeArrayWithUniques(Node* uniqueNodes, int* nodeAddresses, Node* outputNodeArray, Node* childNodeArray ,int numUniqueNodes);
__global__ void generateParentalUniqueNodes(Node* uniqueNodes, Node* nodeArrayD, int numNodesAtDepth, float totalWidth);

//finds all of the neighbors of each node in a bottom up manor
__global__ void computeNeighboringNodes(Node* nodeArray, int numNodes, int depthIndex, int* parentLUT, int* childLUT, int* numNeighbors, int childDepthIndex);

//fills in the vertex array
__global__ void findVertexOwners(Node* nodeArray, int numNodes, int depthIndex, int* vertexLUT, int* numVertices, int* ownerInidices, int* vertexPlacement);
__global__ void fillUniqueVertexArray(Node* nodeArray, Vertex* vertexArray, int numVertices, int vertexIndex,int depthIndex, int depth, float width, int* vertexLUT, int* ownerInidices, int* vertexPlacement);

//fills in the edge array
__global__ void findEdgeOwners(Node* nodeArray, int numNodes, int depthIndex, int* edgeLUT, int* numEdges, int* ownerInidices, int* edgePlacement);
__global__ void fillUniqueEdgeArray(Node* nodeArray, Edge* edgeArray, int numEdges, int edgeIndex,int depthIndex, int depth, float width, int* edgeLUT, int* ownerInidices, int* edgePlacement);

//fills in the face array
__global__ void findFaceOwners(Node* nodeArray, int numNodes, int depthIndex, int* faceLUT, int* numFaces, int* ownerInidices, int* facePlacement);
__global__ void fillUniqueFaceArray(Node* nodeArray, Face* faceArray, int numFaces, int faceIndex,int depthIndex, int depth, float width, int* faceLUT, int* ownerInidices, int* facePlacement);

struct Octree{

  //global variables
  float3* points;
  float3* pointsDevice;
  float3* normals;
  float3* normalsDevice;
  uchar3* colors;
  uchar3* colorsDevice;
  float3 center;
  int numPoints;
  float3 min;
  float3 max;
  float width;
  int depth;

  //the rest of the variables are allocated in methods where they are first used

  //prerequisite variables
  int numFinestUniqueNodes;
  float3* finestNodeCenters;
  int* finestNodePointIndexes;
  int* finestNodeKeys;
  Node* uniqueNodesAtFinestLevel;
  float3* finestNodeCentersDevice;
  int* finestNodePointIndexesDevice;
  int* finestNodeKeysDevice;

  //final octree structures which are all allocated in array generation methods
  int totalNodes;
  Node* finalNodeArray;
  Node* finalNodeArrayDevice;
  int* depthIndex;
  int* pointNodeIndex;
  int totalVertices;
  Vertex* vertexArray;
  Vertex* vertexArrayDevice;
  int totalEdges;
  Edge* edgeArray;
  Edge* edgeArrayDevice;
  int totalFaces;
  Face* faceArray;
  Face* faceArrayDevice;

  /*
  THESE ARE THE LOOK UP TABLES USED IN NEIGHBORHOOD, VERTEX ARRAY,
  EDGE ARRAY, and FACE ARRAY GENERATION (indirect pointers)
  ***device versions destroyed after being used***
  TODO make these constant cuda variables as they are never written to
  */
  int parentLUT[8][27];
  int* parentLUTDevice;
  int childLUT[8][27];
  int* childLUTDevice;
  //these LUTS do not include neighbor 13 (nodes relation to itself)
  //all indirect pointers calculated by hand
  int vertexLUT[8][7];
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

  //TODO put something in these methods, atleast memory freeing in ~Octree()
  Octree();
  ~Octree();

  //TODO remove normals as they will be calculated
  void parsePLY(std::string pathToFile);

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
  void sortByKey();
  void compactData();//also instantiates nodeNumbers and nodeAddresses
  void fillUniqueNodesAtFinestLevel();

  /*
  FILL OCTREE METHODS
  */
  void createFinalNodeArray();//also allocates/copies deviceIndices
  void printLUTs();
  void fillLUTs();//aslo allocates the device versions
  void fillNeighborhoods();
  void checkForGeneralNodeErrors();
  void computeVertexArray();
  void computeEdgeArray();
  void computeFaceArray();

  /*
  NORMAL CALCULATION METHODS
  */
  //TODO implement this part of the pipeline
  //currently using plys that have normals
  void computeNormals();

  };

#endif /* OCTREE_CUH */
