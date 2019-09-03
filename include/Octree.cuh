/** \file Octree.cuh
* \brief This file contains everything related to the CUDA Octree
*/
#ifndef OCTREE_CUH
#define OCTREE_CUH

#include "common_includes.h"
#include "cuda_util.cuh"
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
#include "Unity.cuh"

#include "tinyply.h"

namespace ssrlcv{
  /**
  * \brief data parallel octree
  *
  * \detail this class is used for quick near neighbor searches
  * as well as a tool for surface reconstruction
  * \todo update with new unity
  */
  class Octree{

  private:

    /*
    OCTREE GENERATION PREREQUISITE FUNCTIONS
    */
    void createFinestNodes();

    /*
    FILL OCTREE METHODS
    */
    void fillInCoarserDepths();//also allocates/copies deviceIndices
    void fillNeighborhoods();

    /*
    OCTREE UNIT TESTING
    */
    void checkForGeneralNodeErrors();

  public:

    /** \brief most basic part of octree*/
    struct Node{
      uchar3 color;
      int pointIndex;
      float3 center;
      float width;
      int key;
      int numPoints;
      int depth;
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
    /** \brief holds cubic vertex data */
    struct Vertex{
      uchar3 color;
      float3 coord;
      int nodes[8];
      int depth;
      __device__ __host__ Vertex();
    };
    /** \brief holds cubic edge data */
    struct Edge{
      uchar3 color;
      int v1;
      int v2;
      int depth;
      int nodes[4];
      __device__ __host__ Edge();
    };
    /** \brief hold cubic face data */
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

    std::string name;
    std::string pathToFile;

    float3 center;
    float3 min;
    float3 max;
    float width;
    int depth;

    Unity<float3>* points;

    Unity<float3>* normals;

    Unity<Node>* nodes;
    Unity<Vertex>* vertices;
    Unity<Edge>* edges;
    Unity<Face>* faces;

    //length = # points, value = node containing point
    //ie value = index of node point is in
    Unity<unsigned int>* pointNodeIndex;

    //depth index carriers
    Unity<unsigned int>* nodeDepthIndex;
    Unity<unsigned int>* vertexDepthIndex;
    Unity<unsigned int>* edgeDepthIndex;
    Unity<unsigned int>* faceDepthIndex;

    Octree();
    ~Octree();

    Octree(std::string pathToFile, int depth);
    Octree(int numfloat3s, float3* points, int depth, bool createVEF);
    Octree(int numfloat3s, float3* points, float deepestWidth, bool createVEF);

    Octree(Unity<float3>* points, int depth, bool createVEF);
    Octree(Unity<float3>* points, float deepestWidth, bool createVEF);


    void computeVertexArray();
    void computeEdgeArray();
    void computeFaceArray();
    void createVEFArrays();

    /*
    NORMAL CALCULATION METHODS
    */
    //TODO move this to MeshFactory
    void computeNormals(int minNeighForNorms, int maxNeighbors);
    void computeNormals(int minNeighForNorms, int maxNeighbors, unsigned int numCameras, float3* cameraPositions);

    /*
    PLY WRITERS
    */
    void writeVertexPLY(bool binary = false);
    void writeEdgePLY(bool binary = false);
    void writeCenterPLY(bool binary = false);
    void writepointsPLY(bool binary = false);
    void writeNormalPLY(bool binary = false);
    void writeDepthPLY(int d, bool binary = false);
  };

  /* CUDA variable, method and kernel defintions */

  namespace{
    struct is_not_neg{
      __host__ __device__
      bool operator()(const int x)
      {
        return (x >= 0);
      }
    };
  }

  extern __constant__ int3 coordPlacementIdentity[8];
  extern __constant__ int2 vertexEdgeIdentity[12];
  extern __constant__ int4 vertexFaceIdentity[6];
  extern __constant__ int4 edgeFaceIdentity[6];

  __device__ __host__ float3 getVoidCenter(const Octree::Node &node, int neighbor);
  __device__ __host__ float3 getVoidChildCenter(const Octree::Node &parent, int child);
  __device__ __forceinline__ int floatToOrderedInt(float floatVal);
  __device__ __forceinline__ float orderedIntToFloat(int intVal);
  //prints the bits of any data type
  __device__ __host__ void printBits(size_t const size, void const * const ptr);

  //gets the keys of each node in a top down manor
  __global__ void getNodeKeys(float3* points, float3* nodeCenters, int* nodeKeys, float3 c, float W, int numPoints, int D);

  //following methods are used to fill in the node array in a top down manor
  __global__ void findAllNodes(int numUniqueNodes, int* nodeNumbers, Octree::Node* uniqueNodes);
  void calculateNodeAddresses(dim3 grid, dim3 block,int numUniqueNodes, Octree::Node* uniqueNodes, int* nodeAddresses_device, int* nodeNumbers_device);
  __global__ void fillBlankNodeArray(Octree::Node* uniqueNodes, int* nodeNumbers, int* nodeAddresses, Octree::Node* outputNodeArray, int numUniqueNodes, int currentDepth, float totalWidth);
  __global__ void fillFinestNodeArrayWithUniques(Octree::Node* uniqueNodes, int* nodeAddresses, Octree::Node* outputNodeArray, int numUniqueNodes, unsigned int* pointNodeIndex);
  __global__ void fillNodeArrayWithUniques(Octree::Node* uniqueNodes, int* nodeAddresses, Octree::Node* outputNodeArray, Octree::Node* childNodeArray ,int numUniqueNodes);
  __global__ void generateParentalUniqueNodes(Octree::Node* uniqueNodes, Octree::Node* nodeArrayD, int numNodesAtDepth, float totalWidth);
  __global__ void computeNeighboringNodes(Octree::Node* nodeArray, int numNodes, int depthIndex, int* parentLUT, int* childLUT, int childDepthIndex);

  __global__ void findNormalNeighborsAndComputeCMatrix(int numNodesAtDepth, int depthIndex, int maxNeighbors, Octree::Node* nodeArray, float3* points, float* cMatrix, int* neighborIndices, int* numNeighbors);
  __global__ void transposeFloatMatrix(int m, int n, float* matrix);
  __global__ void setNormal(int currentPoint, float* vt, float3* normals);
  __global__ void checkForAbiguity(int numPoints, int numCameras, float3* normals, float3* points, float3* cameraPositions, bool* ambiguous);
  __global__ void reorient(int numNodesAtDepth, int depthIndex, Octree::Node* nodeArray, int* numNeighbors, int maxNeighbors, float3* normals, int* neighborIndices, bool* ambiguous);

  __global__ void findVertexOwners(Octree::Node* nodeArray, int numNodes, int depthIndex, int* vertexLUT, int* numVertices, int* ownerInidices, int* vertexPlacement);
  __global__ void fillUniqueVertexArray(Octree::Node* nodeArray, Octree::Vertex* vertexArray, int numVertices, int vertexIndex,int depthIndex, int depth, float width, int* vertexLUT, int* ownerInidices, int* vertexPlacement);
  __global__ void findEdgeOwners(Octree::Node* nodeArray, int numNodes, int depthIndex, int* edgeLUT, int* numEdges, int* ownerInidices, int* edgePlacement);
  __global__ void fillUniqueEdgeArray(Octree::Node* nodeArray, Octree::Edge* edgeArray, int numEdges, int edgeIndex,int depthIndex, int depth, float width, int* edgeLUT, int* ownerInidices, int* edgePlacement);
  __global__ void findFaceOwners(Octree::Node* nodeArray, int numNodes, int depthIndex, int* faceLUT, int* numFaces, int* ownerInidices, int* facePlacement);
  __global__ void fillUniqueFaceArray(Octree::Node* nodeArray, Octree::Face* faceArray, int numFaces, int faceIndex,int depthIndex, int depth, float width, int* faceLUT, int* ownerInidices, int* facePlacement);

}


#endif /* OCTREE_CUH */
