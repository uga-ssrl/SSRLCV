/** \file Octree.cuh
* \brief This file contains everything related to the CUDA Octree
*/
#pragma once
#ifndef OCTREE_CUH
#define OCTREE_CUH

#include "common_includes.hpp"
#include "fix_thrust_warning.h"
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

#include "tinyply.h"

namespace ssrlcv{
  /**
  * \ingroup trees
  */
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
      int pointIndex; // starting index of points
      float3 center;
      float width;
      int key;
      int numPoints; // numer of points from starting index
      int depth;
      int numFinestChildren;
      int finestChildIndex;
      int parent;
      int children[8];
      int neighbors[27]; // index of neighbor nodes
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

    ssrlcv::ptr::value<ssrlcv::Unity<float3>> points;

    ssrlcv::ptr::value<ssrlcv::Unity<float3>> normals;

    ssrlcv::ptr::value<ssrlcv::Unity<Node>> nodes;
    ssrlcv::ptr::value<ssrlcv::Unity<Vertex>> vertices;
    ssrlcv::ptr::value<ssrlcv::Unity<Edge>> edges;
    ssrlcv::ptr::value<ssrlcv::Unity<Face>> faces;

    //length = # points, value = node containing point
    //ie value = index of node point is in

    /** the index to the leaf nodes that the points are in (gauruneed to contain points)
     * the value at this index is the location of the leaf node in the node array for this points
     * e.g.
     * points->host.get()[5] has index 5 so look at 5
     * pointNodeIndex->host.get()[5] has value 1234
     * nodes->host.get()[1234] this is the leaf node that contains the point originally searched for
     */
    ssrlcv::ptr::value<ssrlcv::Unity<unsigned int>> pointNodeIndex;

    //depth index carriers
    ssrlcv::ptr::value<ssrlcv::Unity<unsigned int>> nodeDepthIndex;
    ssrlcv::ptr::value<ssrlcv::Unity<unsigned int>> vertexDepthIndex;
    ssrlcv::ptr::value<ssrlcv::Unity<unsigned int>> edgeDepthIndex;
    ssrlcv::ptr::value<ssrlcv::Unity<unsigned int>> faceDepthIndex;

    // =============================================================================================================
    //
    // Constructors and Destructors
    //
    // =============================================================================================================

    Octree();

    Octree(std::string pathToFile, int depth);
    Octree(int numfloat3s, ssrlcv::ptr::host<float3> points, int depth, bool createVEF);
    Octree(int numfloat3s, ssrlcv::ptr::host<float3> points, float deepestWidth, bool createVEF);

    Octree(ssrlcv::ptr::value<ssrlcv::Unity<float3>> points, int depth, bool createVEF);
    Octree(ssrlcv::ptr::value<ssrlcv::Unity<float3>> points, float deepestWidth, bool createVEF);

    // =============================================================================================================
    //
    // Octree Host Methods
    //
    // =============================================================================================================

    void computeVertexArray();
    void computeEdgeArray();
    void computeFaceArray();
    void createVEFArrays();

    // =============================================================================================================
    //
    // Filtering Methods
    //
    // =============================================================================================================

    /**
     * calculates the average distance from a point to N of it's neighbors
     * @param the numer of neighbors to consider
     * @return averages a unity float of the average distance for n neighbors per point
     */
    ssrlcv::ptr::value<ssrlcv::Unity<float>> averageNeighboorDistances(int n);

    /**
     * calculates the average distance from a point to N of it's neighbors and finds that average for all points
     * @param the numer of neighbors to consider
     * @return average the average distance from any given point to it's neighbors
     */
    float averageNeighboorDistance(int n);

    /**
     * finds the point indexes that should be removed this is done for each point.
     * returns a NULL index if the point does not need to be removed, returns the actual index if it does need to be
     * @param cutoff is the minimum average distance from a point to N of its neightbors
     * @param n the number of neighbor points to consider
     * @return points returns unity pf float3 points that are densly packed enough within the cutoff
     */
    ssrlcv::ptr::value<ssrlcv::Unity<float3>> removeLowDensityPoints(float cutoff, int n);

    // =============================================================================================================
    //
    // Normal Caclulation Methods
    //
    // =============================================================================================================

    /**
    * Computes normals for the points within the input points cloud
    * @param minNeighForNorms the minimum number of neighbors to consider for normal calculation
    * @param maxNeighbors the maximum number of neightbors to consider for normal calculation
    */
    void computeNormals(int minNeighForNorms, int maxNeighbors);

    /**
    * Computes normals for the points within the input points cloud
    * @param minNeighForNorms the minimum number of neighbors to consider for normal calculation
    * @param maxNeighbors the maximum number of neightbors to consider for normal calculation
    * @param numCameras the total number of cameras which resulted in the point cloud
    * @param cameraPositions the x,y,z coordinates of the cameras
    * \warning This method assumes that all cameras are on one side of the point cloud, so this should be purposed
    * for landscape surface reconstruction and not small object recostruction.
    */
    void computeNormals(int minNeighForNorms, int maxNeighbors, unsigned int numCameras, float3* cameraPositions);

    /**
    * Computes the average normal of the input points. This is only useful if you can make a "planar" assumption about
    * the input points, that is the points are mostly aligned along a plane. For use in reconstructon filtering should occur
    * before one considers using this method
    * @param minNeighForNorms the minimum number of neighbors to consider for normal calculation
    * @param maxNeighbors the maximum number of neightbors to consider for normal calculation
    * @param numCameras the total number of cameras which resulted in the point cloud
    * @param cameraPositions the x,y,z coordinates of the cameras
    */
    ssrlcv::ptr::value<ssrlcv::Unity<float3>> computeAverageNormal(int minNeighForNorms, int maxNeighbors, unsigned int numCameras, float3* cameraPositions);


    // =============================================================================================================
    //
    // PLY writers
    //
    // =============================================================================================================

    void writeVertexPLY(bool binary = false);
    void writeEdgePLY(bool binary = false);
    void writeCenterPLY(bool binary = false);
    void writepointsPLY(bool binary = false);
    void writeNormalPLY(bool binary = false);
    void writeDepthPLY(int d, bool binary = false);
  };

  static const int3 coordPlacementIdentity_host[8] {
    {-1,-1,-1},
    {-1,-1,1},
    {-1,1,-1},
    {-1,1,1},
    {1,-1,-1},
    {1,-1,1},
    {1,1,-1},
    {1,1,1}
  };
  static const int2 vertexEdgeIdentity_host[12] {
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
  static const int4 vertexFaceIdentity_host[6] {
    {0,1,2,3},
    {0,1,4,5},
    {0,2,4,6},
    {1,3,5,7},
    {2,3,6,7},
    {4,5,6,7}
  };
  static const int4 edgeFaceIdentity_host[6] {
    {0,1,2,3},
    {0,4,5,8},
    {1,4,6,9},
    {2,5,7,10},
    {3,6,7,11},
    {8,9,10,11}
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

  // =============================================================================================================
  //
  // Device Kernels
  //
  // =============================================================================================================

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
  __global__ void fillBlankNodeArray(Octree::Node* uniqueNodes, int* nodeNumbers, int* nodeAddresses, Octree::Node* outputNodeArray, int numUniqueNodes, int currentDepth, float totalWidth);
  __global__ void fillFinestNodeArrayWithUniques(Octree::Node* uniqueNodes, int* nodeAddresses, Octree::Node* outputNodeArray, int numUniqueNodes, unsigned int* pointNodeIndex);
  __global__ void fillNodeArrayWithUniques(Octree::Node* uniqueNodes, int* nodeAddresses, Octree::Node* outputNodeArray, Octree::Node* childNodeArray ,int numUniqueNodes);
  __global__ void generateParentalUniqueNodes(Octree::Node* uniqueNodes, Octree::Node* nodeArrayD, int numNodesAtDepth, float totalWidth, const int3* __restrict__ coordPlacementIdentity);
  __global__ void computeNeighboringNodes(Octree::Node* nodeArray, int numNodes, int depthIndex, int* parentLUT, int* childLUT, int childDepthIndex);

  // calculates the average normal
  __global__ void calculateCloudAverageNormal(float3* average, unsigned long num, float3* normals);

  // calculates average distances to N neighbors
  __global__ void computeAverageNeighboorDistances(int* n, unsigned long numpoints, float3* points, unsigned int* pointNodeIndex, Octree::Node* nodes, float* averages);

  // calculates average distance to N neighbors
  __global__ void computeAverageNeighboorDistance(int* n, unsigned long numpoints, float3* points, unsigned int* pointNodeIndex, Octree::Node* nodes, float* averages);

  // finds the point indexes that should be removed
  __global__ void getGoodDensePoints(int* n, float* cutoff, unsigned long numpoints, float3* points, unsigned int* pointNodeIndex, Octree::Node* nodes, float3* indexes);

  __global__ void findNormalNeighborsAndComputeCMatrix(int numNodesAtDepth, int depthIndex, int maxNeighbors, Octree::Node* nodeArray, float3* points, float* cMatrix, int* neighborIndices, int* numNeighbors);
  __global__ void transposeFloatMatrix(int m, int n, float* matrix);
  __global__ void setNormal(int currentPoint, float* vt, float3* normals);
  __global__ void checkForAmbiguity(int numPoints, int numCameras, float3* normals, float3* points, float3* cameraPositions, bool* ambiguous);
  __global__ void reorient(int numNodesAtDepth, int depthIndex, Octree::Node* nodeArray, int* numNeighbors, int maxNeighbors, float3* normals, int* neighborIndices, bool* ambiguous);

  __global__ void findVertexOwners(Octree::Node* nodeArray, int numNodes, int depthIndex, int* vertexLUT, int* numVertices, int* ownerInidices, int* vertexPlacement);
  __global__ void fillUniqueVertexArray(Octree::Node* nodeArray, Octree::Vertex* vertexArray, int numVertices, int vertexIndex,int depthIndex, int depth, float width, int* vertexLUT, int* ownerInidices, int* vertexPlacement, const int3* __restrict__ coordPlacementIdentity);
  __global__ void findEdgeOwners(Octree::Node* nodeArray, int numNodes, int depthIndex, int* edgeLUT, int* numEdges, int* ownerInidices, int* edgePlacement);
  __global__ void fillUniqueEdgeArray(Octree::Node* nodeArray, Octree::Edge* edgeArray, int numEdges, int edgeIndex,int depthIndex, int depth, float width, int* edgeLUT, int* ownerInidices, int* edgePlacement, const int2* __restrict__ vertexEdgeIdentity);
  __global__ void findFaceOwners(Octree::Node* nodeArray, int numNodes, int depthIndex, int* faceLUT, int* numFaces, int* ownerInidices, int* facePlacement);
  __global__ void fillUniqueFaceArray(Octree::Node* nodeArray, Octree::Face* faceArray, int numFaces, int faceIndex,int depthIndex, int depth, float width, int* faceLUT, int* ownerInidices, int* facePlacement, const int4* __restrict__ edgeFaceIdentity);

}


#endif /* OCTREE_CUH */
