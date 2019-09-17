/** \file MeshFactory.cuh
* \brief this file contains all mesh generation methods
*/
#ifndef MESHFACTORY_CUH
#define MESHFACTORY_CUH

#include "common_includes.h"
#include "cuda_util.cuh"
#include "Image.cuh"
#include "Octree.cuh"
#include "Unity.cuh"
#include <thrust/scan.h>
#include <thrust/device_ptr.h>

namespace ssrlcv{
  /**
  * \brief Factory for generating meshes from PointClouds.
  */
  class MeshFactory{

  public:
    Octree* octree;

    //Major variables (do not need host versions other than for error checking)
    float3* surfaceVertices;
    int numSurfaceVertices;
    int3* surfaceTriangles;
    int numSurfaceTriangles;
    float* vertexImplicitDevice;

    MeshFactory(Octree* octree);
    MeshFactory();
    ~MeshFactory();

    void computeVertexImplicitJAX(int focusDepth);
    void adaptiveMarchingCubes();
    void marchingCubes();
    void jaxMeshing();
    void generateMesh(bool binary);
    void generateMesh();
    void generateMeshWithFinestEdges();

  };

  /* CUDA variable, method and kernel defintions */

  namespace{
    struct is_not_neg_int{
      __host__ __device__
      bool operator()(const int x)
      {
        return (x >= 0);
      }
    };
    struct is_not_zero_float{
      __host__ __device__
      bool operator()(const float x)
      {
        return (x != 0.0f);
      }
    };
  }


  extern __constant__ int cubeCategoryTrianglesFromEdges[256][15];
  extern __constant__ int cubeCategoryEdgeIdentity[256];
  extern __constant__ int numTrianglesInCubeCategory[256];

  __device__ __host__ float3 blenderPrime(const float3 &a, const float3 &b, const float &bw);
  __device__ __host__ float3 blenderPrimePrime(const float3 &a, const float3 &b, const float &bw);

  __global__ void vertexImplicitFromNormals(int numVertices, Octree::Vertex* vertexArray, Octree::Node* nodeArray, float3* normals, float3* points, float* vertexImplicit);
  __global__ void calcVertexNumbers(int numEdges, int depthIndex, Octree::Edge* edgeArray, float* vertexImplicit, int* vertexNumbers);

  //adaptive marching cubes
  __global__ void categorizeCubesRecursively_child(int parent, int parentCategory, Octree::Edge* edgeArray, Octree::Node* nodeArray, int* vertexNumbers, int* cubeCategory, int* triangleNumbers);
  __global__ void categorizeCubesRecursively(int firstChildrenIndex, Octree::Edge* edgeArray, Octree::Node* nodeArray, int* vertexNumbers, int* cubeCategory, int* triangleNumbers);
  __global__ void minimizeVertices(int numEdges, Octree::Edge* edgeArray, Octree::Node* nodeArray, int* cubeCategory, int* vertexNumbers);

  //marching cubes
  __global__ void determineCubeCategories(int numNodes, int nodeIndex, int edgeIndex, Octree::Node* nodeArray, int* vertexNumbers, int* cubeCategory, int* triangleNumbers);
  __global__ void generateSurfaceVertices(int numEdges, int depthIndex, Octree::Edge* edgeArray, Octree::Vertex* vertexArray, int* vertexNumbers, int* vertexAddresses, float3* surfaceVertices);
  __global__ void generateSurfaceTriangles(int numNodes, int nodeIndex, int edgeIndex, Octree::Node* nodeArray, int* vertexAddresses, int* triangleAddresses, int* cubeCategory, int3* surfaceTriangles);

}


#endif /* MESHFACTORY_CUH */
