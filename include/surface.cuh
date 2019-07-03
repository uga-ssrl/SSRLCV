#ifndef SURFACE_CUH
#define SURFACE_CUH

#include "common_includes.h"
#include "octree.cuh"
#include "cuda_util.cuh"

#include <thrust/scan.h>
#include <thrust/device_ptr.h>

extern __constant__ int cubeCategoryTrianglesFromEdges[256][15];
extern __constant__ int cubeCategoryEdgeIdentity[256];
extern __constant__ int numTrianglesInCubeCategory[256];

__device__ __host__ float3 blenderPrime(const float3 &a, const float3 &b, const float &bw);
__device__ __host__ float3 blenderPrimePrime(const float3 &a, const float3 &b, const float &bw);

__global__ void vertexImplicitFromNormals(int numVertices, Vertex* vertexArray, Node* nodeArray, float3* normals, float3* points, float* vertexImplicit);
__global__ void calcVertexNumbers(int numEdges, int depthIndex, Edge* edgeArray, float* vertexImplicit, int* vertexNumbers);

//adaptive marching cubes
__global__ void categorizeCubesRecursively_child(int parent, int parentCategory, Edge* edgeArray, Node* nodeArray, int* vertexNumbers, int* cubeCategory, int* triangleNumbers);
__global__ void categorizeCubesRecursively(int firstChildrenIndex, Edge* edgeArray, Node* nodeArray, int* vertexNumbers, int* cubeCategory, int* triangleNumbers);
__global__ void minimizeVertices(int numEdges, Edge* edgeArray, Node* nodeArray, int* cubeCategory, int* vertexNumbers);

//marching cubes
__global__ void determineCubeCategories(int numNodes, int nodeIndex, int edgeIndex, Node* nodeArray, int* vertexNumbers, int* cubeCategory, int* triangleNumbers);
__global__ void generateSurfaceVertices(int numEdges, int depthIndex, Edge* edgeArray, Vertex* vertexArray, int* vertexNumbers, int* vertexAddresses, float3* surfaceVertices);
__global__ void generateSurfaceTriangles(int numNodes, int nodeIndex, int edgeIndex, Node* nodeArray, int* vertexAddresses, int* triangleAddresses, int* cubeCategory, int3* surfaceTriangles);

struct Surface{

  Octree* octree;

  //Major variables (do not need host versions other than for error checking)
  float3* surfaceVertices;
  int numSurfaceVertices;
  int3* surfaceTriangles;
  int numSurfaceTriangles;
  float* vertexImplicitDevice;

  Surface(Octree* octree);
  Surface(std::string pathToPLY, int depthOfOctree);
  Surface();
  ~Surface();

  void computeVertexImplicitJAX(int focusDepth);
  void adaptiveMarchingCubes();
  void marchingCubes();
  void jaxMeshing();
  void generateMesh();
  void generateMeshWithFinestEdges();

};


#endif /* SURFACE_CUH */
