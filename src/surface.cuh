#ifndef SURFACE_CUH
#define SURFACE_CUH

#include "common_includes.h"
#include "octree.cuh"
#include "cuda_util.cuh"

#include <thrust/scan.h>
#include <thrust/device_ptr.h>
#include "magma_v2.h"
#include "magmasparse.h"
#include <cusparse.h>
#include <cublas_v2.h>

extern __constant__ int3 cubeCategoryTrianglesFromEdges[15][4];
extern __constant__ bool cubeCategoryEdgeIdentity[15][12];
extern __constant__ bool cubeCategoryVertexIdentity[15][8];
extern __constant__ int numTrianglesInCubeCategory[15];

__device__ __host__ float3 blenderPrime(const float3 &a, const float3 &b, const float &bw);
__device__ __host__ float3 blenderPrimePrime(const float3 &a, const float3 &b, const float &bw);

/*
DIVERGENCE VECTOR KERNELS
*/
__global__ void computeVectorFeild(Node* nodeArray, int numFinestNodes, float3* vectorField, float3* normals, float3* points);
__global__ void computeDivergenceFine(Node* nodeArray, int numNodes, int depthIndex, float3* vectorField, float* divCoeff, float* fPrimeLUT);
__global__ void findRelatedChildren(Node* nodeArray, int numNodes, int depthIndex, int2* relativityIndicators);
__global__ void computeDivergenceCoarse(Node* nodeArray, int2* relativityIndicators, int currentNode, int depthIndex, float3* vectorField, float* divCoeff, float* fPrimeLUT);

__global__ void computeLd(int depthOfOctree, Node* nodeArray, int numNodes, int depthIndex, float* laplacianValues, int* laplacianIndices, float* fLUT, float* fPrimePrimeLUT);
__global__ void computeLdCSR(int depthOfOctree, Node* nodeArray, int numNodes, int depthIndex, float* laplacianValues, int* laplacianIndices, int* numNonZero, float* fLUT, float* fPrimePrimeLUT);
__global__ void updateDivergence(Node* nodeArray, int numNodes, int depthIndex, float* divCoeff, float* fLUT, float* fPrimePrimeLUT, float* nodeImplicit);


/*
MULTIGRID SOLVER
*/
__global__ void multiplyLdAnd1D(int numNodesAtDepth, float* laplacianValues, int* laplacianIndices, float* matrix1D, float* result);
__global__ void computeAlpha(int numNodesAtDepth, float* r, float* pTL, float* p, float* numerator, float* denominator);
__global__ void updateX(int numNodesAtDepth, int depthIndex, float* x, float alpha, float* p);
__global__ void computeRNew(int numNodesAtDepth, float* r, float alpha, float* temp);
__global__ void computeBeta(int numNodesAtDepth, float* r, float* rNew, float* numerator, float* denominator);
__global__ void updateP(int numNodesAtDepth, float* rNew, float beta, float* p);

__global__ void pointSumImplicitTraversal(int numPoints, float3* points, Node* nodeArray, int root, float* nodeImplicit, float* sumImplicit);
__global__ void vertexSumImplicitTraversal(int numVertices, Vertex* vertexArray, float* nodeImplicit, float* vertexImplicit, float* avgImplict, int numPoints);

__global__ void vertexImplicitFromNormals(int numVertices, Vertex* vertexArray, Node* nodeArray, float3* normals, float3* points, float* vertexImplicit);

__global__ void calcVertexNumbers(int numEdges, Edge* edgeArray, float* vertexImplicit, int* vertexNumbers);
__global__ void determineCubeCategories(int numNodes, Node* nodeArray, float* vertexImplicit, int* cubeCategory, int* triangleNumbers);
__global__ void generateSurfaceVertices(int numEdges, Edge* edgeArray, Vertex* vertexArray, int* vertexNumbers, int* vertexAddresses, float3* surfaceVertices);
__global__ void generateSurfaceTriangles(int numNodes, Node* nodeArray, int* vertexAddresses, int* triangleNumbers, int* triangleAddresses, int* cubeCategory, int3* surfaceTriangles);

struct Surface{

  Octree* octree;

  //TODO make sure that these are indeed supposed to be floats
  //TODO make these constant cuda variables
  //Look up tables
  float* fLUT;
  float* fLUTDevice;
  float* fPrimeLUT;
  float* fPrimeLUTDevice;
  float* fPrimePrimeLUT;
  float* fPrimePrimeLUTDevice;

  //Major variables (do not need host versions other than for error checking)
  float* divergenceVectorDevice;
  float* nodeImplicitDevice;
  float3* surfaceVertices;
  int numSurfaceVertices;
  int3* surfaceTriangles;
  int numSurfaceTriangles;
  float* vertexImplicitDevice;

  Surface(std::string pathToPLY, int depthOfOctree);
  Surface();
  ~Surface();

  void computeLUTs();
  void computeDivergenceVector();

  void computeImplicitFunction();
  void computeImplicitMagma();
  void computeImplicitCuSPSolver();
  void computeImplicitEasy();

  void computeVertexImplicit();

  void marchingCubes();

  void generateMesh();

};


#endif /* SURFACE_CUH */
