#ifndef SURFACE_CUH
#define SURFACE_CUH

#include "common_includes.h"
#include "Octree.cuh"
#include "cuda_util.cuh"

#include <thrust/scan.h>
#include <thrust/device_ptr.h>
#include "magma_v2.h"
#include "magmasparse.h"
#include <cusparse.h>
#include <cublas_v2.h>

extern __constant__ int cubeCategoryTrianglesFromEdges[256][15];
extern __constant__ int cubeCategoryEdgeIdentity[256];
extern __constant__ int numTrianglesInCubeCategory[256];

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

struct Poisson{

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

  Surface();
  ~Surface();

  void computeLUTs();
  void computeDivergenceVector();

  void computeImplicitFunction();
  void computeImplicitMagma();
  void computeImplicitCuSPSolver();
  void computeImplicitEasy(int focusDepth);
  void computeVertexImplicit();
};


#endif /* SURFACE_CUH */
