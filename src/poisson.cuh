#ifndef POISSON_CUH
#define POISSON_CUH

#include "common_includes.h"
#include "octree.cuh"

__device__ __host__ float3 operator+(const float3 &a, const float3 &b);
__device__ __host__ float3 operator-(const float3 &a, const float3 &b);
__device__ __host__ float3 operator/(const float3 &a, const float3 &b);
__device__ __host__ float3 operator*(const float3 &a, const float3 &b);
__device__ __host__ float dotProduct(const float3 &a, const float3 &b);
__device__ __host__ float3 operator+(const float3 &a, const float &b);
__device__ __host__ float3 operator-(const float3 &a, const float &b);
__device__ __host__ float3 operator/(const float3 &a, const float &b);
__device__ __host__ float3 operator*(const float3 &a, const float &b);
__device__ __host__ float3 operator+(const float &a, const float3 &b);
__device__ __host__ float3 operator-(const float &a, const float3 &b);
__device__ __host__ float3 operator/(const float &a, const float3 &b);
__device__ __host__ float3 operator*(const float &a, const float3 &b);

__device__ __host__ float3 blender(const float3 &a, const float3 &b, const float &bw);
__device__ __host__ float3 blenderPrime(const float3 &a, const float3 &b, const float &bw);
__device__ __host__ float3 blenderPrimePrime(const float3 &a, const float3 &b, const float &bw);

/*
DIVERGENCE VECTOR KERNELS
*/
__global__ void computeVectorFeild(Node* nodeArray, int numFinestNodes, float3* vectorField, float3* normals, float3* points);
__global__ void computeDivergenceFine(Node* nodeArray, int numNodes, int depthIndex, float3* vectorField, float* divCoeff, float* fPrimeLUT);
__global__ void findRelatedChildren(Node* nodeArray, int numNodes, int depthIndex, int2* relativityIndicators);
__global__ void computeDivergenceCoarse(Node* nodeArray, int2* relativityIndicators, int currentNode, int depthIndex, float3* vectorField, float* divCoeff, float* fPrimeLUT);

/*
MULTIGRID SOLVER
*/
__global__ void updateDivergence(Node* nodeArray, int numNodes, int depthIndex, float* divCoeff, float* fLUT, float* fPrimePrimeLUT, float* nodeImplicit);

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

  Poisson(Octree* octree);
  ~Poisson();

  void computeLUTs();
  void computeDivergenceVector();
  void multiGridSolver();
  void computeImplicitFunction();
  void marchingCubes();

};


#endif /* POISSON_CUH */
