#ifndef POISSON_CUH
#define POISSON_CUH

#include "common_includes.h"
#include "octree.cuh"

/*
HELPER STACK DATA STRUCTURE
*/
struct stack_uint{
  std::vector<uint> data;
  int maxSize;

  stack_uint();
  stack_uint(uint maxSize);
  int pop();
  void push(uint i);
};

__device__ float3 operator+(const float3 &a, const float3 &b);
__device__ float3 operator-(const float3 &a, const float3 &b);
__device__ float3 operator/(const float3 &a, const float3 &b);
__device__ float3 operator*(const float3 &a, const float3 &b);
__device__ float dotProduct(const float3 &a, const float3 &b);
__device__ float3 operator+(const float3 &a, const float &b);
__device__ float3 operator-(const float3 &a, const float &b);
__device__ float3 operator/(const float3 &a, const float &b);
__device__ float3 operator*(const float3 &a, const float &b);
__device__ float3 operator+(const float &a, const float3 &b);
__device__ float3 operator-(const float &a, const float3 &b);
__device__ float3 operator/(const float &a, const float3 &b);
__device__ float3 operator*(const float &a, const float3 &b);
/*
CUDA KERNELS
*/

__global__ void computeVectorFeild(Node* nodeArray, int numFinestNodes, float3* vectorField, float nodeWidth, int* fLUT, int* fPrimePrimeLUT, float3* normals, float3* points);
__global__ void computeDivergenceFine(Node* nodeArray, int numNodes, int depthIndex, float3* vectorField, float* divCoeff, int* fPrimeLUT);
__global__ void findRelatedChildren(Node* nodeArray, int numNodes, int depthIndex, int2* relativityIndicators);
__global__ void computeDivergenceCoarse(Node* nodeArray, int2* relativityIndicators, int currentNode, float3* vectorField, float* divCoeff, int* fPrimeLUT);

/*
POISSON RECONSTRUCTION METHODS
*/

struct Poisson{

  Octree* octree;
  int* fLUT;
  int* fPrimeLUT;
  int* fPrimePrimeLUT;
  float3* vectorField;
  float* divCoeff;

  Poisson(Octree* octree);
  ~Poisson();

  void computeLaplacianMatrix();
  void computeDivergenceVector();
  void computeImplicitFunction();
  void marchingCubes();
  void isosurfaceExtraction();

};


#endif /* POISSON_CUH */
