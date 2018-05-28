#ifndef POISSON_CUH
#define POISSON_CUH

#include "common_includes.h"
#include "octree.cuh"

/*
CUDA KERNELS
*/

__global__ void computeVectorFeild(Node* nodeArray, int numFinestNodes, float* vectorField, float nodeWidth, int* fLUT, int* fPrimePrimeLUT, float3* normals);
__global__ void computeDivergenceFine(Node* nodeArray, int numNodes, int depthIndex, float* vectorField, float* divCoeff, int* fPrimeLUT);
__global__ void findRelatedChildren(Node* nodeArray, int numNodes, int depthIndex, int2* relativityIndicators);
__global__ void computeDivergenceCoarse(Node* nodeArray, int2* relativityIndicators, int currentNode, float* vectorField, float* divCoeff, int* fPrimeLUT);
/*
POISSON RECONSTRUCTION METHODS
*/

struct Poisson{

  Octree octree;
  int* fLUT;
  int* fPrimeLUT;
  int* fPrimePrimeLUT;

  Poisson();
  Poisson(Octree octree);
  ~Poisson();

  void computeLaplacianMatrix();
  void computeDivergenceVector();
  void computeImplicitFunction();
  void marchingCubes();
  void isosurfaceExtraction();

};


#endif /* POISSON_CUH */
