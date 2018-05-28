#include "poisson.cuh"

using namespace std;

// Define this to turn on error checking
#define CUDA_ERROR_CHECK

#define CudaSafeCall( err ) __cudaSafeCall( err, __FILE__, __LINE__ )
#define CudaCheckError()    __cudaCheckError( __FILE__, __LINE__ )


inline void __cudaSafeCall(cudaError err, const char *file, const int line) {
#ifdef CUDA_ERROR_CHECK
  if (cudaSuccess != err) {
      fprintf(stderr, "cudaSafeCall() failed at %s:%i : %s\n",
      file, line, cudaGetErrorString(err));
      exit(-1);
  }
#endif

  return;
}
inline void __cudaCheckError(const char *file, const int line) {
#ifdef CUDA_ERROR_CHECK
  cudaError err = cudaGetLastError();
  if (cudaSuccess != err) {
    fprintf(stderr, "cudaCheckError() failed at %s:%i : %s\n",
    file, line, cudaGetErrorString(err));
    exit(-1);
  }

  // More careful checking. However, this will affect performance.
  // Comment away if needed.
  //err = cudaDeviceSynchronize();
  if (cudaSuccess != err) {
    fprintf(stderr, "cudaCheckError() with sync failed at %s:%i : %s\n",
    file, line, cudaGetErrorString(err));
    exit(-1);
  }
#endif

  return;
}

__global__ void computeVectorFeild(Node* nodeArray, int numFinestNodes, float* vectorField, float nodeWidth, int* fLUT, int* fPrimePrimeLUT, float3* normals){
  int blockID = blockIdx.y * gridDim.x + blockIdx.x;
  if(blockID < numFinestNodes){
    __shared__ float vec;
    vec = 0;
    __syncthreads();
    int neighborIndex = nodeArray[blockID].neighbors[threadIdx.x];
    if(neighborIndex != -1){
      int currentPoint = nodeArray[neighborIndex].pointIndex;
      int stopIndex = nodeArray[neighborIndex].numPoints + currentPoint;
      for(int i = currentPoint; i < stopIndex; ++i){
        //atomicAdd(&vec, normals[i]*function of point);
      }
    }
    __syncthreads();
    if(threadIdx.x == 0){
      vectorField[blockID] = vec;
    }
  }
}
__global__ void computeDivergenceFine(Node* nodeArray, int numNodes, int depthIndex, float* vectorField, float* divCoeff, int* fPrimeLUT){
  int blockID = blockIdx.y * gridDim.x + blockIdx.x;
  if(blockID < numNodes){
    __shared__ float coeff;
    int neighborIndex = nodeArray[blockID + depthIndex].neighbors[threadIdx.x];
    int numFinestChildren = nodeArray[neighborIndex].numFinestChildren;
    int finestChildIndex = nodeArray[neighborIndex].finestChildIndex;
    for(int i = finestChildIndex; i < finestChildIndex + numFinestChildren; ++i){
      //<Fo(q), divFo'> = fPrimeLUT[??]
      //atomicAdd(&coeff, vectorField[i]*);
    }
  }
}
__global__ void findRelatedChildren(Node* nodeArray, int numNodes, int depthIndex, int2* relativityIndicators){
  int blockID = blockIdx.y * gridDim.x + blockIdx.x;
  if(blockID < numNodes){
    __shared__ int numRelativeChildren;
    __shared__ int firstRelativeChild;
    numRelativeChildren = 0;
    int neighborIndex = nodeArray[blockID + depthIndex].neighbors[threadIdx.x];
    atomicAdd(&numRelativeChildren, nodeArray[neighborIndex].numFinestChildren);
    atomicMin(&firstRelativeChild, nodeArray[neighborIndex].finestChildIndex);
    __syncthreads();
    if(threadIdx.x == 0){
      relativityIndicators[blockID].x = firstRelativeChild;
      relativityIndicators[blockID].y = numRelativeChildren;
    }
  }
}

__global__ void computeDivergenceCoarse(Node* nodeArray, int2* relativityIndicators, int currentNode, float* vectorField, float* divCoeff, int* fPrimeLUT){
  int globalID = blockIdx.x * blockDim.x + threadIdx.x;
  if(globalID < relativityIndicators[currentNode].y){
    globalID += relativityIndicators[currentNode].x;
    //<Fo(q), divFo'> = fPrimeLUT[??]
    //atomicAdd(&divCoeff[currentNode], vectorField[i]*);
  }
}


//TODO implement these

Poisson::Poisson(){

}

Poisson::Poisson(Octree octree){
  octree = this->octree;
}

Poisson::~Poisson(){
  //delete octree;
}

//TODO get LUT tables filled in. Each should be (2^(depth + 1)) - 1 = n nxn matrix that is symmetric
void Poisson::computeLaplacianMatrix(){

}

void Poisson::computeDivergenceVector(){

}
void Poisson::computeImplicitFunction(){

}
void Poisson::marchingCubes(){

}
void Poisson::isosurfaceExtraction(){

}
