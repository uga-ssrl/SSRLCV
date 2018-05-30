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


stack_uint::stack_uint(){
  this->maxSize = 80;
}

stack_uint::stack_uint(uint maxSize){
  this->maxSize = maxSize;
}

int stack_uint::pop(){
  if(this->data.size() == 0){
    cout<<"STACK UNDERFLOW...size of stack is 0"<<endl;
    exit(-1);
  }
  int i = this->data[this->data.size() -1];
  this->data.erase(this->data.begin() + (this->data.size() - 1));
  return i;
}

void stack_uint::push(uint i){
  if(this->data.size() == this->maxSize){
    cout<<"STACK OVERFLOW..."<<this->maxSize<<" is the defined maximum size of this stack"<<endl;
    exit(-1);
  }
  this->data.push_back(i);
}

__device__ float3 operator+(const float3 &a, const float3 &b) {
  return {a.x+b.x, a.y+b.y, a.z+b.z};
}
__device__ float3 operator-(const float3 &a, const float3 &b) {
  return {a.x-b.x, a.y-b.y, a.z-b.z};
}
__device__ float3 operator/(const float3 &a, const float3 &b) {
  return {a.x/b.x, a.y/b.y, a.z/b.z};
}
__device__ float3 operator*(const float3 &a, const float3 &b) {
  return {a.x*b.x, a.y*b.y, a.z*b.z};
}
__device__ float dotProduct(const float3 &a, const float3 &b){
  return (a.x*b.x) + (a.y*b.y) + (a.z*b.z);
}
__device__ float3 operator+(const float3 &a, const float &b){
  return {a.x+b, a.y+b, a.z+b};
}
__device__ float3 operator-(const float3 &a, const float &b){
  return {a.x-b, a.y-b, a.z-b};

}
__device__ float3 operator/(const float3 &a, const float &b){
  return {a.x/b, a.y/b, a.z/b};

}
__device__ float3 operator*(const float3 &a, const float &b){
  return {a.x*b, a.y*b, a.z*b};
}
__device__ float3 operator+(const float &a, const float3 &b) {
  return {a+b.x, a+b.y, a+b.z};
}
__device__ float3 operator-(const float &a, const float3 &b) {
  return {a-b.x, a-b.y, a-b.z};
}
__device__ float3 operator/(const float &a, const float3 &b) {
  return {a/b.x, a/b.y, a/b.z};
}
__device__ float3 operator*(const float &a, const float3 &b) {
  return {a*b.x, a*b.y, a*b.z};
}


__global__ void computeVectorFeild(Node* nodeArray, int numFinestNodes, float3* vectorField, float nodeWidth, int* fLUT, int* fPrimePrimeLUT, float3* normals, float3* points){
  int blockID = blockIdx.y * gridDim.x + blockIdx.x;
  if(blockID < numFinestNodes){
    __shared__ float3 vec;
    vec = {0.0f, 0.0f, 0.0f};
    __syncthreads();
    int neighborIndex = nodeArray[blockID].neighbors[threadIdx.x];
    if(neighborIndex != -1){
      int currentPoint = nodeArray[neighborIndex].pointIndex;
      int stopIndex = nodeArray[neighborIndex].numPoints + currentPoint;
      float3 blend = {0.0f,0.0f,0.0f};
      float width = nodeArray[blockID].width;
      float3 center = nodeArray[blockID].center;
      float2 bounds = {-1.5f,1.5f};
      for(int i = currentPoint; i < stopIndex; ++i){
        blend = (points[i] - center)/width;
        //TODO determine if you want to change cube from [-1,1]^3 to [-1.5,1.5]^3
        //TODO make sure that t = q - o.c / o.w
        if((blend.x >= bounds.x && blend.x <= 0.0f) &&
        (blend.y >= bounds.x && blend.y <= 0.0f) &&
        (blend.z >= bounds.x && blend.z <= 0.0f)){
          //n = 2 Fo(q) make bounds {-1.0f, 1.0f}
          //blend = 1.0f - blend;

          //n = 3 Fo(q) make bounds {-1.5f, 1.5f}
          blend = (0.5*blend*blend*blend) + (2.5*blend*blend) + (4.0f*blend) + 2;

          blend = normals[i]*(blend/(width*width*width));
        }
        else if((blend.x > 0.0f && blend.x <= bounds.y) &&
        (blend.y > 0.0f && blend.y <= bounds.y) &&
        (blend.z > 0.0f && blend.z <= bounds.y)){
          //n = 2 Fo(q) make bounds {-1.0f, 1.0f}
          blend = blend + 1.0f;

          //n = 3 Fo(q) make bounds {-1.5f, 1.5f}
          blend = (-0.5*blend*blend*blend) + (2.5*blend*blend) + (-4.0f*blend) + 2;

          blend = normals[i]*(blend/(width*width*width));
        }
        else{
          continue;
        }

        atomicAdd(&vec.x, blend.x);
        atomicAdd(&vec.y, blend.y);
        atomicAdd(&vec.z, blend.z);
      }
    }
    __syncthreads();
    if(threadIdx.x == 0){
      vectorField[blockID] = vec;
    }
  }
}
__global__ void computeDivergenceFine(Node* nodeArray, int numNodes, int depthIndex, float3* vectorField, float* divCoeff, int* fPrimeLUT){
  int blockID = blockIdx.y * gridDim.x + blockIdx.x;
  if(blockID < numNodes){
    __shared__ float coeff;
    int neighborIndex = nodeArray[blockID + depthIndex].neighbors[threadIdx.x];
    int numFinestChildren = nodeArray[neighborIndex].numFinestChildren;
    int finestChildIndex = nodeArray[neighborIndex].finestChildIndex;
    for(int i = finestChildIndex; i < finestChildIndex + numFinestChildren; ++i){
      //<Fo(q), divFo'> = fPrimeLUT[??]
      //atomicAdd(&coeff, dotProduct(vectorField[i], ));
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

__global__ void computeDivergenceCoarse(Node* nodeArray, int2* relativityIndicators, int currentNode, float3* vectorField, float* divCoeff, int* fPrimeLUT){
  int globalID = blockIdx.x * blockDim.x + threadIdx.x;
  if(globalID < relativityIndicators[currentNode].y){
    globalID += relativityIndicators[currentNode].x;
    //<Fo(q), divFo'> = fPrimeLUT[??]
    //atomicAdd(&divCoeff[currentNode], vectorField[i]*);
  }
}


//TODO implement these

Poisson::Poisson(Octree* octree){
  this->octree = octree;
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
