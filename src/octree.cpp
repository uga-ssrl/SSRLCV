#include "Octree.h"

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


//pretty much just a binary search in each dimension performed by threads
__global__ void getKeys(float3* points, float3* centers, int* keys, float3 c, float W, int N, int D){

  int tx = threadIdx.x;
  int bx = blockIdx.x;
  int globalID = bx * blockDim.x + tx;
  if(globalID < N){
    float x = points[i].x;
    float y = points[i].y;
    float z = points[i].z;
    float leftx = c.x-W/2.0f, rightx = c.x + W/2.0f;
    float lefty = c.y-W/2.0f, righty = c.y + W/2.0f;
    float leftz = c.z-W/2.0f, rightz = c.z + W/2.0f;
    int key = 0;
    int depth = 1;
    while(depth <= D){
      if(x < c.x){
        key <<= 1;
        rightx = c.x;
        c.x = (leftx + rightx)/2.0f;
      }
      else{
        key = (key << 1) + 1;
        leftx = c.x;
        c.x = (leftx + rightx)/2.0f;
      }
      if(y < c.y){
        key <<= 1;
        righty = c.y;
        c.y = (lefty + righty)/2.0f;
      }
      else{
        key = (key << 1) + 1;
        lefty = c.y;
        c.y = (lefty + righty)/2.0f;
      }
      if(z < c.z){
        key <<= 1;
        rightz = c.z;
        c.z = (leftz + rightz)/2.0f;
      }
      else{
        key = (key << 1) + 1;
        leftz = c.z;
        c.z = (leftz + rightz)/2.0f;
      }
      depth++;
    }
    keys[i] = key;
    centers[i].x = c.x;
    centers[i].y = c.y;
    centers[i].z = c.z;
  }
}


Octree::Octree(){

}
Octree::Octree(float3* points, float3* normals, int numPoints, int depth){
  this->points = points;
  this->normals = normals;
  this->centers = new float3[numPoints];
  this->keys = new int[numPoints];
  this->numPoints = numPoints;
  this->depth = depth;
}
void Octree::findMinMax(){
  this->min = this->points[i];
  this->max = this->points[i];
  float3 currentPoint;
  for(int i = 0; i < numPoints; ++i){
    currentPoint = this->points[i];
    if(currentPoint.x < this->min.x){
      this->min.x = currentPoint;
    }
    if(currentPoint.x > this->max.x){
      this->max.x = currentPoint;
    }
    if(currentPoint.y < this->min.y){
      this->min.y = currentPoint;
    }
    if(currentPoint.y > this->max.y){
      this->max.y = currentPoint;
    }
    if(currentPoint.z < this->min.z){
      this->min.z = currentPoint;
    }
    if(currentPoint.z > this->max.z){
      this->max.z = currentPoint;
    }
    if(currentPoint.z < this->min.z){
      this->min.x = currentPoint;
    }
  }
}

void Octree::allocateDeviceVariables(){
  CudaSafeCall(cudaMalloc((void**)&this->pointsDevice, this->numPoints * sizeof(float3)));
  CudaSafeCall(cudaMalloc((void**)&this->centersDevice, this->numPoints * sizeof(float3)));
  CudaSafeCall(cudaMalloc((void**)&this->keysDevice, this->numPoints * sizeof(int)));
  CudaSafeCall(cudaMalloc((void**)&this->normalsDevice, this->numPoints * sizeof(int)));

}

void Octree::executeKeyRetrieval(){


  CudaSafeCall(cudaMemcpy(this->pointsDevice, this->points, this->numPoints * sizeof(float3), cudaMemcpyHostToDevice));
  CudaSafeCall(cudaMemcpy(this->centersDevice, this->centers, this->numPoints * sizeof(float3), cudaMemcpyHostToDevice));
  CudaSafeCall(cudaMemcpy(this->keysDevice, this->keys, this->numPoints * sizeof(int), cudaMemcpyHostToDevice));

  getKeys<<<1,1>>>(this->pointsDevice, this->centersDevice, this->keysDevice, this->width, this->numPoints, this->depth);
  CudaCheckError();
  CudaSafeCall(cudaMemcpy(this->centers, this->centersDevice, this->numPoints * sizeof(float3), cudaMemcpyDeviceToHost));
  CudaSafeCall(cudaMemcpy(this->keys, this->keysDevice, this->numPoints * sizeof(int), cudaMemcpyDeviceToHost));

}

void Octree::cudaFreeMemory(){
  CudaSafeCall(cudaFree(this->keysDevice));
  CudaSafeCall(cudaFree(this->centerDevice));
  CudaSafeCall(cudaFree(this->pointsDevice));
}
