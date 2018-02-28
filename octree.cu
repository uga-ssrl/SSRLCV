#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <cuda.h>
#include <iostream>
#include <string>
#include <stdio.h>
#include <ctype.h>
#include <vector>
#include <algorithm>
#include <thrust/sort.h>
#include <thrust/pair.h>
#include <thrust/gather.h>

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

}

struct Octree{
  float3* points;
  float3* centers;
  float3* normals;
  int* keys;
  int numPoints;
  Octree();
  Octree(float3* points, float3* normals, int numPoints){
      this->points = points;
      this->normals = normals;
      this->centers = new float3[numPoints];
      this->keys = new int[numPoints];
      this->numPoints = numPoints;
  }
};

int main(){

  //0. find mins and maxs {minX,minY,minZ} {maxX, maxY, maxZ}
  //1. find keys
  //2. sort keys, points, normals, and device_launch_parameters
  //3. compact the keys
  int numPoints = 10000;
  float3* points = new float3[numPoints];
  float3* normals = new float3[numPoints];
  Octree octree = Octree(points, normals, numPoints);

  float width;
  int depth;

  float3* pointsDevice;
  float3* centersDevice;
  int* keysDevice;

  CudaSafeCall(cudaMalloc((void**)&pointsDevice, numPoints * sizeof(float3)));
  CudaSafeCall(cudaMalloc((void**)&centersDevice, numPoints * sizeof(float3)));
  CudaSafeCall(cudaMalloc((void**)&keysDevice, numPoints * sizeof(int)));

  CudaSafeCall(cudaMemcpy(pointsDevice, octree.points, numPoints * sizeof(float3), cudaMemcpyHostToDevice));
  CudaSafeCall(cudaMemcpy(centersDevice, octree.centers, numPoints * sizeof(float3), cudaMemcpyHostToDevice));
  CudaSafeCall(cudaMemcpy(keysDevice, octree.keys, numPoints * sizeof(int), cudaMemcpyHostToDevice));

  getKeys<<<1,1>>>(pointsDevice, centersDevice, keysDevice, width, numPoints, depth);
  CudaCheckError();
  CudaSafeCall(cudaMemcpy(octree.centers, centerDevice, numPoints * sizeof(float3), cudaMemcpyDeviceToHost));
  CudaSafeCall(cudaMemcpy(octree.keys, keysDevice, numPoints * sizeof(int), cudaMemcpyDeviceToHost));

  CudaSafeCall(cudaFree(keysDevice));
  CudaSafeCall(cudaFree(centerDevice));
  CudaSafeCall(cudaFree(pointsDevice));







  return 0;
}
