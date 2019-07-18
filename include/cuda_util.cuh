#ifndef CUDA_UTIL_CUH
#define CUDA_UTIL_CUH

#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <cuda.h>
#include <cusolverDn.h>
#include <stdio.h>
#include <iostream>

/*
NOTE cannot use device inline statements (nvcc link issue)
*/

//TODO do this for all vector types and related type operations
/*
u&char1,2,3,4
u&short1,2,3,4
u&int1,2,3,4
u&long1,2,3,4
u&longlong1,2,3,4
float1,2,3,4
double1,2,3,4
*/

struct is_not_neg{
  __host__ __device__
  bool operator()(const int x)
  {
    return (x >= 0);
  }
};

__device__ __host__ void printBits(size_t const size, void const * const ptr);
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

__device__ __host__ float2 operator+(const float2 &a, const float2 &b);
__device__ __host__ float2 operator-(const float2 &a, const float2 &b);
__device__ __host__ float2 operator/(const float2 &a, const float2 &b);
__device__ __host__ float2 operator*(const float2 &a, const float2 &b);
__device__ __host__ float dotProduct(const float2 &a, const float2 &b);
__device__ __host__ float2 operator+(const float2 &a, const float &b);
__device__ __host__ float2 operator-(const float2 &a, const float &b);
__device__ __host__ float2 operator/(const float2 &a, const float &b);
__device__ __host__ float2 operator*(const float2 &a, const float &b);
__device__ __host__ float2 operator+(const float &a, const float2 &b);
__device__ __host__ float2 operator-(const float &a, const float2 &b);
__device__ __host__ float2 operator/(const float &a, const float2 &b);
__device__ __host__ float2 operator*(const float &a, const float2 &b);
__device__ __host__ bool operator==(const float2 &a, const float2 &b);

__device__ __host__ float2 operator+(const float2 &a, const int2 &b);
__device__ __host__ float2 operator-(const float2 &a, const int2 &b);
__device__ __host__ float2 operator/(const float2 &a, const int2 &b);
__device__ __host__ float2 operator*(const float2 &a, const int2 &b);
__device__ __host__ float2 operator+(const int2 &a, const float2 &b);
__device__ __host__ float2 operator-(const int2 &a, const float2 &b);
__device__ __host__ float2 operator/(const int2 &a, const float2 &b);
__device__ __host__ float2 operator*(const int2 &a, const float2 &b);

__device__ __host__ int2 operator+(const int2 &a, const int2 &b);
__device__ __host__ int2 operator-(const int2 &a, const int2 &b);
__device__ __host__ float2 operator/(const float2 &a, const int2 &b);
__device__ __host__ int2 operator*(const int2 &a, const int2 &b);
__device__ __host__ int dotProduct(const int2 &a, const int2 &b);
__device__ __host__ float2 operator/(const int2 &a, const float &b);

__device__ __host__ uint2 operator+(const uint2 &a, const uint2 &b);
__device__ __host__ uint2 operator-(const uint2 &a, const uint2 &b);
__device__ __host__ uint2 operator*(const uint2 &a, const uint2 &b);
__device__ __host__ uint2 operator/(const uint2 &a, const uint2 &b);
__device__ __host__ int2 operator+(const int2 &a, const uint2 &b);
__device__ __host__ int2 operator-(const int2 &a, const uint2 &b);
__device__ __host__ int2 operator*(const int2 &a, const uint2 &b);
__device__ __host__ int2 operator/(const int2 &a, const uint2 &b);
__device__ __host__ int2 operator+(const uint2 &a, const int2 &b);
__device__ __host__ int2 operator-(const uint2 &a, const int2 &b);
__device__ __host__ int2 operator*(const uint2 &a, const int2 &b);
__device__ __host__ int2 operator/(const uint2 &a, const int2 &b);

__device__ __host__ float2 operator+(const int2 &a, const float &b);
__device__ __host__ float2 operator-(const int2 &a, const float &b);
__device__ __host__ float2 operator/(const float2 &a, const float &b);
__device__ __host__ float2 operator*(const int2 &a, const float &b);
__device__ __host__ float2 operator/(const int2 &a, const float &b);

__device__ __host__ ulong2 operator+(const ulong2 &a, const int2 &b);
__device__ __host__ ulong2 operator-(const ulong2 &a, const int2 &b);
__device__ __host__ ulong2 operator/(const ulong2 &a, const int2 &b);
__device__ __host__ ulong2 operator*(const ulong2 &a, const int2 &b);
__device__ __host__ ulong2 operator/(const ulong2 &a, const int2 &b);

__device__ __host__ ulong2 operator+(const int2 &a, const ulong2 &b);
__device__ __host__ ulong2 operator-(const int2 &a, const ulong2 &b);
__device__ __host__ ulong2 operator/(const int2 &a, const ulong2 &b);
__device__ __host__ ulong2 operator*(const int2 &a, const ulong2 &b);
__device__ __host__ ulong2 operator/(const int2 &a, const ulong2 &b);

__device__ __host__ bool operator>(const float2 &a, const float &b);
__device__ __host__ bool operator<(const float2 &a, const float &b);
__device__ __host__ bool operator>(const float2 &a, const float2 &b);
__device__ __host__ bool operator<(const float2 &a, const float2 &b);
__device__ __host__ bool operator>(const float2 &a, const int2 &b);
__device__ __host__ bool operator<(const float2 &a, const int2 &b);

/*
must be in the compilation unit

__device__ __forceinline__ unsigned long getGlobalIdx_1D_1D(){
  return blockIdx.x *blockDim.x + threadIdx.x;
}
__device__ __forceinline__ unsigned long getGlobalIdx_1D_2D(){
  return blockIdx.x * blockDim.x * blockDim.y +
    threadIdx.y * blockDim.x + threadIdx.x;
}
__device__ __forceinline__ unsigned long getGlobalIdx_1D_3D(){
  return blockIdx.x * blockDim.x * blockDim.y * blockDim.z +
    threadIdx.z * blockDim.y * blockDim.x +
    threadIdx.y * blockDim.x + threadIdx.x;
}
__device__ __forceinline__ unsigned long getGlobalIdx_2D_1D(){
  unsigned long blockId = blockIdx.y * gridDim.x + blockIdx.x;
  unsigned long threadId = blockId * blockDim.x + threadIdx.x;
  return threadId;
}
__device__ __forceinline__ unsigned long getGlobalIdx_2D_2D(){
  unsigned long blockId = blockIdx.x + blockIdx.y * gridDim.x;
  unsigned long threadId = blockId * (blockDim.x * blockDim.y) +
    (threadIdx.y * blockDim.x) + threadIdx.x;
  return threadId;
}
__device__ __forceinline__ unsigned long getGlobalIdx_2D_3D(){
  unsigned long blockId = blockIdx.x + blockIdx.y * gridDim.x;
  unsigned long threadId = blockId * (blockDim.x * blockDim.y * blockDim.z) +
    (threadIdx.z * (blockDim.x * blockDim.y)) +
    (threadIdx.y * blockDim.x) + threadIdx.x;
  return threadId;
}
__device__ __forceinline__ unsigned long getGlobalIdx_3D_1D(){
  unsigned long blockId = blockIdx.x + blockIdx.y * gridDim.x +
    gridDim.x * gridDim.y * blockIdx.z;
  unsigned long threadId = blockId * blockDim.x + threadIdx.x;
  return threadId;
}
__device__ __forceinline__ unsigned long getGlobalIdx_3D_2D(){
  unsigned long blockId = blockIdx.x + blockIdx.y * gridDim.x +
    gridDim.x * gridDim.y * blockIdx.z;
  unsigned long threadId = blockId * (blockDim.x * blockDim.y) +
    (threadIdx.y * blockDim.x) + threadIdx.x;
  return threadId;
}
__device__ __forceinline__ unsigned long getGlobalIdx_3D_3D(){
  unsigned long blockId = blockIdx.x + blockIdx.y * gridDim.x +
    gridDim.x * gridDim.y * blockIdx.z;
  unsigned long threadId = blockId * (blockDim.x * blockDim.y * blockDim.z) +
    (threadIdx.z * (blockDim.x * blockDim.y)) +
    (threadIdx.y * blockDim.x) + threadIdx.x;
  return threadId;
}
*/

//TODO make grid and block setting max occupancy and add more overloaded methods for various situations
//currently only supports up to grid.x grid.y
void getFlatGridBlock(unsigned long size, dim3 &grid, dim3 &block);
void getGrid(unsigned long size, dim3 &grid);
__host__ void cusolverCheckError(cusolverStatus_t cusolver_status);
void printDeviceProperties();

#endif /* CUDA_UTIL_CUH */
