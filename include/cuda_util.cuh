/** \file cuda_util.cuh
* \brief file for simple utility methods surrounding CUDA device code
*/
#ifndef CUDA_UTIL_CUH
#define CUDA_UTIL_CUH

#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <cuda.h>
#include <cusolverDn.h>
#include <stdio.h>
#include <cuda_occupancy.h>
#include <iostream>
#include <map>
#include <string>

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
__device__ void orderInt3(int3 &toOrder);
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
__device__ __host__ uint2 operator+(const uint2 &a, const int &b);
__device__ __host__ uint2 operator-(const uint2 &a, const int &b);
__device__ __host__ uint2 operator*(const uint2 &a, const int &b);
__device__ __host__ uint2 operator/(const uint2 &a, const int &b);
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

__device__ __host__ int2 operator+(const int2 &a, const int &b);
__device__ __host__ int2 operator-(const int2 &a, const int &b);

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

template<typename... Types>
void getFlatGridBlock(unsigned long numElements, dim3 &grid, dim3 &block, void (*kernel)(Types...), size_t dynamicSharedMem = 0, int device = 0){
  cudaDeviceProp prop;
  cudaGetDeviceProperties(&prop, device);
  
  int blockSize;
  int minGridSize;
  cudaOccupancyMaxPotentialBlockSize(
    &minGridSize,
    &blockSize,
    kernel,
    dynamicSharedMem,
    numElements
  );
  block = {blockSize,1,1}; 
  unsigned long gridSize = (numElements + (unsigned long)blockSize - 1) / (unsigned long)blockSize;
  if(gridSize > prop.maxGridSize[0]){
    if(gridSize >= 65535L*65535L*65535L){
      grid = {65535,65535,65535};
    }
    else{
      gridSize = (gridSize/65535L) + 1;
      grid.x = 65535;
      if(gridSize > 65535){
        grid.z = (grid.y/65535) + 1;
        grid.y = 65535; 
      }
      else{
        grid.y = 65535;
        grid.z = 1;
      }
    }
  }
  else{
    grid = {gridSize,1,1};
  }
}
template<typename... Types>
void get2DGridBlock(uint2 size, dim3 &grid, dim3 &block,  void (*kernel)(Types...), size_t dynamicSharedMem = 0, int device = 0){
  cudaDeviceProp prop;
  cudaGetDeviceProperties(&prop, device);
  
  int blockSize;
  int minGridSize;
  cudaOccupancyMaxPotentialBlockSize(
    &minGridSize,
    &blockSize,
    kernel,
    dynamicSharedMem,
    size.x*size.y
  );
  block = {sqrt(blockSize),1,1}; 
  block.y = block.x;
  grid = {(size.x + (unsigned long)block.x - 1) / (unsigned long)block.x,
    (size.y + (unsigned long)block.y - 1) / (unsigned long)block.y,
    1};
  
}
void getGrid(unsigned long numElements, dim3 &grid, int device = 0);
void checkDims(dim3 grid, dim3 block, int device = 0);  


__host__ void cusolverCheckError(cusolverStatus_t cusolver_status);
void printDeviceProperties();

/* We can overload the following
+ 	- 	* 	/ 	% 	^
& 	| 	~ 	! 	, 	=
< 	> 	<= 	>= 	++ 	--
<< 	>> 	== 	!= 	&& 	||
+= 	-= 	/= 	%= 	^= 	&=
|= 	*= 	<<= 	>>= 	[] 	()
-> 	->* 	new 	new [] 	delete 	delete []
*/


#endif /* CUDA_UTIL_CUH */
