/** 
* \file cuda_util.cuh
* \brief This file contains simple cuda utility functions.
* \details The primary contents of this file will be cuda vector 
* operator overloads as well as grid and block configurers. 
* \note Even though inline methods for calculating cannot be 
* used from outside the compilation unit, they are listed at the bottom 
* of the cuda_util.cuh header. 
*/
#ifndef CUDA_UTIL_CUH
#define CUDA_UTIL_CUH

#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <cuda.h>
#include <thrust/device_ptr.h>
#include <thrust/sort.h>
#include <thrust/remove.h>
#include "Unity.cuh"

#include <cusolverDn.h>
#include <stdio.h>
#include <cuda_occupancy.h>
#include <iostream>
#include <map>
#include <string>


/**
* \ingroup cuda_util
* \defgroup nvidia_gpu_compatibility
* \{
*/

/**
* \brief Method for getting grid and block for a 1D kernel.
* \details This method calculates a grid and block configuration 
* in an attempt to achieve high levels of CUDA occupancy as well 
* as ensuring there will be enough threads for a specified number of elements. 
* Methods for determining globalThreadID's from the returned grid and block 
* can be found at the bottom of cuda_util.h but must be placed in the same 
* compilational unit.
* \param numElements - number of elements that will be threaded in kernel
* \param grid - dim3 grid argument to be set withint this function
* \param block - dim3 block argument to be set within this function
* \param kernel - function pointer to the kernel that is going to use the grid and block 
* \param dynamicSharedMem - size of dynamic shared memory used in kernel (optional parameter - will default to 0)
* \param device - the NVIDIA GPU device ID (optional parameter - will default to 0)
* \warning This creates grid and block dimensions that guarantee coverage of numElements. 
* This likely means that there will be more threads that necessary, so make sure to check that 
* globalThreadID < numElements in you kernel. Otherwise there will be an illegal memory access. 
*/
template<typename... Types>
void getFlatGridBlock(unsigned long numElements, dim3 &grid, dim3 &block, void (*kernel)(Types...), size_t dynamicSharedMem = 0, int device = 0){
  grid = {1,1,1};
  block = {1,1,1};
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
  block = {(unsigned int)blockSize,1,1}; 
  unsigned int gridSize = (numElements + (unsigned int)blockSize - 1) / (unsigned int)blockSize;
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
void getGridWMaxBlock(unsigned long numElements, dim3 &grid, dim3 &block, void (*kernel)(Types...), size_t dynamicSharedMem = 0, int device = 0){
  grid = {1,1,1};
  block = {1,1,1};
  cudaDeviceProp prop;
  cudaGetDeviceProperties(&prop, device);
  
  int blockSize;
  int minGridSize;
  cudaOccupancyMaxPotentialBlockSize(
    &minGridSize,
    &blockSize,
    kernel,
    dynamicSharedMem,
    0
  );
  block = {(unsigned int)blockSize,1,1}; 
  if(numElements > prop.maxGridSize[0]){
    if(numElements >= 65535L*65535L*65535L){
      grid = {65535,65535,65535};
    }
    else{
      numElements = (numElements/65535L) + 1;
      grid.x = 65535;
      if(numElements > 65535){
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
    grid = {numElements,1,1};
  }
}
template<typename... Types>
void getGridAndBlock(unsigned long numElements, dim3 &grid, unsigned int desiredBlockSize, dim3 &block, void (*kernel)(Types...), size_t dynamicSharedMem = 0, int device = 0){
  grid = {1,1,1};
  block = {1,1,1};
  cudaDeviceProp prop;
  cudaGetDeviceProperties(&prop, device);
  
  int blockSize;
  int minGridSize;
  cudaOccupancyMaxPotentialBlockSize(
    &minGridSize,
    &blockSize,
    kernel,
    dynamicSharedMem,
    0
  );
  block = {(blockSize > desiredBlockSize) ? desiredBlockSize : (unsigned int) blockSize,1,1}; 

  if(numElements > prop.maxGridSize[0]){
    if(numElements >= 65535L*65535L*65535L){
      grid = {65535,65535,65535};
    }
    else{
      numElements = (numElements/65535L) + 1;
      grid.x = 65535;
      if(numElements > 65535){
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
    grid = {numElements,1,1};
  }
}


/**
* \brief Method for getting grid and block for a 2D kernel.
* \details This method calculates a grid and block configuration 
* in an attempt to achieve high levels of CUDA occupancy as well 
* as ensuring there will be enough threads for the specified 2D size. 
* \note User will have to calculate x and y threadIds in their kernels for this to be useful.
* \param size - 2D size for threading over data
* \param grid - dim3 grid argument to be set within this function
* \param block - dim3 block argument to be set within this function
* \param kernel - function pointer to the kernel that is going to use the grid and block 
* \param dynamicSharedMem - size of dynamic shared memory used in kernel (optional parameter - will default to 0)
* \param device - the NVIDIA GPU device ID (optional parameter - will default to 0)
* \warning This creates grid and block dimensions that guarantee coverage of 2D sizes. 
* This likely means that there will be more threads that necessary, so make sure to check that 
* globalThreadID.x < size.x && globalThreadID.y < size.y in you kernel. 
* Otherwise there will be an illegal memory access. 
*/
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
  block = {(unsigned int)sqrt(blockSize),1,1}; 
  block.y = block.x;
  grid = {(size.x + block.x - 1) / block.x,
    (size.y + block.y - 1) / block.y,
    1
  };
  
}
/**
* \brief Method for getting grid for a 1D kernel.
* \details This method is used when the user need a grid 
* configuration to loop over a set of elements when the block 
* has been configured by the user for another threading purpose. 
* \param numElements - number of elements to be threaded
* \param grid - dim3 grid argument to be set within this function
* \param device - the NVIDIA GPU device ID (optional parameter - will be default to 0)
* \warning The block configuration set by the user must be within 
* the capabilities of the device used or an exception will be thrown. 
* Methods like get2DGridBlock() and getFlatGridBlock make sure that 
* this does not happen. 
*/
void getGrid(unsigned long numElements, dim3 &grid, int device = 0);
/**
* \brief Method for checking if grid and block to not exceed limits.
* \details This method will check that grid and block configurations 
* do not exceed the theoretical limit of a specified GPU.
* \param grid - dim3 grid configuration to check
* \param block - dim3 block configuration to check
* \param device - the NVIDIA GPU device ID (optional parameter - will be default to 0)
* \todo Update to take in a kernel to ensure that the 
* configuration will work for the specified kernel. 
* \warning This currently does not guarentee that the configuration 
* will work for a resource greedy kernel. (completing todo will resolve this)
*/
void checkDims(dim3 grid, dim3 block, int device = 0);  


/**
* \brief Cusolver Error Checker
* \details This method will check for cusolver errors when using the cusolver library.
* \param cusolver_status - variable returned by cusolver methods for error analysis
*/
__host__ void cusolverCheckError(cusolverStatus_t cusolver_status);
/**
* \brief Method for printing details on all available NVIDIA GPUs.
*/
void printDeviceProperties();

/**
* \}
*/

/**
* \ingroup cuda_util
* \see Unity
* \brief universal sort method for unity if < operator is overloader
* \todo add option for > or < operator usage
*/
template<typename T>
void sort(ssrlcv::Unity<T>* array){
  if(array == nullptr){
    throw ssrlcv::UnityException("passed array == nullptr in sort(Unity<T>* array)");
  }
  ssrlcv::MemoryState origin = array->getMemoryState();
  ssrlcv::MemoryState fore = array->getFore();
  if(origin == ssrlcv::cpu || fore == ssrlcv::cpu) array->transferMemoryTo(ssrlcv::gpu);
  thrust::device_ptr<T> array_ptr(array->device);
  thrust::device_ptr<T> new_end = thrust::sort(array_ptr,array_ptr+array->size());
  CudaCheckError();
  array->setFore(ssrlcv::gpu);
  if(origin != ssrlcv::gpu) array->setMemoryState(origin);
}
/**
* \ingroup cuda_util
* \see Unity
* \brief universal sort method for unity if < operator is overloader
*/
template<typename T>
void sort(ssrlcv::Unity<T>* array, bool (*compare)(const T&,const T&)){
  if(array == nullptr){
    throw ssrlcv::UnityException("passed array == nullptr in sort(Unity<T>* array)");
  }
  ssrlcv::MemoryState origin = array->getMemoryState();
  ssrlcv::MemoryState fore = array->getFore();
  if(origin == ssrlcv::cpu || fore == ssrlcv::cpu) array->transferMemoryTo(ssrlcv::gpu);
  thrust::device_ptr<T> array_ptr(array->device);
  thrust::device_ptr<T> new_end = thrust::sort(array_ptr,array_ptr+array->size(),compare);
  CudaCheckError();
  array->setFore(ssrlcv::gpu);
  if(origin != ssrlcv::gpu) array->setMemoryState(origin);
}
namespace{
  template<typename T> 
  __host__ __device__ bool check_if_valid(const T &var){
    return var.invalid;
  } 
}


/**
* \brief remove elements of a unity
*/
template<typename T>
void remove(ssrlcv::Unity<T>* array, bool (*validate)(const T&)){
  if(array == nullptr){
    throw ssrlcv::UnityException("passed array == nullptr in remove(Unity<T>* array)");
  }
  ssrlcv::MemoryState origin = array->getMemoryState();
  ssrlcv::MemoryState fore = array->getFore();
  if(origin == ssrlcv::cpu || fore == ssrlcv::cpu) array->transferMemoryTo(ssrlcv::gpu);
  thrust::device_ptr<T> array_ptr(array->device);
  thrust::device_ptr<T> new_end = thrust::remove_if(array_ptr,array_ptr+array->size(),validate);
  CudaCheckError();
  unsigned long numElements = new_end - array_ptr;
  if(numElements == 0){
    std::clog<<"remove(Unity<T>, bool(*validate)(const T&)) led to all elements being removed (array deleted)"<<std::endl;
    delete array;
    return;
  }
  else if(numElements != array->size()){
    T* array_device = nullptr;
    CudaSafeCall(cudaMalloc((void**)&array_device,numElements));
    CudaSafeCall(cudaMemcpy(array_device,array->device,numElements*sizeof(T),cudaMemcpyDeviceToDevice));
    array->setData(array_device,numElements,ssrlcv::gpu);
  }
  if(origin != ssrlcv::gpu) array->setMemoryState(origin);
}

/**
* \brief struct for use in thrust::compaction methods
* \details When calling a thrust::compaction method using 
* is_not_neg() as the protocol will move all non negative values 
* to the end of the array for easy removal. 
*/
struct is_not_neg{
  __host__ __device__
  bool operator()(const char x)
  {
    return (x >= 0);
  }
  __host__ __device__
  bool operator()(const short x)
  {
    return (x >= 0);
  }
  __host__ __device__
  bool operator()(const int x)
  {
    return (x >= 0);
  }
  __host__ __device__
  bool operator()(const float x)
  {
    return (x >= 0);
  }
  __host__ __device__
  bool operator()(const long x)
  {
    return (x >= 0);
  }
  __host__ __device__
  bool operator()(const double x)
  {
    return (x >= 0);
  }
  __host__ __device__
  bool operator()(const long long x)
  {
    return (x >= 0);
  }
};

__device__ __host__ void printBits(size_t const size, void const * const ptr);


/*
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



#endif /* CUDA_UTIL_CUH */