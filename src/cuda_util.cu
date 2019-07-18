#include "cuda_util.cuh"

#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <cuda.h>
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
__device__ __host__ void printBits(size_t const size, void const * const ptr){
  unsigned char *b = (unsigned char*) ptr;
  unsigned char byte;
  int i, j;
  printf("bits - ");
  for (i=size-1;i>=0;i--){
    for (j=7;j>=0;j--){
      byte = (b[i] >> j) & 1;
      printf("%u", byte);
    }
  }
  printf("\n");
}

__device__ __host__ float3 operator+(const float3 &a, const float3 &b) {
  return {a.x+b.x, a.y+b.y, a.z+b.z};
}
__device__ __host__ float3 operator-(const float3 &a, const float3 &b) {
  return {a.x-b.x, a.y-b.y, a.z-b.z};
}
__device__ __host__ float3 operator/(const float3 &a, const float3 &b) {
  return {a.x/b.x, a.y/b.y, a.z/b.z};
}
__device__ __host__ float3 operator*(const float3 &a, const float3 &b) {
  return {a.x*b.x, a.y*b.y, a.z*b.z};
}
__device__ __host__ float dotProduct(const float3 &a, const float3 &b){
  return (a.x*b.x) + (a.y*b.y) + (a.z*b.z);
}
__device__ __host__ float3 operator+(const float3 &a, const float &b){
  return {a.x+b, a.y+b, a.z+b};
}
__device__ __host__ float3 operator-(const float3 &a, const float &b){
  return {a.x-b, a.y-b, a.z-b};
}
__device__ __host__ float3 operator/(const float3 &a, const float &b){
  return {a.x/b, a.y/b, a.z/b};
}
__device__ __host__ float3 operator*(const float3 &a, const float &b){
  return {a.x*b, a.y*b, a.z*b};
}
__device__ __host__ float3 operator+(const float &a, const float3 &b) {
  return {a+b.x, a+b.y, a+b.z};
}
__device__ __host__ float3 operator-(const float &a, const float3 &b) {
  return {a-b.x, a-b.y, a-b.z};
}
__device__ __host__ float3 operator/(const float &a, const float3 &b) {
  return {a/b.x, a/b.y, a/b.z};
}
__device__ __host__ float3 operator*(const float &a, const float3 &b) {
  return {a*b.x, a*b.y, a*b.z};
}
__device__ __host__ bool operator==(const float3 &a, const float3 &b){
  return (a.x==b.x)&&(a.y==b.y)&&(a.z==b.z);
}

__device__ __host__ float2 operator+(const float2 &a, const float2 &b){
  return {a.x + b.x, a.y + b.y};
}
__device__ __host__ float2 operator-(const float2 &a, const float2 &b){
  return {a.x - b.x, a.y - b.y};
}
__device__ __host__ float2 operator/(const float2 &a, const float2 &b){
  return {a.x / b.x, a.y / b.y};
}
__device__ __host__ float2 operator*(const float2 &a, const float2 &b){
  return {a.x * b.x, a.y * b.y};
}
__device__ __host__ float dotProduct(const float2 &a, const float2 &b){
  return (a.x*b.x) + (a.y*b.y);
}
__device__ __host__ float2 operator+(const float2 &a, const float &b){
  return {a.x + b, a.y + b};
}
__device__ __host__ float2 operator-(const float2 &a, const float &b){
  return {a.x - b, a.y - b};
}
__device__ __host__ float2 operator/(const float2 &a, const float &b){
  return {a.x / b, a.y / b};
}
__device__ __host__ float2 operator*(const float2 &a, const float &b){
  return {a.x * b, a.y * b};
}
__device__ __host__ float2 operator+(const float &a, const float2 &b){
  return {a + b.x, a + b.y};
}
__device__ __host__ float2 operator-(const float &a, const float2 &b){
  return {a - b.x, a - b.y};
}
__device__ __host__ float2 operator/(const float &a, const float2 &b){
  return {a / b.x, a / b.y};
}
__device__ __host__ float2 operator*(const float &a, const float2 &b){
  return {a * b.x, a * b.y};
}
__device__ __host__ bool operator==(const float2 &a, const float2 &b){
  return a.x == b.x && a.y == b.y;
}

__device__ __host__ float2 operator+(const float2 &a, const int2 &b){
  return {a.x + ((float) b.x), a.y + ((float) b.y)};
}
__device__ __host__ float2 operator-(const float2 &a, const int2 &b){
  return {a.x - ((float) b.x), a.y - ((float) b.y)};
}
__device__ __host__ float2 operator/(const float2 &a, const int2 &b){
  return {a.x / ((float) b.x), a.y / ((float) b.y)};
}
__device__ __host__ float2 operator*(const float2 &a, const int2 &b){
  return {a.x * ((float) b.x), a.y * ((float) b.y)};
}
__device__ __host__ float2 operator+(const int2 &a, const float2 &b){
  return {((float) a.x) + b.x, ((float) a.y) + b.y};
}
__device__ __host__ float2 operator-(const int2 &a, const float2 &b){
  return {((float) a.x) - b.x, ((float) a.y) - b.y};
}
__device__ __host__ float2 operator/(const int2 &a, const float2 &b){
  return {((float) a.x) / b.x, ((float) a.y) / b.y};
}
__device__ __host__ float2 operator*(const int2 &a, const float2 &b){
  return {(float) a.x * b.x, ((float) a.y) * b.y};
}

__device__ __host__ int2 operator+(const int2 &a, const int2 &b){
  return {a.x + b.x, a.y + b.y};
}
__device__ __host__ int2 operator-(const int2 &a, const int2 &b){
  return {a.x - b.x, a.y - b.y};
}
__device__ __host__ float2 operator/(const int2 &a, const int2 &b){
  return {((float) a.x) / ((float) b.x), ((float) a.y) / ((float) b.y)};
}
__device__ __host__ int2 operator*(const int2 &a, const int2 &b){
  return {a.x * b.x, a.y * b.y};
}
__device__ __host__ int dotProduct(const int2 &a, const int2 &b){
  return (a.x*b.x) + (a.y*b.y);
}
__device__ __host__ float2 operator/(const int2 &a, const float &b){
  return {((float)a.x)/b, ((float)a.y)/b};
}
__device__ __host__ float2 operator+(const int2 &a, const float &b){
  return {((float)a.x) + b, ((float)a.y) + b};
}
__device__ __host__ float2 operator-(const int2 &a, const float &b){
  return {((float)a.x) - b, ((float)a.y) - b};
}

__device__ __host__ uint2 operator+(const uint2 &a, const uint2 &b){
  return {a.x + b.x, a.y + b.y};
}
__device__ __host__ uint2 operator-(const uint2 &a, const uint2 &b){
  return {a.x - b.x, a.y - b.y};
}
__device__ __host__ uint2 operator*(const uint2 &a, const uint2 &b){
  return {a.x * b.x, a.y * b.y};
}
__device__ __host__ uint2 operator/(const uint2 &a, const uint2 &b){
  return {a.x / b.x, a.y / b.y};
}
__device__ __host__ int2 operator+(const int2 &a, const uint2 &b){
  return {a.x + b.x, a.y + b.y};
}
__device__ __host__ int2 operator-(const int2 &a, const uint2 &b){
  return {a.x - b.x, a.y - b.y};
}
__device__ __host__ int2 operator*(const int2 &a, const uint2 &b){
  return {a.x * b.x, a.y * b.y};
}
__device__ __host__ int2 operator/(const int2 &a, const uint2 &b){
  return {a.x / b.x, a.y / b.y};
}
__device__ __host__ int2 operator+(const uint2 &a, const int2 &b){
  return {a.x + b.x, a.y + b.y};
}
__device__ __host__ int2 operator-(const uint2 &a, const int2 &b){
  return {a.x - b.x, a.y - b.y};
}
__device__ __host__ int2 operator*(const uint2 &a, const int2 &b){
  return {a.x * b.x, a.y * b.y};
}
__device__ __host__ int2 operator/(const uint2 &a, const int2 &b){
  return {a.x / b.x, a.y / b.y};
}





__device__ __host__ ulong2 operator+(const ulong2 &a, const int2 &b){
  return {a.x + (unsigned long)b.x, a.y + (unsigned long)b.y};
}
__device__ __host__ ulong2 operator-(const ulong2 &a, const int2 &b){
  return  {a.x - (unsigned long)b.x, a.y - (unsigned long)b.y};
}
__device__ __host__ ulong2 operator*(const ulong2 &a, const int2 &b){
  return  {a.x * (unsigned long)b.x, a.y * (unsigned long)b.y};
}

__device__ __host__ ulong2 operator+(const int2 &a, const ulong2 &b){
  return {(unsigned long)a.x + b.x, (unsigned long)a.y + b.y};
}
__device__ __host__ ulong2 operator-(const int2 &a, const ulong2 &b){
  return {(unsigned long)a.x - b.x, (unsigned long)a.y - b.y};
}
__device__ __host__ ulong2 operator*(const int2 &a, const ulong2 &b){
  return {(unsigned long)a.x * b.x, (unsigned long)a.y * b.y};
}

__device__ __host__ float2 operator*(const int2 &a, const float &b){
  return {((float)a.x) * b, ((float)a.y) * b};
}

__device__ __host__ bool operator>(const float2 &a, const float &b){
  return (a.x > b) && (a.y > b);
}

__device__ __host__ bool operator<(const float2 &a, const float &b){
  return (a.x < b) && (a.y < b);
}

__device__ __host__ bool operator>(const float2 &a, const float2 &b){
  return (a.x > b.x) && (a.y > b.y);
}

__device__ __host__ bool operator<(const float2 &a, const float2 &b){
  return (a.x < b.x) && (a.y < b.y);
}

__device__ __host__ bool operator>(const float2 &a, const int2 &b){
  return (a.x > b.x) && (a.y > b.y);
}

__device__ __host__ bool operator<(const float2 &a, const int2 &b){
  return (a.x < b.x) && (a.y < b.y);
}




void max_occupancy(dim3 &grid, dim3 &block, const int &gridDim, const int &blockDim, const uint3 &forceBlock, const long &valueToAchieve){

}

void getFlatGridBlock(unsigned long size, dim3 &grid, dim3 &block) {
  if(2147483647 > size){
    grid.x = size;
  }
  else if((unsigned long) 2147483647 * 1024 > size){
    grid.x = 2147483647;
    block.x = 1024;
    while(block.x * grid.x > size){
      block.x--;
    }
    block.x++;
  }
  else{
    grid.x = 65535;
    block.x = 1024;
    grid.y = 1;
    while(grid.x * grid.y * block.x < size){
      grid.y++;
    }
  }
}
void getGrid(unsigned long size, dim3 &grid) {
  if(2147483647 > size){
    grid.x = size;
  }
  else{
    grid.x = 65535;
    grid.y = 1;
    while(grid.x * grid.y * grid.y < size){
      grid.y++;
    }
  }
}

__host__ void cusolverCheckError(cusolverStatus_t cusolver_status){
  switch (cusolver_status){
      case CUSOLVER_STATUS_SUCCESS:
          std::cout<<"CUSOLVER_SUCCESS"<<std::endl;
          break;

      case CUSOLVER_STATUS_NOT_INITIALIZED:
          std::cout<<"CUSOLVER_STATUS_NOT_INITIALIZED"<<std::endl;
          exit(-1);

      case CUSOLVER_STATUS_ALLOC_FAILED:
          std::cout<<"CUSOLVER_STATUS_ALLOC_FAILED"<<std::endl;
          exit(-1);

      case CUSOLVER_STATUS_INVALID_VALUE:
          std::cout<<"CUSOLVER_STATUS_INVALID_VALUE"<<std::endl;
          exit(-1);

      case CUSOLVER_STATUS_ARCH_MISMATCH:
          std::cout<<"CUSOLVER_STATUS_ARCH_MISMATCH"<<std::endl;
          exit(-1);

      case CUSOLVER_STATUS_EXECUTION_FAILED:
          std::cout<<"CUSOLVER_STATUS_EXECUTION_FAILED"<<std::endl;
          exit(-1);

      case CUSOLVER_STATUS_INTERNAL_ERROR:
          std::cout<<"CUSOLVER_STATUS_INTERNAL_ERROR"<<std::endl;
          exit(-1);

      case CUSOLVER_STATUS_MATRIX_TYPE_NOT_SUPPORTED:
          std::cout<<"CUSOLVER_STATUS_MATRIX_TYPE_NOT_SUPPORTED"<<std::endl;
          exit(-1);
  }
}

//prints tx2 info relevant to cuda devlopment
//citation: this method comes from the nvidia formums, and has been modified slightly
void printDeviceProperties(){
	std::cout<<"\n---------------START OF DEVICE PROPERTIES---------------\n"<<std::endl;

  int nDevices;
  cudaGetDeviceCount(&nDevices);      //find num of devices on tx2

  for (int i = 0; i < nDevices; i++)  //print info on each device
	{
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, i);

    printf("Device Number: %d\n", i);
    printf(" -Device name: %s\n\n", prop.name);
    printf(" -Memory\n  -Memory Clock Rate (KHz): %d\n", prop.memoryClockRate);
    printf("  -Memory Bus Width (bits): %d\n",prop.memoryBusWidth);
    printf("  -Peak Memory Bandwidth (GB/s): %f\n",2.0 * prop.memoryClockRate * (prop.memoryBusWidth / 8) / 1.0e6);
    printf("  -Total Global Memory (bytes): %lo\n", prop.totalGlobalMem);
    printf("  -Total Const Memory (bytes): %lo\n", prop.totalConstMem);
    printf("  -Max pitch allowed for memcpy in regions allocated by cudaMallocPitch() (bytes): %lo\n\n", prop.memPitch);
    printf("  -Shared Memory per block (bytes): %lo\n", prop.sharedMemPerBlock);
    printf("  -Max number of threads per block: %d\n",prop.maxThreadsPerBlock);
    printf("  -Max number of blocks: %dx%dx%d\n",prop.maxGridSize[0], prop.maxGridSize[1], prop.maxGridSize[2]);
    printf("  -32bit Registers per block: %d\n", prop.regsPerBlock);
    printf("  -Threads per warp: %d\n\n", prop.warpSize);
    printf("  -Total number of Multiprocessors: %d\n",prop.multiProcessorCount);
    printf("  -Shared Memory per Multiprocessor (bytes): %lo\n",prop.sharedMemPerMultiprocessor);
    printf("  -32bit Registers per Multiprocessor: %d\n\n", prop.regsPerMultiprocessor);
    printf("  -Number of asynchronous engines: %d\n", prop.asyncEngineCount);
    printf("  -Texture alignment requirement (bytes): %lo\n  -Texture base addresses that are aligned to "
    "textureAlignment bytes do not need an offset applied to texture fetches.\n\n", prop.textureAlignment);
    printf(" -Device Compute Capability:\n  -Major revision #: %d\n  -Minor revision #: %d\n", prop.major, prop.minor);

		printf(" -Run time limit for kernels that get executed on this device: ");
		if(prop.kernelExecTimeoutEnabled)
		{
      printf("YES\n");
    }
    else
		{
      printf("NO\n");
    }

    printf(" -Device is ");
    if(prop.integrated)
		{
      printf("integrated. (motherboard)\n");
    }
    else
		{
      printf("discrete. (card)\n\n");
    }

    if(prop.isMultiGpuBoard)
		{
      printf(" -Device is on a MultiGPU configurations.\n\n");
    }

    switch(prop.computeMode)
		{
      case(0):
        printf(" -Default compute mode (Multiple threads can use cudaSetDevice() with this device)\n");
        break;
      case(1):
        printf(" -Compute-exclusive-thread mode (Only one thread in one processwill be able to use\n cudaSetDevice() with this device)\n");
        break;
      case(2):
        printf(" -Compute-prohibited mode (No threads can use cudaSetDevice() with this device)\n");
        break;
      case(3):
        printf(" -Compute-exclusive-process mode (Many threads in one process will be able to use\n cudaSetDevice() with this device)\n");
        break;
      default:
        printf(" -GPU in unknown compute mode.\n");
        break;
    }

    if(prop.canMapHostMemory)
		{
      printf("\n -The device can map host memory into the CUDA address space for use with\n cudaHostAlloc() or cudaHostGetDevicePointer().\n\n");
    }
    else
		{
      printf("\n -The device CANNOT map host memory into the CUDA address space.\n\n");
    }

    printf(" -ECC support: ");
    if(prop.ECCEnabled)
		{
      printf(" ON\n");
    }
    else
		{
      printf(" OFF\n");
    }

    printf(" -PCI Bus ID: %d\n", prop.pciBusID);
    printf(" -PCI Domain ID: %d\n", prop.pciDomainID);
    printf(" -PCI Device (slot) ID: %d\n", prop.pciDeviceID);

    printf(" -Using a TCC Driver: ");
    if(prop.tccDriver)
		{
      printf("YES\n");
    }
    else
		{
      printf("NO\n");
    }
  }
  std::cout<<"\n----------------END OF DEVICE PROPERTIES----------------\n"<<std::endl;
}
