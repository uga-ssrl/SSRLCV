#include "cuda_util.cuh"

/*
NVIDIA GPU COMPATIBILITY
*/
void getGrid(unsigned long numElements, dim3 &grid, int device){
  cudaDeviceProp prop;
  cudaGetDeviceProperties(&prop, device);
    
  grid = {(unsigned int)prop.maxGridSize[0],(unsigned int)prop.maxGridSize[1],(unsigned int)prop.maxGridSize[2]};
  if(numElements < grid.x){
    grid.x = numElements;
    grid.y = 1;
    grid.z = 1;
  }
  else{
    grid.x = 65536;
    if(numElements < grid.x*grid.y){
      grid.y = numElements/grid.x;
      grid.y++;
      grid.z = 1;
    }
    else if(numElements < grid.x*grid.y*grid.z){
      grid.z = numElements/(grid.x*grid.y);
      grid.z++;
    }
  }
}
void checkDims(dim3 grid, dim3 block, int device){
  cudaDeviceProp prop;
  cudaGetDeviceProperties(&prop, device);
  bool goodDims = true;
  if(grid.x > prop.maxGridSize[0]){
    goodDims = false;
  }
  else if(grid.y > prop.maxGridSize[1]){
    goodDims = false;
  }
  else if(grid.z > prop.maxGridSize[2]){
    goodDims = false;
  }
  else if(block.x > prop.maxThreadsDim[0]){
    goodDims = false;
  }
  else if(block.y > prop.maxThreadsDim[1]){
    goodDims = false;
  }
  else if(block.z > prop.maxThreadsDim[2]){
    goodDims = false;
  }
  else if(block.x*block.y*block.z > prop.maxThreadsPerBlock){
    goodDims = false;
  }
  if(!goodDims){
    std::cerr<<"ERROR: grid or block dims are invalid for given device"<<std::endl;
    exit(-1);
    //TODO replace with exception and make more specific
    //maybe make macro like CudaSafeCall()
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
    printf("  -Max Threads per Multiprocessor: %d\n", prop.maxThreadsPerMultiProcessor);
    printf("  -Total number of Multiprocessors: %d\n",prop.multiProcessorCount);
    printf("  -Shared Memory per Multiprocessor (bytes): %lo\n",prop.sharedMemPerMultiprocessor);
    printf("  -32bit Registers per Multiprocessor: %d\n", prop.regsPerMultiprocessor);
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

/*
SIMPLE CUDA METHODS
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