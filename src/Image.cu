#include "Image.cuh"
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
  // err = cudaDeviceSynchronize();
  if (cudaSuccess != err) {
    fprintf(stderr, "cudaCheckError() with sync failed at %s:%i : %s\n",
    file, line, cudaGetErrorString(err));
    exit(-1);
  }
#endif

  return;
}

__device__ __forceinline__ unsigned long getGlobalIdx_2D_1D(){
  unsigned long blockId = blockIdx.y * gridDim.x + blockIdx.x;
  unsigned long threadId = blockId * blockDim.x + threadIdx.x;
  return threadId;
}
__device__ __forceinline__ unsigned char bwaToBW(const uchar2 &color){
  return color.x;
}
__device__ __forceinline__ unsigned char rgbToBW(const uchar3 &color){
  return (color.x/4) + (color.y/2) + (color.z/4);
}
__device__ __forceinline__ unsigned char rgbaToBW(const uchar4 &color){
  return rgbToBW({color.x,color.y,color.z});
}

__global__ void generateBW(int numPixels, unsigned char colorDepth, unsigned char* colorPixels, unsigned char* pixels){
  unsigned long globalID = getGlobalIdx_2D_1D();
  if(globalID < numPixels){
    int numValues = colorDepth;
    switch(numValues){
      case 2:
        pixels[globalID] = bwaToBW({colorPixels[globalID*numValues],colorPixels[globalID*numValues + 1]});
        break;
      case 3:
        pixels[globalID] = rgbToBW({colorPixels[globalID*numValues],colorPixels[globalID*numValues + 1], colorPixels[globalID*numValues + 2]});
        break;
      case 4:
        pixels[globalID] = rgbToBW({colorPixels[globalID*numValues],colorPixels[globalID*numValues + 1], colorPixels[globalID*numValues + 2]});
        break;
      default:
        printf("ERROR colorDepth of %u is not supported\n",numValues);
        asm("trap;");
    }
  }
}

__device__ __host__ Image_Descriptor::Image_Descriptor(){
  this->id = 0;
  this->size = {0,0};
  this->cam_vec = {0.0f,0.0f,0.0f};
  this->cam_pos = {0.0f,0.0f,0.0f};
  this->fov = 0;
  this->foc = 0;
}
__device__ __host__ Image_Descriptor::Image_Descriptor(int id, int2 size){
  this->id = id;
  this->size = size;
  this->cam_vec = {0.0f,0.0f,0.0f};
  this->cam_pos = {0.0f,0.0f,0.0f};
  this->fov = 0;
  this->foc = 0;
}
__device__ __host__ Image_Descriptor::Image_Descriptor(int id, int2 size, float3 cam_pos, float3 camp_dir){
  this->id = id;
  this->size = size;
  this->cam_pos = cam_pos;
  this->cam_vec = cam_vec;
  this->fov = 0;
  this->foc = 0;
}

Image::Image(){
  this->arrayStates[0] = cpu;
  this->arrayStates[1] = cpu;
  this->arrayStates[2] = cpu;
  this->descriptor.id = -1;
  this->features = NULL;
  this->featureDescriptors = NULL;
  this->numFeatures = 0;
  this->numDescriptorsPerFeature = 0;
  this->features_device = NULL;
  this->featureDescriptors_device = NULL;
  this->feature_size = 0;
  this->featureDescriptor_size = 0;
  this->pixels = NULL;
  this->pixels_device = NULL;
}
Image::Image(std::string filePath){
  this->filePath = filePath;
  this->arrayStates[0] = cpu;
  this->arrayStates[1] = cpu;
  this->arrayStates[2] = cpu;
  //read image
  this->pixels = readPNG(filePath.c_str(), this->descriptor.size.y, this->descriptor.size.x, this->colorDepth);

  this->totalPixels = this->descriptor.size.x*this->descriptor.size.y;
  this->pixels_device = NULL;

  this->descriptor.id = -1;
  this->features = NULL;
  this->featureDescriptors = NULL;
  this->numFeatures = 0;
  this->numDescriptorsPerFeature = 0;
  this->features_device = NULL;
  this->featureDescriptors_device = NULL;
  this->feature_size = 0;
  this->featureDescriptor_size = 0;
}
Image::Image(std::string filePath, int id){
  this->filePath = filePath;
  this->arrayStates[0] = cpu;
  this->arrayStates[1] = cpu;
  this->arrayStates[2] = cpu;
  //read image
  this->pixels = readPNG(filePath.c_str(), this->descriptor.size.y, this->descriptor.size.x, this->colorDepth);

  this->totalPixels = this->descriptor.size.x*this->descriptor.size.y;
  this->pixels_device = NULL;

  this->descriptor.id = id;
  this->features = NULL;
  this->featureDescriptors = NULL;
  this->numFeatures = 0;
  this->numDescriptorsPerFeature = 0;
  this->features_device = NULL;
  this->featureDescriptors_device = NULL;
  this->feature_size = 0;
  this->featureDescriptor_size = 0;
}
Image::Image(std::string filePath, int id, MemoryState arrayStates[3]){
  this->filePath = filePath;
  this->arrayStates[0] = arrayStates[0];
  this->arrayStates[1] = arrayStates[1];
  this->arrayStates[2] = arrayStates[2];
  this->pixels_device = NULL;
  for(int i = 1; i < 3; ++i){
    if(arrayStates[i] != cpu && arrayStates[i] != gpu){
      std::cout<<"ERROR invalid MemoryState (cpu and gpu only allowed)"<<std::endl;
      exit(-1);
    }
  }

  //read image
  this->pixels = readPNG(filePath.c_str(), this->descriptor.size.y, this->descriptor.size.x, this->colorDepth);
  this->totalPixels = this->descriptor.size.x*this->descriptor.size.y;
  if(this->arrayStates[0] == gpu){//gpu then

    CudaSafeCall(cudaMalloc((void**)&this->pixels_device, this->totalPixels*((int)this->colorDepth)*sizeof(unsigned char)));
    CudaSafeCall(cudaMemcpy(this->pixels_device, this->pixels, this->totalPixels*((int)this->colorDepth)*sizeof(unsigned char), cudaMemcpyHostToDevice));
    delete[] this->pixels;
  }
  else{//both
    CudaSafeCall(cudaMalloc((void**)&this->pixels_device, this->totalPixels*((int)this->colorDepth)*sizeof(unsigned char)));
    CudaSafeCall(cudaMemcpy(this->pixels_device, this->pixels, this->totalPixels*((int)this->colorDepth)*sizeof(unsigned char), cudaMemcpyHostToDevice));
  }

  this->descriptor.id = id;
  this->features = NULL;
  this->featureDescriptors = NULL;
  this->numFeatures = 0;
  this->numDescriptorsPerFeature = 0;
  this->features_device = NULL;
  this->featureDescriptors_device = NULL;
  this->feature_size = 0;
  this->featureDescriptor_size = 0;
}
Image::~Image(){

}

void Image::setPixelState(MemoryState pixelState){
  if(this->arrayStates[0] != pixelState){
    if(this->arrayStates[0] == cpu && this->pixels != NULL){
      if(pixelState == both){
        CudaSafeCall(cudaMalloc((void**)&this->pixels_device, this->totalPixels*((int)this->colorDepth)*sizeof(unsigned char)));
        CudaSafeCall(cudaMemcpy(this->pixels_device, this->pixels, this->totalPixels*((int)this->colorDepth)*sizeof(unsigned char), cudaMemcpyHostToDevice));
      }
      if(pixelState == gpu){
        delete[] this->pixels;
      }
    }
    else if(this->arrayStates[0] == gpu && this->pixels_device != NULL){
      if(pixelState == both){
        this->pixels = new unsigned char[this->totalPixels*((int)this->colorDepth)];
        CudaSafeCall(cudaMemcpy(this->pixels, this->pixels_device, this->totalPixels*((int)this->colorDepth)*sizeof(unsigned char), cudaMemcpyDeviceToHost));
      }
      if(pixelState == cpu){
        CudaSafeCall(cudaFree(this->pixels_device));
      }
    }
    else if(this->pixels_device != NULL && this->pixels != NULL){
      if(pixelState == cpu){
        CudaSafeCall(cudaFree(this->pixels_device));
      }
      else{
        delete[] this->pixels;
      }
    }
    else{
      std::cout<<"ERROR changing array state for pixels"<<std::endl;
      exit(-1);
    }
  }
  this->arrayStates[0] = pixelState;
}
unsigned char* Image::readColorPixels(){
  if(this->filePath.length() == 0){
    std::cout<<"ERROR must set filePath before trying to read in pixels"<<std::endl;
    exit(-1);
  }
  return readPNG(this->filePath.c_str(), this->descriptor.size.y, this->descriptor.size.x, this->colorDepth);
}
void Image::convertToBW(){
  if(this->colorDepth == 1){
    std::cout<<"Pixels are already bw"<<std::endl;
    return;
  }

  int numPixels = this->totalPixels;

  unsigned char* colorPixels_device;
  CudaSafeCall(cudaMalloc((void**)&colorPixels_device, ((int)this->colorDepth)*numPixels*sizeof(unsigned char)));

  switch(this->arrayStates[0]){
    case cpu:
      CudaSafeCall(cudaMemcpy(colorPixels_device, this->pixels, ((int)this->colorDepth)*numPixels*sizeof(unsigned char), cudaMemcpyHostToDevice));
      delete[] this->pixels;
      this->pixels = NULL;
      break;
    case gpu:
      CudaSafeCall(cudaMemcpy(colorPixels_device, this->pixels_device, ((int)this->colorDepth)*numPixels*sizeof(unsigned char), cudaMemcpyDeviceToDevice));
      CudaSafeCall(cudaFree(this->pixels_device));
      break;
    case both:
      CudaSafeCall(cudaMemcpy(colorPixels_device, this->pixels_device, ((int)this->colorDepth)*numPixels*sizeof(unsigned char), cudaMemcpyDeviceToDevice));
      CudaSafeCall(cudaFree(this->pixels_device));
      delete[] this->pixels;
      this->pixels = NULL;
      this->pixels_device = NULL;
      break;
    default:
      std::cout<<"ERROR invalid pixel array state (neither gpu or cpu specified)"<<std::endl;
      exit(-1);
  }
  std::cout<<"Converting image "<<this->descriptor.id<<" to grayscale"<<std::endl;

  CudaSafeCall(cudaMalloc((void**)&this->pixels_device, numPixels*sizeof(unsigned char)));

  dim3 grid;
  dim3 block;
  getFlatGridBlock(numPixels, grid, block);
  generateBW<<<grid,block>>>(numPixels, this->colorDepth, colorPixels_device, this->pixels_device);
  CudaCheckError();
  CudaSafeCall(cudaFree(colorPixels_device));
  if(this->arrayStates[0] == both){//both
    this->pixels = new unsigned char[numPixels];
    CudaSafeCall(cudaMemcpy(this->pixels, this->pixels_device, numPixels*sizeof(unsigned char), cudaMemcpyDeviceToHost));
  }
  if(this->arrayStates[0] == cpu){
    CudaSafeCall(cudaFree(this->pixels_device));
    this->pixels_device = NULL;
  }
  this->colorDepth = 1;
}
