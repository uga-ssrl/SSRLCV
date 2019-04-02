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
  this->dpix = 0.0f;
}
__device__ __host__ Image_Descriptor::Image_Descriptor(int id, int2 size){
  this->id = id;
  this->size = size;
  this->cam_vec = {0.0f,0.0f,0.0f};
  this->cam_pos = {0.0f,0.0f,0.0f};
  this->fov = 0;
  this->foc = 0;
  this->dpix = 0.0f;
}
__device__ __host__ Image_Descriptor::Image_Descriptor(int id, int2 size, float3 cam_pos, float3 camp_dir){
  this->id = id;
  this->size = size;
  this->cam_pos = cam_pos;
  this->cam_vec = cam_vec;
  this->fov = 0;
  this->foc = 0;
  this->dpix = 0.0f;
}

void get_cam_params2view(Image_Descriptor &cam1, Image_Descriptor &cam2, std::string infile){
  std::ifstream input(infile);
  std::string line;
  float res = 0.0f;
  while(std::getline(input, line)) {
    std::istringstream iss(line);
    std::string param;
    float arg1;
    float arg2;
    float arg3;
    iss >> param >> arg1;
    if(param.compare("foc") == 0) {
      cam1.foc = arg1;
      cam2.foc = arg1;
    }
    else if(param.compare("fov") == 0) {
      cam1.fov = arg1;
      cam2.fov = arg1;
    }
    else if(param.compare("res") == 0) {
      res = arg1;
    }
    else if(param.compare("cam1C") == 0) {
      iss >> arg2 >> arg3;
      cam1.cam_pos.x = arg1;
      cam1.cam_pos.y = arg2;
      cam1.cam_pos.z = arg3;
    }
    else if(param.compare("cam1V") == 0) {
      iss >> arg2 >> arg3;
      cam1.cam_vec.x = arg1;
      cam1.cam_vec.y = arg2;
      cam1.cam_vec.z = arg3;
    }
    else if(param.compare("cam2C") == 0) {
      iss >> arg2 >> arg3;
      cam2.cam_pos.x = arg1;
      cam2.cam_pos.y = arg2;
      cam2.cam_pos.z = arg3;
    }
    else if(param.compare("cam2V") == 0) {
      iss >> arg2 >> arg3;
      cam2.cam_vec.x = arg1;
      cam2.cam_vec.y = arg2;
      cam2.cam_vec.z = arg3;
    }
  }
  cam1.dpix = (cam1.foc*tan(cam1.fov/2))/(res/2);
  cam2.dpix = (cam2.foc*tan(cam2.fov/2))/(res/2);
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

// TODO remove pass thru variables, they are not needed
void Image::segment(int x_num, int y_num, int x_size, int y_size){
  // TODO can this be done in a kernel?

  int total_x = x_num * x_size;
  int total_y = y_num * y_size;

  // build the segments
  this->segments = new Image[x_num*y_num];

  std::cout << "setting up..." << std::endl;
  for (int i = 0; i < x_num*y_num; i++){
    segments[i] = new Image()[0]; // why did this work?
    // needed for all segments
    segments[i].descriptor.foc = 0.160;
    segments[i].descriptor.fov = (11.4212*PI/180);
    segments[i].descriptor.cam_pos = {7.81417, 0.0f, 44.3630};
    segments[i].descriptor.cam_vec = {-0.173648, 0.0f, -0.984808};
    // fill in the segment helper for this guy
    segments[i].segment_helper.is_segment = true;
    segments[i].segment_helper.segment_number = i;
    segments[i].segment_helper.size.x = x_size;
    segments[i].segment_helper.size.y = y_size;
    segments[i].segment_helper.pix_filled = 0; // same thing as where to put next pix
    segments[i].pixels = new unsigned char[x_size,y_size]; // empty image segments
  }
  //std::cout << "wow!" << std::endl;
  //int total = total_x * total_y;
  std::cout << "about to fill segs..." << std::endl;
  unsigned char pix;
  int tofill, seg_num, index;
  for (int i = 0; i < total_y; i++) {
    for (int j = 0; j < total_x; j++) {
      index = j + i*total_x;
      std::cout << "getting index..." << std::endl;
      pix = this->pixels[index];
      std::cout << "getting seg num..." << std::endl;
      seg_num = getSegNum(j,i);
      std::cout << "getting and setting..." << std::endl;
      tofill = segments[seg_num].segment_helper.pix_filled;
      segments[seg_num].pixels[tofill] = pix;
      segments[seg_num].segment_helper.pix_filled++;
    }
  }

}

bool Image::isInSegment(int x_run, int y_run){
  return false;
}

int Image::getSegNum(int x, int y){
  int x_i = (x + 1) / this->descriptor.segment_size.x;
  int y_i = (y + 1) / this->descriptor.segment_size.y;
  return x_i + x_i * y_i;
  // return -1;
}






































//yee
