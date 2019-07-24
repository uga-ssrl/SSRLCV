#include "Image.cuh"




__device__ __host__ ssrlcv::Image_Descriptor::Image_Descriptor(){
  this->id = 0;
  this->size = {0,0};
  this->cam_vec = {0.0f,0.0f,0.0f};
  this->cam_pos = {0.0f,0.0f,0.0f};
  this->fov = 0;
  this->foc = 0;
  this->dpix = 0.0f;
}
__device__ __host__ ssrlcv::Image_Descriptor::Image_Descriptor(int id, uint2 size){
  this->id = id;
  this->size = size;
  this->cam_vec = {0.0f,0.0f,0.0f};
  this->cam_pos = {0.0f,0.0f,0.0f};
  this->fov = 0;
  this->foc = 0;
  this->dpix = 0.0f;
}
__device__ __host__ ssrlcv::Image_Descriptor::Image_Descriptor(int id, uint2 size, float3 cam_pos, float3 camp_dir){
  this->id = id;
  this->size = size;
  this->cam_pos = cam_pos;
  this->cam_vec = cam_vec;
  this->fov = 0;
  this->foc = 0;
  this->dpix = 0.0f;
}

void ssrlcv::get_cam_params2view(Image_Descriptor &cam1, Image_Descriptor &cam2, std::string infile){
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

ssrlcv::Image::Image(){
  this->colorDepth = 0;
  this->descriptor.id = -1;
  this->pixels = nullptr;
  this->quadtree = nullptr;
}

ssrlcv::Image::Image(std::string filePath, int id){
  this->filePath = filePath;

  unsigned char* pixels_host = readPNG(filePath.c_str(), this->descriptor.size.y, this->descriptor.size.x, this->colorDepth);
  this->pixels = new Unity<unsigned char>(pixels_host,this->descriptor.size.y*this->descriptor.size.x*this->colorDepth,cpu);

  this->descriptor.id = id;
  this->quadtree = nullptr;
}

ssrlcv::Image::~Image(){
  if(this->quadtree != nullptr){
    delete this->quadtree;
  }
  std::cout<<"destructing"<<std::endl;
  if(this->pixels != nullptr){
    delete this->pixels;
  }
  std::cout<<"destructing"<<std::endl;
}
void ssrlcv::Image::generateQuadtree(){
  uint2 depth = {0,0};
  int2 border = {0,0};
  if(this->descriptor.size.x > this->descriptor.size.y){
    depth.y = (unsigned int)ceil(log2((float)this->descriptor.size.x));
  }
  else{
    depth.y = (unsigned int)ceil(log2((float)this->descriptor.size.y));
  }
  border.x = (pow(2,depth.y) - this->descriptor.size.x)/2;
  border.y = (pow(2,depth.y) - this->descriptor.size.y)/2;
  depth.x = (depth.y <= 4) ? 0 : 4;
  this->quadtree = new Quadtree<unsigned char>(this->descriptor.size,depth,this->pixels,this->colorDepth,border);
}
void ssrlcv::Image::convertToBW(){
  if(this->colorDepth == 1){
    std::cout<<"Pixels are already bw"<<std::endl;
    return;
  }

  MemoryState origin = this->pixels->state;
  this->pixels->transferMemoryTo(gpu);

  unsigned int numPixels = (this->pixels->numElements/this->colorDepth);

  unsigned char* bwPixels_device;
  CudaSafeCall(cudaMalloc((void**)&bwPixels_device, numPixels*sizeof(unsigned char)));

  dim3 grid;
  dim3 block;
  getFlatGridBlock(numPixels, grid, block);
  generateBW<<<grid,block>>>(numPixels, this->colorDepth, this->pixels->device, bwPixels_device);
  CudaCheckError();

  this->pixels->setData(bwPixels_device, numPixels, gpu);
  this->pixels->transferMemoryTo(origin);
  if(origin == cpu){
    this->pixels->clear(gpu);
  }
  this->colorDepth = 1;
}
__device__ __forceinline__ unsigned long ssrlcv::getGlobalIdx_2D_1D(){
  unsigned long blockId = blockIdx.y * gridDim.x + blockIdx.x;
  unsigned long threadId = blockId * blockDim.x + threadIdx.x;
  return threadId;
}
__device__ __forceinline__ unsigned char ssrlcv::bwaToBW(const uchar2 &color){
  return color.x;
}
__device__ __forceinline__ unsigned char ssrlcv::rgbToBW(const uchar3 &color){
  return (color.x/4) + (color.y/2) + (color.z/4);
}
__device__ __forceinline__ unsigned char ssrlcv::rgbaToBW(const uchar4 &color){
  return rgbToBW({color.x,color.y,color.z});
}

__global__ void ssrlcv::generateBW(int numPixels, unsigned int colorDepth, unsigned char* colorPixels, unsigned char* pixels){
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
        pixels[globalID] = rgbaToBW({colorPixels[globalID*numValues],colorPixels[globalID*numValues + 1], colorPixels[globalID*numValues + 2], colorPixels[globalID*numValues + 3]});
        break;
      default:
        printf("ERROR colorDepth of %u is not supported\n",numValues);
        asm("trap;");
    }
  }
}
