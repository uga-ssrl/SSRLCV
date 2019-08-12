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

ssrlcv::Image::Image(){
  this->descriptor.id = -1;
  this->quadtree = nullptr;
  this->filePath = "n/a";
}
ssrlcv::Image::Image(std::string filePath, int id){
  this->filePath = filePath;
  this->descriptor.id = id;
  unsigned int colorDepth = 1;
  unsigned char* pixels_host = readPNG(filePath.c_str(), this->descriptor.size.y, this->descriptor.size.x, colorDepth);
  Unity<unsigned char>* pixels = new Unity<unsigned char>(pixels_host,this->descriptor.size.y*this->descriptor.size.x*colorDepth,cpu);
  unsigned int depth = 0;
  int2 border = {0,0};
  if(this->descriptor.size.x > this->descriptor.size.y){
    depth = (unsigned int)ceil(log2((float)this->descriptor.size.x));
  }
  else{
    depth = (unsigned int)ceil(log2((float)this->descriptor.size.y));
  }
  border.x = (pow(2,depth) - this->descriptor.size.x)/2;
  border.y = (pow(2,depth) - this->descriptor.size.y)/2;
  this->quadtree = new Quadtree<unsigned char>(this->descriptor.size,depth,pixels,colorDepth,border);
}
ssrlcv::Image::Image(std::string filePath, unsigned int convertColorDepthTo, int id){
  this->filePath = filePath;
  this->descriptor.id = id;
  unsigned int colorDepth = 1;
  unsigned char* pixels_host = readPNG(filePath.c_str(), this->descriptor.size.y, this->descriptor.size.x, colorDepth);
  Unity<unsigned char>* pixels = new Unity<unsigned char>(pixels_host,this->descriptor.size.y*this->descriptor.size.x*colorDepth,cpu);
  if(convertColorDepthTo == 1){
    convertToBW(pixels,colorDepth);
    colorDepth = 1;
  }
  else if(convertColorDepthTo != 0){
    std::cerr<<"ERROR: Image() does not currently support conversion to anything but BW"<<std::endl;
    exit(-1);
  }
  unsigned int depth = 0;
  int2 border = {0,0};
  if(this->descriptor.size.x > this->descriptor.size.y){
    depth = (unsigned int)ceil(log2((float)this->descriptor.size.x));
  }
  else{
    depth = (unsigned int)ceil(log2((float)this->descriptor.size.y));
  }
  border.x = (pow(2,depth) - this->descriptor.size.x)/2;
  border.y = (pow(2,depth) - this->descriptor.size.y)/2;
  this->quadtree = new Quadtree<unsigned char>(this->descriptor.size,depth,pixels,colorDepth,border);
}
ssrlcv::Image::Image(std::string filePath, unsigned int convertColorDepthTo, unsigned int quadtreeBinDepth, int id){
  this->filePath = filePath;
  this->descriptor.id = id;
  unsigned int colorDepth = 1;
  unsigned char* pixels_host = readPNG(filePath.c_str(), this->descriptor.size.y, this->descriptor.size.x, colorDepth);
  Unity<unsigned char>* pixels = new Unity<unsigned char>(pixels_host,this->descriptor.size.y*this->descriptor.size.x*colorDepth,cpu);
  if(convertColorDepthTo == 1){
    convertToBW(pixels,colorDepth);
    colorDepth = 1;
  }
  else if(convertColorDepthTo != 0){
    std::cerr<<"ERROR: Image() does not currently support conversion to anything but BW"<<std::endl;
    exit(-1);
  }
  unsigned int depth = 0;
  int2 border = {0,0};
  if(this->descriptor.size.x > this->descriptor.size.y){
    depth = (unsigned int)ceil(log2((float)this->descriptor.size.x));
  }
  else{
    depth = (unsigned int)ceil(log2((float)this->descriptor.size.y));
  }
  border.x = (pow(2,depth) - this->descriptor.size.x)/2;
  border.y = (pow(2,depth) - this->descriptor.size.y)/2;
  if(quadtreeBinDepth == depth){
    std::cerr<<"ERROR: invalid quadtree depth of "<<quadtreeBinDepth<<std::endl;
    exit(-1);
  }
  depth -= quadtreeBinDepth;
  this->quadtree = new Quadtree<unsigned char>(this->descriptor.size,depth,pixels,colorDepth,border);
}

ssrlcv::Image::~Image(){
  if(this->quadtree != nullptr){
    delete this->quadtree;
  }
}
void ssrlcv::calcFundamentalMatrix_2View(Image_Descriptor query, Image_Descriptor target, float3 *F){
  if(query.fov != target.fov || query.foc != target.foc){
    std::cout<<"ERROR calculating fundamental matrix for 2view needs to bet taken with same camera (foc&fov are same)"<<std::endl;
    exit(-1);
  }
  float angle1;
  if(abs(query.cam_vec.z) < .00001) {
    if(query.cam_vec.y > 0)  angle1 = PI/2;
    else       angle1 = -1*PI/2;
  }
  else {
    angle1 = atan(query.cam_vec.y / query.cam_vec.z);
    if(query.cam_vec.z<0 && query.cam_vec.y>=0) {
      angle1 += PI;
    }
    if(query.cam_vec.z<0 && query.cam_vec.y<0) {
      angle1 -= PI;
    }
  }
  float3 A1[3] = {
    {1, 0, 0},
    {0, cos(angle1), -sin(angle1)},
    {0, sin(angle1), cos(angle1)}
  };

  float3 temp = multiply3x3x1(A1, query.cam_vec);

  float angle2 = 0.0f;
  if(abs(temp.z) < .00001) {
    if(temp.x <= 0)  angle1 = PI/2;
    else       angle1 = -1*PI/2;
  }
  else {
    angle2 = atan(-1*temp.x / temp.z);
    if(temp.z<0 && temp.x<0) {
      angle1 += PI;
    }
    if(temp.z<0 && temp.x>0) {
      angle2 -= PI;
    }
  }
  float3 B1[3] = {
    {cos(angle2), 0, sin(angle2)},
    {0, 1, 0},
    {-sin(angle2), 0, cos(angle2)}
  };

  float3 temp2 = multiply3x3x1(B1, temp);
  float3 rot1[3];
  multiply3x3(B1, A1, rot1);
  float3 rot1Transpose[3];
  transpose3x3(rot1,rot1Transpose);
  temp = multiply3x3x1(rot1Transpose, temp2);

  angle1 = 0.0f;
  if(abs(target.cam_vec.z) < .00001) {
    if(target.cam_vec.y > 0)  angle1 = PI/2;
    else       angle1 = -1*PI/2;
  }
  else {
    angle1 = atan(target.cam_vec.y / target.cam_vec.z);
    if(target.cam_vec.z<0 && target.cam_vec.y>=0) {
      angle1 += PI;
    }
    if(target.cam_vec.z<0 && target.cam_vec.y<0) {
      angle1 -= PI;
    }
  }
  float3 A2[3] = {
    {1, 0, 0},
    {0, cos(angle1), -sin(angle1)},
    {0, sin(angle1), cos(angle1)}
  };
  temp2 = multiply3x3x1(A2, target.cam_vec);

  angle2 = 0.0f;
  if(abs(temp2.z) < .00001) {
    if(temp2.x <= 0)  angle1 = PI/2;
    else       angle1 = -1*PI/2;
  }
  else {
    angle2 = atan(-1*temp2.x / temp2.z);
    if(temp2.z<0 && temp2.x<0) {
      angle1 += PI;
    }
    if(temp2.z<0 && temp2.x>0) {
      angle2 -= PI;
    }
  }
  float3 B2[3] = {
    {cos(angle2), 0, sin(angle2)},
    {0, 1, 0},
    {-sin(angle2), 0, cos(angle2)}
  };

  temp = multiply3x3x1(B2, temp2);

  float3 rot2[3];
  multiply3x3(B2, A2, rot2);
  float3 rot2Transpose[3];
  transpose3x3(rot2, rot2Transpose);

  temp2 = multiply3x3x1(rot2Transpose, temp);

  float2 dpix = {query.foc*tan(query.fov/2)/(query.size.x/2),
    query.foc*tan(query.fov/2)/(query.size.y/2)};

  float3 K[3] = {
    {query.foc/dpix.x, 0, ((float)query.size.x)/2.0f},
    {0, query.foc/dpix.y, ((float)query.size.y)/2.0f},
    {0, 0, 1}
  };
  float3 K_inv[3];
  inverse3x3(K,K_inv);
  float3 K_invTranspose[3];
  transpose3x3(K_inv,K_invTranspose);

  float3 R[3];
  multiply3x3(rot2Transpose, rot1, R);
  float3 S[3] = {
    {0, query.cam_pos.z - target.cam_pos.z, target.cam_pos.y - query.cam_pos.y},
    {query.cam_pos.z - target.cam_pos.z,0, query.cam_pos.x - target.cam_pos.x},
    {query.cam_pos.y - target.cam_pos.y, target.cam_pos.x - query.cam_pos.x, 0}
  };
  float3 E[3];;
  multiply3x3(R,S,E);
  float3 tempF[3];
  multiply3x3(K_invTranspose, E,tempF);
  multiply3x3(tempF, K_inv, F);
  std::cout << std::endl <<"between image "<<query.id<<" and "<<target.id
  <<" the final fundamental matrix result is: " << std::endl;
  for(int r = 0; r < 3; ++r) {
    std::cout << F[r].x << "  " << F[r].y << " "<<  F[r].z << std::endl;
  }
  std::cout<<std::endl;
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
void ssrlcv::convertToBW(Unity<unsigned char>* pixels, unsigned int colorDepth){
  if(colorDepth == 1){
    std::cout<<"Pixels are already bw"<<std::endl;
    return;
  }

  MemoryState origin = pixels->state;
  pixels->transferMemoryTo(gpu);

  unsigned int numPixels = (pixels->numElements/colorDepth);

  unsigned char* bwPixels_device;
  CudaSafeCall(cudaMalloc((void**)&bwPixels_device, numPixels*sizeof(unsigned char)));

  dim3 grid;
  dim3 block;
  getFlatGridBlock(numPixels, grid, block);
  generateBW<<<grid,block>>>(numPixels, colorDepth, pixels->device, bwPixels_device);
  CudaCheckError();

  pixels->setData(bwPixels_device, numPixels, gpu);
  pixels->transferMemoryTo(origin);
  if(origin == cpu){
    pixels->clear(gpu);
  }
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
