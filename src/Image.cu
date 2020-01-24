#include "Image.cuh"

__device__ __host__ ssrlcv::Image::Camera::Camera(){
  this->cam_vec = {0.0f,0.0f,0.0f};
  this->cam_pos = {0.0f,0.0f,0.0f};
  this->axangle = 0.0f;
  this->fov = 0;
  this->foc = 0;
  this->dpix = {0.0f,0.0f};
  this->size = {0,0};
}
__device__ __host__ ssrlcv::Image::Camera::Camera(uint2 size){
  this->cam_vec = {0.0f,0.0f,0.0f};
  this->cam_pos = {0.0f,0.0f,0.0f};
  this->axangle = 0.0f;
  this->fov = 0;
  this->foc = 0;
  this->dpix = {0.0f,0.0f};
  this->size = {0,0};
}
__device__ __host__ ssrlcv::Image::Camera::Camera(uint2 size, float3 cam_pos, float3 camp_dir){
  this->cam_pos = cam_pos;
  this->cam_vec = cam_vec;
  this->axangle = 0.0f;
  this->fov = 0;
  this->foc = 0;
  this->dpix = {0.0f,0.0f};
  this->size = size;
}

ssrlcv::Image::Image(){
  this->id = -1;
  this->filePath = "n/a";
}

ssrlcv::Image::Image(uint2 size, unsigned int colorDepth, Unity<unsigned char>* pixels){
  this->filePath = "n/a";
  this->id = -1;
  this->colorDepth = colorDepth;
  this->pixels = pixels;
  this->camera.size = size;
  this->size = size;
}


ssrlcv::Image::Image(std::string filePath, int id){
  this->filePath = filePath;
  this->id = id;
  this->colorDepth = 1;
  unsigned char* pixels_host = readPNG(filePath.c_str(), this->size.y, this->size.x, this->colorDepth);
  this->camera.size = this->size;
  this->size = size;
  this->pixels = new Unity<unsigned char>(pixels_host,this->size.y*this->size.x*this->colorDepth,cpu);
}
ssrlcv::Image::Image(std::string filePath, unsigned int convertColorDepthTo, int id){
  this->filePath = filePath;
  this->id = id;
  this->colorDepth = 1;
  unsigned char* pixels_host = readPNG(filePath.c_str(), this->size.y, this->size.x, this->colorDepth);
  this->camera.size = this->size;
  this->size = size;
  this->pixels = new Unity<unsigned char>(pixels_host,this->size.y*this->size.x*this->colorDepth,cpu);
  if(convertColorDepthTo == 1){
    convertToBW(this->pixels, this->colorDepth);
    this->colorDepth = 1;
  }
  else if(convertColorDepthTo != 0){
    std::cerr<<"ERROR: Image() does not currently support conversion to anything but BW"<<std::endl;
    exit(-1);
  }
}

ssrlcv::Image::~Image(){
  if(this->pixels != nullptr){
    delete this->pixels;
  }
}

void ssrlcv::Image::convertColorDepthTo(unsigned int colorDepth){
  std::cout<<"Converting pixel depth to "<<colorDepth<<" from "<<this->colorDepth<<std::endl;
  if(colorDepth == 1){
    convertToBW(this->pixels,this->colorDepth);
    this->colorDepth = 1;
  }
  else if (colorDepth == 3){
    convertToRGB(this->pixels,this->colorDepth);
    this->colorDepth = 3;
  }
  else{
    std::cerr<<colorDepth<<" is currently not supported in convertColorDepthTo"<<std::endl;
    exit(-1);
  }
}
ssrlcv::Unity<int2>* ssrlcv::Image::getPixelGradients(){
  return generatePixelGradients(this->size,this->pixels);
}
void ssrlcv::Image::alterSize(int scalingFactor){
  if(scalingFactor == 0){
    std::cerr<<"using a binDepth of 0 results in no binning or upsampling\nuse binDepth>0 for binning and binDepth<0 for upsampling"<<std::endl;
    return;
  }
  else if((float)this->size.x/powf(2.0f,scalingFactor) < 1.0f ||(float)this->size.y/powf(2.0f,scalingFactor) < 1.0f){
    std::cerr<<"ERROR binning "<<scalingFactor<<" many times cannot be done on and image of size "<<this->size.x<<"x"<<this->size.y<<std::endl;
    exit(-1);
  }
  MemoryState origin = this->pixels->state;
  if(origin == cpu || this->pixels->fore == cpu) this->pixels->transferMemoryTo(gpu);

  uint2 scaler = {2,2};
  if(scalingFactor < 0){//upsampling
    for(int i = 0; i < abs(scalingFactor); ++i){
      this->pixels->setData(upsample(this->size,this->colorDepth,this->pixels)->device,this->size.x*this->size.y*this->colorDepth*4,gpu);
      this->size = this->size*scaler;
    }
  }
  else{//downsampling
    for(int i = 0; i < scalingFactor; ++i){
      this->pixels->setData(bin(this->size,this->colorDepth,this->pixels)->device,(this->size.x*this->size.y*this->colorDepth)/4,gpu);
      this->size = this->size/scaler;
    }
  }
  if(origin == cpu) this->pixels->setMemoryState(cpu);
}


ssrlcv::Unity<unsigned char>* ssrlcv::addBufferBorder(uint2 size, ssrlcv::Unity<unsigned char>* pixels, int2 border){
  if(border.x == 0|| border.y == 0){
    std::cerr<<"ERROR border cannot be 0"<<std::endl;
    exit(-1);
  }
  if(border.x*2 + (int) size.x < 0 || border.y*2 + (int)size.y < 0){
    std::cerr<<"ERROR border causes negative dimensions"<<std::endl;
    exit(-1);
  }
  if(pixels->numElements%((int)size.x*size.y) != 0){
    std::cerr<<"ERROR color depth cannot be determined due to pixels->numElements%(size.x*size.y) != 0"<<std::endl;
  }
  MemoryState origin = pixels->state;
  if(origin == cpu || pixels->fore == cpu) pixels->setMemoryState(cpu);
  uint2 newSize = {size.x + (border.x*2),size.y + (border.y*2)};
  int colorDepth = pixels->numElements/((int)size.x*size.y);
  unsigned char* bufferedPixels_host = new unsigned char[newSize.x*newSize.y*colorDepth]();
  Unity<unsigned char>* bufferedPixels = new Unity<unsigned char>(bufferedPixels_host,newSize.x*newSize.y*colorDepth,cpu);
  bufferedPixels->setMemoryState(gpu);
  for(int y = border.y; y < (int)size.y + border.y; ++y){
    CudaSafeCall(cudaMemcpy(bufferedPixels->device + (y*newSize.x) + border.x,pixels->device + (y*size.x),size.x*sizeof(unsigned char),cudaMemcpyDeviceToDevice));
  }
  bufferedPixels->fore = gpu;
  if(origin != gpu){
    bufferedPixels->setMemoryState(origin);
    pixels->setMemoryState(origin);
  }
}
ssrlcv::Unity<float>* ssrlcv::addBufferBorder(uint2 size, ssrlcv::Unity<float>* pixels, int2 border){
  if(border.x == 0|| border.y == 0){
    std::cerr<<"ERROR border cannot be 0"<<std::endl;
    exit(-1);
  }
  if(border.x*2 + (int) size.x < 0 || border.y*2 + (int)size.y < 0){
    std::cerr<<"ERROR border causes negative dimensions"<<std::endl;
    exit(-1);
  }
  if(pixels->numElements%((int)size.x*size.y) != 0){
    std::cerr<<"ERROR color depth cannot be determined due to pixels->numElements%(size.x*size.y) != 0"<<std::endl;
  }
  MemoryState origin = pixels->state;
  if(origin == cpu || pixels->fore == cpu) pixels->setMemoryState(cpu);
  uint2 newSize = {size.x + (border.x*2),size.y + (border.y*2)};
  int colorDepth = pixels->numElements/((int)size.x*size.y);
  float* bufferedPixels_host = new float[newSize.x*newSize.y*colorDepth]();
  Unity<float>* bufferedPixels = new Unity<float>(bufferedPixels_host,newSize.x*newSize.y*colorDepth,cpu);
  bufferedPixels->setMemoryState(gpu);
  std::cout<<"applying border"<<std::endl;
  for(int y = 0; y < (int)size.y; ++y){
    CudaSafeCall(cudaMemcpy(bufferedPixels->device + ((y+border.y)*newSize.x) + border.x,pixels->device + (y*size.x),size.x*sizeof(float),cudaMemcpyDeviceToDevice));
  }
  bufferedPixels->fore = gpu;
  if(origin != gpu){
    bufferedPixels->setMemoryState(origin);
    pixels->setMemoryState(origin);
  }
}

ssrlcv::Unity<unsigned char>* ssrlcv::convertImageToChar(Unity<float>* pixels){
  MemoryState origin = pixels->state;
  if(origin == cpu || pixels->fore == cpu) pixels->setMemoryState(gpu);
  normalizeImage(pixels);
  dim3 grid = {1,1,1};
  dim3 block = {1,1,1};
  getFlatGridBlock(pixels->numElements,grid,block);
  Unity<unsigned char>* castPixels = new Unity<unsigned char>(nullptr,pixels->numElements,gpu);
  convertToCharImage<<<grid,block>>>(pixels->numElements,castPixels->device,pixels->device);
  cudaDeviceSynchronize();
  CudaCheckError();

  if(origin == cpu){
    pixels->setMemoryState(cpu);
    castPixels->setMemoryState(cpu);
  }
  return castPixels;
}
ssrlcv::Unity<float>* ssrlcv::convertImageToFlt(Unity<unsigned char>* pixels){
  MemoryState origin = pixels->state;
  if(origin == cpu || pixels->fore == cpu) pixels->transferMemoryTo(gpu);
  dim3 grid = {1,1,1};
  dim3 block = {1,1,1};
  getFlatGridBlock(pixels->numElements,grid,block);
  Unity<float>* castPixels = new Unity<float>(nullptr,pixels->numElements,gpu);
  convertToFltImage<<<grid,block>>>(pixels->numElements,pixels->device,castPixels->device);
  cudaDeviceSynchronize();
  CudaCheckError();
  if(origin == cpu){
    pixels->setMemoryState(cpu);
    castPixels->setMemoryState(cpu);
  }
  return castPixels;
}

//todo use cuda reduction instead of cpu loop for min max finding
//todo add support for color depth
void ssrlcv::normalizeImage(Unity<float>* pixels){
  MemoryState origin = pixels->state;
  float2 minMax = {FLT_MAX,-FLT_MAX};
  if(origin == gpu || pixels->fore == gpu) pixels->transferMemoryTo(cpu);
  for(int i = 0; i < pixels->numElements; ++i){
      if(minMax.x > pixels->host[i]) minMax.x = pixels->host[i];
      if(minMax.y < pixels->host[i]) minMax.y = pixels->host[i];
  }
  if(origin == cpu || pixels->fore == cpu) pixels->setMemoryState(gpu);
  dim3 grid = {1,1,1};
  dim3 block = {1,1,1};
  getFlatGridBlock(pixels->numElements,grid,block);
  normalize<<<grid,block>>>(pixels->numElements,pixels->device,minMax);
  cudaDeviceSynchronize();
  CudaCheckError();

  pixels->fore = gpu;

  if(origin == cpu) pixels->setMemoryState(cpu);
}
void ssrlcv::normalizeImage(Unity<float>* pixels, float2 minMax){
  MemoryState origin = pixels->state;
  if(origin == cpu || pixels->fore == cpu) pixels->setMemoryState(gpu);
  dim3 grid = {1,1,1};
  dim3 block = {1,1,1};
  getFlatGridBlock(pixels->numElements,grid,block);
  normalize<<<grid,block>>>(pixels->numElements,pixels->device,minMax);
  cudaDeviceSynchronize();
  CudaCheckError();

  pixels->fore = gpu;

  if(origin == cpu) pixels->setMemoryState(cpu);
}

void ssrlcv::convertToBW(Unity<unsigned char>* pixels, unsigned int colorDepth){
  if(colorDepth == 1){
    std::cout<<"Pixels are already bw"<<std::endl;
    return;
  }

  MemoryState origin = pixels->state;
  if(origin == cpu || pixels->fore == cpu) pixels->transferMemoryTo(gpu);

  unsigned int numPixels = (pixels->numElements/colorDepth);

  unsigned char* bwPixels_device;
  CudaSafeCall(cudaMalloc((void**)&bwPixels_device, numPixels*sizeof(unsigned char)));

  dim3 grid = {1,1,1};
  dim3 block = {1,1,1};
  getFlatGridBlock(numPixels, grid, block);
  generateBW<<<grid,block>>>(numPixels, colorDepth, pixels->device, bwPixels_device);
  cudaDeviceSynchronize();
  CudaCheckError();

  pixels->setData(bwPixels_device, numPixels, gpu);
  if(origin == cpu) pixels->setMemoryState(origin);
}
void ssrlcv::convertToRGB(Unity<unsigned char>* pixels, unsigned int colorDepth){
  if(colorDepth == 3){
    std::cout<<"Pixels are already rgb"<<std::endl;
    return;
  }

  MemoryState origin = pixels->state;
  if(origin == cpu || pixels->fore == cpu) pixels->transferMemoryTo(gpu);

  unsigned int numPixels = (pixels->numElements/colorDepth);

  unsigned char* rgbPixels_device;
  CudaSafeCall(cudaMalloc((void**)&rgbPixels_device, numPixels*3*sizeof(unsigned char)));

  dim3 grid;
  dim3 block;
  getFlatGridBlock(numPixels, grid, block);
  generateRGB<<<grid,block>>>(numPixels, colorDepth, pixels->device, rgbPixels_device);
  cudaDeviceSynchronize();
  CudaCheckError();

  pixels->setData(rgbPixels_device, 3*numPixels, gpu);
  if(origin == cpu) pixels->setMemoryState(origin);
}
//TODO implement
void calcFundamentalMatrix_2View(float cam0[3][3], float cam1[3][3], float (&F)[3][3]){

}
void ssrlcv::calcFundamentalMatrix_2View(Image* query, Image* target, float3 (&F)[3]){
  if(query->camera.foc != target->camera.foc){
    std::cout<<"ERROR calculating fundamental matrix for 2view needs to bet taken with same camera (foc&fov are same)"<<std::endl;
    exit(-1);
  }
  float angle1;
  if(abs(query->camera.cam_vec.z) < .00001) {
    if(query->camera.cam_vec.y > 0)  angle1 = PI/2;
    else       angle1 = -1*PI/2;
  }
  else {
    angle1 = atan(query->camera.cam_vec.y / query->camera.cam_vec.z);
    if(query->camera.cam_vec.z<0 && query->camera.cam_vec.y>=0) {
      angle1 += PI;
    }
    if(query->camera.cam_vec.z<0 && query->camera.cam_vec.y<0) {
      angle1 -= PI;
    }
  }
  float3 A1[3] = {
    {1, 0, 0},
    {0, cos(angle1), -sin(angle1)},
    {0, sin(angle1), cos(angle1)}
  };

  float3 temp = {0.0f,0.0f,0.0f};
  multiply(A1, query->camera.cam_vec,temp);

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

  float3 temp2 = {0.0f,0.0f,0.0f};
  multiply(B1, temp, temp2);
  float3 rot1[3];
  multiply(B1, A1, rot1);
  float3 rot1Transpose[3];
  transpose(rot1,rot1Transpose);
  multiply(rot1Transpose, temp2, temp);

  angle1 = 0.0f;
  if(abs(target->camera.cam_vec.z) < .00001) {
    if(target->camera.cam_vec.y > 0)  angle1 = PI/2;
    else       angle1 = -1*PI/2;
  }
  else {
    angle1 = atan(target->camera.cam_vec.y / target->camera.cam_vec.z);
    if(target->camera.cam_vec.z<0 && target->camera.cam_vec.y>=0) {
      angle1 += PI;
    }
    if(target->camera.cam_vec.z<0 && target->camera.cam_vec.y<0) {
      angle1 -= PI;
    }
  }
  float3 A2[3] = {
    {1, 0, 0},
    {0, cos(angle1), -sin(angle1)},
    {0, sin(angle1), cos(angle1)}
  };
  multiply(A2, target->camera.cam_vec,temp2);

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

  multiply(B2, temp2, temp);

  float3 rot2[3];
  multiply(B2, A2, rot2);
  float3 rot2Transpose[3];
  transpose(rot2, rot2Transpose);

  multiply(rot2Transpose, temp, temp2);

  float3 K[3] = {
    {query->camera.foc, 0, ((float)query->size.x)/2.0f},//NOTE the foc was divided by dpix.x and dpix.y but currently using foc in pixels
    {0, query->camera.foc, ((float)query->size.y)/2.0f},//NOTE the foc was divided by dpix.x and dpix.y but currently using foc in pixels
    {0, 0, 1}
  };
  float3 K_inv[3];
  inverse(K,K_inv);
  float3 K_invTranspose[3];
  transpose(K_inv,K_invTranspose);

  float3 R[3];
  multiply(rot2Transpose, rot1, R);
  float3 S[3] = {
    {0, query->camera.cam_pos.z - target->camera.cam_pos.z, target->camera.cam_pos.y - query->camera.cam_pos.y},
    {query->camera.cam_pos.z - target->camera.cam_pos.z,0, query->camera.cam_pos.x - target->camera.cam_pos.x},
    {query->camera.cam_pos.y - target->camera.cam_pos.y, target->camera.cam_pos.x - query->camera.cam_pos.x, 0}
  };
  float3 E[3];
  multiply(R,S,E);
  float3 tempF[3];
  multiply(K_invTranspose, E,tempF);
  multiply(tempF, K_inv, F);
  std::cout << std::endl <<"between image "<<query->id<<" and "<<target->id
  <<" the final fundamental matrix result is: " << std::endl;
  for(int r = 0; r < 3; ++r) {
    std::cout << F[r].x << "  " << F[r].y << " "<<  F[r].z << std::endl;
  }
  std::cout<<std::endl;
}
void ssrlcv::get_cam_params2view(Image* cam1, Image* cam2, std::string infile){
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
      cam1->camera.foc = arg1;
      cam2->camera.foc = arg1;
    }
    else if(param.compare("fov") == 0) {
      cam1->camera.fov = arg1;
      cam2->camera.fov = arg1;
    }
    else if(param.compare("res") == 0) {
      res = arg1;
    }
    else if(param.compare("cam1C") == 0) {
      iss >> arg2 >> arg3;
      cam1->camera.cam_pos.x = arg1;
      cam1->camera.cam_pos.y = arg2;
      cam1->camera.cam_pos.z = arg3;
    }
    else if(param.compare("cam1V") == 0) {
      iss >> arg2 >> arg3;
      cam1->camera.cam_vec.x = arg1;
      cam1->camera.cam_vec.y = arg2;
      cam1->camera.cam_vec.z = arg3;
    }
    else if(param.compare("cam2C") == 0) {
      iss >> arg2 >> arg3;
      cam2->camera.cam_pos.x = arg1;
      cam2->camera.cam_pos.y = arg2;
      cam2->camera.cam_pos.z = arg3;
    }
    else if(param.compare("cam2V") == 0) {
      iss >> arg2 >> arg3;
      cam2->camera.cam_vec.x = arg1;
      cam2->camera.cam_vec.y = arg2;
      cam2->camera.cam_vec.z = arg3;
    }
  }

  cam1->camera.dpix = {cam1->camera.foc*tan(cam1->camera.fov/2)/(cam1->size.x/2),
    cam1->camera.foc*tan(cam1->camera.fov/2)/(cam1->size.y/2)};
  cam2->camera.dpix = {cam2->camera.foc*tan(cam2->camera.fov/2)/(cam2->size.x/2),
    cam2->camera.foc*tan(cam2->camera.fov/2)/(cam2->size.y/2)};
}

ssrlcv::Unity<int2>* ssrlcv::generatePixelGradients(uint2 imageSize, Unity<unsigned char>* pixels){
  MemoryState origin = pixels->state;
  if(origin == cpu || pixels->fore == cpu){
    pixels->transferMemoryTo(gpu);
  }
  int2* gradients_device = nullptr;
  CudaSafeCall(cudaMalloc((void**)&gradients_device,pixels->numElements*sizeof(int2)));
  dim3 grid = {1,1,1};
  dim3 block = {1,1,1};
  getFlatGridBlock(pixels->numElements,grid,block);
  calculatePixelGradients<<<grid,block>>>(imageSize,pixels->device,gradients_device);
  CudaCheckError();

  if(origin == cpu) pixels->setMemoryState(cpu);

  return new Unity<int2>(gradients_device,pixels->numElements,gpu);
}
ssrlcv::Unity<float2>* ssrlcv::generatePixelGradients(uint2 imageSize, Unity<float>* pixels){
  MemoryState origin = pixels->state;
  if(origin == cpu || pixels->fore == cpu){
    pixels->transferMemoryTo(gpu);
  }
  float2* gradients_device = nullptr;
  CudaSafeCall(cudaMalloc((void**)&gradients_device,pixels->numElements*sizeof(float2)));
  dim3 grid = {1,1,1};
  dim3 block = {1,1,1};
  getFlatGridBlock(pixels->numElements,grid,block);
  calculatePixelGradients<<<grid,block>>>(imageSize,pixels->device,gradients_device);
  CudaCheckError();

  if(origin == cpu) pixels->setMemoryState(cpu);

  return new Unity<float2>(gradients_device,pixels->numElements,gpu);
}

ssrlcv::Unity<unsigned char>* ssrlcv::bin(uint2 imageSize, unsigned int colorDepth, Unity<unsigned char>* pixels){
  MemoryState origin = pixels->state;

  if(origin == cpu || pixels->fore == cpu){
    pixels->transferMemoryTo(gpu);
  }
  Unity<unsigned char>* binnedImage = new Unity<unsigned char>(nullptr, pixels->numElements/4, gpu);
  dim3 grid = {(imageSize.x/64)+1,(imageSize.y/64)+1,1};
  dim3 block = {32,32,1};

  binImage<<<grid,block>>>(imageSize,colorDepth,pixels->device,binnedImage->device);
  cudaDeviceSynchronize();
  CudaCheckError();


  if(origin == cpu) pixels->setMemoryState(cpu);
  if(origin != gpu) binnedImage->setMemoryState(origin);
  return binnedImage;
}
ssrlcv::Unity<float>* ssrlcv::bin(uint2 imageSize, unsigned int colorDepth, Unity<float>* pixels){
  MemoryState origin = pixels->state;

  if(origin == cpu || pixels->fore == cpu){
    pixels->transferMemoryTo(gpu);
  }
  Unity<float>* binnedImage = new Unity<float>(nullptr, pixels->numElements/4, gpu);
  dim3 grid = {(imageSize.x/64)+1,(imageSize.y/64)+1,1};
  dim3 block = {32,32,1};

  binImage<<<grid,block>>>(imageSize,colorDepth,pixels->device,binnedImage->device);
  cudaDeviceSynchronize();
  CudaCheckError();


  if(origin == cpu) pixels->setMemoryState(cpu);
  if(origin != gpu) binnedImage->setMemoryState(origin);
  return binnedImage;
}
ssrlcv::Unity<unsigned char>* ssrlcv::upsample(uint2 imageSize, unsigned int colorDepth, Unity<unsigned char>* pixels){
  MemoryState origin = pixels->state;

  if(origin == cpu || pixels->fore == cpu){
    pixels->transferMemoryTo(gpu);
  }

  dim3 grid = {(imageSize.x/16)+1,(imageSize.y/16)+1,1};
  dim3 block = {32,32,1};

  Unity<unsigned char>* upsampledImage = new Unity<unsigned char>(nullptr, pixels->numElements*4, gpu);

  upsampleImage<<<grid,block>>>(imageSize,colorDepth,pixels->device,upsampledImage->device);
  cudaDeviceSynchronize();
  CudaCheckError();

  if(origin == cpu) pixels->setMemoryState(cpu);
  if(origin != gpu) upsampledImage->setMemoryState(origin);
  return upsampledImage;

}
ssrlcv::Unity<float>* ssrlcv::upsample(uint2 imageSize, unsigned int colorDepth, Unity<float>* pixels){
  MemoryState origin = pixels->state;

  if(origin == cpu || pixels->fore == cpu){
    pixels->transferMemoryTo(gpu);
  }

  dim3 grid = {(imageSize.x/16)+1,(imageSize.y/16)+1,1};
  dim3 block = {32,32,1};

  Unity<float>* upsampledImage = new Unity<float>(nullptr, pixels->numElements*4, gpu);

  upsampleImage<<<grid,block>>>(imageSize,colorDepth,pixels->device,upsampledImage->device);
  cudaDeviceSynchronize();
  CudaCheckError();

  if(origin == cpu) pixels->setMemoryState(cpu);
  if(origin != gpu) upsampledImage->setMemoryState(origin);
  return upsampledImage;

}
ssrlcv::Unity<unsigned char>* ssrlcv::scaleImage(uint2 imageSize, unsigned int colorDepth, Unity<unsigned char>* pixels, float outputPixelWidth){
  MemoryState origin = pixels->state;

  if(origin == cpu || pixels->fore != gpu){
    pixels->transferMemoryTo(gpu);
  }
  unsigned char* sampledImage_device = nullptr;

  dim3 grid = {(imageSize.x/(32*outputPixelWidth))+1,(imageSize.y/(32*outputPixelWidth))+1,1};
  dim3 block = {32,32,1};

  CudaSafeCall(cudaMalloc((void**)&sampledImage_device,pixels->numElements*4*sizeof(unsigned char)));
  bilinearInterpolation<<<grid,block>>>(imageSize,colorDepth,pixels->device,sampledImage_device,outputPixelWidth);
  cudaDeviceSynchronize();
  CudaCheckError();

  Unity<unsigned char>* sampledImage = new Unity<unsigned char>(sampledImage_device, pixels->numElements/(outputPixelWidth*outputPixelWidth), gpu);

  if(origin == cpu){
    pixels->setMemoryState(cpu);
    sampledImage->transferMemoryTo(cpu);
  }

  return sampledImage;
}
ssrlcv::Unity<float>* ssrlcv::scaleImage(uint2 imageSize, unsigned int colorDepth, Unity<float>* pixels, float outputPixelWidth){
  MemoryState origin = pixels->state;

  if(origin == cpu || pixels->fore != gpu){
    pixels->transferMemoryTo(gpu);
  }
  float* sampledImage_device = nullptr;

  dim3 grid = {(imageSize.x/(32*outputPixelWidth))+1,(imageSize.y/(32*outputPixelWidth))+1,1};
  dim3 block = {32,32,1};

  CudaSafeCall(cudaMalloc((void**)&sampledImage_device,pixels->numElements*4*sizeof(float)));
  bilinearInterpolation<<<grid,block>>>(imageSize,colorDepth,pixels->device,sampledImage_device,outputPixelWidth);
  cudaDeviceSynchronize();
  CudaCheckError();

  Unity<float>* sampledImage = new Unity<float>(sampledImage_device, pixels->numElements/(outputPixelWidth*outputPixelWidth), gpu);

  if(origin == cpu){
    pixels->setMemoryState(cpu);
    sampledImage->transferMemoryTo(cpu);
  }

  return sampledImage;
}


ssrlcv::Unity<float>* ssrlcv::convolve(uint2 imageSize, Unity<unsigned char>* pixels, unsigned int colorDepth, int2 kernelSize, float* kernel, bool symmetric){
  if(kernelSize.x%2 == 0 || kernelSize.y%2 == 0){
    std::cerr<<"ERROR kernel for image convolution must have an odd dimension"<<std::endl;
    exit(-1);
  }
  MemoryState origin = pixels->state;
  if(origin == cpu) pixels->transferMemoryTo(gpu);
  Unity<float>* convolvedImage = new Unity<float>(nullptr,imageSize.x*imageSize.y*colorDepth,gpu);
  float* kernel_device = nullptr;
  CudaSafeCall(cudaMalloc((void**)&kernel_device,kernelSize.x*kernelSize.y*sizeof(float)));
  CudaSafeCall(cudaMemcpy(kernel_device,kernel,kernelSize.x*kernelSize.y*sizeof(float),cudaMemcpyHostToDevice));
  dim3 grid = {(imageSize.x/32)+1,(imageSize.y/32)+1,colorDepth};
  dim3 block = {32,32,1};
  float2 minMax = {FLT_MAX,-FLT_MAX};
  float* min = nullptr;

  if(symmetric){
    convolveImage_symmetric<<<grid,block>>>(imageSize, pixels->device, colorDepth, kernelSize, kernel_device, convolvedImage->device);
  }
  else{
    convolveImage<<<grid,block>>>(imageSize, pixels->device, colorDepth, kernelSize, kernel_device, convolvedImage->device);
  }
  cudaDeviceSynchronize();
  CudaCheckError();

  CudaSafeCall(cudaFree(kernel_device));

  if(origin == cpu) pixels->setMemoryState(cpu);
  return convolvedImage;
}
ssrlcv::Unity<float>* ssrlcv::convolve(uint2 imageSize, Unity<float>* pixels, unsigned int colorDepth, int2 kernelSize, float* kernel, bool symmetric){
  if(kernelSize.x%2 == 0 || kernelSize.y%2 == 0){
    std::cerr<<"ERROR kernel for image convolution must have an odd dimension"<<std::endl;
    exit(-1);
  }
  MemoryState origin = pixels->state;
  if(origin == cpu) pixels->transferMemoryTo(gpu);
  Unity<float>* convolvedImage = new Unity<float>(nullptr,imageSize.x*imageSize.y*colorDepth,gpu);
  float* kernel_device = nullptr;
  CudaSafeCall(cudaMalloc((void**)&kernel_device,kernelSize.x*kernelSize.y*sizeof(float)));
  CudaSafeCall(cudaMemcpy(kernel_device,kernel,kernelSize.x*kernelSize.y*sizeof(float),cudaMemcpyHostToDevice));
  dim3 grid = {(imageSize.x/32)+1,(imageSize.y/32)+1,colorDepth};
  dim3 block = {32,32,1};

  if(symmetric){
    convolveImage_symmetric<<<grid,block>>>(imageSize, pixels->device, colorDepth, kernelSize, kernel_device, convolvedImage->device);
  }
  else{
    convolveImage<<<grid,block>>>(imageSize, pixels->device, colorDepth, kernelSize, kernel_device, convolvedImage->device);
  }
  cudaDeviceSynchronize();
  CudaCheckError();

  CudaSafeCall(cudaFree(kernel_device));

  if(origin == cpu) pixels->setMemoryState(cpu);
  return convolvedImage;
}

__device__ __host__ __forceinline__ int ssrlcv::getSymmetrizedCoord(int i, unsigned int l){
  int ll = 2*l;
  i = (i+ll)%ll;
  return (i>l-1) ? i = ll - 1 - i : i;
}
__device__ __forceinline__ unsigned char ssrlcv::bwaToBW(const uchar2 &color){
  return (1-color.y)*color.x + color.y*color.x;
}
__device__ __forceinline__ unsigned char ssrlcv::rgbToBW(const uchar3 &color){
  return (color.x/4) + (color.y/2) + (color.z/4);
}
__device__ __forceinline__ unsigned char ssrlcv::rgbaToBW(const uchar4 &color){
  return rgbToBW(rgbaToRGB(color));
}
__device__ __forceinline__ uchar3 ssrlcv::bwToRGB(const unsigned char &color){
  int colorTemp = (int) color*10;
  return {(unsigned char)colorTemp/4,(unsigned char)colorTemp/2,(unsigned char)colorTemp/4};
}
__device__ __forceinline__ uchar3 ssrlcv::bwaToRGB(const uchar2 &color){
  return {color.x,color.y,(color.x/3)*2 + (color.y/3)};
}
__device__ __forceinline__ uchar3 ssrlcv::rgbaToRGB(const uchar4 &color){
  return {
    (1-color.w)*color.x + color.w*color.x,
    (1-color.w)*color.y + color.w*color.y,
    (1-color.w)*color.z + color.w*color.z,
  };
}

__global__ void ssrlcv::generateBW(int numPixels, unsigned int colorDepth, unsigned char* colorPixels, unsigned char* pixels){
  unsigned int globalID = (blockIdx.y* gridDim.x+ blockIdx.x)*blockDim.x + threadIdx.x;
  if(globalID < numPixels){
    int numValues = (int) colorDepth;
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
__global__ void ssrlcv::generateRGB(int numPixels, unsigned int colorDepth, unsigned char* colorPixels, unsigned char* pixels){
  unsigned int globalID = (blockIdx.y* gridDim.x+ blockIdx.x)*blockDim.x + threadIdx.x;
  if(globalID < numPixels){
    int numValues = colorDepth;
    uchar3 value;
    switch(numValues){
      case 1:
        value = bwToRGB(colorPixels[globalID]);
        break;
      case 2:
        value = bwaToRGB({colorPixels[globalID*numValues],colorPixels[globalID*numValues + 1]});
        break;
      case 4:
        value = rgbaToRGB({colorPixels[globalID*numValues],colorPixels[globalID*numValues + 1], colorPixels[globalID*numValues + 2], colorPixels[globalID*numValues + 3]});
        break;
      default:
        printf("ERROR colorDepth of %u is not supported\n",numValues);
        asm("trap;");
    }
    pixels[globalID*3] = value.x;
    pixels[globalID*3 + 1] = value.y;
    pixels[globalID*3 + 2] = value.z;
  }
}

__global__ void ssrlcv::binImage(uint2 imageSize, unsigned int colorDepth, unsigned char* pixels, unsigned char* binnedImage){
  unsigned int x = blockIdx.x*blockDim.x + threadIdx.x;
  unsigned int y = blockIdx.y*blockDim.y + threadIdx.y;
  if(x < imageSize.x/2 && y < imageSize.y/2){
    for(int d = 0; d < colorDepth; ++d){
      float sumPix = pixels[y*colorDepth*2*imageSize.x + (x*2*colorDepth) + d] +
      pixels[(y*2+1)*colorDepth*imageSize.x + (x*2*colorDepth) + d] +
      pixels[y*2*colorDepth*imageSize.x + ((x*2+1)*colorDepth) + d] +
      pixels[(y*2+1)*colorDepth*imageSize.x + ((x*2+1)*colorDepth) + d];
      binnedImage[y*colorDepth*(imageSize.x/2) + (x*colorDepth) + d] = (unsigned char) roundf(sumPix/4.0f);
    }
  }
}
__global__ void ssrlcv::upsampleImage(uint2 imageSize, unsigned int colorDepth, unsigned char* pixels, unsigned char* upsampledImage){
  unsigned int i = blockIdx.x*blockDim.x + threadIdx.x;
  unsigned int j = blockIdx.y*blockDim.y + threadIdx.y;
  if(i < imageSize.x*2 && j < imageSize.y*2){
    float x = i*0.5f;
    float y = j*0.5f;
    int xm = getSymmetrizedCoord((int)x,imageSize.x);
    int xp = getSymmetrizedCoord((int)x + 1,imageSize.x);
    int ym = getSymmetrizedCoord((int)y,imageSize.y);
    int yp = getSymmetrizedCoord((int)y + 1,imageSize.y);

    float2 interLocDiff = {x-floor(x),y-floor(y)};

    for(int d = 0; d < colorDepth; ++d){
      float sumPix = interLocDiff.x*interLocDiff.y*((float)pixels[yp*colorDepth*imageSize.x + xp*colorDepth + d]);
      sumPix += (1.0f-interLocDiff.x)*interLocDiff.y*((float)pixels[yp*colorDepth*imageSize.x + xm*colorDepth + d]);
      sumPix += interLocDiff.x*(1-interLocDiff.y)*((float)pixels[ym*colorDepth*imageSize.x + xp*colorDepth + d]);
      sumPix += (1-interLocDiff.x)*(1-interLocDiff.y)*((float)pixels[ym*colorDepth*imageSize.x + xm*colorDepth + d]);
      upsampledImage[j*colorDepth*(imageSize.x*2) + i*colorDepth + d] = (unsigned char) sumPix;
    }
  }
}
__global__ void ssrlcv::bilinearInterpolation(uint2 imageSize, unsigned int colorDepth, unsigned char* pixels, unsigned char* outputPixels, float outputPixelWidth){
  unsigned int i = blockIdx.x*blockDim.x + threadIdx.x;
  unsigned int j = blockIdx.y*blockDim.y + threadIdx.y;
  if(i < imageSize.x/outputPixelWidth && j < imageSize.y/outputPixelWidth){
    float x = i*outputPixelWidth;
    float y = j*outputPixelWidth;
    int xm = getSymmetrizedCoord((int)x,imageSize.x);
    int xp = getSymmetrizedCoord((int)x + 1,imageSize.x);
    int ym = getSymmetrizedCoord((int)y,imageSize.y);
    int yp = getSymmetrizedCoord((int)y + 1,imageSize.y);

    float2 interLocDiff = {x-floor(x),y-floor(y)};

    for(int d = 0; d < colorDepth; ++d){
      float sumPix = interLocDiff.x*interLocDiff.y*((float)pixels[yp*colorDepth*imageSize.x + xp*colorDepth + d]);
      sumPix += (1.0f-interLocDiff.x)*interLocDiff.y*((float)pixels[yp*colorDepth*imageSize.x + xm*colorDepth + d]);
      sumPix += interLocDiff.x*(1-interLocDiff.y)*((float)pixels[ym*colorDepth*imageSize.x + xp*colorDepth + d]);
      sumPix += (1-interLocDiff.x)*(1-interLocDiff.y)*((float)pixels[ym*colorDepth*imageSize.x + xm*colorDepth + d]);
      outputPixels[j*colorDepth*llroundf(imageSize.x/outputPixelWidth) + i*colorDepth + d] = (unsigned char) sumPix;
    }
  }
}

__global__ void ssrlcv::binImage(uint2 imageSize, unsigned int colorDepth, float* pixels, float* binnedImage){
  unsigned int x = blockIdx.x*blockDim.x + threadIdx.x;
  unsigned int y = blockIdx.y*blockDim.y + threadIdx.y;
  if(x < imageSize.x/2 && y < imageSize.y/2){
    for(int d = 0; d < colorDepth; ++d){
      float sumPix = pixels[y*colorDepth*2*imageSize.x + x*2*colorDepth + d] +
      pixels[(y*2+1)*colorDepth*imageSize.x + x*2*colorDepth + d] +
      pixels[y*2*colorDepth*imageSize.x + (x*2+1)*colorDepth + d] +
      pixels[(y*2+1)*colorDepth*imageSize.x + (x*2+1)*colorDepth + d];
      binnedImage[y*colorDepth*(imageSize.x/2) + x*colorDepth + d] = sumPix/4.0f;
    }
  }
}
__global__ void ssrlcv::upsampleImage(uint2 imageSize, unsigned int colorDepth, float* pixels, float* upsampledImage){
  unsigned int i = blockIdx.x*blockDim.x + threadIdx.x;
  unsigned int j = blockIdx.y*blockDim.y + threadIdx.y;
  if(i < imageSize.x*2 && j < imageSize.y*2){
    float x = i*0.5f;
    float y = j*0.5f;
    int xm = getSymmetrizedCoord((int)x,imageSize.x);
    int xp = getSymmetrizedCoord((int)x + 1,imageSize.x);
    int ym = getSymmetrizedCoord((int)y,imageSize.y);
    int yp = getSymmetrizedCoord((int)y + 1,imageSize.y);

    float2 interLocDiff = {x-floor(x),y-floor(y)};

    for(int d = 0; d < colorDepth; ++d){
      float sumPix = interLocDiff.x*interLocDiff.y*((float)pixels[yp*colorDepth*imageSize.x + xp*colorDepth + d]);
      sumPix += (1.0f-interLocDiff.x)*interLocDiff.y*((float)pixels[yp*colorDepth*imageSize.x + xm*colorDepth + d]);
      sumPix += interLocDiff.x*(1-interLocDiff.y)*((float)pixels[ym*colorDepth*imageSize.x + xp*colorDepth + d]);
      sumPix += (1-interLocDiff.x)*(1-interLocDiff.y)*((float)pixels[ym*colorDepth*imageSize.x + xm*colorDepth + d]);
      upsampledImage[j*colorDepth*(imageSize.x*2) + i*colorDepth + d] = sumPix;
    }
  }
}
__global__ void ssrlcv::bilinearInterpolation(uint2 imageSize, unsigned int colorDepth, float* pixels, float* outputPixels, float outputPixelWidth){
  unsigned int i = blockIdx.x*blockDim.x + threadIdx.x;
  unsigned int j = blockIdx.y*blockDim.y + threadIdx.y;
  if(i < imageSize.x/outputPixelWidth && j < imageSize.y/outputPixelWidth){
    float x = i*outputPixelWidth;
    float y = j*outputPixelWidth;
    int xm = getSymmetrizedCoord((int)x,imageSize.x);
    int xp = getSymmetrizedCoord((int)x + 1,imageSize.x);
    int ym = getSymmetrizedCoord((int)y,imageSize.y);
    int yp = getSymmetrizedCoord((int)y + 1,imageSize.y);

    float2 interLocDiff = {x-floor(x),y-floor(y)};

    for(int d = 0; d < colorDepth; ++d){
      float sumPix = interLocDiff.x*interLocDiff.y*((float)pixels[yp*colorDepth*imageSize.x + xp*colorDepth + d]);
      sumPix += (1.0f-interLocDiff.x)*interLocDiff.y*((float)pixels[yp*colorDepth*imageSize.x + xm*colorDepth + d]);
      sumPix += interLocDiff.x*(1-interLocDiff.y)*((float)pixels[ym*colorDepth*imageSize.x + xp*colorDepth + d]);
      sumPix += (1-interLocDiff.x)*(1-interLocDiff.y)*((float)pixels[ym*colorDepth*imageSize.x + xm*colorDepth + d]);
      outputPixels[j*colorDepth*llroundf(imageSize.x/outputPixelWidth) + i*colorDepth + d] = sumPix;
    }
  }
}

__global__ void ssrlcv::convolveImage(uint2 imageSize, unsigned char* pixels, unsigned int colorDepth, int2 kernelSize, float* kernel, float* convolvedImage){
  unsigned int x = blockIdx.x*blockDim.x + threadIdx.x;
  unsigned int y = blockIdx.y*blockDim.y + threadIdx.y;
  unsigned int color = blockIdx.z*blockDim.z + threadIdx.z;
  if(x < imageSize.x && y < imageSize.y){
    if(x + (kernelSize.x/2) >= imageSize.x || x < kernelSize.x/2 || y + (kernelSize.y/2) >= imageSize.y || y < kernelSize.y/2){
      convolvedImage[(y*imageSize.x + x)*colorDepth + color] = 0;
    }
    else{
      float sum = 0.0f;
      for(int ky = -kernelSize.y/2; ky <= kernelSize.y/2; ++ky){
        for(int kx = -kernelSize.x/2; kx <= kernelSize.x/2; ++kx){
          sum += ((float)pixels[((y+ky)*imageSize.x + (x+kx))*colorDepth + color])*kernel[(ky+(kernelSize.y/2))*kernelSize.x + (kx+(kernelSize.x/2))];
        }
      }
      convolvedImage[(y*imageSize.x + x)*colorDepth + color] = sum;
    }
  }
}
__global__ void ssrlcv::convolveImage(uint2 imageSize, float* pixels, unsigned int colorDepth, int2 kernelSize, float* kernel, float* convolvedImage){
  unsigned int x = blockIdx.x*blockDim.x + threadIdx.x;
  unsigned int y = blockIdx.y*blockDim.y + threadIdx.y;
  unsigned int color = blockIdx.z*blockDim.z + threadIdx.z;
  if(x < imageSize.x && y < imageSize.y){
    if(x + (kernelSize.x/2) >= imageSize.x || x < kernelSize.x/2 || y + (kernelSize.y/2) >= imageSize.y || y < kernelSize.y/2){
      convolvedImage[(y*imageSize.x + x)*colorDepth + color] = 0;
    }
    else{
      float sum = 0.0f;
      for(int ky = -kernelSize.y/2; ky <= kernelSize.y/2; ++ky){
        for(int kx = -kernelSize.x/2; kx <= kernelSize.x/2; ++kx){
          sum += pixels[((y+ky)*imageSize.x + (x+kx))*colorDepth + color]*kernel[(ky+(kernelSize.y/2))*kernelSize.x + (kx+(kernelSize.x/2))];
        }
      }
      convolvedImage[(y*imageSize.x + x)*colorDepth + color] = sum;
    }
  }
}
__global__ void ssrlcv::convolveImage_symmetric(uint2 imageSize, unsigned char* pixels, unsigned int colorDepth, int2 kernelSize, float* kernel, float* convolvedImage){
  unsigned int x = blockIdx.x*blockDim.x + threadIdx.x;
  unsigned int y = blockIdx.y*blockDim.y + threadIdx.y;
  unsigned int color = blockIdx.z*blockDim.z + threadIdx.z;
  if(x < imageSize.x && y < imageSize.y){
    int2 symmetricCoord = {0,0};
    float sum = 0.0f;
    for(int ky = -kernelSize.y/2; ky <= kernelSize.y/2; ++ky){
      for(int kx = -kernelSize.x/2; kx <= kernelSize.x/2; ++kx){
        symmetricCoord = {getSymmetrizedCoord(x+kx,(int)imageSize.x),getSymmetrizedCoord(y+ky,(int)imageSize.y)};
        sum += ((float)pixels[((symmetricCoord.y)*imageSize.x + (symmetricCoord.x))*colorDepth + color])*kernel[(ky+(kernelSize.y/2))*kernelSize.x + (kx+(kernelSize.x/2))];
      }
    }
    convolvedImage[(y*imageSize.x + x)*colorDepth + color] = sum;
  }
}
__global__ void ssrlcv::convolveImage_symmetric(uint2 imageSize, float* pixels, unsigned int colorDepth, int2 kernelSize, float* kernel, float* convolvedImage){
  unsigned int x = blockIdx.x*blockDim.x + threadIdx.x;
  unsigned int y = blockIdx.y*blockDim.y + threadIdx.y;
  unsigned int color = blockIdx.z*blockDim.z + threadIdx.z;
  if(x < imageSize.x && y < imageSize.y){
    int2 symmetricCoord = {0,0};
    float sum = 0.0f;
    for(int ky = -kernelSize.y/2; ky <= kernelSize.y/2; ++ky){
      for(int kx = -kernelSize.x/2; kx <= kernelSize.x/2; ++kx){
        symmetricCoord = {getSymmetrizedCoord(x+kx,(int)imageSize.x),getSymmetrizedCoord(y+ky,(int)imageSize.y)};
        sum += pixels[((symmetricCoord.y)*imageSize.x + (symmetricCoord.x))*colorDepth + color]*kernel[(ky+(kernelSize.y/2))*kernelSize.x + (kx+(kernelSize.x/2))];
      }
    }
    convolvedImage[(y*imageSize.x + x)*colorDepth + color] = sum;
  }
}

__global__ void ssrlcv::convertToCharImage(unsigned int numPixels, unsigned char* pixels, float* fltPixels){
  unsigned int globalID = (blockIdx.y* gridDim.x+ blockIdx.x)*blockDim.x + threadIdx.x;
  if(globalID < numPixels){
    pixels[globalID] = (unsigned char) 255.0f*fltPixels[globalID];
  }
}
__global__ void ssrlcv::convertToFltImage(unsigned int numPixels, unsigned char* pixels, float* fltPixels){
  unsigned int globalID = (blockIdx.y* gridDim.x+ blockIdx.x)*blockDim.x + threadIdx.x;
  if(globalID < numPixels){
    fltPixels[globalID] = (float) pixels[globalID];
  }
}
__global__ void ssrlcv::normalize(unsigned long numPixels, float* pixels, float2 minMax){
  unsigned int globalID = (blockIdx.y* gridDim.x+ blockIdx.x)*blockDim.x + threadIdx.x;
  if(globalID < numPixels){
    pixels[globalID] = (pixels[globalID] - minMax.x)/(minMax.y - minMax.x);
  }
}

__global__ void ssrlcv::calculatePixelGradients(uint2 imageSize, unsigned char* pixels, int2* gradients){
  unsigned long globalID = (blockIdx.y* gridDim.x+ blockIdx.x)*blockDim.x + threadIdx.x;
  if(globalID < imageSize.x*imageSize.y){
    int2 loc = {(int)globalID%imageSize.x,(int)globalID/imageSize.x};
    int2 xContrib = {loc.x + 1,loc.x - 1};
    int2 yContrib = {loc.y + 1,loc.y - 1};
    if(xContrib.y == -1) xContrib = xContrib + 1;
    else if(xContrib.x == imageSize.x) xContrib = xContrib - 1;
    if(yContrib.y == -1) yContrib = yContrib + 1;
    else if(yContrib.x == imageSize.y) yContrib = yContrib - 1;
    gradients[globalID] = {
      (int)pixels[loc.y*imageSize.x + xContrib.x] - (int)pixels[loc.y*imageSize.x + xContrib.y],
      (int)pixels[yContrib.x*imageSize.x + loc.x] - (int)pixels[yContrib.y*imageSize.x + loc.x]
    };
  }
}
__global__ void ssrlcv::calculatePixelGradients(uint2 imageSize, float* pixels, float2* gradients){
  unsigned long globalID = (blockIdx.y* gridDim.x+ blockIdx.x)*blockDim.x + threadIdx.x;
  if(globalID < imageSize.x*imageSize.y){
    int2 loc = {(int)globalID%imageSize.x,(int)globalID/imageSize.x};
    int2 xContrib = {loc.x + 1,loc.x - 1};
    int2 yContrib = {loc.y + 1,loc.y - 1};
    if(xContrib.y == -1) xContrib = xContrib + 1;
    else if(xContrib.x == imageSize.x) xContrib = xContrib - 1;
    if(yContrib.y == -1) yContrib = yContrib + 1;
    else if(yContrib.x == imageSize.y) yContrib = yContrib - 1;
    gradients[globalID] = {
      pixels[loc.y*imageSize.x + xContrib.x] - pixels[loc.y*imageSize.x + xContrib.y],
      pixels[yContrib.x*imageSize.x + loc.x] - pixels[yContrib.y*imageSize.x + loc.x]
    };
  }
}
