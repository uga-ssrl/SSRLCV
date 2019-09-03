#include "Image.cuh"




__device__ __host__ ssrlcv::Image_Descriptor::Image_Descriptor(){
  this->id = 0;
  this->size = {0,0};
  this->cam_vec = {0.0f,0.0f,0.0f};
  this->cam_pos = {0.0f,0.0f,0.0f};
  this->fov = 0;
  this->foc = 0;
  this->dpix = 0.0f;
  this->colorDepth = 1;
}
__device__ __host__ ssrlcv::Image_Descriptor::Image_Descriptor(int id, uint2 size){
  this->id = id;
  this->size = size;
  this->cam_vec = {0.0f,0.0f,0.0f};
  this->cam_pos = {0.0f,0.0f,0.0f};
  this->fov = 0;
  this->foc = 0;
  this->dpix = 0.0f;
  this->colorDepth = 1;
}
__device__ __host__ ssrlcv::Image_Descriptor::Image_Descriptor(int id, uint2 size, float3 cam_pos, float3 camp_dir){
  this->id = id;
  this->size = size;
  this->cam_pos = cam_pos;
  this->cam_vec = cam_vec;
  this->fov = 0;
  this->foc = 0;
  this->dpix = 0.0f;
  this->colorDepth = 1;
}

ssrlcv::Image::Image(){
  this->descriptor.id = -1;
  this->filePath = "n/a";
}
ssrlcv::Image::Image(std::string filePath, int id){
  this->filePath = filePath;
  this->descriptor.id = id;
  this->descriptor.colorDepth = 1;
  unsigned char* pixels_host = readPNG(filePath.c_str(), this->descriptor.size.y, this->descriptor.size.x, this->descriptor.colorDepth);
  this->pixels = new Unity<unsigned char>(pixels_host,this->descriptor.size.y*this->descriptor.size.x*this->descriptor.colorDepth,cpu);
}
ssrlcv::Image::Image(std::string filePath, unsigned int convertColorDepthTo, int id){
  this->filePath = filePath;
  this->descriptor.id = id;
  this->descriptor.colorDepth = 1;
  unsigned char* pixels_host = readPNG(filePath.c_str(), this->descriptor.size.y, this->descriptor.size.x, this->descriptor.colorDepth);
  this->pixels = new Unity<unsigned char>(pixels_host,this->descriptor.size.y*this->descriptor.size.x*this->descriptor.colorDepth,cpu);
  if(convertColorDepthTo == 1){
    convertToBW(this->pixels, this->descriptor.colorDepth);
    this->descriptor.colorDepth = 1;
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

void ssrlcv::Image::alterSize(int binDepth){
  if(binDepth <= 0){
    std::cerr<<"Image::alterSize does not currently support upsampling"<<std::endl;
    exit(0);
  }
  MemoryState origin = this->pixels->state;
  if(origin == cpu || this->pixels->fore == cpu) this->pixels->transferMemoryTo(gpu);

  Unity<unsigned char>* alteredPixels = bin(this->descriptor.size,this->descriptor.colorDepth,this->pixels);
  delete this->pixels;
  this->pixels = alteredPixels;
  this->descriptor.size.x /= pow(2,binDepth);
  this->descriptor.size.y /= pow(2,binDepth);

  this->pixels->fore = gpu;
  if(origin == cpu) this->pixels->setMemoryState(cpu);
}


ssrlcv::Unity<unsigned char>* ssrlcv::bin(uint2 imageSize, unsigned int colorDepth, Unity<unsigned char>* pixels){
  MemoryState origin = pixels->state;

  if(origin == cpu || pixels->fore != gpu){
    pixels->transferMemoryTo(gpu);
  }
  unsigned char* binnedImage_device = nullptr;

  CudaSafeCall(cudaMalloc((void**)&binnedImage_device,(pixels->numElements/4)*colorDepth*sizeof(unsigned char)));
  binImage<<<{(imageSize.x/32)+1,(imageSize.y/32)+1,1},{32,32,1}>>>(imageSize,colorDepth,pixels->device,binnedImage_device);
  CudaCheckError();

  if(origin == cpu){
    pixels->setMemoryState(cpu);
  }

  Unity<unsigned char>* binnedImage = new Unity<unsigned char>(binnedImage_device, colorDepth*pixels->numElements/4, gpu);
  binnedImage->transferMemoryTo(cpu);

  return binnedImage;
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

ssrlcv::Unity<unsigned int>* ssrlcv::applyBorder(Image* image, float2 border){

  dim3 grid = {1,1,1};
  dim3 block = {1,1,1};
  getFlatGridBlock(image->pixels->numElements, grid, block);
  unsigned int* pixelNumbers_device = nullptr;
  unsigned int* pixelAddresses_device = nullptr;
  CudaSafeCall(cudaMalloc((void**)&pixelNumbers_device,image->pixels->numElements*sizeof(unsigned int)));
  CudaSafeCall(cudaMalloc((void**)&pixelAddresses_device,image->pixels->numElements*sizeof(unsigned int)));

  applyBorder<<<grid,block>>>(image->descriptor.size, pixelNumbers_device, pixelAddresses_device, border);
  cudaDeviceSynchronize();
  CudaCheckError();

  thrust::device_ptr<unsigned int> pN(pixelNumbers_device);
  thrust::inclusive_scan(pN, pN + image->pixels->numElements, pN);

  int numValidPixels = 0;
  CudaSafeCall(cudaMemcpy(&numValidPixels,pixelNumbers_device + (image->pixels->numElements - 1), sizeof(unsigned int), cudaMemcpyDeviceToHost));
  CudaSafeCall(cudaFree(pixelNumbers_device));

  thrust::device_ptr<unsigned int> pA(pixelAddresses_device);
  thrust::remove(pA, pA + image->pixels->numElements, 0);

  unsigned int* pixelAddressesReduced_device = nullptr;
  CudaSafeCall(cudaMalloc((void**)&pixelAddressesReduced_device,numValidPixels*sizeof(unsigned int)));
  CudaSafeCall(cudaMemcpy(pixelAddressesReduced_device,pixelAddresses_device,numValidPixels*sizeof(unsigned int),cudaMemcpyDeviceToDevice));

  CudaSafeCall(cudaFree(pixelAddresses_device));

  return new Unity<unsigned int>(pixelAddressesReduced_device,numValidPixels,gpu);
}
ssrlcv::Unity<unsigned int>* ssrlcv::applyBorder(uint2 imageSize, float2 border){

  dim3 grid = {1,1,1};
  dim3 block = {1,1,1};
  getFlatGridBlock(imageSize.x*imageSize.y, grid, block);
  unsigned int* pixelNumbers_device = nullptr;
  unsigned int* pixelAddresses_device = nullptr;
  CudaSafeCall(cudaMalloc((void**)&pixelNumbers_device,imageSize.x*imageSize.y*sizeof(unsigned int)));
  CudaSafeCall(cudaMalloc((void**)&pixelAddresses_device,imageSize.x*imageSize.y*sizeof(unsigned int)));

  applyBorder<<<grid,block>>>(imageSize, pixelNumbers_device, pixelAddresses_device, border);
  cudaDeviceSynchronize();
  CudaCheckError();

  thrust::device_ptr<unsigned int> pN(pixelNumbers_device);
  thrust::inclusive_scan(pN, pN + imageSize.x*imageSize.y, pN);

  int numValidPixels = 0;
  CudaSafeCall(cudaMemcpy(&numValidPixels,pixelNumbers_device + (imageSize.x*imageSize.y - 1), sizeof(unsigned int), cudaMemcpyDeviceToHost));
  CudaSafeCall(cudaFree(pixelNumbers_device));

  thrust::device_ptr<unsigned int> pA(pixelAddresses_device);
  thrust::remove(pA, pA + imageSize.x*imageSize.y, 0);

  unsigned int* pixelAddressesReduced_device = nullptr;
  CudaSafeCall(cudaMalloc((void**)&pixelAddressesReduced_device,numValidPixels*sizeof(unsigned int)));
  CudaSafeCall(cudaMemcpy(pixelAddressesReduced_device,pixelAddresses_device,numValidPixels*sizeof(unsigned int),cudaMemcpyDeviceToDevice));

  CudaSafeCall(cudaFree(pixelAddresses_device));

  return new Unity<unsigned int>(pixelAddressesReduced_device,numValidPixels,gpu);
}
ssrlcv::Unity<float2>* ssrlcv::getLocationsWithinBorder(Image* image, float2 border){
  Unity<unsigned int>* pixelAddresses = applyBorder(image, border);
  float2* pixelCenters_device = nullptr;
  CudaSafeCall(cudaMalloc((void**)&pixelCenters_device,pixelAddresses->numElements*sizeof(float2)));
  dim3 grid = {1,1,1};
  dim3 block = {1,1,1};
  getFlatGridBlock(pixelAddresses->numElements,grid,block);
  getPixelCenters<<<grid,block>>>(pixelAddresses->numElements,image->descriptor.size,pixelAddresses->device,pixelCenters_device);
  CudaCheckError();
  unsigned long numPixels = pixelAddresses->numElements;
  delete pixelAddresses;
  return new Unity<float2>(pixelCenters_device,numPixels,gpu);
}
ssrlcv::Unity<float2>* ssrlcv::getLocationsWithinBorder(uint2 imageSize, float2 border){
  Unity<unsigned int>* pixelAddresses = applyBorder(imageSize, border);
  float2* pixelCenters_device = nullptr;
  CudaSafeCall(cudaMalloc((void**)&pixelCenters_device,pixelAddresses->numElements*sizeof(float2)));
  dim3 grid = {1,1,1};
  dim3 block = {1,1,1};
  getFlatGridBlock(pixelAddresses->numElements,grid,block);
  getPixelCenters<<<grid,block>>>(pixelAddresses->numElements,imageSize,pixelAddresses->device,pixelCenters_device);
  CudaCheckError();
  unsigned long numPixels = pixelAddresses->numElements;
  delete pixelAddresses;
  return new Unity<float2>(pixelCenters_device,numPixels,gpu);
}

ssrlcv::Unity<int2>* ssrlcv::generatePixelGradients(Image* image){
  MemoryState origin = image->pixels->state;
  if(origin == cpu || image->pixels->fore == cpu){
    image->pixels->transferMemoryTo(gpu);
  }
  int2* gradients_device = nullptr;
  CudaSafeCall(cudaMalloc((void**)&gradients_device,image->pixels->numElements*sizeof(int2)));
  dim3 grid = {1,1,1};
  dim3 block = {1,1,1};
  getFlatGridBlock(image->pixels->numElements,grid,block);
  calculatePixelGradients<<<grid,block>>>(image->descriptor.size,image->pixels->device,gradients_device);
  CudaCheckError();
  if(origin == cpu) image->pixels->setMemoryState(cpu);

  return new Unity<int2>(gradients_device,image->pixels->numElements,gpu);
}
ssrlcv::Unity<int2>* ssrlcv::generatePixelGradients(uint2 imageSize, Unity<unsigned char>* pixels){
  MemoryState origin = pixels->state;
  if(origin == cpu || pixels->fore == cpu){
    pixels->transferMemoryTo(gpu);
  }
  int2* gradients_device = nullptr;
  CudaSafeCall(cudaMalloc((void**)gradients_device,pixels->numElements*sizeof(int2)));
  dim3 grid = {1,1,1};
  dim3 block = {1,1,1};
  getFlatGridBlock(pixels->numElements,grid,block);
  calculatePixelGradients<<<grid,block>>>(imageSize,pixels->device,gradients_device);
  CudaCheckError();

  if(origin == cpu) pixels->setMemoryState(cpu);

  return new Unity<int2>(gradients_device,pixels->numElements,gpu);
}

//convolve host method goes here - will call convolveImage

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

__global__ void ssrlcv::binImage(uint2 imageSize, unsigned int colorDepth, unsigned char* pixels, unsigned char* binnedImage){
  unsigned int x = blockIdx.x*blockDim.x + threadIdx.x;
  unsigned int y = blockIdx.y*blockDim.y + threadIdx.y;
  if(x < imageSize.x/2 && y < imageSize.y/2){
    for(int d = 0; d < colorDepth; ++d){
      float sumPix = pixels[y*colorDepth*2*imageSize.x + x*2*colorDepth + d] +
      pixels[(y + 1)*colorDepth*2*imageSize.x + x*2*colorDepth + d] +
      pixels[y*colorDepth*2*imageSize.x + (x+1)*2*colorDepth + d] +
      pixels[(y+1)*colorDepth*2*imageSize.x + (x+1)*2*colorDepth + d];
      binnedImage[y*colorDepth*(imageSize.x/2) + x*colorDepth + d] = (unsigned char) (sumPix/4);
    }
  }
}

__global__ void ssrlcv::convolveImage(uint2 imageSize, unsigned char* pixels, unsigned int colorDepth, unsigned int kernelSize, float* kernel, unsigned char* convolvedImage){
  unsigned int x = blockIdx.x*blockDim.x + threadIdx.x;
  unsigned int y = blockIdx.y*blockDim.y + threadIdx.y;
  unsigned int color = blockIdx.z*blockDim.z + threadIdx.z;
  if(x < imageSize.x && y < imageSize.y){
    if(x + (kernelSize/2) >= imageSize.x || x < kernelSize/2 || y + (kernelSize/2) >= imageSize.y || y < kernelSize/2){
      convolvedImage[(y*imageSize.x + x)*colorDepth + color] = 0;
    }
    else{
      float* sums = new float[colorDepth];
      for(int kx = -kernelSize/2; kx <= kernelSize/2; ++kx){
        for(int ky = -kernelSize/2; ky <= kernelSize/2; ++ky){
          for(int c = 0; c < colorDepth; ++c){
            sums[c] += (pixels[((y+ky)*imageSize.x + (x+kx))*colorDepth + c]*kernel[(ky+(kernelSize/2))*kernelSize + (kx+(kernelSize/2))]);
          }
        }
      }
      for(int c = 0; c < colorDepth; ++c){
        convolvedImage[(y*imageSize.x + x)*colorDepth + c] = (unsigned char) sums[c]/(kernelSize*kernelSize);
      }
    }
  }
}

__global__ void ssrlcv::applyBorder(uint2 imageSize, unsigned int* featureNumbers, unsigned int* featureAddresses, float2 border){
  unsigned int globalID = (blockIdx.y* gridDim.x+ blockIdx.x)*blockDim.x + threadIdx.x;
  if(globalID < imageSize.x*imageSize.y){
    unsigned int x = globalID%imageSize.x;
    unsigned int y = globalID/imageSize.x;
    if(x >= border.x && x < imageSize.x - border.x && y >= border.y && y < imageSize.y - border.y){
      featureNumbers[globalID] = 1;
      featureAddresses[globalID] = globalID;
    }
    else{
      featureNumbers[globalID] = 0;
      featureAddresses[globalID] = 0;
    }
  }
}
__global__ void ssrlcv::getPixelCenters(unsigned int numValidPixels, uint2 imageSize, unsigned int* pixelAddresses, float2* pixelCenters){
  unsigned int globalID = (blockIdx.y* gridDim.x+ blockIdx.x)*blockDim.x + threadIdx.x;
  if(globalID < numValidPixels){
    pixelCenters[globalID] = {((float)(pixelAddresses[globalID]%imageSize.x)),((float)(pixelAddresses[globalID]/imageSize.x))};
  }
}

__global__ void ssrlcv::calculatePixelGradients(uint2 imageSize, unsigned char* pixels, int2* gradients){
  unsigned long globalID = (blockIdx.y* gridDim.x+ blockIdx.x)*blockDim.x + threadIdx.x;
  if(globalID < imageSize.x*imageSize.y){
    uint2 loc = {globalID%imageSize.x,globalID/imageSize.x};
    if(loc.x == 0 || loc.x == imageSize.x - 1 || loc.y == 0 || loc.y == imageSize.y - 1){
      gradients[globalID] = {0,0};
    }
    else{
      gradients[globalID].x = (int)pixels[loc.y*imageSize.x + loc.x + 1] - (int)pixels[loc.y*imageSize.x + loc.x - 1];
      gradients[globalID].y = (int)pixels[(loc.y + 1)*imageSize.x + loc.x] - (int)pixels[(loc.y - 1)*imageSize.x + loc.x];
    }
  }
}
