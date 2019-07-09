#include "FeatureFactory.cuh"

/*
HOST METHODS
*/
//Base feature factory


ssrlcv::FeatureFactory::FeatureFactory(){

}

ssrlcv::SIFT_FeatureFactory::SIFT_FeatureFactory(){

}

void ssrlcv::SIFT_FeatureFactory::fillDescriptors(ssrlcv::Image* image, ssrlcv::Unity<ssrlcv::Feature<ssrlcv::SIFT_Descriptor>>* features){
  dim3 grid = {1,1,1};
  dim3 block = {9,1,1};
  getGrid(features->numElements, grid);

  if(image->pixels->fore == cpu){
    image->pixels->transferMemoryTo(gpu);
  }
  if(features->fore == cpu){
    features->transferMemoryTo(gpu);
  }

  std::cout<<"computing thetas for feature descriptors..."<<std::endl;
  clock_t timer = clock();
  computeThetas<<<grid, block>>>(features->numElements, image->descriptor, image->pixels->device, features->device);
  cudaDeviceSynchronize();
  CudaCheckError();
  printf("done in %f seconds.\n\n",((float) clock() -  timer)/CLOCKS_PER_SEC);

  block = {18,18,1};
  std::cout<<"generating feature descriptors..."<<std::endl;
  timer = clock();
  fillDescriptorsDensly<<<grid,block>>>(features->numElements, image->descriptor, image->pixels->device, features->device);
  cudaDeviceSynchronize();
  CudaCheckError();
  printf("done in %f seconds.\n\n",((float) clock() -  timer)/CLOCKS_PER_SEC);
}
ssrlcv::Unity<ssrlcv::Feature<ssrlcv::SIFT_Descriptor>>* ssrlcv::SIFT_FeatureFactory::generateFeaturesDensly(ssrlcv::Image* image){
  Unity<Feature<SIFT_Descriptor>>* features = nullptr;
  std::cout<<"generating features"<<std::endl;
  if(image->pixels == nullptr || image->pixels->state == null){
    std::cout<<"ERROR must have pixels for generating features"<<std::endl;
    exit(-1);
  }
  MemoryState origin = image->pixels->state;
  if(origin != gpu || image->pixels->fore == cpu){
    image->pixels->transferMemoryTo(gpu);
  }
  if(image->colorDepth != 1){
    image->convertToBW();
  }

  Feature<SIFT_Descriptor>* features_device = nullptr;
  CudaSafeCall(cudaMalloc((void**)&features_device, image->pixels->numElements*sizeof(Feature<SIFT_Descriptor>)));
  features = new Unity<Feature<SIFT_Descriptor>>(features_device, image->pixels->numElements, gpu);


  dim3 grid = {(unsigned int)image->descriptor.size.x,(unsigned int)image->descriptor.size.y,1};
  dim3 block = {1,1,1};
  clock_t timer = clock();
  initFeatureArray<<<grid, block>>>(features->numElements, image->descriptor, features->device);
  cudaDeviceSynchronize();
  CudaCheckError();

  printf("done in %f seconds.\n\n",((float) clock() -  timer)/CLOCKS_PER_SEC);

  this->fillDescriptors(image, features);

  image->pixels->transferMemoryTo(origin);
  if(origin == cpu){
    image->pixels->clear(gpu);
  }

  return features;
}

/*
CUDA implementations
*/

__constant__ float ssrlcv::pi = 3.1415926535897932384626433832795028841971693993751058209749445923078164062;
__constant__ int2 ssrlcv::immediateNeighbors[9] = {
  {-1, -1},
  {-1, 0},
  {-1, 1},
  {0, -1},
  {0, 0},
  {0, 1},
  {1, -1},
  {1, 0},
  {1, 1}
};

/*
DEVICE METHODS
*/
__device__ __forceinline__ unsigned long ssrlcv::getGlobalIdx_2D_1D(){
  unsigned long blockId = blockIdx.y * gridDim.x + blockIdx.x;
  unsigned long threadId = blockId * blockDim.x + threadIdx.x;
  return threadId;
}
__device__ __forceinline__ float ssrlcv::getMagnitude(const int2 &vector){
  return sqrtf(dotProduct(vector, vector));
}
__device__ __forceinline__ float ssrlcv::getTheta(const int2 &vector){
  float theta = atan2f((float)vector.y, (float)vector.x) + pi;
  return fmod(theta,2.0f*pi);
}
__device__ __forceinline__ float ssrlcv::getTheta(const float2 &vector){
  float theta = atan2f(vector.y, vector.x) + pi;
  return fmod(theta,2.0f*pi);
}
__device__ __forceinline__ float ssrlcv::getTheta(const float2 &vector, const float &offset){
  float theta = (atan2f(vector.y, vector.x) + pi) - offset;
  return fmod(theta + 2.0f*pi,2.0f*pi);
}
__device__ void ssrlcv::trickleSwap(const float2 &compareWValue, float2* &arr, int index, const int &length){
  for(int i = index; i < length; ++i){
    if(compareWValue.x > arr[i].x){
      float2 temp = arr[i];
      arr[i] = compareWValue;
      if((temp.x == 0.0f && temp.y == 0.0f)|| index + 1 == length) return;
      return trickleSwap(temp, arr, index + 1, length);
    }
  }
}
__device__ __forceinline__ long4 ssrlcv::getOrientationContributers(const long2 &loc, const int2 &imageSize){
  long4 orientationContributers;
  long pixelIndex = loc.y*imageSize.x + loc.x;
  orientationContributers.x = (loc.x == imageSize.x - 1) ? -1 : pixelIndex + 1;
  orientationContributers.y = (loc.x == 0) ? -1 : pixelIndex - 1;
  orientationContributers.z = (loc.y == imageSize.y - 1) ? -1 : (loc.y + 1)*imageSize.x + loc.x;
  orientationContributers.w = (loc.y == 0) ? -1 : (loc.y - 1)*imageSize.x + loc.x;
  return orientationContributers;
}
__device__ __forceinline__ int ssrlcv::floatToOrderedInt(float floatVal){
 int intVal = __float_as_int( floatVal );
 return (intVal >= 0 ) ? intVal : intVal ^ 0x7FFFFFFF;
}
__device__ __forceinline__ float ssrlcv::orderedIntToFloat(int intVal){
 return __int_as_float( (intVal >= 0) ? intVal : intVal ^ 0x7FFFFFFF);
}
__device__ __forceinline__ float ssrlcv::atomicMinFloat (float * addr, float value) {
  float old;
  old = (value >= 0) ? __int_as_float(atomicMin((int *)addr, __float_as_int(value))) :
    __uint_as_float(atomicMax((unsigned int *)addr, __float_as_uint(value)));
  return old;
}
__device__ __forceinline__ float ssrlcv::atomicMaxFloat (float * addr, float value) {
  float old;
  old = (value >= 0) ? __int_as_float(atomicMax((int *)addr, __float_as_int(value))) :
    __uint_as_float(atomicMin((unsigned int *)addr, __float_as_uint(value)));
  return old;
}
__device__ __forceinline__ float ssrlcv::modulus(const float &x, const float &y){
    float z = x;
    int n;
    if(z < 0){
        n = (int)((-z)/y)+1;
        z += n*y;
    }
    n = (int)(z/y);
    z -= n*y;
    return z;
}
__device__ __forceinline__ float2 ssrlcv::rotateAboutPoint(const int2 &loc, const float &theta, const float2 &origin){
  float2 rotatedPoint = {(float) loc.x, (float) loc.y};
  rotatedPoint = rotatedPoint - origin;
  float2 temp = rotatedPoint;

  rotatedPoint.x = (temp.x*cosf(theta)) - (temp.y*sinf(theta)) + origin.x;
  rotatedPoint.y = (temp.x*sinf(theta)) + (temp.y*cosf(theta)) + origin.y;

  return rotatedPoint;
}
/*
KERNELS
*/

__global__ void ssrlcv::initFeatureArray(unsigned long totalFeatures, ssrlcv::Image_Descriptor image, ssrlcv::Feature<ssrlcv::SIFT_Descriptor>* features){
  float2 locationInParent = {((float)blockIdx.y) + 0.5f,((float)blockIdx.x) + 0.5f};
  features[blockIdx.y*gridDim.x + blockIdx.x] = Feature<SIFT_Descriptor>(locationInParent, SIFT_Descriptor());
}
//possible to use this kernel for multiple orientations
//NOTE currently not doing so
__global__ void ssrlcv::computeThetas(unsigned long totalFeatures, ssrlcv::Image_Descriptor image, unsigned char* pixels, ssrlcv::Feature<ssrlcv::SIFT_Descriptor>* features){
  unsigned long blockId = blockIdx.y * gridDim.x + blockIdx.x;
  if(blockId < totalFeatures){
    ssrlcv::Feature<SIFT_Descriptor> feature = features[blockId];
    __shared__ int2 orientationVectors[9];
    ssrlcv::Image_Descriptor parentRef = image;

    //vector = (x+1) - (x-1), (y+1) - (y-1)
    long2 orientLoc = {lrintf(feature.loc.x + immediateNeighbors[threadIdx.x].x),lrintf(feature.loc.y + immediateNeighbors[threadIdx.x].y)};
    long4 orientationContributers = getOrientationContributers(orientLoc, parentRef.size);
    orientationVectors[threadIdx.x].x = pixels[orientationContributers.x] - pixels[orientationContributers.y];
    orientationVectors[threadIdx.x].y = pixels[orientationContributers.z] - pixels[orientationContributers.w];

    __syncthreads();

    if(threadIdx.x != 0) return;

    //commented portions can be used to utilize multiple orientations
    //float2* bestMagWThetas = new float2[numOrientations];
    //for(int i = 0; i < numOrientations; ++i) bestMagWThetas[i] = {0.0f,0.0f};
    float2 bestMagWTheta = {0.0f,0.0f};//would need to replace with above variable to utilize multiple orientations
    float2 tempMagWTheta = {0.0f,0.0f};
    int2 currentOrientationVector = {0,0};
    for(int i = 0; i < 9; ++i){
      currentOrientationVector = orientationVectors[i];
      tempMagWTheta = {getMagnitude(currentOrientationVector), getTheta(currentOrientationVector)};
      //trickleSwap(tempMagWTheta, bestMagWThetas, 0, regNumOrient);
      if(tempMagWTheta.x > bestMagWTheta.x) bestMagWTheta = tempMagWTheta;
    }
    features[blockId].descriptor.theta = bestMagWTheta.y;
    //delete[] bestMagWThetas;
  }
}
__global__ void ssrlcv::fillDescriptorsDensly(unsigned long totalFeatures, ssrlcv::Image_Descriptor image, unsigned char* pixels, ssrlcv::Feature<ssrlcv::SIFT_Descriptor>* features){
  unsigned long blockId = blockIdx.y * gridDim.x + blockIdx.x;
  if(blockId < totalFeatures){

    ssrlcv::Feature<SIFT_Descriptor> feature = features[blockId];
    __shared__ float2 descriptorGrid[18][18];
    __shared__ float localMax;
    __shared__ float localMin;
    descriptorGrid[threadIdx.x][threadIdx.y] = {0.0f,0.0f};
    localMax = 0.0f;
    localMin = FLT_MAX;
    __syncthreads();
    /*
    FIRST DEFINE HOG GRID
    (x,y) = [(-8.5,-8.5),(8.5,8.5)]
      x' = xcos(theta) - ysin(theta) + feature.x
      y' = ycos(theta) + xsin(theta) + feature.y

    */
    ssrlcv::Image_Descriptor parentRef = image;
    float theta = feature.descriptor.theta;

    float2 descriptorGridPoint = {0.0f, 0.0f};
    descriptorGridPoint.x = (((threadIdx.x - 8.5f)*cosf(theta)) - ((threadIdx.y - 8.5f)*sinf(theta))) + feature.loc.x;
    descriptorGridPoint.y = (((threadIdx.x - 8.5f)*sinf(theta)) + ((threadIdx.y - 8.5f)*cosf(theta))) + feature.loc.y;
    float2 pixValueLoc = {((threadIdx.x - 8.5f) + feature.loc.x), ((threadIdx.y - 8.5f) + feature.loc.y)};

    float newValue = 0.0f;
    ulong4 pixContributers;
    pixContributers.x = (((int)(pixValueLoc.y - 0.5f))*parentRef.size.x) + ((int)(pixValueLoc.x - 0.5f));
    pixContributers.y = (((int)(pixValueLoc.y - 0.5f))*parentRef.size.x) + ((int)(pixValueLoc.x + 0.5f));
    pixContributers.z = (((int)(pixValueLoc.y + 0.5f))*parentRef.size.x) + ((int)(pixValueLoc.x - 0.5f));
    pixContributers.w = (((int)(pixValueLoc.y + 0.5f))*parentRef.size.x) + ((int)(pixValueLoc.x + 0.5f));
    newValue += pixels[pixContributers.x];
    newValue += pixels[pixContributers.y];
    newValue += pixels[pixContributers.z];
    newValue += pixels[pixContributers.w];
    newValue /= 4.0f;

    descriptorGrid[threadIdx.x][threadIdx.y].x = newValue;
    __syncthreads();

    float2 grad = {0.0f, 0.0f};//magnitude, orientation
    if(threadIdx.x > 0 && threadIdx.x < 17 && threadIdx.y > 0 && threadIdx.y < 17){
      float2 vector = {descriptorGrid[threadIdx.x + 1][threadIdx.y].x -
        descriptorGrid[threadIdx.x - 1][threadIdx.y].x,
        descriptorGrid[threadIdx.x][threadIdx.y + 1].x -
        descriptorGrid[threadIdx.x][threadIdx.y - 1].x};

      //expf stuff is the gaussian weighting function
      float expPow = -sqrtf(dotProduct(descriptorGridPoint - feature.loc,descriptorGridPoint - feature.loc))/16.0f;
      grad.x = sqrtf(dotProduct(vector, vector))*expf(expPow);
      grad.y = getTheta(vector,theta);
    }
    __syncthreads();
    /*
    NOW CREATE HOG AND GET DESCRIPTOR
    */
    descriptorGrid[threadIdx.x][threadIdx.y] = grad;
    if(threadIdx.x >= 4 || threadIdx.y >= 4) return;
    int2 gradDomain = {((int) threadIdx.x*4) + 1, ((int) threadIdx.x + 1)*4};
    int2 gradRange = {((int) threadIdx.y*4) + 1, ((int) threadIdx.y + 1)*4};

    float bin_descriptors[8] = {0.0f};
    float rad45 = 45.0f*(pi/180.0f);
    for(int x = gradDomain.x; x <= gradDomain.y; ++x){
      for(int y = gradRange.x; y <= gradRange.y; ++y){
        for(int o = 1; o < 9; ++o){
          if(o*rad45 > descriptorGrid[x][y].y){
            bin_descriptors[o - 1] += descriptorGrid[x][y].x;
            break;
          }
        }
      }
    }
    __syncthreads();
    /*
    NORMALIZE
    */
    for(int d = 0; d < 8; ++d){
      atomicMinFloat(&localMin, bin_descriptors[d]);
      atomicMaxFloat(&localMax, bin_descriptors[d]);
    }
    __syncthreads();
    for(int d = 0; d < 8; ++d){
      feature.descriptor.values[(threadIdx.y*4 + threadIdx.x)*8 + d] =
        __float2int_rn(255*(bin_descriptors[d]-localMin)/(localMax-localMin));
    }
    features[blockId] = feature;
  }
}
