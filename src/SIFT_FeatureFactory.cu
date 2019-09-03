#include "SIFT_FeatureFactory.cuh"


ssrlcv::SIFT_FeatureFactory::SIFT_FeatureFactory(bool dense, unsigned int maxOrientations, float orientationThreshold) :
dense(dense), maxOrientations(maxOrientations), orientationThreshold(orientationThreshold){
  this->orientationContribWidth = 1.5;
  this->descriptorContribWidth = 6.0;
}

void ssrlcv::SIFT_FeatureFactory::setDensity(bool dense){
  this->dense = dense;
}
void ssrlcv::SIFT_FeatureFactory::setMaxOrientations(unsigned int maxOrientations){
  if(maxOrientations == 0){
    std::cerr<<"ERROR cannot set maxOrientations to 0"<<std::endl;
    exit(-1);
  }
  this->maxOrientations = maxOrientations;
}
void ssrlcv::SIFT_FeatureFactory::setOrientationThreshold(float orientationThreshold){
  if(orientationThreshold == 0.0f){
    std::cerr<<"ERROR cannot set orientation threshold to 0.0f"<<std::endl;
    exit(-1);
  }
  this->orientationThreshold = orientationThreshold;
}

void ssrlcv::SIFT_FeatureFactory::setOrientationContribWidth(float orientationContribWidth){
  this->orientationContribWidth = orientationContribWidth;
}

void ssrlcv::SIFT_FeatureFactory::setDescriptorContribWidth(float descriptorContribWidth){
  this->descriptorContribWidth = descriptorContribWidth;
}



//binDepth currently not being used
ssrlcv::Unity<ssrlcv::Feature<ssrlcv::SIFT_Descriptor>>* ssrlcv::SIFT_FeatureFactory::generateFeatures(ssrlcv::Image* image){
  Unity<Feature<SIFT_Descriptor>>* features = nullptr;

  if(image->pixels->fore == cpu){
    image->pixels->transferMemoryTo(gpu);
  }
  if(image->colorDepth != 1){
    convertToBW(image->pixels,image->colorDepth);
    image->colorDepth = 1;
  }

  if(this->dense){
    dim3 grid = {1,1,1};
    dim3 block = {1,1,1};

    clock_t timer = clock();

    Unity<int2>* gradients = generatePixelGradients(image);
    Unity<float2>* keyPoints = getLocationsWithinBorder(image->size, {12.0f,12.0f});

    printf("\nDense SIFT prep done in %f seconds.\n\n",((float) clock() -  timer)/CLOCKS_PER_SEC);

    image->pixels->clear(gpu);//may want to nullify pixels for space

    features = this->createFeatures(image->size,1,sqrtf(2),gradients,keyPoints);

    delete gradients;
    delete keyPoints;
  }
  else{

  }


  return features;
}

ssrlcv::Unity<ssrlcv::Feature<ssrlcv::SIFT_Descriptor>>* ssrlcv::SIFT_FeatureFactory::createFeatures(uint2 imageSize, float pixelWidth, float sigma, Unity<int2>* gradients, Unity<float2>* keyPoints){

  clock_t timer = clock();

  int* thetaNumbers_device = nullptr;
  CudaSafeCall(cudaMalloc((void**)&thetaNumbers_device,keyPoints->numElements*this->maxOrientations*sizeof(int)));
  float* thetas_device = nullptr;
  CudaSafeCall(cudaMalloc((void**)&thetas_device,keyPoints->numElements*this->maxOrientations*sizeof(float)));
  float* thetas_host = new float[keyPoints->numElements*this->maxOrientations];
  for(int i = 0; i < keyPoints->numElements*this->maxOrientations; ++i){
    thetas_host[i] = -1.0f;
  }
  CudaSafeCall(cudaMemcpy(thetas_device,thetas_host,keyPoints->numElements*this->maxOrientations*sizeof(float),cudaMemcpyHostToDevice));

  dim3 grid = {1,1,1};
  dim3 block = {1,1,1};

  getFlatGridBlock(keyPoints->numElements,grid,block);

  computeThetas<<<grid,block>>>(keyPoints->numElements,imageSize.x,sigma,pixelWidth,
    this->orientationContribWidth,3.0f*sigma*this->orientationContribWidth,
    keyPoints->device, gradients->device, thetaNumbers_device, this->maxOrientations,
    this->orientationThreshold,thetas_device);
  cudaDeviceSynchronize();
  CudaCheckError();

  printf("compute thetas done in %f seconds.\n\n",((float) clock() -  timer)/CLOCKS_PER_SEC);
  timer = clock();

  thrust::device_ptr<int> tN(thetaNumbers_device);
  thrust::device_ptr<int> end = thrust::remove(tN, tN + keyPoints->numElements*this->maxOrientations, -1);
  int numFeatures = end - tN;

  thrust::device_ptr<float> t(thetas_device);
  thrust::device_ptr<float> new_end = thrust::remove(t, t + keyPoints->numElements*this->maxOrientations, -1.0f);

  printf("theta compaction done in %f seconds.\n\n",((float) clock() -  timer)/CLOCKS_PER_SEC);
  timer = clock();

  grid = {1,1,1};
  block = {4,4,8};
  getGrid(numFeatures,grid);

  Feature<SIFT_Descriptor>* features_device = nullptr;
  CudaSafeCall(cudaMalloc((void**)&features_device,numFeatures*sizeof(Feature<SIFT_Descriptor>)));
  fillDescriptors<<<grid,block>>>(numFeatures,imageSize.x,features_device,sigma,pixelWidth,
    this->descriptorContribWidth,1.25f*sqrtf(2.0f)*sigma*this->descriptorContribWidth,
    thetas_device,thetaNumbers_device,keyPoints->device,gradients->device);
  cudaDeviceSynchronize();
  CudaCheckError();

  CudaSafeCall(cudaFree(thetas_device));
  CudaSafeCall(cudaFree(thetaNumbers_device));

  printf("fill descriptors done in %f seconds.\n\n",((float) clock() -  timer)/CLOCKS_PER_SEC);

  return new Unity<Feature<SIFT_Descriptor>>(features_device,numFeatures,gpu);
}



/*
CUDA implementations
*/

__constant__ float ssrlcv::pi = 3.1415927;

/*
const long double PI = 3.141592653589793238L;
const double PI = 3.141592653589793;
const float PI = 3.1415927;
*/

/*
DEVICE METHODS
*/
__device__ __forceinline__ unsigned long ssrlcv::getGlobalIdx_2D_1D(){
  unsigned long blockId = blockIdx.y * gridDim.x + blockIdx.x;
  unsigned long threadId = blockId * blockDim.x + threadIdx.x;
  return threadId;
}
__device__ __forceinline__ float ssrlcv::getMagnitude(const int2 &vector){
  return sqrtf((float)dotProduct(vector, vector));
}
__device__ __forceinline__ float ssrlcv::getMagnitude(const float2 &vector){
  return sqrtf(dotProduct(vector, vector));
}
__device__ __forceinline__ float ssrlcv::getMagnitudeSq(const int2 &vector){
  return (float)dotProduct(vector, vector);
}
__device__ __forceinline__ float ssrlcv::getMagnitudeSq(const float2 &vector){
  return dotProduct(vector, vector);
}
__device__ __forceinline__ float ssrlcv::getTheta(const int2 &vector){
  return fmodf(atan2f((float)vector.y, (float)vector.x) + pi,2.0f*pi);
}
__device__ __forceinline__ float ssrlcv::getTheta(const float2 &vector){
  return fmodf(atan2f(vector.y, vector.x) + pi,2.0f*pi);
}
__device__ __forceinline__ float ssrlcv::getTheta(const float2 &vector, const float &offset){
  return fmodf((atan2f(vector.y, vector.x) + pi) - offset,2.0f*pi);
}
__device__ void ssrlcv::trickleSwap(const float2 &compareWValue, float2* arr, const int &index, const int &length){
  for(int i = index; i < length; ++i){
    if(compareWValue.x > arr[i].x){
      float2 temp = arr[i];
      arr[i] = compareWValue;
      if((temp.x == 0.0f && temp.y == 0.0f)|| index + 1 == length) return;
      return trickleSwap(temp, arr, index + 1, length);
    }
  }
}
__device__ __forceinline__ long4 ssrlcv::getOrientationContributers(const long2 &loc, const uint2 &imageSize){
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


//will make a hog of certain size based on sigma
__global__ void ssrlcv::computeThetas(const unsigned long numKeyPoints, const unsigned int imageWidth, const float sigma,
    const float pixelWidth, const float lambda, const float windowWidth, const float2* __restrict__ keyPointLocations,
    const int2* gradients, int* __restrict__ thetaNumbers, const unsigned int maxOrientations, const float orientationThreshold,
    float* __restrict__ thetas){

  unsigned long globalId = getGlobalIdx_2D_1D();

  if(globalId < numKeyPoints){
    float2 keyPoint = keyPointLocations[globalId];
    float orientationHist[36] = {0.0f};
    float maxHist = 0.0f;
    int regNumOrient = maxOrientations;
    float2 gradient = {0.0f,0.0f};
    float2 temp2 = {0.0f,0.0f};
    for(float y = (keyPoint.y - windowWidth)/pixelWidth; y <= (keyPoint.y + windowWidth)/pixelWidth; y+=1.0f){
      for(float x = (keyPoint.x - windowWidth)/pixelWidth; x <= (keyPoint.x + windowWidth)/pixelWidth; x+=1.0f){
        gradient = {
          (float)gradients[llroundf(y)*imageWidth + llroundf(x)].x,
          (float)gradients[llroundf(y)*imageWidth + llroundf(x)].y
        };
        temp2 = {x*pixelWidth - keyPoint.x,y*pixelWidth - keyPoint.y};
        orientationHist[llroundf(36.0f*getTheta(gradient)/(2.0f*pi))] += expf(-getMagnitude(temp2)/(2.0f*lambda*lambda*sigma*sigma))*getMagnitude(gradient);
      }
    }
    float3 convHelper = {orientationHist[35],orientationHist[0],orientationHist[1]};
    for(int i = 0; i < 6; ++i){
      temp2.x = orientationHist[0];//need to hold on to this for id = 35 conv
      for(int id = 1; id < 36; ++id){
        orientationHist[id] = (convHelper.x+convHelper.y+convHelper.z)/3.0f;
        convHelper.x = convHelper.y;
        convHelper.y = convHelper.z;
        convHelper.z = (id < 35) ? orientationHist[id+1] : temp2.x;
        if(i == 5){
          if(orientationHist[id] > maxHist){
            maxHist = orientationHist[id];
          }
        }
      }
    }
    maxHist *= orientationThreshold;//% of max orientation value

    float2* bestMagWThetas = new float2[regNumOrient]();
    float2 tempMagWTheta = {0.0f,0.0f};
    for(int b = 0; b < 36; ++b){
      if(orientationHist[b] < maxHist ||
        (b > 0 && orientationHist[b] < orientationHist[b-1]) ||
        (b < 35 && orientationHist[b] < orientationHist[b+1]) ||
        (orientationHist[b] < bestMagWThetas[regNumOrient-1].x)) continue;

      tempMagWTheta.x = orientationHist[b];

      if(b == 0){
        tempMagWTheta.y = (orientationHist[35]-orientationHist[1])/(orientationHist[35]-(2.0f*orientationHist[0])+orientationHist[1]);
      }
      else if(b == 35){
        tempMagWTheta.y = (orientationHist[34]-orientationHist[0])/(orientationHist[34]-(2.0f*orientationHist[35])+orientationHist[0]);
      }
      else{
        tempMagWTheta.y = (orientationHist[b-1]-orientationHist[b+1])/(orientationHist[b-1]-(2.0f*orientationHist[b])+orientationHist[b+1]);
      }

      tempMagWTheta.y *= (pi/36.0f);
      tempMagWTheta.y += (float)b*(pi/18.0f);
      if(tempMagWTheta.y < 0.0f){
        tempMagWTheta.y += 2.0f*pi;
      }

      for(int i = 0; i < regNumOrient; ++i){
        if(tempMagWTheta.x > bestMagWThetas[i].x){
          for(int ii = i; ii < regNumOrient; ++ii){
            temp2 = bestMagWThetas[ii];
            bestMagWThetas[ii] = tempMagWTheta;
            tempMagWTheta = temp2;
          }
        }
      }
    }
    for(int i = 0; i < regNumOrient; ++i){
      if(bestMagWThetas[i].x == 0.0f){
        thetaNumbers[globalId*regNumOrient + i] = -1;
        thetas[globalId*regNumOrient + i] = -1.0f;
      }
      else{
        thetaNumbers[globalId*regNumOrient + i] = globalId;
        thetas[globalId*regNumOrient + i] = bestMagWThetas[i].y;
      }
    }
    delete[] bestMagWThetas;
  }
}

__global__ void ssrlcv::fillDescriptors(const unsigned long numFeatures, const unsigned int imageWidth, Feature<SIFT_Descriptor>* features,
    const float sigma, const float pixelWidth, const float lambda, const float windowWidth, const float* __restrict__ thetas,
    const int* __restrict__ keyPointAddresses, const float2* __restrict__ keyPointLocations, const int2* __restrict__ gradients){

  unsigned long blockId = blockIdx.y* gridDim.x+ blockIdx.x;
  if(blockId < numFeatures){
    __shared__ float normSq;
    __shared__ float bin_descriptors[4][4][8];
    bin_descriptors[threadIdx.x][threadIdx.y][threadIdx.z] = 0.0f;
    float2 keyPoint = keyPointLocations[keyPointAddresses[blockId]];
    float theta = thetas[blockId];
    normSq = 0.0f;
    __syncthreads();

    /*
    FIRST DEFINE HOG GRID
    (x,y) = [(-8.5,-8.5),(8.5,8.5)]
      x' = xcos(theta) - ysin(theta) + feature.x
      y' = ycos(theta) + xsin(theta) + feature.y

    */

    float2 descriptorGridPoint = {0.0f,0.0f};

    float2 contribLoc = {0.0f,0.0f};
    float2 gradient = {0.0f,0.0f};
    float2 temp = {0.0f,0.0f};
    float2 histLoc = {0.0f,0.0f};
    bool histFound = false;
    for(float y = ((keyPoint.y-windowWidth)/pixelWidth) + (float)threadIdx.y; y <= (keyPoint.y+windowWidth)/pixelWidth; y+=(float)blockDim.y){
      if(threadIdx.z != 0) break;
      for(float x = ((keyPoint.x-windowWidth)/pixelWidth) + (float)threadIdx.x; x <= (keyPoint.x+windowWidth)/pixelWidth; x+=(float)blockDim.x){
        contribLoc.x = (((x*pixelWidth - keyPoint.x)*cosf(theta)) + ((y*pixelWidth - keyPoint.y)*sinf(theta)))/sigma;
        contribLoc.y = ((-(x*pixelWidth - keyPoint.x)*sinf(theta)) + ((y*pixelWidth - keyPoint.y)*cosf(theta)))/sigma;
        if(abs(contribLoc.x) > lambda*1.25f || abs(contribLoc.y) > lambda*1.25f) continue;
        gradient = {
          (float)gradients[llroundf(contribLoc.y + keyPoint.y)*imageWidth + llroundf(contribLoc.x + keyPoint.x)].x,
          (float)gradients[llroundf(contribLoc.y + keyPoint.y)*imageWidth + llroundf(contribLoc.x + keyPoint.x)].y
        };
        //calculate expow
        temp = {contribLoc.x*pixelWidth - keyPoint.x,contribLoc.y*pixelWidth - keyPoint.y};
        descriptorGridPoint.x = getMagnitude(gradient)*expf(-getMagnitude(temp)/(2.0f*lambda*lambda*sigma*sigma));
        descriptorGridPoint.y = fmodf((atan2f(gradient.y,gradient.x)+pi)-theta+(2.0f*pi),2.0f*pi);//getTheta(gradient,theta);
        histFound = false;
        for(float nx = 0; nx < 4.0f && !histFound; nx+=1.0f){
          for(float ny = 0; ny < 4.0f && !histFound; ny+=1.0f){
            histLoc = {(nx*0.5f - 0.75f)*lambda,(ny*0.5f - 0.75f)*lambda};
            if(abs(histLoc.x - contribLoc.x) <= lambda*0.5f && abs(histLoc.y - contribLoc.y) <= lambda*0.5f){
              histFound = true;
              for(float k = 0; k < 8.0f; k+=1.0f){
                if(abs(fmodf(descriptorGridPoint.y-(k*45.0f*(pi/180.0f)),2.0f*pi)) < 45.0f*(pi/180.0f)){
                  //TODO find solution to rounding here
                  atomicAdd(&bin_descriptors[(int)nx][(int)ny][(int)k],roundf((1.0f-(2.0f/lambda)*abs(histLoc.x - contribLoc.x))*
                    (1.0f-(2.0f/lambda)*abs(histLoc.y - contribLoc.y))*
                    (1.0f-(2.0f/pi)*abs(fmodf(descriptorGridPoint.y-(k*45.0f*(pi/180.0f)),2.0f*pi)))*descriptorGridPoint.x));
                }
              }
            }
          }
        }
      }
    }
    /*
    NORMALIZE
    */
    __syncthreads();
    atomicAdd(&normSq, bin_descriptors[threadIdx.x][threadIdx.y][threadIdx.z]*bin_descriptors[threadIdx.x][threadIdx.y][threadIdx.z]);
    __syncthreads();

    temp.x = bin_descriptors[threadIdx.x][threadIdx.y][threadIdx.z];
    temp.x = (temp.x*temp.x < 0.2f*0.2f*normSq) ? temp.x : 0.2f*sqrtf(normSq);
    temp.x = roundf(512.0f*temp.x/sqrtf(normSq));
    features[blockId].descriptor.values[(threadIdx.y*4 + threadIdx.x)*8 + threadIdx.z] = (temp.x < 255) ? (unsigned char) temp.x : 255;
    if(threadIdx.x == 0 && threadIdx.y == 0 && threadIdx.z == 0){
      features[blockId].descriptor.theta = theta;
      features[blockId].descriptor.sigma = sigma;
      features[blockId].loc = keyPoint;
    }
  }
}
