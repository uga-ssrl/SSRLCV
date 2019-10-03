#include "SIFT_FeatureFactory.cuh"

ssrlcv::SIFT_FeatureFactory::SIFT_FeatureFactory(float orientationContribWidth, float descriptorContribWidth){
  this->orientationContribWidth = orientationContribWidth;
  this->descriptorContribWidth = descriptorContribWidth;
}

//binDepth currently not being used
ssrlcv::Unity<ssrlcv::Feature<ssrlcv::SIFT_Descriptor>>* ssrlcv::SIFT_FeatureFactory::generateFeatures(ssrlcv::Image* image, bool dense, unsigned int maxOrientations, float orientationThreshold){
  Unity<Feature<SIFT_Descriptor>>* features = nullptr;

  if(image->pixels->fore == cpu){
    image->pixels->transferMemoryTo(gpu);
  }
  if(image->colorDepth != 1){
    convertToBW(image->pixels,image->colorDepth);
    image->colorDepth = 1;
  }

  if(dense){
    dim3 grid = {1,1,1};
    dim3 block = {1,1,1};

    clock_t timer = clock();
    Unity<int2>* gradients = image->getPixelGradients();
    //12x12 border
    Unity<float2>* keyPoints = new Unity<float2>(nullptr,(image->size.x-24)*(image->size.y-24),cpu);

    for(int y = 0; y < image->size.y-24; ++y){
      for(int x = 0; x < image->size.x-24; ++x){
        keyPoints->host[y*(image->size.x-24) + x] = {x + 12.0f, y + 12.0f};
      }
    }
    //will free cpu memory and instantiate gpu memory
    keyPoints->setMemoryState(gpu);

    printf("\nDense SIFT prep done in %f seconds.\n\n",((float) clock() -  timer)/CLOCKS_PER_SEC);

    image->pixels->clear(gpu);//may want to nullify pixels for space

    features = this->createFeatures(image->size,orientationThreshold,maxOrientations,sqrtf(2),gradients,keyPoints);

    delete gradients;
    delete keyPoints;
  }
  else{
    uint2 scaleSpaceDim = {4,6};
    //int nspo = scaleSpaceDim.y - 3;//num scale space/octave in dog made from a {4,6} ScaleSpace
    float noiseThreshold = 0.015f;//*(powf(2,1.0f/nspo)-1)/(powf(2,1.0f/3.0f)-1);
    float edgeThreshold = 12.1f;//12.1 = (10.0f + 1)^2 / 10.0f //formula = (r+1)^2/r from lowes paper where r = 10

    DOG* dog = new DOG(image,-1,scaleSpaceDim,sqrtf(2.0f)/2.0f,{2,sqrtf(2.0f)},{8,8});
    //std::string dump = "out/scalespace";
    //dog->dumpData(dump);
    std::cout<<"converting to dog..."<<std::endl;
    dog->convertToDOG();
    //dump = "out/dog";
    //dog->dumpData(dump);
    std::cout<<"looking for keypoints..."<<std::endl;
    dog->findKeyPoints(noiseThreshold,edgeThreshold,true); 


    ScaleSpace::Octave* currentOctave = nullptr;
    ScaleSpace::Octave::Blur* currentBlur = nullptr;
    int numFeaturesProduced = 0;
    MemoryState origin[2];
    unsigned int numKeyPointsInBlur = 0;
    dim3 grid = {1,1,1};
    dim3 block = {1,1,1};

    std::cout<<"removing border keyPoints for SIFT_Descriptor Generation..."<<std::endl;

    //maybe replace with remove border
    for(int o = 0; o < dog->depth.x; ++o){
      currentOctave = dog->octaves[o];
      if(currentOctave->extrema == nullptr) continue;
      origin[0] = currentOctave->extrema->state;
      if(origin[0] == cpu || currentOctave->extrema->fore == cpu){
        currentOctave->extrema->setMemoryState(gpu);
      }
      for(int b = 0; b < dog->depth.y; ++b){
        if(b + 1 == dog->depth.y){
          numKeyPointsInBlur = currentOctave->extrema->numElements - currentOctave->extremaBlurIndices[b];
        }
        else{
          numKeyPointsInBlur = currentOctave->extremaBlurIndices[b+1] - currentOctave->extremaBlurIndices[b];
        }
        if(numKeyPointsInBlur == 0) continue;

        currentBlur = currentOctave->blurs[b];
        grid = {1,1,1};
        block = {1,1,1};
        getFlatGridBlock(numKeyPointsInBlur,grid,block);

        checkKeyPoints<<<grid,block>>>(numKeyPointsInBlur,currentOctave->extremaBlurIndices[b],currentBlur->size, currentOctave->pixelWidth,
          this->descriptorContribWidth,currentOctave->extrema->device);
        cudaDeviceSynchronize();
        CudaCheckError();
      }
      currentOctave->extrema->fore = gpu;//just to make sure
      currentOctave->discardExtrema();
      if(currentOctave->extrema != nullptr){
        if(origin[0] == cpu) currentOctave->extrema->setMemoryState(cpu);
      }
    }

    std::cout<<"computing keypoint orientations..."<<std::endl;

    //problem is here
    dog->computeKeyPointOrientations(orientationThreshold,maxOrientations,this->orientationContribWidth,true);

    //then create features from each of the keyPoints
    unsigned int numKeyPoints = 0;
    for(int o = 0; o < dog->depth.x; ++o){
      if(dog->octaves[o]->extrema == nullptr) continue;
      numKeyPoints += dog->octaves[o]->extrema->numElements;
    }
    features = new Unity<Feature<SIFT_Descriptor>>(nullptr,numKeyPoints,gpu);
    //fill descriptors based on SSKeyPoint information
    block = {4,4,8};


    std::cout<<"creating features from keypoints..."<<std::endl;
    for(int o = 0; o < dog->depth.x; ++o){
      currentOctave = dog->octaves[o];
      if(currentOctave->extrema == nullptr) continue;
      origin[0] = currentOctave->extrema->state;
      if(origin[0] == cpu || currentOctave->extrema->fore == cpu){
        currentOctave->extrema->setMemoryState(gpu);
      }
      for(int b = 0; b < dog->depth.y; ++b){
        if(b + 1 == dog->depth.y){
          numKeyPointsInBlur = currentOctave->extrema->numElements - currentOctave->extremaBlurIndices[b];
        }
        else{
          numKeyPointsInBlur = currentOctave->extremaBlurIndices[b+1] - currentOctave->extremaBlurIndices[b];
        }
        if(numKeyPointsInBlur == 0) continue;
        currentBlur = currentOctave->blurs[b];
        origin[1] = currentBlur->gradients->state;
        if(origin[1] == cpu || currentBlur->gradients->fore == cpu){
          currentBlur->gradients->setMemoryState(gpu);
        }
        grid = {1,1,1};
        getGrid(numKeyPointsInBlur,grid);
        fillDescriptors<<<grid,block>>>(numKeyPointsInBlur,currentBlur->size, 
          features->device + numFeaturesProduced, currentOctave->pixelWidth, this->descriptorContribWidth,
          currentOctave->extrema->device + currentOctave->extremaBlurIndices[b], currentBlur->gradients->device);
        cudaDeviceSynchronize();
        CudaCheckError();

        numFeaturesProduced += numKeyPointsInBlur;
        if(origin[1] == cpu) currentBlur->gradients->setMemoryState(cpu);
      }
      if(origin[0] == cpu) currentOctave->extrema->setMemoryState(cpu);
      std::cout<<"features created from octave "<<o<<std::endl;
    }
    delete dog;
  }


  return features;
}

//dense sift
ssrlcv::Unity<ssrlcv::Feature<ssrlcv::SIFT_Descriptor>>* ssrlcv::SIFT_FeatureFactory::createFeatures(uint2 imageSize,float orientationThreshold, unsigned int maxOrientations, float pixelWidth, Unity<int2>* gradients, Unity<float2>* keyPoints){

  clock_t timer = clock();

  int* thetaNumbers_device = nullptr;
  CudaSafeCall(cudaMalloc((void**)&thetaNumbers_device,keyPoints->numElements*maxOrientations*sizeof(int)));
  float* thetas_device = nullptr;
  CudaSafeCall(cudaMalloc((void**)&thetas_device,keyPoints->numElements*maxOrientations*sizeof(float)));
  float* thetas_host = new float[keyPoints->numElements*maxOrientations];
  for(int i = 0; i < keyPoints->numElements*maxOrientations; ++i){
    thetas_host[i] = -1.0f;
  }
  CudaSafeCall(cudaMemcpy(thetas_device,thetas_host,keyPoints->numElements*maxOrientations*sizeof(float),cudaMemcpyHostToDevice));

  dim3 grid = {1,1,1};
  dim3 block = {1,1,1};

  getFlatGridBlock(keyPoints->numElements,grid,block);

  computeThetas<<<grid,block>>>(keyPoints->numElements,imageSize.x,pixelWidth,
    this->orientationContribWidth,3.0f*this->orientationContribWidth,
    keyPoints->device, gradients->device, thetaNumbers_device, maxOrientations,
    orientationThreshold,thetas_device);
  cudaDeviceSynchronize();
  CudaCheckError();

  printf("compute thetas done in %f seconds.\n\n",((float) clock() -  timer)/CLOCKS_PER_SEC);
  timer = clock();

  thrust::device_ptr<int> tN(thetaNumbers_device);
  thrust::device_ptr<int> end = thrust::remove(tN, tN + keyPoints->numElements*maxOrientations, -1);
  int numFeatures = end - tN;

  thrust::device_ptr<float> t(thetas_device);
  thrust::device_ptr<float> new_end = thrust::remove(t, t + keyPoints->numElements*maxOrientations, -1.0f);

  printf("theta compaction done in %f seconds.\n\n",((float) clock() -  timer)/CLOCKS_PER_SEC);
  timer = clock();

  grid = {1,1,1};
  block = {4,4,8};
  getGrid(numFeatures,grid);

  Feature<SIFT_Descriptor>* features_device = nullptr;
  CudaSafeCall(cudaMalloc((void**)&features_device,numFeatures*sizeof(Feature<SIFT_Descriptor>)));
  fillDescriptors<<<grid,block>>>(numFeatures,imageSize.x,features_device,pixelWidth,
    this->descriptorContribWidth,1.25f*sqrtf(2.0f)*this->descriptorContribWidth,
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

/*
DEVICE METHODS
*/

//reimplemented as these are inline functions

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

__device__ __forceinline__ float ssrlcv::atomicMinFloat (float * addr, float value){
  float old;
  old = (value >= 0) ? __int_as_float(atomicMin((int *)addr, __float_as_int(value))) :
    __uint_as_float(atomicMax((unsigned int *)addr, __float_as_uint(value)));
  return old;
}
__device__ __forceinline__ float ssrlcv::atomicMaxFloat (float * addr, float value){
  float old;
  old = (value >= 0) ? __int_as_float(atomicMax((int *)addr, __float_as_int(value))) :
    __uint_as_float(atomicMin((unsigned int *)addr, __float_as_uint(value)));
  return old;
}
__device__ __forceinline__ float ssrlcv::edgeness(const float (&hessian)[2][2]){
    float e = trace(hessian);
    return e*e/determinant(hessian);    
}


/*
KERNELS
*/


__global__ void ssrlcv::computeThetas(const unsigned long numKeyPoints, const unsigned int imageWidth,
    const float pixelWidth, const float lambda, const float windowWidth, const float2* __restrict__ keyPointLocations,
    const int2* gradients, int* __restrict__ thetaNumbers, const unsigned int maxOrientations, const float orientationThreshold,
    float* __restrict__ thetas){

  unsigned long globalID = (blockIdx.y* gridDim.x+ blockIdx.x)*blockDim.x + threadIdx.x;

  if(globalID < numKeyPoints){
    float2 keyPoint = keyPointLocations[globalID];
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
        orientationHist[llroundf(36.0f*getTheta(gradient)/(2.0f*pi))] += expf(-getMagnitude(temp2)/(2.0f*lambda*lambda))*getMagnitude(gradient);
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
        thetaNumbers[globalID*regNumOrient + i] = -1;
        thetas[globalID*regNumOrient + i] = -1.0f;
      }
      else{
        thetaNumbers[globalID*regNumOrient + i] = globalID;
        thetas[globalID*regNumOrient + i] = bestMagWThetas[i].y;
      }
    }
    delete[] bestMagWThetas;
  }
}

__global__ void ssrlcv::fillDescriptors(const unsigned long numFeatures, const unsigned int imageWidth, Feature<SIFT_Descriptor>* features,
    const float pixelWidth, const float lambda, const float windowWidth, const float* __restrict__ thetas,
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
    for(float y = ((keyPoint.y-windowWidth)/1.00) + (float)threadIdx.y; y <= (keyPoint.y+windowWidth)/1.00; y+=(float)blockDim.y){
      if(threadIdx.z != 0) break;
      for(float x = ((keyPoint.x-windowWidth)/1.00) + (float)threadIdx.x; x <= (keyPoint.x+windowWidth)/1.00; x+=(float)blockDim.x){
        contribLoc.x = (((x*1.00 - keyPoint.x)*cosf(theta)) + ((y*1.00 - keyPoint.y)*sinf(theta)));
        contribLoc.y = ((-(x*1.00 - keyPoint.x)*sinf(theta)) + ((y*1.00 - keyPoint.y)*cosf(theta)));
        if(abs(contribLoc.x) > lambda*1.25f || abs(contribLoc.y) > lambda*1.25f) continue;
        gradient = {
          (float)gradients[llroundf(contribLoc.y + keyPoint.y)*imageWidth + llroundf(contribLoc.x + keyPoint.x)].x,
          (float)gradients[llroundf(contribLoc.y + keyPoint.y)*imageWidth + llroundf(contribLoc.x + keyPoint.x)].y
        };
        //calculate expow
        temp = {contribLoc.x*pixelWidth - keyPoint.x,contribLoc.y*pixelWidth - keyPoint.y};
        descriptorGridPoint.x = getMagnitude(gradient)*expf(-getMagnitude(temp)/(2.0f*lambda*lambda));
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
      features[blockId].descriptor.sigma = 1.0f;//dense
      features[blockId].loc = keyPoint;
    }
  }
}


__global__ void ssrlcv::checkKeyPoints(unsigned int numKeyPoints, unsigned int keyPointIndex, uint2 imageSize, float pixelWidth, float lambda, FeatureFactory::ScaleSpace::SSKeyPoint* keyPoints){
  unsigned int globalID = (blockIdx.y* gridDim.x+ blockIdx.x)*blockDim.x + threadIdx.x;
  if(globalID < numKeyPoints){
    FeatureFactory::ScaleSpace::SSKeyPoint kp = keyPoints[globalID + keyPointIndex];
    float windowWidth = kp.sigma*lambda/pixelWidth;
    if((kp.loc.x - windowWidth) < 0.0f || 
    (kp.loc.y - windowWidth) < 0.0f || 
    (kp.loc.x + windowWidth) >= imageSize.x ||
    (kp.loc.y + windowWidth) >= imageSize.y){
      keyPoints[globalID + keyPointIndex].discard = true;
    }
  }
}


__global__ void ssrlcv::fillDescriptors(unsigned int numFeatures, uint2 imageSize, Feature<SIFT_Descriptor>* features,
float pixelWidth, float lambda, FeatureFactory::ScaleSpace::SSKeyPoint* keyPoints, float2* gradients){
  unsigned long blockId = blockIdx.y* gridDim.x+ blockIdx.x;
  if(blockId < numFeatures){
    __shared__ float normSq;
    __shared__ float bin_descriptors[4][4][8];
    bin_descriptors[threadIdx.x][threadIdx.y][threadIdx.z] = 0.0f;
    FeatureFactory::ScaleSpace::SSKeyPoint kp = keyPoints[blockId];
    float2 keyPoint = kp.loc;
    float windowWidth = kp.sigma*lambda/pixelWidth;
    float theta = kp.theta;

    normSq = 0.0f;
    __syncthreads();

    float2 descriptorGridPoint = {0.0f,0.0f};
    int imageWidth = imageSize.x;

    float2 contribLoc = {0.0f,0.0f};
    float2 gradient = {0.0f,0.0f};
    float2 histLoc = {0.0f,0.0f};
    float temp = 0.0f;
    float binWidth = windowWidth/2.0f;
    float angle = 0.0f;

    float rad45 = 45.0f*(pi/180.0f);
    for(float y = (float)threadIdx.y - windowWidth; y <= windowWidth; y+=(float)blockDim.y){
      if(threadIdx.z != 0) break;
      for(float x = (float)threadIdx.x - windowWidth; x <= windowWidth; x+=(float)blockDim.x){
        contribLoc = {(x*cosf(theta)) + (y*sinf(theta)),(-x*sinf(theta)) + (y*cosf(theta))};
        if(abs(contribLoc.x) > windowWidth || abs(contribLoc.y) > windowWidth) continue;
        gradient = gradients[llroundf(contribLoc.y + keyPoint.y)*imageWidth + llroundf(contribLoc.x + keyPoint.x)];//this might need to be an interpolation
        //calculate expow
        descriptorGridPoint.x = getMagnitude(gradient)*expf(-getMagnitude(contribLoc)/(2.0f*windowWidth*windowWidth))/2.0f/pi/windowWidth/windowWidth; 
        descriptorGridPoint.y = getTheta(gradient,theta);

        for(float nx = 0; nx < 4.0f; nx+=1.0f){
          for(float ny = 0; ny < 4.0f; ny+=1.0f){
            histLoc = {(nx*0.5f - 0.75f)*lambda,(ny*0.5f - 0.75f)*lambda};
            histLoc = {abs(histLoc.x - contribLoc.x),abs(histLoc.y - contribLoc.y)};
            if(histLoc.x <= binWidth && histLoc.y <= binWidth){
              histLoc = histLoc/binWidth;
              for(float k = 0; k < 8.0*rad45; k+=rad45){
                angle = abs(fmodf(descriptorGridPoint.y-k,2.0f*pi));
                if(angle < rad45){
                  angle /= (rad45*0.5f);
                  //rounding to nearest int would help with rounding errors in atomicAdd
                  temp = (1.0f-histLoc.x)*(1.0f-histLoc.y)*(1.0f-angle)*descriptorGridPoint.x;
                  atomicAdd(&bin_descriptors[(int)nx][(int)ny][(int)k],temp);
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
    
    bin_descriptors[threadIdx.x][threadIdx.y][threadIdx.z] /= sqrtf(normSq);
    if(bin_descriptors[threadIdx.x][threadIdx.y][threadIdx.z] > 0.2f) bin_descriptors[threadIdx.x][threadIdx.y][threadIdx.z] = 0.2f;
    
    __syncthreads();
    normSq = 0.0f;
    __syncthreads();
    atomicAdd(&normSq, bin_descriptors[threadIdx.x][threadIdx.y][threadIdx.z]*bin_descriptors[threadIdx.x][threadIdx.y][threadIdx.z]);
    __syncthreads();
    
    bin_descriptors[threadIdx.x][threadIdx.y][threadIdx.z] /= sqrtf(normSq);
    features[blockId].descriptor.values[(threadIdx.y*4 + threadIdx.x)*8 + threadIdx.z] = bin_descriptors[threadIdx.x][threadIdx.y][threadIdx.z]*255;
    if(threadIdx.x == 0 && threadIdx.y == 0 && threadIdx.z == 0){
      features[blockId].descriptor.theta = kp.theta;
      features[blockId].descriptor.sigma = kp.sigma;
      features[blockId].loc = kp.loc*pixelWidth;//absolute location on image
    }
  }
}