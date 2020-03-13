#include "SIFT_FeatureFactory.cuh"

ssrlcv::SIFT_FeatureFactory::SIFT_FeatureFactory(float orientationContribWidth, float descriptorContribWidth){
  this->orientationContribWidth = orientationContribWidth;
  this->descriptorContribWidth = descriptorContribWidth;
}

ssrlcv::Unity<ssrlcv::Feature<ssrlcv::SIFT_Descriptor>>* ssrlcv::SIFT_FeatureFactory::generateFeatures(ssrlcv::Image* image, bool dense, unsigned int maxOrientations, float orientationThreshold){
  std::cout<<"Generating SIFT features for image "<<image->id<<std::endl<<"\t";
  Unity<Feature<SIFT_Descriptor>>* features = nullptr;
  MemoryState origin = image->pixels->getMemoryState();
  if(origin != gpu) image->pixels->setMemoryState(gpu);

  if(image->colorDepth != 1){
    convertToBW(image->pixels,image->colorDepth);
    image->colorDepth = 1;
  }
  if(dense){

    clock_t timer = clock();
    Unity<float>* pixelsFLT = convertImageToFlt(image->pixels);
    if(origin != gpu) image->pixels->setMemoryState(origin);//no longer need to force pixels on gpu
    normalizeImage(pixelsFLT);
    Unity<float2>* gradients = generatePixelGradients(image->size, pixelsFLT);
    delete pixelsFLT;
    //12x12 border
    Unity<float2>* keyPoints = new Unity<float2>(nullptr,(image->size.x-24)*(image->size.y-24),cpu);
    for(int y = 0; y < image->size.y-24; ++y){
      for(int x = 0; x < image->size.x-24; ++x){
        keyPoints->host[y*(image->size.x-24) + x] = {(float)x + 12.0f, (float)y + 12.0f};
      }
    }

    //will free cpu memory and instantiate gpu memory
    keyPoints->setMemoryState(gpu);

    printf("\nDense SIFT prep done in %f seconds.\n\n",((float) clock() -  timer)/CLOCKS_PER_SEC);

    features = this->createFeatures(image->size,orientationThreshold,maxOrientations,1.0f,gradients,keyPoints);
    delete gradients;
    delete keyPoints;
  }
  else{
    uint2 scaleSpaceDim = {4,6};
    //int nspo = scaleSpaceDim.y - 3;//num scale space/octave in dog made from a {4,6} ScaleSpace
    float noiseThreshold = 0.015f;//*(powf(2,1.0f/nspo)-1)/(powf(2,1.0f/3.0f)-1);//if 0.15 there is a segfault
    float edgeThreshold = 12.1f;//12.1 = (10.0f + 1)^2 / 10.0f //formula = (r+1)^2/r from lowes paper where r = 10
    DOG* dog = new DOG(image,-1,scaleSpaceDim,sqrtf(2.0f)/2.0f,{2,sqrtf(2.0f)},{8,8},true);//last true specifies dog conversion
    std::cout<<"\tdog created"<<std::endl;
    if(origin != gpu) image->pixels->setMemoryState(origin);//no longer need to force pixels on gpu
    // std::string dump = "out/dog";
    // dog->dumpData(dump);
    dog->findKeyPoints(noiseThreshold,edgeThreshold,true); 

    ScaleSpace::Octave* currentOctave = nullptr;
    ScaleSpace::Octave::Blur* currentBlur = nullptr;
    int numFeaturesProduced = 0;
    MemoryState* extremaOrigin = new MemoryState[dog->depth.x];
    unsigned int numKeyPointsInBlur = 0;
    dim3 grid = {1,1,1};
    dim3 block = {1,1,1};

    for(int o = 0; o < dog->depth.x; ++o){
      currentOctave = dog->octaves[o];
      if(currentOctave->extrema == nullptr) continue;
      extremaOrigin[o] = currentOctave->extrema->getMemoryState(); 
      if(extremaOrigin[o] != gpu) currentOctave->extrema->setMemoryState(gpu);
      
      for(int b = 0; b < dog->depth.y; ++b){
        if(b + 1 == dog->depth.y){
          numKeyPointsInBlur = currentOctave->extrema->size() - currentOctave->extremaBlurIndices[b];
        }
        else{
          numKeyPointsInBlur = currentOctave->extremaBlurIndices[b+1] - currentOctave->extremaBlurIndices[b];
        }
        if(numKeyPointsInBlur == 0) continue;

        currentBlur = currentOctave->blurs[b];
        grid = {1,1,1};
        block = {1,1,1};
        getFlatGridBlock(numKeyPointsInBlur,grid,block,checkKeyPoints);

        checkKeyPoints<<<grid,block>>>(numKeyPointsInBlur,currentOctave->extremaBlurIndices[b],currentBlur->size, currentOctave->pixelWidth,
          this->descriptorContribWidth,currentOctave->extrema->device);
        cudaDeviceSynchronize();
        CudaCheckError();
      }
      currentOctave->discardExtrema();
    }
    //have not transfered any extrema back to extremaOrigin

    dog->computeKeyPointOrientations(orientationThreshold,maxOrientations,this->orientationContribWidth,true);

    //then create features from each of the keyPoints
    unsigned int numKeyPoints = 0;
    for(int o = 0; o < dog->depth.x; ++o){
      if(dog->octaves[o]->extrema == nullptr) continue;
      numKeyPoints += dog->octaves[o]->extrema->size();
    }
    if(numKeyPoints == 0){
      std::cerr<<"ERROR: something went wrong and there are 0 keypoints"<<std::endl;
      exit(0);
    }
    std::cout<<"total keypoints found = "<<numKeyPoints<<std::endl;
    std::cout<<"creating features from keypoints..."<<std::endl;
    features = new Unity<Feature<SIFT_Descriptor>>(nullptr,numKeyPoints,gpu);
    //fill descriptors based on SSKeyPoint information
    block = {4,4,8};

    MemoryState gradientsOrigin;

    for(int o = 0; o < dog->depth.x; ++o){
      currentOctave = dog->octaves[o];
      if(currentOctave->extrema == nullptr) continue;
      //extrema should already be on gpu from last loop
      for(int b = 0; b < dog->depth.y; ++b){
        if(b + 1 == dog->depth.y){
          numKeyPointsInBlur = currentOctave->extrema->size() - currentOctave->extremaBlurIndices[b];
        }
        else{
          numKeyPointsInBlur = currentOctave->extremaBlurIndices[b+1] - currentOctave->extremaBlurIndices[b];
        }
        if(numKeyPointsInBlur == 0) continue;
  
        currentBlur = currentOctave->blurs[b];
        gradientsOrigin = currentBlur->gradients->getMemoryState();
        if(gradientsOrigin != gpu) currentBlur->gradients->setMemoryState(gpu);
        grid = {1,1,1};
        getGrid(numKeyPointsInBlur,grid);
        fillDescriptors<<<grid,block>>>(numKeyPointsInBlur,currentBlur->size, 
          features->device + numFeaturesProduced, currentOctave->pixelWidth, this->descriptorContribWidth,
          currentOctave->extrema->device + currentOctave->extremaBlurIndices[b], currentBlur->gradients->device);
        cudaDeviceSynchronize();
        CudaCheckError();

        numFeaturesProduced += numKeyPointsInBlur;
        if(gradientsOrigin != gpu) currentBlur->gradients->setMemoryState(gradientsOrigin);
      }
      if(extremaOrigin[o] != gpu) currentOctave->extrema->setMemoryState(extremaOrigin[o]);
      std::cout<<"\tfeatures created from octave "<<o<<std::endl;
    }
    delete[] extremaOrigin;
    delete dog;
  }
  std::cout<<"\n\n";
  return features;
}

ssrlcv::Unity<ssrlcv::Feature<ssrlcv::SIFT_Descriptor>>* ssrlcv::SIFT_FeatureFactory::createFeatures(uint2 imageSize,float orientationThreshold, unsigned int maxOrientations, float pixelWidth, Unity<float2>* gradients, Unity<float2>* keyPoints){

  clock_t timer = clock();

  int* thetaNumbers_device = nullptr;
  CudaSafeCall(cudaMalloc((void**)&thetaNumbers_device,keyPoints->size()*maxOrientations*sizeof(int)));
  float* thetas_device = nullptr;
  CudaSafeCall(cudaMalloc((void**)&thetas_device,keyPoints->size()*maxOrientations*sizeof(float)));
  float* thetas_host = new float[keyPoints->size()*maxOrientations];
  for(int i = 0; i < keyPoints->size()*maxOrientations; ++i){
    thetas_host[i] = -1.0f;
  }
  CudaSafeCall(cudaMemcpy(thetas_device,thetas_host,keyPoints->size()*maxOrientations*sizeof(float),cudaMemcpyHostToDevice));

  dim3 grid = {1,1,1};
  dim3 block = {1,1,1};

  void (*fp)(const unsigned long, const unsigned int,
    const float, const float, const float, const float2*,
    const float2*, int*, const unsigned int, const float,
    float*) = &computeThetas;

  getFlatGridBlock(keyPoints->size(),grid,block,fp);

  computeThetas<<<grid,block>>>(keyPoints->size(),imageSize.x,pixelWidth,
    this->orientationContribWidth,ceil(3.0f*this->orientationContribWidth/pixelWidth),
    keyPoints->device, gradients->device, thetaNumbers_device, maxOrientations,
    orientationThreshold,thetas_device);
  cudaDeviceSynchronize();
  CudaCheckError();

  printf("compute thetas done in %f seconds.\n\n",((float) clock() -  timer)/CLOCKS_PER_SEC);
  timer = clock();

  thrust::device_ptr<int> tN(thetaNumbers_device);
  thrust::device_ptr<int> end = thrust::remove(tN, tN + keyPoints->size()*maxOrientations, -1);
  int numFeatures = end - tN;

  thrust::device_ptr<float> t(thetas_device);
  thrust::device_ptr<float> new_end = thrust::remove(t, t + keyPoints->size()*maxOrientations, -FLT_MAX);

  printf("theta compaction done in %f seconds.\n\n",((float) clock() -  timer)/CLOCKS_PER_SEC);
  timer = clock();

  grid = {1,1,1};
  block = {4,4,8};
  getGrid(numFeatures,grid);

  Feature<SIFT_Descriptor>* features_device = nullptr;
  CudaSafeCall(cudaMalloc((void**)&features_device,numFeatures*sizeof(Feature<SIFT_Descriptor>)));
  fillDescriptors<<<grid,block>>>(numFeatures,imageSize.x,features_device,pixelWidth,
    this->descriptorContribWidth,ceil(this->descriptorContribWidth/pixelWidth),
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
    const float2* gradients, int* __restrict__ thetaNumbers, const unsigned int maxOrientations, const float orientationThreshold,
    float* __restrict__ thetas){

  unsigned long globalID = (blockIdx.y* gridDim.x+ blockIdx.x)*blockDim.x + threadIdx.x;

  if(globalID < numKeyPoints){
    float2 keyPoint = keyPointLocations[globalID];
    float orientationHist[36] = {0.0f};
    float maxHist = 0.0f;
    int regNumOrient = maxOrientations;
    float2 gradient = {0.0f,0.0f};
    float2 temp2 = {0.0f,0.0f};
    float weight = 2*(windowWidth*pixelWidth/3.0f)*(windowWidth*pixelWidth/3.0f);
    float angle = 0.0f;
    float rad10 = pi/18.0f;
    for(float y = (keyPoint.y - windowWidth)/pixelWidth; y <= (keyPoint.y + windowWidth)/pixelWidth; y+=1.0f){
      for(float x = (keyPoint.x - windowWidth)/pixelWidth; x <= (keyPoint.x + windowWidth)/pixelWidth; x+=1.0f){
        gradient = gradients[llroundf(y)*imageWidth + llroundf(x)];//interpolation?
        temp2 = {x*pixelWidth - keyPoint.x,y*pixelWidth - keyPoint.y};
        angle = fmodf(atan2f(gradient.y,gradient.x) + (2.0f*pi),2.0f*pi);//atan2f returns from -pi tp pi
        orientationHist[(int)floor(angle/rad10)] += getMagnitude(gradient)*expf(-((temp2.x*temp2.x)+(temp2.y*temp2.y))/weight);//pi/weight;
      }
    }
    // float3 convHelper = {orientationHist[35],orientationHist[0],orientationHist[1]};
    // for(int i = 0; i < 6; ++i){
    //  temp2.x = orientationHist[0];//need to hold on to this for id = 35 conv
    //  for(int id = 1; id < 36; ++id){
    //    orientationHist[id] = (convHelper.x+convHelper.y+convHelper.z)/3.0f;
    //    convHelper.x = convHelper.y;
    //    convHelper.y = convHelper.z;
    //    convHelper.z = (id < 35) ? orientationHist[id+1] : temp2.x;
    //    if(i == 5){
    //      if(orientationHist[id] > maxHist){
    //        maxHist = orientationHist[id];
    //      }
    //    }
    //  }
    // }
    for(int i = 0; i < 36; ++i){
      if(orientationHist[i] > maxHist) maxHist = orientationHist[i];
    }
    maxHist *= orientationThreshold;//% of max orientation value
    float2* bestMagWThetas = new float2[regNumOrient]();
    float2 tempMagWTheta = {0.0f,0.0f};
    for(int b = 0; b < 36; ++b){
      if(orientationHist[b] < maxHist ||
        (b > 0 && orientationHist[b] < orientationHist[b-1]) ||
        (b < 35 && orientationHist[b] < orientationHist[b+1]) ||
        (b == 0 && orientationHist[b] < orientationHist[35]) || 
        (b == 35 && orientationHist[b] < orientationHist[0]) ||
        (orientationHist[b] < bestMagWThetas[regNumOrient-1].x)){
        continue;
      } 

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
      tempMagWTheta.y += (b*rad10);
      tempMagWTheta.y = fmodf(tempMagWTheta.y + (2.0f*pi),2.0f*pi);

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
        thetas[globalID*regNumOrient + i] = -FLT_MAX;
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
    const int* __restrict__ keyPointAddresses, const float2* __restrict__ keyPointLocations, const float2* __restrict__ gradients){

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
    float2 histLoc = {0.0f,0.0f};
    float temp = 0.0f;
    float binWidth = windowWidth/2.0f;
    float angle = 0.0f;

    float rad45 = pi/4.0f;
    for(float y = (float)threadIdx.y - windowWidth; y <= windowWidth; y+=(float)blockDim.y){
      if(threadIdx.z != 0) break;
      for(float x = (float)threadIdx.x - windowWidth; x <= windowWidth; x+=(float)blockDim.x){
        contribLoc = {(x*cosf(-theta)) + (y*sinf(-theta)),(-x*sinf(-theta)) + (y*cosf(-theta))};
        if(abs(contribLoc.x) > windowWidth || abs(contribLoc.y) > windowWidth) continue;
        //should interpolate to get proper gradient???
        gradient = gradients[llroundf(contribLoc.y + keyPoint.y)*imageWidth + llroundf(contribLoc.x + keyPoint.x)];
        descriptorGridPoint.x = getMagnitude(gradient)*expf(-((contribLoc.x*contribLoc.x)+(contribLoc.y*contribLoc.y))/(2.0f*windowWidth*windowWidth));///2.0f/pi/windowWidth/windowWidth; 
        descriptorGridPoint.y = fmodf(atan2f(gradient.y,gradient.x) - theta + (2.0f*pi),2.0f*pi);

        for(float nx = 0; nx < 4.0f; nx+=1.0f){
          for(float ny = 0; ny < 4.0f; ny+=1.0f){
            histLoc = {(nx*0.5f - 0.75f)*windowWidth,(ny*0.5f - 0.75f)*windowWidth};
            histLoc = {(histLoc.x*cosf(-theta)) + (histLoc.y*sinf(-theta)),(-histLoc.x*sinf(-theta)) + (histLoc.y*cosf(-theta))};
            histLoc = {abs(histLoc.x - contribLoc.x),abs(histLoc.y - contribLoc.y)};
            if(histLoc.x <= binWidth && histLoc.y <= binWidth){
              histLoc = histLoc/binWidth;
              for(int k = 0; k < 8; ++k){
                angle = abs(descriptorGridPoint.y-((float)k*rad45));
                if(angle < rad45){
                  angle /= rad45;
                  temp = (1.0f-histLoc.x)*(1.0f-histLoc.y)*(1.0f-angle)*descriptorGridPoint.x;
                  atomicAdd(&bin_descriptors[(int)nx][(int)ny][k],temp);
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
    features[blockId].descriptor.values[(threadIdx.y*4 + threadIdx.x)*8 + threadIdx.z] = (unsigned char) roundf(255.0f*bin_descriptors[threadIdx.x][threadIdx.y][threadIdx.z]/sqrtf(normSq));
    if(threadIdx.x == 0 && threadIdx.y == 0 && threadIdx.z == 0){
      features[blockId].descriptor.theta = theta;
      features[blockId].descriptor.sigma = 1.0f;
      features[blockId].loc = keyPoint*pixelWidth;//absolute location on image
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
    (kp.loc.x + windowWidth) >= imageSize.x - 1||
    (kp.loc.y + windowWidth) >= imageSize.y - 1){
      keyPoints[globalID + keyPointIndex].discard = true;
    }
  }
}


__global__ void ssrlcv::fillDescriptors(unsigned int numFeatures, uint2 imageSize, Feature<SIFT_Descriptor>* features,
float pixelWidth, float lambda, FeatureFactory::ScaleSpace::SSKeyPoint* keyPoints, float2* gradients){
  unsigned long blockId = blockIdx.y* gridDim.x+ blockIdx.x;
  if(blockId < numFeatures){
    __shared__ float norm;
    norm = 0.0f;
    __shared__ float bin_descriptors[4][4][8];
    bin_descriptors[threadIdx.x][threadIdx.y][threadIdx.z] = 0.0f;
    FeatureFactory::ScaleSpace::SSKeyPoint kp = keyPoints[blockId];
    float2 keyPoint = kp.loc;
    float windowWidth = ceil(kp.sigma*lambda/pixelWidth);
    float theta = kp.theta;
    __syncthreads();

    float2 descriptorGridPoint = {0.0f,0.0f};
    int imageWidth = imageSize.x;

    float2 contribLoc = {0.0f,0.0f};
    float2 gradient = {0.0f,0.0f};
    float2 histLoc = {0.0f,0.0f};
    float temp = 0.0f;
    float binWidth = windowWidth/2.0f;
    float angle = 0.0f;

    float rad45 = pi/4.0f;
    for(float y = (float)threadIdx.y - windowWidth; y <= windowWidth; y+=(float)blockDim.y){
      if(threadIdx.z != 0) break;
      for(float x = (float)threadIdx.x - windowWidth; x <= windowWidth; x+=(float)blockDim.x){
        contribLoc = {(x*cosf(-theta)) + (y*sinf(-theta)),(-x*sinf(-theta)) + (y*cosf(-theta))};
        if(abs(contribLoc.x) > windowWidth || abs(contribLoc.y) > windowWidth) continue;
        gradient = gradients[llroundf(contribLoc.y + keyPoint.y)*imageWidth + llroundf(contribLoc.x + keyPoint.x)];//this might need to be an interpolation
        descriptorGridPoint.x = getMagnitude(gradient)*expf(-((contribLoc.x*contribLoc.x)+(contribLoc.y*contribLoc.y))/(2.0f*windowWidth*windowWidth));//2.0f/pi/windowWidth/windowWidth; 
        descriptorGridPoint.y = fmodf(atan2f(gradient.y,gradient.x) - theta + (2.0f*pi),2.0f*pi);

        for(float nx = 0; nx < 4.0f; nx+=1.0f){
          for(float ny = 0; ny < 4.0f; ny+=1.0f){
            histLoc = {(nx*0.5f - 0.75f)*windowWidth,(ny*0.5f - 0.75f)*windowWidth};
            histLoc = {(histLoc.x*cosf(-theta)) + (histLoc.y*sinf(-theta)),(-histLoc.x*sinf(-theta)) + (histLoc.y*cosf(-theta))};
            histLoc = {abs(histLoc.x - contribLoc.x),abs(histLoc.y - contribLoc.y)};
            if(histLoc.x <= binWidth && histLoc.y <= binWidth){
              histLoc = histLoc/binWidth;
              for(int k = 0; k < 8; ++k){
                angle = abs(descriptorGridPoint.y-((float)k*rad45));
                if(angle < rad45){
                  angle /= rad45;
                  temp = (1.0f-histLoc.x)*(1.0f-histLoc.y)*(1.0f-angle)*descriptorGridPoint.x;
                  atomicAdd(&bin_descriptors[(int)nx][(int)ny][k],temp);
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
    atomicAdd(&norm, bin_descriptors[threadIdx.x][threadIdx.y][threadIdx.z]*bin_descriptors[threadIdx.x][threadIdx.y][threadIdx.z]);
    __syncthreads();
    bin_descriptors[threadIdx.x][threadIdx.y][threadIdx.z] /= sqrtf(norm);
    if(bin_descriptors[threadIdx.x][threadIdx.y][threadIdx.z] > 0.2f) bin_descriptors[threadIdx.x][threadIdx.y][threadIdx.z] = 0.2f;    
    __syncthreads();
    norm = 0.0f;    
    __syncthreads();
    atomicAdd(&norm,bin_descriptors[threadIdx.x][threadIdx.y][threadIdx.z]*bin_descriptors[threadIdx.x][threadIdx.y][threadIdx.z]);
    __syncthreads();
    features[blockId].descriptor.values[(threadIdx.y*4 + threadIdx.x)*8 + threadIdx.z] = (unsigned char) roundf(255.0f*bin_descriptors[threadIdx.x][threadIdx.y][threadIdx.z]/sqrtf(norm));
    if(threadIdx.x == 0 && threadIdx.y == 0 && threadIdx.z == 0){
      features[blockId].descriptor.theta = kp.theta;
      features[blockId].descriptor.sigma = kp.sigma;
      features[blockId].loc = kp.loc*pixelWidth;//absolute location on image
    }
  }
}
