#include "MatchFactory.cuh"

template class ssrlcv::MatchFactory<ssrlcv::SIFT_Descriptor>;


template<typename T>
ssrlcv::MatchFactory<T>::MatchFactory(){

}

template<typename T>
void ssrlcv::MatchFactory<T>::refineMatches(ssrlcv::Unity<ssrlcv::Match>* matches, float cutoffRatio){
  if(cutoffRatio == 0.0f){
    std::cout<<"ERROR illegal value used for cutoff ratio: 0.0"<<std::endl;
    exit(-1);
  }
  MemoryState origin = matches->state;
  if(origin != both){
    matches->transferMemoryTo(both);
  }

  float max = 0.0f;
  float min = FLT_MAX;
  for(int i = 0; i < matches->numElements; ++i){
    if(matches->host[i].distance < min) min = matches->host[i].distance;
    if(matches->host[i].distance > max) max = matches->host[i].distance;
  }
  if(origin == gpu) matches->clear(cpu);

  printf("max dist = %f || min dist = %f\n",max,min);
  int* matchCounter_device = nullptr;
  CudaSafeCall(cudaMalloc((void**)&matchCounter_device, matches->numElements*sizeof(int)));

  dim3 grid = {1,1,1};
  dim3 block = {1,1,1};
  getFlatGridBlock(matches->numElements, grid, block);
  refineWCutoffRatio<<<grid,block>>>(matches->numElements, matches->device, matchCounter_device, {min, max}, cutoffRatio);
  cudaDeviceSynchronize();
  CudaCheckError();

  thrust::device_ptr<int> sum(matchCounter_device);
  thrust::inclusive_scan(sum, sum + matches->numElements, sum);
  unsigned long afterCompaction = 0;
  CudaSafeCall(cudaMemcpy(&(afterCompaction),matchCounter_device + (matches->numElements - 1), sizeof(int), cudaMemcpyDeviceToHost));

  Match* minimizedMatches_device = nullptr;
  CudaSafeCall(cudaMalloc((void**)&minimizedMatches_device, afterCompaction*sizeof(Match)));

  copyMatches<<<grid,block>>>(matches->numElements, matchCounter_device, minimizedMatches_device, matches->device);
  cudaDeviceSynchronize();
  CudaCheckError();
  unsigned long beforeCompaction = matches->numElements;

  matches->setData(minimizedMatches_device, afterCompaction, gpu);

  printf("numMatches after eliminating base on %f cutoffRatio = %d (was %d)\n",cutoffRatio,matches->numElements,beforeCompaction);

  matches->transferMemoryTo(origin);
  if(origin == cpu){
    matches->clear(gpu);
  }
}

template<typename T>
void ssrlcv::MatchFactory<T>::refineMatches(ssrlcv::Unity<ssrlcv::FeatureMatch<T>>* matches, float cutoffRatio){
  if(cutoffRatio == 0.0f){
    std::cout<<"ERROR illegal value used for cutoff ratio: 0.0"<<std::endl;
    exit(-1);
  }
  MemoryState origin = matches->state;
  if(origin != both){
    matches->transferMemoryTo(both);
  }

  float max = 0.0f;
  float min = FLT_MAX;
  for(int i = 0; i < matches->numElements; ++i){
    if(matches->host[i].distance < min) min = matches->host[i].distance;
    if(matches->host[i].distance > max) max = matches->host[i].distance;
  }
  if(origin == gpu) matches->clear(cpu);

  printf("max dist = %f || min dist = %f\n",max,min);
  int* matchCounter_device = nullptr;
  CudaSafeCall(cudaMalloc((void**)&matchCounter_device, matches->numElements*sizeof(int)));

  dim3 grid = {1,1,1};
  dim3 block = {1,1,1};
  getFlatGridBlock(matches->numElements, grid, block);
  refineWCutoffRatio<T><<<grid,block>>>(matches->numElements, matches->device, matchCounter_device, {min, max}, cutoffRatio);
  cudaDeviceSynchronize();
  CudaCheckError();

  thrust::device_ptr<int> sum(matchCounter_device);
  thrust::inclusive_scan(sum, sum + matches->numElements, sum);
  unsigned long afterCompaction = 0;
  CudaSafeCall(cudaMemcpy(&(afterCompaction),matchCounter_device + (matches->numElements - 1), sizeof(int), cudaMemcpyDeviceToHost));

  FeatureMatch<T>* minimizedMatches_device = nullptr;
  CudaSafeCall(cudaMalloc((void**)&minimizedMatches_device, afterCompaction*sizeof(FeatureMatch<T>)));

  copyMatches<T><<<grid,block>>>(matches->numElements, matchCounter_device, minimizedMatches_device, matches->device);
  cudaDeviceSynchronize();
  CudaCheckError();
  unsigned long beforeCompaction = matches->numElements;

  matches->setData(minimizedMatches_device, afterCompaction, gpu);

  printf("numMatches after eliminating base on %f cutoffRatio = %d (was %d)\n",cutoffRatio,matches->numElements,beforeCompaction);

  matches->transferMemoryTo(origin);
  if(origin == cpu){
    matches->clear(gpu);
  }
}

template<typename T>
ssrlcv::Unity<ssrlcv::Match>* ssrlcv::MatchFactory<T>::sortMatches(Unity<Match>* allMatches){
  unsigned long len = allMatches->numElements;
  // insertion sort
  // each match element is accessed with allMatches->host[]
  unsigned long i = 0;
  unsigned long j = 0;
  ssrlcv::Match temp;
  while (i < len){
    j = i;
    while (j > 0 && allMatches->host[j-1].distance > allMatches->host[j].distance){
      temp = allMatches->host[j];
      allMatches->host[j] = allMatches->host[j-1];
      allMatches->host[j-1] = temp;
      j--;
    }
    i++;
  }
  return allMatches;
}

template<typename T>
ssrlcv::Unity<ssrlcv::FeatureMatch<T>>* ssrlcv::MatchFactory<T>::sortMatches(Unity<FeatureMatch<T>>* allMatches){
  unsigned long len = allMatches->numElements;
  // insertion sort
  // each match element is accessed with allMatches->host[]
  unsigned long i = 0;
  unsigned long j = 0;
  ssrlcv::FeatureMatch<T> temp;
  while (i < len){
    j = i;
    while (j > 0 && allMatches->host[j-1].distance > allMatches->host[j].distance){
      temp = allMatches->host[j];
      allMatches->host[j] = allMatches->host[j-1];
      allMatches->host[j-1] = temp;
      j--;
    }
    i++;
  }
  return allMatches;
}

template<typename T>
ssrlcv::Unity<ssrlcv::Match>* ssrlcv::MatchFactory<T>::getRawMatches(Unity<FeatureMatch<T>>* matches){
  Match* rawMatches = new Match[matches->numElements];
  MemoryState origin = matches->state;
  if(origin == gpu) matches->transferMemoryTo(cpu);
  for(int i = 0; i < matches->numElements; ++i){
    for(int f = 0; f < 2; ++f){
      rawMatches[i].parentIds[f] = matches->host[i].parentIds[f];
      rawMatches[i].locations[f] = matches->host[i].locations[f];
    }
    rawMatches[i].distance = matches->host[i].distance;
  }
  if(origin == gpu) matches->setMemoryState(gpu);
  return new Unity<Match>(rawMatches, matches->numElements, cpu);
}



template<typename T>
ssrlcv::Unity<ssrlcv::FeatureMatch<T>>* ssrlcv::MatchFactory<T>::generateMatchesBruteForce(ssrlcv::Image* query, ssrlcv::Unity<ssrlcv::Feature<T>>* queryFeatures,
ssrlcv::Image* target, ssrlcv::Unity<ssrlcv::Feature<T>>* targetFeatures){

  MemoryState origin[2] = {queryFeatures->state, targetFeatures->state};

  if(queryFeatures->fore == cpu) queryFeatures->transferMemoryTo(gpu);
  if(targetFeatures->fore == cpu) targetFeatures->transferMemoryTo(gpu);

  unsigned int numPossibleMatches = queryFeatures->numElements;

  FeatureMatch<T>* matches_device = nullptr;
  CudaSafeCall(cudaMalloc((void**)&matches_device, numPossibleMatches*sizeof(FeatureMatch<T>)));

  Unity<FeatureMatch<T>>* matches = new Unity<FeatureMatch<T>>(matches_device, numPossibleMatches, gpu);

  dim3 grid = {1,1,1};
  dim3 block = {1024,1,1};
  getGrid(matches->numElements,grid);

  clock_t timer = clock();

  matchFeaturesBruteForce<T><<<grid, block>>>(query->id, queryFeatures->numElements, queryFeatures->device,
    target->id, targetFeatures->numElements, targetFeatures->device, matches->device);

  cudaDeviceSynchronize();
  CudaCheckError();

  printf("done in %f seconds.\n\n",((float) clock() -  timer)/CLOCKS_PER_SEC);

  matches->transferMemoryTo(cpu);
  matches->clear(gpu);

  if(origin[0] != queryFeatures->state){
    queryFeatures->setMemoryState(origin[0]);
  }
  if(origin[1] != targetFeatures->state){
    targetFeatures->setMemoryState(origin[1]);
  }

  return matches;
}

template<typename T>
ssrlcv::Unity<ssrlcv::FeatureMatch<T>>* ssrlcv::MatchFactory<T>::generateMatchesConstrained(ssrlcv::Image* query, ssrlcv::Unity<ssrlcv::Feature<T>>* queryFeatures,
ssrlcv::Image* target, ssrlcv::Unity<ssrlcv::Feature<T>>* targetFeatures, float epsilon){

  MemoryState origin[2] = {queryFeatures->state, targetFeatures->state};

  if(queryFeatures->fore == cpu) queryFeatures->transferMemoryTo(gpu);
  if(targetFeatures->fore == cpu) targetFeatures->transferMemoryTo(gpu);

  unsigned int numPossibleMatches = queryFeatures->numElements;

  FeatureMatch<T>* matches_device = nullptr;
  CudaSafeCall(cudaMalloc((void**)&matches_device, numPossibleMatches*sizeof(FeatureMatch<T>)));

  Unity<FeatureMatch<T>>* matches = new Unity<FeatureMatch<T>>(matches_device, numPossibleMatches, gpu);

  dim3 grid = {1,1,1};
  dim3 block = {1024,1,1};
  getGrid(matches->numElements,grid);

  clock_t timer = clock();
  float3* fundamental = new float3[3];
  calcFundamentalMatrix_2View(query, target, fundamental);

  float3* fundamental_device;
  CudaSafeCall(cudaMalloc((void**)&fundamental_device, 3*sizeof(float3)));
  CudaSafeCall(cudaMemcpy(fundamental_device, fundamental, 3*sizeof(float3), cudaMemcpyHostToDevice));

  matchFeaturesConstrained<T><<<grid, block>>>(query->id, queryFeatures->numElements, queryFeatures->device,
    target->id, targetFeatures->numElements, targetFeatures->device, matches->device, epsilon, fundamental_device);
  cudaDeviceSynchronize();
  CudaCheckError();

  CudaSafeCall(cudaFree(fundamental_device));

  printf("done in %f seconds.\n\n",((float) clock() -  timer)/CLOCKS_PER_SEC);

  matches->transferMemoryTo(cpu);
  matches->clear(gpu);

  queryFeatures->transferMemoryTo(origin[0]);
  if(origin[0] == cpu){
    queryFeatures->clear(gpu);
  }
  targetFeatures->transferMemoryTo(origin[1]);
  if(origin[1] == cpu){
    targetFeatures->clear(gpu);
  }

  return matches;

}

template<typename T>
ssrlcv::Unity<ssrlcv::FeatureMatch<T>>* ssrlcv::MatchFactory<T>::generateSubPixelMatchesBruteForce(ssrlcv::Image* query, ssrlcv::Unity<ssrlcv::Feature<T>>* queryFeatures,
ssrlcv::Image* target, ssrlcv::Unity<ssrlcv::Feature<T>>* targetFeatures){

  MemoryState origin[2] = {queryFeatures->state, targetFeatures->state};

  if(queryFeatures->fore == cpu) queryFeatures->transferMemoryTo(gpu);
  if(targetFeatures->fore == cpu) targetFeatures->transferMemoryTo(gpu);

  Unity<FeatureMatch<T>>* matches = this->generateMatchesBruteForce(query, queryFeatures, target, targetFeatures);
  matches->transferMemoryTo(gpu);

  SubpixelM7x7* subDescriptors_device;
  CudaSafeCall(cudaMalloc((void**)&subDescriptors_device, matches->numElements*sizeof(SubpixelM7x7)));

  dim3 grid = {1,1,1};
  dim3 block = {9,9,1};
  getGrid(matches->numElements, grid);
  std::cout<<"initializing subPixelMatches..."<<std::endl;
  clock_t timer = clock();
  initializeSubPixels<T><<<grid, block>>>(matches->numElements, matches->device, subDescriptors_device,
    query->size, queryFeatures->numElements, queryFeatures->device,
    target->size, targetFeatures->numElements, targetFeatures->device);

  cudaDeviceSynchronize();
  CudaCheckError();
  printf("done in %f seconds.\n\n",((float) clock() -  timer)/CLOCKS_PER_SEC);

  Spline* splines_device;
  CudaSafeCall(cudaMalloc((void**)&splines_device, matches->numElements*2*sizeof(Spline)));

  grid = {1,1,1};
  block = {6,6,4};
  getGrid(matches->numElements*2, grid);

  std::cout<<"filling bicubic splines..."<<std::endl;
  timer = clock();
  fillSplines<<<grid,block>>>(matches->numElements, subDescriptors_device, splines_device);
  cudaDeviceSynchronize();
  CudaCheckError();
  printf("done in %f seconds.\n\n",((float) clock() -  timer)/CLOCKS_PER_SEC);
  CudaSafeCall(cudaFree(subDescriptors_device));

  std::cout<<"determining subpixel locations..."<<std::endl;
  timer = clock();
  determineSubPixelLocationsBruteForce<T><<<grid,block>>>(0.1, matches->numElements, matches->device, splines_device);
  cudaDeviceSynchronize();
  CudaCheckError();
  printf("done in %f seconds.\n\n",((float) clock() -  timer)/CLOCKS_PER_SEC);
  CudaSafeCall(cudaFree(splines_device));

  matches->transferMemoryTo(cpu);
  matches->clear(gpu);

  queryFeatures->transferMemoryTo(origin[0]);
  if(origin[0] == cpu){
    queryFeatures->clear(gpu);
  }
  targetFeatures->transferMemoryTo(origin[1]);
  if(origin[1] == cpu){
    targetFeatures->clear(gpu);
  }

  return matches;
}

template<typename T>
ssrlcv::Unity<ssrlcv::FeatureMatch<T>>* ssrlcv::MatchFactory<T>::generateSubPixelMatchesConstrained(ssrlcv::Image* query, ssrlcv::Unity<ssrlcv::Feature<T>>* queryFeatures,
ssrlcv::Image* target, ssrlcv::Unity<ssrlcv::Feature<T>>* targetFeatures, float epsilon){
  MemoryState origin[2] = {queryFeatures->state, targetFeatures->state};

  if(queryFeatures->fore == cpu) queryFeatures->transferMemoryTo(gpu);
  if(targetFeatures->fore == cpu) targetFeatures->transferMemoryTo(gpu);

  Unity<FeatureMatch<T>>* matches = this->generateMatchesConstrained(query, queryFeatures, target, targetFeatures, epsilon);
  matches->transferMemoryTo(gpu);

  SubpixelM7x7* subDescriptors_device;
  CudaSafeCall(cudaMalloc((void**)&subDescriptors_device, matches->numElements*sizeof(SubpixelM7x7)));

  dim3 grid = {1,1,1};
  dim3 block = {9,9,1};
  getGrid(matches->numElements, grid);
  std::cout<<"initializing subPixelMatches..."<<std::endl;
  clock_t timer = clock();
  initializeSubPixels<T><<<grid, block>>>(matches->numElements, matches->device, subDescriptors_device,
    query->size, queryFeatures->numElements, queryFeatures->device,
    target->size, targetFeatures->numElements, targetFeatures->device);

  cudaDeviceSynchronize();
  CudaCheckError();
  printf("done in %f seconds.\n\n",((float) clock() -  timer)/CLOCKS_PER_SEC);

  Spline* splines_device;
  CudaSafeCall(cudaMalloc((void**)&splines_device, matches->numElements*2*sizeof(Spline)));

  grid = {1,1,1};
  block = {6,6,4};
  getGrid(matches->numElements*2, grid);

  std::cout<<"filling bicubic splines..."<<std::endl;
  timer = clock();
  fillSplines<<<grid,block>>>(matches->numElements, subDescriptors_device, splines_device);
  cudaDeviceSynchronize();
  CudaCheckError();
  printf("done in %f seconds.\n\n",((float) clock() -  timer)/CLOCKS_PER_SEC);
  CudaSafeCall(cudaFree(subDescriptors_device));

  std::cout<<"determining subpixel locations..."<<std::endl;
  timer = clock();
  determineSubPixelLocationsBruteForce<T><<<grid,block>>>(0.1, matches->numElements, matches->device, splines_device);
  cudaDeviceSynchronize();
  CudaCheckError();
  printf("done in %f seconds.\n\n",((float) clock() -  timer)/CLOCKS_PER_SEC);
  CudaSafeCall(cudaFree(splines_device));

  matches->transferMemoryTo(cpu);
  matches->clear(gpu);

  queryFeatures->transferMemoryTo(origin[0]);
  if(origin[0] == cpu){
    queryFeatures->clear(gpu);
  }
  targetFeatures->transferMemoryTo(origin[1]);
  if(origin[1] == cpu){
    targetFeatures->clear(gpu);
  }

  return matches;
}

/*
CUDA implementations
*/

__constant__ int ssrlcv::splineHelper[4][4] = {
  {1,0,0,0},
  {0,0,1,0},
  {-3,3,-2,-1},
  {2,-2,1,1}
};
__constant__ int ssrlcv::splineHelperInv[4][4] = {
  {1,0,-3,2},
  {0,0,3,-2},
  {0,1,-2,1},
  {0,0,-1,1}
};

__device__ __host__ __forceinline__ float ssrlcv::sum(const float3 &a){
  return a.x + a.y + a.z;
}
__device__ __forceinline__ float ssrlcv::square(const float &a){
  return a*a;
}
__device__ __forceinline__ float ssrlcv::atomicMinFloat (float * addr, float value) {
  float old;
  old = (value >= 0) ? __int_as_float(atomicMin((int *)addr, __float_as_int(value))) :
    __uint_as_float(atomicMax((unsigned int *)addr, __float_as_uint(value)));
  return old;
}
__device__ __forceinline__ float ssrlcv::findSubPixelContributer(const float2 &loc, const int &width){
  return ((loc.y - 12)*(width - 24)) + (loc.x - 12);
}

__device__ __forceinline__ float ssrlcv::calcElucidSq(const Feature<SIFT_Descriptor>& a, const Feature<SIFT_Descriptor>& b){
  float dist = 0.0f;
  for(int i = 0; i < 128; ++i){
    dist += square(((float)a.descriptor.values[i]-b.descriptor.values[i]));
  }
  dist += a.descriptor.theta - b.descriptor.theta;
  dist += a.descriptor.sigma - b.descriptor.sigma;
  dist += dotProduct(a.loc - b.loc,a.loc - b.loc);
  return dist;
}
__device__ __forceinline__ float ssrlcv::calcElucidSq(const Feature<SIFT_Descriptor>& a, const Feature<SIFT_Descriptor>& b, const float &bestMatch){
  float dist = 0.0f;
  for(int i = 0; i < 128 && dist < bestMatch; ++i){
    dist += square(((float)a.descriptor.values[i]-b.descriptor.values[i]));
  }
  if(dist < bestMatch) dist += a.descriptor.theta - b.descriptor.theta;
  else return dist;
  if(dist < bestMatch) dist += a.descriptor.sigma - b.descriptor.sigma;
  else return dist;
  if(dist < bestMatch) dist += dotProduct(a.loc - b.loc,a.loc - b.loc);
  return dist;
}


/*
matching
*/
template<typename T>
__global__ void ssrlcv::matchFeaturesBruteForce(unsigned int queryImageID, unsigned long numFeaturesQuery,
ssrlcv::Feature<T>* featuresQuery, unsigned int targetImageID, unsigned long numFeaturesTarget,
ssrlcv::Feature<T>* featuresTarget, ssrlcv::FeatureMatch<T>* matches){
  unsigned long blockId = blockIdx.y * gridDim.x + blockIdx.x;
  if(blockId < numFeaturesQuery){
    Feature<T> feature = featuresQuery[blockId];
    __shared__ int localMatch[1024];
    __shared__ float localDist[1024];
    localMatch[threadIdx.x] = -1;
    localDist[threadIdx.x] = FLT_MAX;
    __syncthreads();
    float currentDist = 0.0f;
    unsigned long numFeaturesTarget_register = numFeaturesQuery;
    for(int f = threadIdx.x; f < numFeaturesTarget_register; f += 1024){
      currentDist = calcElucidSq(feature,featuresTarget[f],localDist[threadIdx.x]);
      if(localDist[threadIdx.x] > currentDist){
        localDist[threadIdx.x] = currentDist;
        localMatch[threadIdx.x] = f;
      }
    }
    __syncthreads();
    if(threadIdx.x != 0) return;
    currentDist = FLT_MAX;
    int matchIndex = -1;
    for(int i = 0; i < 1024; ++i){
      if(currentDist > localDist[i]){
        currentDist = localDist[i];
        matchIndex = localMatch[i];
      }
    }
    FeatureMatch<T> match;
    match.descriptors[0] = feature.descriptor;
    match.descriptors[1] = featuresTarget[matchIndex].descriptor;
    match.locations[0] = feature.loc;
    match.locations[1] = featuresTarget[matchIndex].loc;
    match.parentIds[0] = queryImageID;
    match.parentIds[1] = targetImageID;
    match.distance = currentDist;
    matches[blockId] = match;
  }
}

template<typename T>
__global__ void ssrlcv::matchFeaturesConstrained(unsigned int queryImageID, unsigned long numFeaturesQuery,
ssrlcv::Feature<T>* featuresQuery, unsigned int targetImageID, unsigned long numFeaturesTarget,
ssrlcv::Feature<T>* featuresTarget, ssrlcv::FeatureMatch<T>* matches, float epsilon, float3 fundamental[3]){
  unsigned long blockId = blockIdx.y * gridDim.x + blockIdx.x;
  if(blockId < numFeaturesQuery){
    Feature<T> feature = featuresQuery[blockId];
    __shared__ int localMatch[1024];
    __shared__ float localDist[1024];
    localMatch[threadIdx.x] = -1;
    localDist[threadIdx.x] = FLT_MAX;
    __syncthreads();
    float currentDist = 0.0f;
    unsigned long numFeaturesTarget_register = numFeaturesQuery;

    float3 epipolar = {0.0f,0.0f,0.0f};
    epipolar.x = (fundamental[0].x*feature.loc.x) + (fundamental[0].y*feature.loc.y) + fundamental[0].z;
    epipolar.y = (fundamental[1].x*feature.loc.x) + (fundamental[1].y*feature.loc.y) + fundamental[1].z;
    epipolar.z = (fundamental[2].x*feature.loc.x) + (fundamental[2].y*feature.loc.y) + fundamental[2].z;

    float p = 0.0f;

    Feature<T> currentFeature;
    float regEpsilon = epsilon;

    for(int f = threadIdx.x; f < numFeaturesTarget_register; f += 1024){

      currentFeature = featuresTarget[f];
      //ax + by + c = 0
      p = -1*((epipolar.x*currentFeature.loc.x) + epipolar.z)/epipolar.y;
      if(abs(currentFeature.loc.y - p) >= regEpsilon) continue;
      currentDist = calcElucidSq(feature,currentFeature,localDist[threadIdx.x]);
      if(localDist[threadIdx.x] > currentDist){
        localDist[threadIdx.x] = currentDist;
        localMatch[threadIdx.x] = f;
      }
    }
    __syncthreads();
    if(threadIdx.x != 0) return;
    currentDist = FLT_MAX;
    int matchIndex = -1;
    for(int i = 0; i < 1024; ++i){
      if(currentDist > localDist[i]){
        currentDist = localDist[i];
        matchIndex = localMatch[i];
      }
    }
    FeatureMatch<T> match;
    match.descriptors[0] = feature.descriptor;
    match.descriptors[1] = featuresTarget[matchIndex].descriptor;
    match.locations[0] = feature.loc;
    match.locations[1] = featuresTarget[matchIndex].loc;
    match.parentIds[0] = queryImageID;
    match.parentIds[1] = targetImageID;
    match.distance = currentDist;
    matches[blockId] = match;
  }
}


/*
subpixel stuff
*/
//TODO overload this kernel for different types of descriptors

//NOTE THIS MIGHT ONLY WORK FOR DENSE SIFT
template<typename T>
__global__ void ssrlcv::initializeSubPixels(unsigned long numMatches, ssrlcv::FeatureMatch<T>* matches, ssrlcv::SubpixelM7x7* subPixelDescriptors,
uint2 querySize, unsigned long numFeaturesQuery, ssrlcv::Feature<T>* featuresQuery,
uint2 targetSize, unsigned long numFeaturesTarget, ssrlcv::Feature<T>* featuresTarget){
  unsigned long blockId = blockIdx.y * gridDim.x + blockIdx.x;
  if(blockId < numMatches){
    __shared__ SubpixelM7x7 subDescriptor;
    FeatureMatch<T> match = matches[blockId];

    //this now needs to be actual indices to contributers
    int2 contrib = {((int)threadIdx.x) - 4, ((int)threadIdx.y) - 4};
    int contribQuery = findSubPixelContributer(match.locations[0] + contrib, querySize.x);
    int contribTarget = findSubPixelContributer(match.locations[1] + contrib, targetSize.x);

    int pairedMatchIndex = findSubPixelContributer(match.locations[1], targetSize.x);

    bool foundM1 = false;
    bool foundM2 = false;

    if(contribTarget >= 0 && contribTarget < numFeaturesTarget){
      subDescriptor.M1[threadIdx.x][threadIdx.y] = calcElucidSq(featuresQuery[blockId], featuresTarget[contribTarget]);
      foundM1 = true;
    }
    if(contribQuery >= 0 && contribQuery < numFeaturesQuery){
      subDescriptor.M2[threadIdx.x][threadIdx.y] = calcElucidSq(featuresQuery[contribQuery], featuresTarget[pairedMatchIndex]);
      foundM2 = true;
    }
    __syncthreads();
    //COME up with better way to do this
    if(!foundM1){
      float val = 0.0f;
      for(int x = 0; x < 9; ++x){
        for(int y = 0; y < 9; ++y){
          val += subDescriptor.M1[x][y];
        }
      }
      subDescriptor.M1[threadIdx.x][threadIdx.y] = val/81;
    }
    if(!foundM2){
      float val = 0.0f;
      for(int x = 0; x < 9; ++x){
        for(int y = 0; y < 9; ++y){
          val += subDescriptor.M2[x][y];
        }
      }
      subDescriptor.M2[threadIdx.x][threadIdx.y] = val/81;
    }
    __syncthreads();
    if(threadIdx.x == 0 && threadIdx.y == 0){
      subPixelDescriptors[blockId] = subDescriptor;
    }
  }
}

__global__ void ssrlcv::fillSplines(unsigned long numMatches, SubpixelM7x7* subPixelDescriptors, ssrlcv::Spline* splines){
  unsigned long blockId = blockIdx.y * gridDim.x + blockIdx.x;
  if(blockId < numMatches*2){
    float descriptor[9][9];
    for(int x = 0; x < 9; ++x){
      for(int y = 0; y < 9; ++y){
        descriptor[x][y] = (blockId%2 == 0) ? subPixelDescriptors[blockId/2].M1[x][y] : subPixelDescriptors[blockId/2].M2[x][y];
      }
    }

    __shared__ Spline spline;
    int2 corner = {
      ((int)threadIdx.z)%2,
      ((int)threadIdx.z)/2
    };
    int2 contributer = {
      ((int)threadIdx.x) + 2 + corner.x,
      ((int)threadIdx.y) + 2 + corner.y
    };
    float4 localCoeff;
    localCoeff.x = descriptor[contributer.x][contributer.y];
    localCoeff.y = descriptor[contributer.x + 1][contributer.y] - descriptor[contributer.x - 1][contributer.y];
    localCoeff.z = descriptor[contributer.x][contributer.y + 1] - descriptor[contributer.x][contributer.y - 1];
    localCoeff.w = descriptor[contributer.x + 1][contributer.y + 1] - descriptor[contributer.x - 1][contributer.y - 1];

    spline.coeff[threadIdx.x][threadIdx.y][corner.x][corner.y] = localCoeff.x;
    spline.coeff[threadIdx.x][threadIdx.y][corner.x][corner.y + 2] = localCoeff.y;
    spline.coeff[threadIdx.x][threadIdx.y][corner.x + 2][corner.y] = localCoeff.z;
    spline.coeff[threadIdx.x][threadIdx.y][corner.x + 2][corner.y + 2] = localCoeff.z;

    // Multiplying matrix a and b and storing in array mult.
    if(threadIdx.z != 0) return;
    float mult[4][4] = {0.0f};
    for(int i = 0; i < 4; ++i){
      for(int j = 0; j < 4; ++j){
        for(int c = 0; c < 4; ++c){
          mult[i][j] += splineHelper[i][c]*spline.coeff[threadIdx.x][threadIdx.y][c][j];
        }
      }
    }
    for(int i = 0; i < 4; ++i){
      for(int j = 0; j < 4; ++j){
        spline.coeff[threadIdx.x][threadIdx.y][i][j] = 0.0f;
      }
    }
    for(int i = 0; i < 4; ++i){
      for(int j = 0; j < 4; ++j){
        for(int c = 0; c < 4; ++c){
          spline.coeff[threadIdx.x][threadIdx.y][i][j] += mult[i][c]*splineHelperInv[c][j];
        }
      }
    }

    __syncthreads();
    splines[blockId] = spline;
  }
}

//NOTE THIS UPDATES FEATURE.LOC in MATCH
template<typename T>
__global__ void ssrlcv::determineSubPixelLocationsBruteForce(float increment, unsigned long numMatches, ssrlcv::FeatureMatch<T>* matches, ssrlcv::Spline* splines){
  unsigned long blockId = blockIdx.y * gridDim.x + blockIdx.x;
  if(blockId < numMatches*2){
    __shared__ float minimum;
    minimum = FLT_MAX;
    __syncthreads();
    float localCoeff[4][4];
    for(int i = 0; i < 4; ++i){
      for(int j = 0; j < 4; ++j){
        localCoeff[i][j] = splines[blockId].coeff[threadIdx.x][threadIdx.y][i][j];
      }
    }
    float value = 0.0f;
    float localMin = FLT_MAX;
    float2 localSubLoc = {0.0f,0.0f};
    for(float x = -1.0f; x <= 1.0f; x+=increment){
      for(float y = -1.0f; y <= 1.0f; y+=increment){
        value = 0.0f;
        for(int i = 0; i < 4; ++i){
          for(int j = 0; j < 4; ++j){
            value += (localCoeff[i][j]*powf(x,i)*powf(y,j));
          }
        }
        if(value < localMin){
          localMin = value;
          localSubLoc = {x,y};
        }
      }
    }
    atomicMinFloat(&minimum, localMin);
    __syncthreads();
    if(localMin == minimum){
      if(blockId%2 == 0) matches[blockId/2].locations[0]  = localSubLoc + matches[blockId/2].locations[0];
      else matches[blockId/2].locations[1] = localSubLoc + matches[blockId/2].locations[1];
    }
    else return;
  }
}

/*
MATCH REFINEMENT
*/

//NOTE may be able to replace this with thrust stream compaction
__global__ void ssrlcv::refineWCutoffRatio(unsigned long numMatches, ssrlcv::Match* matches, int* matchCounter, float2 minMax, float cutoffRatio){
  unsigned long globalId = blockIdx.x*blockDim.x + threadIdx.x;
  if(globalId < numMatches){
    float2 regMinMax = minMax;
    if((matches[globalId].distance - regMinMax.x)/(regMinMax.y-regMinMax.x) < cutoffRatio){
      matchCounter[globalId] = 1;
    }
    else{
      matchCounter[globalId] = 0;
    }
  }
}


__global__ void ssrlcv::copyMatches(unsigned long numMatches, int* matchCounter, ssrlcv::Match* minimizedMatches, ssrlcv::Match* matches){
  unsigned long globalId = blockIdx.x*blockDim.x + threadIdx.x;
  if(globalId < numMatches){
    int counterVal = matchCounter[globalId];
    if(counterVal != 0 && (globalId == 0 || counterVal > matchCounter[globalId - 1])){
      minimizedMatches[counterVal - 1] = matches[globalId];
    }
  }
}

template<typename T>
__global__ void ssrlcv::refineWCutoffRatio(unsigned long numMatches, ssrlcv::FeatureMatch<T>* matches, int* matchCounter, float2 minMax, float cutoffRatio){
  unsigned long globalId = blockIdx.x*blockDim.x + threadIdx.x;
  if(globalId < numMatches){
    float2 regMinMax = minMax;
    if((matches[globalId].distance - regMinMax.x)/(regMinMax.y-regMinMax.x) < cutoffRatio){
      matchCounter[globalId] = 1;
    }
    else{
      matchCounter[globalId] = 0;
    }
  }
}

template<typename T>
__global__ void ssrlcv::copyMatches(unsigned long numMatches, int* matchCounter, ssrlcv::FeatureMatch<T>* minimizedMatches, ssrlcv::FeatureMatch<T>* matches){
  unsigned long globalId = blockIdx.x*blockDim.x + threadIdx.x;
  if(globalId < numMatches){
    int counterVal = matchCounter[globalId];
    if(counterVal != 0 && (globalId == 0 || counterVal > matchCounter[globalId - 1])){
      minimizedMatches[counterVal - 1] = matches[globalId];
    }
  }
}
