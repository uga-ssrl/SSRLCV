#include "MatchFactory.cuh"

template class ssrlcv::MatchFactory<ssrlcv::SIFT_Descriptor>;


template<typename T>
ssrlcv::MatchFactory<T>::MatchFactory(){

}

template<typename T>
void ssrlcv::MatchFactory<T>::refineMatches(ssrlcv::Unity<ssrlcv::DMatch>* matches, float cutoffRatio){
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

  thrust::device_ptr<DMatch> needsCompacting(matches->device);
  thrust::device_ptr<DMatch> end = thrust::remove_if(needsCompacting, needsCompacting + matches->numElements, match_dist_thresholder((max-min)*cutoffRatio + min));
  unsigned int numElementsBelowThreshold = end - needsCompacting;

  printf("%d matches have been refined to %d matches using a cutoff of %f (%f percentile)\n",matches->numElements,numElementsBelowThreshold,(max-min)*cutoffRatio + min,cutoffRatio);

  FeatureMatch<T>* compactedMatches_device = nullptr;
  CudaSafeCall(cudaMalloc((void**)&compactedMatches_device,numElementsBelowThreshold*sizeof(DMatch)));
  CudaSafeCall(cudaMemcpy(compactedMatches_device,matches->device,numElementsBelowThreshold*sizeof(DMatch),cudaMemcpyDeviceToDevice));

  matches->setData(compactedMatches_device,numElementsBelowThreshold,gpu);

  if(origin == cpu) matches->setMemoryState(cpu);
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

  thrust::device_ptr<FeatureMatch<T>> needsCompacting(matches->device);
  thrust::device_ptr<FeatureMatch<T>> end = thrust::remove_if(needsCompacting, needsCompacting + matches->numElements, match_dist_thresholder((max-min)*cutoffRatio + min));
  unsigned int numElementsBelowThreshold = end - needsCompacting;

  printf("%d matches have been refined to %d matches using a cutoff of %f (%f percentile)\n",matches->numElements,numElementsBelowThreshold,(max-min)*cutoffRatio + min,cutoffRatio);

  FeatureMatch<T>* compactedMatches_device = nullptr;
  CudaSafeCall(cudaMalloc((void**)&compactedMatches_device,numElementsBelowThreshold*sizeof(FeatureMatch<T>)));
  CudaSafeCall(cudaMemcpy(compactedMatches_device,matches->device,numElementsBelowThreshold*sizeof(FeatureMatch<T>),cudaMemcpyDeviceToDevice));

  matches->setData(compactedMatches_device,numElementsBelowThreshold,gpu);

  if(origin == cpu) matches->setMemoryState(cpu);
}

template<typename T>
void ssrlcv::MatchFactory<T>::sortMatches(Unity<DMatch>* matches){
  if(matches->state == gpu || matches->fore == gpu){
    thrust::device_ptr<DMatch> toSort(matches->device);
    thrust::sort(toSort, toSort + matches->numElements,match_dist_comparator());
    matches->fore = gpu;
    if(matches->state == both) matches->transferMemoryTo(cpu);
  }
  else{
    unsigned long len = matches->numElements;
    // insertion sort
    // each match element is accessed with allMatches->host[]
    unsigned long i = 0;
    unsigned long j = 0;
    ssrlcv::DMatch temp;
    while (i < len){
      j = i;
      while (j > 0 && matches->host[j-1].distance > matches->host[j].distance){
        temp = matches->host[j];
        matches->host[j] = matches->host[j-1];
        matches->host[j-1] = temp;
        j--;
      }
      i++;
    }
    matches->fore = cpu;
    if(matches->state == both) matches->transferMemoryTo(gpu);
  }
}
template<typename T>
void ssrlcv::MatchFactory<T>::sortMatches(Unity<FeatureMatch<T>>* matches){
  if(matches->state == gpu || matches->fore == gpu){
    thrust::device_ptr<FeatureMatch<T>> toSort(matches->device);
    thrust::sort(toSort, toSort + matches->numElements,match_dist_comparator());
    matches->fore = gpu;
    if(matches->state == both) matches->transferMemoryTo(cpu);
  }
  else{
    unsigned long len = matches->numElements;
    // insertion sort
    // each match element is accessed with allMatches->host[]
    unsigned long i = 0;
    unsigned long j = 0;
    ssrlcv::FeatureMatch<T> temp;
    while (i < len){
      j = i;
      while (j > 0 && matches->host[j-1].distance > matches->host[j].distance){
        temp = matches->host[j];
        matches->host[j] = matches->host[j-1];
        matches->host[j-1] = temp;
        j--;
      }
      i++;
    }
    matches->fore = cpu;
    if(matches->state == both) matches->transferMemoryTo(gpu);
  }
}

template<typename T>
ssrlcv::Unity<ssrlcv::Match>* ssrlcv::MatchFactory<T>::getRawMatches(Unity<DMatch>* matches){
  if(matches->state == gpu || matches->fore == gpu){
    Match* rawMatches_device = nullptr;
    CudaSafeCall(cudaMalloc((void**)&rawMatches_device, matches->numElements*sizeof(DMatch)));
    dim3 grid = {1,1,1};
    dim3 block = {1,1,1};
    getFlatGridBlock(matches->numElements,grid,block);
    convertMatchToRaw<<<grid,block>>>(matches->numElements,rawMatches_device,matches->device);
    cudaDeviceSynchronize();
    CudaCheckError();
    return new Unity<Match>(rawMatches_device,matches->numElements,gpu);
  }
  else{
    Match* rawMatches_host = new Match[matches->numElements];
    for(int i = 0; i < matches->numElements; ++i){
      for(int f = 0; f < 2; ++f){
        rawMatches_host[i] = Match(matches->host[i]);
      }
    }
    return new Unity<Match>(rawMatches_host, matches->numElements, cpu);
  }
}
template<typename T>
ssrlcv::Unity<ssrlcv::Match>* ssrlcv::MatchFactory<T>::getRawMatches(Unity<FeatureMatch<T>>* matches){
  if(matches->state == gpu || matches->fore == gpu){
    Match* rawMatches_device = nullptr;
    CudaSafeCall(cudaMalloc((void**)&rawMatches_device, matches->numElements*sizeof(Match)));
    dim3 grid = {1,1,1};
    dim3 block = {1,1,1};
    getFlatGridBlock(matches->numElements,grid,block);
    convertMatchToRaw<<<grid,block>>>(matches->numElements,rawMatches_device,matches->device);
    cudaDeviceSynchronize();
    CudaCheckError();
    return new Unity<Match>(rawMatches_device,matches->numElements,gpu);
  }
  else{
    Match* rawMatches_host = new Match[matches->numElements];
    for(int i = 0; i < matches->numElements; ++i){
      for(int f = 0; f < 2; ++f){
        rawMatches_host[i] = Match(matches->host[i]);
      }
    }
    return new Unity<Match>(rawMatches_host, matches->numElements, cpu);
  }
}

template<typename T>
ssrlcv::Unity<ssrlcv::Match>* ssrlcv::MatchFactory<T>::generateMatches(Image* query, Unity<Feature<T>>* queryFeatures, Image* target, Unity<Feature<T>>* targetFeatures){
  MemoryState origin[2] = {queryFeatures->state, targetFeatures->state};

  if(queryFeatures->fore == cpu) queryFeatures->setMemoryState(gpu);
  if(targetFeatures->fore == cpu) targetFeatures->setMemoryState(gpu);

  unsigned int numPossibleMatches = queryFeatures->numElements;

  Match* matches_device = nullptr;
  CudaSafeCall(cudaMalloc((void**)&matches_device, numPossibleMatches*sizeof(Match)));

  Unity<Match>* matches = new Unity<Match>(matches_device, numPossibleMatches, gpu);

  dim3 grid = {1,1,1};
  dim3 block = {1024,1,1};
  getGrid(matches->numElements,grid);

  clock_t timer = clock();

  matchFeaturesBruteForce<<<grid, block>>>(query->id, queryFeatures->numElements, queryFeatures->device,
    target->id, targetFeatures->numElements, targetFeatures->device, matches->device);

  cudaDeviceSynchronize();
  CudaCheckError();

  printf("done in %f seconds.\n\n",((float) clock() -  timer)/CLOCKS_PER_SEC);

  if(origin[0] != queryFeatures->state) queryFeatures->setMemoryState(origin[0]);
  if(origin[1] != targetFeatures->state) targetFeatures->setMemoryState(origin[1]);

  return matches;
}
template<typename T>
ssrlcv::Unity<ssrlcv::Match>* ssrlcv::MatchFactory<T>::generateMatchesConstrained(Image* query, Unity<Feature<T>>* queryFeatures, Image* target, Unity<Feature<T>>* targetFeatures, float epsilon){
  MemoryState origin[2] = {queryFeatures->state, targetFeatures->state};

  if(queryFeatures->fore == cpu) queryFeatures->setMemoryState(gpu);
  if(targetFeatures->fore == cpu) targetFeatures->setMemoryState(gpu);

  unsigned int numPossibleMatches = queryFeatures->numElements;

  Match* matches_device = nullptr;
  CudaSafeCall(cudaMalloc((void**)&matches_device, numPossibleMatches*sizeof(Match)));

  Unity<Match>* matches = new Unity<Match>(matches_device, numPossibleMatches, gpu);

  dim3 grid = {1,1,1};
  dim3 block = {1024,1,1};
  getGrid(matches->numElements,grid);

  clock_t timer = clock();
  float3* fundamental = new float3[3];
  calcFundamentalMatrix_2View(query, target, fundamental);

  float3* fundamental_device;
  CudaSafeCall(cudaMalloc((void**)&fundamental_device, 3*sizeof(float3)));
  CudaSafeCall(cudaMemcpy(fundamental_device, fundamental, 3*sizeof(float3), cudaMemcpyHostToDevice));

  matchFeaturesConstrained<<<grid, block>>>(query->id, queryFeatures->numElements, queryFeatures->device,
    target->id, targetFeatures->numElements, targetFeatures->device, matches->device, epsilon, fundamental_device);
  cudaDeviceSynchronize();
  CudaCheckError();

  CudaSafeCall(cudaFree(fundamental_device));

  printf("done in %f seconds.\n\n",((float) clock() -  timer)/CLOCKS_PER_SEC);

  if(origin[0] != queryFeatures->state) queryFeatures->setMemoryState(origin[0]);
  if(origin[1] != targetFeatures->state) targetFeatures->setMemoryState(origin[1]);

  return matches;
}
template<typename T>
ssrlcv::Unity<ssrlcv::DMatch>*ssrlcv::MatchFactory<T>:: generateDistanceMatches(Image* query, Unity<Feature<T>>* queryFeatures, Image* target, Unity<Feature<T>>* targetFeatures){
  MemoryState origin[2] = {queryFeatures->state, targetFeatures->state};

  if(queryFeatures->fore == cpu) queryFeatures->setMemoryState(gpu);
  if(targetFeatures->fore == cpu) targetFeatures->setMemoryState(gpu);

  unsigned int numPossibleMatches = queryFeatures->numElements;

  DMatch* matches_device = nullptr;
  CudaSafeCall(cudaMalloc((void**)&matches_device, numPossibleMatches*sizeof(DMatch)));

  Unity<DMatch>* matches = new Unity<DMatch>(matches_device, numPossibleMatches, gpu);

  dim3 grid = {1,1,1};
  dim3 block = {1024,1,1};
  getGrid(matches->numElements,grid);

  clock_t timer = clock();

  matchFeaturesBruteForce<<<grid, block>>>(query->id, queryFeatures->numElements, queryFeatures->device,
    target->id, targetFeatures->numElements, targetFeatures->device, matches->device);

  cudaDeviceSynchronize();
  CudaCheckError();

  printf("done in %f seconds.\n\n",((float) clock() -  timer)/CLOCKS_PER_SEC);

  if(origin[0] != queryFeatures->state) queryFeatures->setMemoryState(origin[0]);
  if(origin[1] != targetFeatures->state) targetFeatures->setMemoryState(origin[1]);

  return matches;
}
template<typename T>
ssrlcv::Unity<ssrlcv::DMatch>*ssrlcv::MatchFactory<T>:: generateDistanceMatchesConstrained(Image* query, Unity<Feature<T>>* queryFeatures, Image* target, Unity<Feature<T>>* targetFeatures, float epsilon){
  MemoryState origin[2] = {queryFeatures->state, targetFeatures->state};

  if(queryFeatures->fore == cpu) queryFeatures->setMemoryState(gpu);
  if(targetFeatures->fore == cpu) targetFeatures->setMemoryState(gpu);

  unsigned int numPossibleMatches = queryFeatures->numElements;

  DMatch* matches_device = nullptr;
  CudaSafeCall(cudaMalloc((void**)&matches_device, numPossibleMatches*sizeof(DMatch)));

  Unity<DMatch>* matches = new Unity<DMatch>(matches_device, numPossibleMatches, gpu);

  dim3 grid = {1,1,1};
  dim3 block = {1024,1,1};
  getGrid(matches->numElements,grid);

  clock_t timer = clock();
  float3* fundamental = new float3[3];
  calcFundamentalMatrix_2View(query, target, fundamental);

  float3* fundamental_device;
  CudaSafeCall(cudaMalloc((void**)&fundamental_device, 3*sizeof(float3)));
  CudaSafeCall(cudaMemcpy(fundamental_device, fundamental, 3*sizeof(float3), cudaMemcpyHostToDevice));

  matchFeaturesConstrained<<<grid, block>>>(query->id, queryFeatures->numElements, queryFeatures->device,
    target->id, targetFeatures->numElements, targetFeatures->device, matches->device, epsilon, fundamental_device);
  cudaDeviceSynchronize();
  CudaCheckError();

  CudaSafeCall(cudaFree(fundamental_device));

  printf("done in %f seconds.\n\n",((float) clock() -  timer)/CLOCKS_PER_SEC);

  if(origin[0] != queryFeatures->state) queryFeatures->setMemoryState(origin[0]);
  if(origin[1] != targetFeatures->state) targetFeatures->setMemoryState(origin[1]);

  return matches;
}
template<typename T>
ssrlcv::Unity<ssrlcv::FeatureMatch<T>>* ssrlcv::MatchFactory<T>::generateFeatureMatches(ssrlcv::Image* query, ssrlcv::Unity<ssrlcv::Feature<T>>* queryFeatures,
ssrlcv::Image* target, ssrlcv::Unity<ssrlcv::Feature<T>>* targetFeatures){

  MemoryState origin[2] = {queryFeatures->state, targetFeatures->state};

  if(queryFeatures->fore == cpu) queryFeatures->setMemoryState(gpu);
  if(targetFeatures->fore == cpu) targetFeatures->setMemoryState(gpu);

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

  if(origin[0] != queryFeatures->state) queryFeatures->setMemoryState(origin[0]);
  if(origin[1] != targetFeatures->state) targetFeatures->setMemoryState(origin[1]);

  return matches;
}
template<typename T>
ssrlcv::Unity<ssrlcv::FeatureMatch<T>>* ssrlcv::MatchFactory<T>::generateFeatureMatchesConstrained(ssrlcv::Image* query, ssrlcv::Unity<ssrlcv::Feature<T>>* queryFeatures,
ssrlcv::Image* target, ssrlcv::Unity<ssrlcv::Feature<T>>* targetFeatures, float epsilon){

  MemoryState origin[2] = {queryFeatures->state, targetFeatures->state};

  if(queryFeatures->fore == cpu) queryFeatures->setMemoryState(gpu);
  if(targetFeatures->fore == cpu) targetFeatures->setMemoryState(gpu);

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

  if(origin[0] != queryFeatures->state) queryFeatures->setMemoryState(origin[0]);
  if(origin[1] != targetFeatures->state) targetFeatures->setMemoryState(origin[1]);

  return matches;

}

template<typename T>
ssrlcv::Unity<ssrlcv::FeatureMatch<T>>* ssrlcv::MatchFactory<T>::generateSubPixelMatches(ssrlcv::Image* query, ssrlcv::Unity<ssrlcv::Feature<T>>* queryFeatures,
ssrlcv::Image* target, ssrlcv::Unity<ssrlcv::Feature<T>>* targetFeatures){

  MemoryState origin[2] = {queryFeatures->state, targetFeatures->state};

  if(queryFeatures->fore == cpu) queryFeatures->setMemoryState(gpu);
  if(targetFeatures->fore == cpu) targetFeatures->setMemoryState(gpu);

  Unity<FeatureMatch<T>>* matches = this->generateFeatureMatches(query, queryFeatures, target, targetFeatures);
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

  if(origin[0] != queryFeatures->state) queryFeatures->setMemoryState(origin[0]);
  if(origin[1] != targetFeatures->state) targetFeatures->setMemoryState(origin[1]);

  return matches;
}
template<typename T>
ssrlcv::Unity<ssrlcv::FeatureMatch<T>>* ssrlcv::MatchFactory<T>::generateSubPixelMatchesConstrained(ssrlcv::Image* query, ssrlcv::Unity<ssrlcv::Feature<T>>* queryFeatures,
ssrlcv::Image* target, ssrlcv::Unity<ssrlcv::Feature<T>>* targetFeatures, float epsilon){
  MemoryState origin[2] = {queryFeatures->state, targetFeatures->state};

  if(queryFeatures->fore == cpu) queryFeatures->setMemoryState(gpu);
  if(targetFeatures->fore == cpu) targetFeatures->setMemoryState(gpu);

  Unity<FeatureMatch<T>>* matches = this->generateFeatureMatchesConstrained(query, queryFeatures, target, targetFeatures, epsilon);
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

  if(origin[0] != queryFeatures->state) queryFeatures->setMemoryState(origin[0]);
  if(origin[1] != targetFeatures->state) targetFeatures->setMemoryState(origin[1]);

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
__device__ __forceinline__ float ssrlcv::calcElucidSq(const SIFT_Descriptor& a, const SIFT_Descriptor& b){
  float dist = 0.0f;
  for(int i = 0; i < 128; ++i){
    dist += square(((float)a.values[i]-b.values[i]));
  }
  dist += a.theta - b.theta;
  dist += a.sigma - b.sigma;
  return dist;
}
__device__ __forceinline__ float ssrlcv::calcElucidSq(const SIFT_Descriptor& a, const SIFT_Descriptor& b, const float &bestMatch){
  float dist = 0.0f;
  for(int i = 0; i < 128 && dist < bestMatch; ++i){
    dist += square(((float)a.values[i]-b.values[i]));
  }
  if(dist < bestMatch) dist += a.theta - b.theta;
  else return dist;
  if(dist < bestMatch) dist += a.sigma - b.sigma;
  else return dist;
  return dist;
}


/*
matching
*/
//base matching kernels
template<typename T>
__global__ void ssrlcv::matchFeaturesBruteForce(unsigned int queryImageID, unsigned long numFeaturesQuery,
ssrlcv::Feature<T>* featuresQuery, unsigned int targetImageID, unsigned long numFeaturesTarget,
ssrlcv::Feature<T>* featuresTarget, Match* matches){
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
    Match match;
    match.keyPoints[0].loc = feature.loc;
    match.keyPoints[1].loc = featuresTarget[matchIndex].loc;
    match.keyPoints[0].parentId = queryImageID;
    match.keyPoints[1].parentId = targetImageID;
    matches[blockId] = match;
  }
}
template<typename T>
__global__ void ssrlcv::matchFeaturesBruteForce(unsigned int queryImageID, unsigned long numFeaturesQuery,
ssrlcv::Feature<T>* featuresQuery, unsigned int targetImageID, unsigned long numFeaturesTarget,
ssrlcv::Feature<T>* featuresTarget, DMatch* matches){
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
    DMatch match;
    match.keyPoints[0].loc = feature.loc;
    match.keyPoints[1].loc = featuresTarget[matchIndex].loc;
    match.keyPoints[0].parentId = queryImageID;
    match.keyPoints[1].parentId = targetImageID;
    match.distance = currentDist;
    matches[blockId] = match;
  }
}
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
    match.keyPoints[0].loc = feature.loc;
    match.keyPoints[1].loc = featuresTarget[matchIndex].loc;
    match.keyPoints[0].parentId = queryImageID;
    match.keyPoints[1].parentId = targetImageID;
    match.distance = currentDist;
    matches[blockId] = match;
  }
}

template<typename T>
__global__ void ssrlcv::matchFeaturesConstrained(unsigned int queryImageID, unsigned long numFeaturesQuery,
ssrlcv::Feature<T>* featuresQuery, unsigned int targetImageID, unsigned long numFeaturesTarget,
ssrlcv::Feature<T>* featuresTarget, Match* matches, float epsilon, float3 fundamental[3]){
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
    Match match;
    match.keyPoints[0].loc = feature.loc;
    match.keyPoints[1].loc = featuresTarget[matchIndex].loc;
    match.keyPoints[0].parentId = queryImageID;
    match.keyPoints[1].parentId = targetImageID;
    matches[blockId] = match;
  }
}
template<typename T>
__global__ void ssrlcv::matchFeaturesConstrained(unsigned int queryImageID, unsigned long numFeaturesQuery,
ssrlcv::Feature<T>* featuresQuery, unsigned int targetImageID, unsigned long numFeaturesTarget,
ssrlcv::Feature<T>* featuresTarget, DMatch* matches, float epsilon, float3 fundamental[3]){
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
    DMatch match;
    match.keyPoints[0].loc = feature.loc;
    match.keyPoints[1].loc = featuresTarget[matchIndex].loc;
    match.keyPoints[0].parentId = queryImageID;
    match.keyPoints[1].parentId = targetImageID;
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
    match.keyPoints[0].loc = feature.loc;
    match.keyPoints[1].loc = featuresTarget[matchIndex].loc;
    match.keyPoints[0].parentId = queryImageID;
    match.keyPoints[1].parentId = targetImageID;
    match.distance = currentDist;
    matches[blockId] = match;
  }
}


//subpixel kernels
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
    int contribQuery = findSubPixelContributer(match.keyPoints[0].loc + contrib, querySize.x);
    int contribTarget = findSubPixelContributer(match.keyPoints[1].loc + contrib, targetSize.x);

    int pairedMatchIndex = findSubPixelContributer(match.keyPoints[1].loc, targetSize.x);

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
      if(blockId%2 == 0) matches[blockId/2].keyPoints[0].loc  = localSubLoc + matches[blockId/2].keyPoints[0].loc;
      else matches[blockId/2].keyPoints[1].loc = localSubLoc + matches[blockId/2].keyPoints[1].loc;
    }
    else return;
  }
}


//utility kernels
__global__ void ssrlcv::convertMatchToRaw(unsigned long numMatches, ssrlcv::Match* rawMatches, ssrlcv::Match* matches){
  unsigned long globalID = (blockIdx.y* gridDim.x+ blockIdx.x)*blockDim.x + threadIdx.x;
  if(globalID < numMatches){
    rawMatches[globalID] = Match(matches[globalID]);
  }
}
