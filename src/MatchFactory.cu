#include "MatchFactory.cuh"

template class ssrlcv::MatchFactory<ssrlcv::SIFT_Descriptor>;
template class ssrlcv::MatchFactory<ssrlcv::Window_3x3>;
template class ssrlcv::MatchFactory<ssrlcv::Window_9x9>;
template class ssrlcv::MatchFactory<ssrlcv::Window_15x15>;
template class ssrlcv::MatchFactory<ssrlcv::Window_25x25>;
template class ssrlcv::MatchFactory<ssrlcv::Window_31x31>;


template<typename T>
ssrlcv::MatchFactory<T>::MatchFactory(){
  this->relativeThreshold = 0.8f;
  this->absoluteThreshold = 250.0f;
  this->seedFeatures = nullptr;
}
template<typename T>
ssrlcv::MatchFactory<T>::MatchFactory(float relativeThreshold, float absoluteThreshold) :
relativeThreshold(relativeThreshold), absoluteThreshold(absoluteThreshold)
{
  this->seedFeatures = nullptr;
}
template<typename T>
void ssrlcv::MatchFactory<T>::setSeedFeatures(Unity<Feature<T>>* seedFeatures){
  this->seedFeatures = seedFeatures;
}
template<typename T>
void ssrlcv::MatchFactory<T>::validateMatches(ssrlcv::Unity<ssrlcv::Match>* matches){
  MemoryState origin = matches->state;
  if(origin == cpu || matches->fore == cpu){
    matches->transferMemoryTo(gpu);
  }

  thrust::device_ptr<Match> needsValidating(matches->device);
  thrust::device_ptr<Match> new_end = thrust::remove_if(needsValidating,needsValidating+matches->numElements,validate());
  cudaDeviceSynchronize();
  CudaCheckError();
  int numMatchesLeft = new_end - needsValidating;
  if(numMatchesLeft == 0){
    std::cout<<"No valid matches found"<<std::endl;
    delete matches;
    matches = nullptr;
    return;
  }
  

  printf("%d valid matches found out of %d original matches\n",numMatchesLeft,matches->numElements);

  Match* validatedMatches_device = nullptr;
  CudaSafeCall(cudaMalloc((void**)&validatedMatches_device,numMatchesLeft*sizeof(Match)));
  CudaSafeCall(cudaMemcpy(validatedMatches_device,matches->device,numMatchesLeft*sizeof(Match),cudaMemcpyDeviceToDevice));

  matches->setData(validatedMatches_device,numMatchesLeft,gpu);

  if(origin == cpu) matches->setMemoryState(cpu);
}
template<typename T>
void ssrlcv::MatchFactory<T>::validateMatches(ssrlcv::Unity<ssrlcv::DMatch>* matches){
  MemoryState origin = matches->state;
  if(origin == cpu || matches->fore == cpu){
    matches->transferMemoryTo(gpu);
  }

  thrust::device_ptr<DMatch> needsValidating(matches->device);
  thrust::device_ptr<DMatch> new_end = thrust::remove_if(needsValidating,needsValidating+matches->numElements,validate());
  cudaDeviceSynchronize();
  CudaCheckError();
  int numMatchesLeft = new_end - needsValidating;
  if(numMatchesLeft == 0){
    std::cout<<"No valid matches found"<<std::endl;
    delete matches;
    matches = nullptr;
    return;
  }
  

  printf("%d valid matches found out of %d original matches\n",numMatchesLeft,matches->numElements);

  DMatch* validatedMatches_device = nullptr;
  CudaSafeCall(cudaMalloc((void**)&validatedMatches_device,numMatchesLeft*sizeof(DMatch)));
  CudaSafeCall(cudaMemcpy(validatedMatches_device,matches->device,numMatchesLeft*sizeof(DMatch),cudaMemcpyDeviceToDevice));

  matches->setData(validatedMatches_device,numMatchesLeft,gpu);

  if(origin == cpu) matches->setMemoryState(cpu);
}
template<typename T>
void ssrlcv::MatchFactory<T>::validateMatches(ssrlcv::Unity<ssrlcv::FeatureMatch<T>>* matches){
  MemoryState origin = matches->state;
  if(origin == cpu || matches->fore == cpu){
    matches->transferMemoryTo(gpu);
  }

  thrust::device_ptr<FeatureMatch<T>> needsValidating(matches->device);
  thrust::device_ptr<FeatureMatch<T>> new_end = thrust::remove_if(needsValidating,needsValidating+matches->numElements,validate());
  cudaDeviceSynchronize();
  CudaCheckError();
  int numMatchesLeft = new_end - needsValidating;
  if(numMatchesLeft == 0){
    std::cout<<"No valid matches found"<<std::endl;
    delete matches;
    matches = nullptr;
    return;
  }
  

  printf("%d valid matches found out of %d original matches\n",numMatchesLeft,matches->numElements);

  FeatureMatch<T>* validatedMatches_device = nullptr;
  CudaSafeCall(cudaMalloc((void**)&validatedMatches_device,numMatchesLeft*sizeof(FeatureMatch<T>)));
  CudaSafeCall(cudaMemcpy(validatedMatches_device,matches->device,numMatchesLeft*sizeof(FeatureMatch<T>),cudaMemcpyDeviceToDevice));

  matches->setData(validatedMatches_device,numMatchesLeft,gpu);

  if(origin == cpu) matches->setMemoryState(cpu);

}
template<typename T>
void ssrlcv::MatchFactory<T>::refineMatches(ssrlcv::Unity<ssrlcv::DMatch>* matches, float threshold){
  if(threshold == 0.0f){
    std::cout<<"ERROR illegal value used for threshold: 0.0"<<std::endl;
    exit(-1);
  }
  MemoryState origin = matches->state;
  if(origin == cpu || matches->fore == cpu){
    matches->transferMemoryTo(gpu);
  }

  thrust::device_ptr<DMatch> needsCompacting(matches->device);
  thrust::device_ptr<DMatch> end = thrust::remove_if(needsCompacting, needsCompacting + matches->numElements, match_dist_thresholder(threshold));
  unsigned int numElementsBelowThreshold = end - needsCompacting;
  if(numElementsBelowThreshold == 0){
    delete matches;
    matches = nullptr;
    return;
  }

  printf("%d matches have been refined to %d matches using a cutoff of %f\n",matches->numElements,numElementsBelowThreshold,threshold);

  DMatch* compactedMatches_device = nullptr;
  CudaSafeCall(cudaMalloc((void**)&compactedMatches_device,numElementsBelowThreshold*sizeof(DMatch)));
  CudaSafeCall(cudaMemcpy(compactedMatches_device,matches->device,numElementsBelowThreshold*sizeof(DMatch),cudaMemcpyDeviceToDevice));

  matches->setData(compactedMatches_device,numElementsBelowThreshold,gpu);

  if(origin == cpu) matches->setMemoryState(cpu);
}
template<typename T>
void ssrlcv::MatchFactory<T>::refineMatches(ssrlcv::Unity<ssrlcv::FeatureMatch<T>>* matches, float threshold){
  if(threshold == 0.0f){
    std::cout<<"ERROR illegal value used for cutoff ratio: 0.0"<<std::endl;
    exit(-1);
  }
  MemoryState origin = matches->state;
  if(origin == cpu || matches->fore == cpu){
    matches->transferMemoryTo(gpu);
  }

  if(origin == gpu) matches->clear(cpu);

  thrust::device_ptr<FeatureMatch<T>> needsCompacting(matches->device);
  thrust::device_ptr<FeatureMatch<T>> end = thrust::remove_if(needsCompacting, needsCompacting + matches->numElements, match_dist_thresholder(threshold));
  unsigned int numElementsBelowThreshold = end - needsCompacting;
  if(numElementsBelowThreshold == 0){
    delete matches;
    matches = nullptr;
    return;
  }

  printf("%d matches have been refined to %d matches using a cutoff of %f\n",matches->numElements,numElementsBelowThreshold,threshold);

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
    CudaSafeCall(cudaMalloc((void**)&rawMatches_device, matches->numElements*sizeof(Match)));
    dim3 grid = {1,1,1};
    dim3 block = {1,1,1};
    void (*fp)(unsigned long, Match*, DMatch*) = &convertMatchToRaw;
    getFlatGridBlock(matches->numElements,grid,block,fp);
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
    void (*fp)(unsigned long, Match*, FeatureMatch<T>*) = &convertMatchToRaw;
    getFlatGridBlock(matches->numElements,grid,block,fp);
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
ssrlcv::Unity<float>* ssrlcv::MatchFactory<T>::getSeedDistances(Unity<Feature<T>>* features){
  MemoryState origin = features->state;

  if(this->seedFeatures->fore == cpu) this->seedFeatures->setMemoryState(gpu);
  if(features->fore == cpu) features->setMemoryState(gpu);

  unsigned int numPossibleMatches = features->numElements;

  Unity<float>* matchDistances = new Unity<float>(nullptr, numPossibleMatches,gpu);

  dim3 grid = {1,1,1};
  dim3 block = {192,1,1};//may need to make a device query for largest block size
  getGrid(matchDistances->numElements,grid,getSeedMatchDistances<T>);

  clock_t timer = clock();

  getSeedMatchDistances<<<grid, block>>>(features->numElements,features->device,this->seedFeatures->numElements,
    this->seedFeatures->device,matchDistances->device);

  cudaDeviceSynchronize();
  CudaCheckError();

  printf("done in %f seconds.\n\n",((float) clock() -  timer)/CLOCKS_PER_SEC);

  if(origin != features->state) features->setMemoryState(origin);
  
  return matchDistances;
}

template<typename T>
ssrlcv::Unity<ssrlcv::Match>* ssrlcv::MatchFactory<T>::generateMatches(Image* query, Unity<Feature<T>>* queryFeatures, Image* target, Unity<Feature<T>>* targetFeatures, Unity<float>* seedDistances){
  MemoryState origin[2] = {queryFeatures->state, targetFeatures->state};

  if(queryFeatures->fore == cpu) queryFeatures->setMemoryState(gpu);
  if(targetFeatures->fore == cpu) targetFeatures->setMemoryState(gpu);

  unsigned int numPossibleMatches = queryFeatures->numElements;

  Match* matches_device = nullptr;
  CudaSafeCall(cudaMalloc((void**)&matches_device, numPossibleMatches*sizeof(Match)));
  Unity<Match>* matches = new Unity<Match>(matches_device, numPossibleMatches, gpu);

  dim3 grid = {1,1,1};
  dim3 block = {192,1,1};//may need to make a device query for largest block size
  void (*fp)(unsigned int, unsigned long,
    Feature<T>*, unsigned int, unsigned long,
    Feature<T>*, Match*, float) = &matchFeaturesBruteForce;
  getGrid(matches->numElements,grid,fp);

  clock_t timer = clock();

  if(seedDistances == nullptr){
    matchFeaturesBruteForce<<<grid, block>>>(query->id, queryFeatures->numElements, queryFeatures->device,
    target->id, targetFeatures->numElements, targetFeatures->device, matches->device,this->absoluteThreshold);
  }
  else if(seedDistances->numElements != queryFeatures->numElements){
    std::cerr<<"ERROR: seedDistances should have come from matching a seed image to queryFeatures"<<std::endl;
    exit(-1);
  }
  else{
    if(seedDistances->fore != gpu) seedDistances->setMemoryState(gpu);
    matchFeaturesBruteForce<<<grid, block>>>(query->id, queryFeatures->numElements, queryFeatures->device,
    target->id, targetFeatures->numElements, targetFeatures->device, matches->device,seedDistances->device,
    this->relativeThreshold,this->absoluteThreshold);
  }
  
  cudaDeviceSynchronize();
  CudaCheckError();

  this->validateMatches(matches);

  printf("done in %f seconds.\n\n",((float) clock() -  timer)/CLOCKS_PER_SEC);

  if(origin[0] != queryFeatures->state) queryFeatures->setMemoryState(origin[0]);
  if(origin[1] != targetFeatures->state) targetFeatures->setMemoryState(origin[1]);

  return matches;
}
template<typename T>
ssrlcv::Unity<ssrlcv::Match>* ssrlcv::MatchFactory<T>::generateMatchesConstrained(Image* query, Unity<Feature<T>>* queryFeatures, Image* target, Unity<Feature<T>>* targetFeatures, float epsilon, float fundamental[3][3], Unity<float>* seedDistances){
  MemoryState origin[2] = {queryFeatures->state, targetFeatures->state};

  if(queryFeatures->fore == cpu) queryFeatures->setMemoryState(gpu);
  if(targetFeatures->fore == cpu) targetFeatures->setMemoryState(gpu);

  unsigned int numPossibleMatches = queryFeatures->numElements;

  Match* matches_device = nullptr;
  CudaSafeCall(cudaMalloc((void**)&matches_device, numPossibleMatches*sizeof(Match)));

  Unity<Match>* matches = new Unity<Match>(matches_device, numPossibleMatches, gpu);

  dim3 grid = {1,1,1};
  dim3 block = {192,1,1};//may need to make a device query for largest block size
  void (*fp)(unsigned int, unsigned long,
    Feature<T>*, unsigned int, unsigned long,
    Feature<T>*, Match*, float, float*, float) = &matchFeaturesConstrained;
  getGrid(matches->numElements,grid,fp);

  float* fundamental_device = nullptr;
  CudaSafeCall(cudaMalloc((void**)&fundamental_device,9*sizeof(float)));
  CudaSafeCall(cudaMemcpy(fundamental_device,fundamental,9*sizeof(float),cudaMemcpyHostToDevice));

  clock_t timer = clock();

  if(seedDistances == nullptr){
    matchFeaturesConstrained<<<grid, block>>>(query->id, queryFeatures->numElements, queryFeatures->device,
    target->id, targetFeatures->numElements, targetFeatures->device, matches->device, epsilon,fundamental_device,this->absoluteThreshold);
  }
  else if(seedDistances->numElements != queryFeatures->numElements){
    std::cerr<<"ERROR: seedDistances should have come from matching a seed image to queryFeatures"<<std::endl;
    exit(-1);
  }
  else{
    if(seedDistances->fore != gpu) seedDistances->setMemoryState(gpu);
    matchFeaturesConstrained<<<grid, block>>>(query->id, queryFeatures->numElements, queryFeatures->device,
    target->id, targetFeatures->numElements, targetFeatures->device, matches->device,epsilon,fundamental_device,seedDistances->device,
    this->relativeThreshold,this->absoluteThreshold);
  }

  cudaDeviceSynchronize();
  CudaCheckError();

  CudaSafeCall(cudaFree(fundamental_device));

  this->validateMatches(matches);

  printf("done in %f seconds.\n\n",((float) clock() -  timer)/CLOCKS_PER_SEC);

  if(origin[0] != queryFeatures->state) queryFeatures->setMemoryState(origin[0]);
  if(origin[1] != targetFeatures->state) targetFeatures->setMemoryState(origin[1]);

  return matches;
}


template<typename T>
ssrlcv::Unity<ssrlcv::DMatch>*ssrlcv::MatchFactory<T>:: generateDistanceMatches(Image* query, Unity<Feature<T>>* queryFeatures, Image* target, Unity<Feature<T>>* targetFeatures, Unity<float>* seedDistances){
  MemoryState origin[2] = {queryFeatures->state, targetFeatures->state};

  if(queryFeatures->fore == cpu) queryFeatures->setMemoryState(gpu);
  if(targetFeatures->fore == cpu) targetFeatures->setMemoryState(gpu);

  unsigned int numPossibleMatches = queryFeatures->numElements;

  Unity<DMatch>* matches = new Unity<DMatch>(nullptr, numPossibleMatches, gpu);

  dim3 grid = {1,1,1};
  dim3 block = {192,1,1};//may need to make a device query for largest block size
  void (*fp)(unsigned int, unsigned long,
    Feature<T>*, unsigned int, unsigned long,
    Feature<T>*, DMatch*, float) = &matchFeaturesBruteForce;
  getGrid(matches->numElements,grid,fp);

  clock_t timer = clock();

  if(seedDistances == nullptr){
    matchFeaturesBruteForce<<<grid, block>>>(query->id, queryFeatures->numElements, queryFeatures->device,
    target->id, targetFeatures->numElements, targetFeatures->device, matches->device,this->absoluteThreshold);
  }
  else if(seedDistances->numElements != queryFeatures->numElements){
    std::cerr<<"ERROR: seedDistances should have come from matching a seed image to queryFeatures"<<std::endl;
    exit(-1);
  }
  else{
    if(seedDistances->fore != gpu) seedDistances->setMemoryState(gpu);
    matchFeaturesBruteForce<<<grid, block>>>(query->id, queryFeatures->numElements, queryFeatures->device,
    target->id, targetFeatures->numElements, targetFeatures->device, matches->device,seedDistances->device,
    this->relativeThreshold,this->absoluteThreshold);
  }
  cudaDeviceSynchronize();
  CudaCheckError();

  this->validateMatches(matches);

  printf("done in %f seconds.\n\n",((float) clock() -  timer)/CLOCKS_PER_SEC);

  if(origin[0] != queryFeatures->state) queryFeatures->setMemoryState(origin[0]);
  if(origin[1] != targetFeatures->state) targetFeatures->setMemoryState(origin[1]);

  return matches;
}
template<typename T>
ssrlcv::Unity<ssrlcv::DMatch>*ssrlcv::MatchFactory<T>:: generateDistanceMatchesConstrained(Image* query, Unity<Feature<T>>* queryFeatures, Image* target, Unity<Feature<T>>* targetFeatures, float epsilon, float fundamental[3][3], Unity<float>* seedDistances){
  MemoryState origin[2] = {queryFeatures->state, targetFeatures->state};

  if(queryFeatures->fore == cpu) queryFeatures->setMemoryState(gpu);
  if(targetFeatures->fore == cpu) targetFeatures->setMemoryState(gpu);

  unsigned int numPossibleMatches = queryFeatures->numElements;

  DMatch* matches_device = nullptr;
  CudaSafeCall(cudaMalloc((void**)&matches_device, numPossibleMatches*sizeof(DMatch)));

  Unity<DMatch>* matches = new Unity<DMatch>(matches_device, numPossibleMatches, gpu);

  dim3 grid = {1,1,1};
  dim3 block = {192,1,1};//may need to make a device query for largest block size
  void (*fp)(unsigned int, unsigned long,
    Feature<T>*, unsigned int, unsigned long,
    Feature<T>*, DMatch*, float, float*, float) = &matchFeaturesConstrained;
  getGrid(matches->numElements,grid,fp);

  float* fundamental_device = nullptr;
  CudaSafeCall(cudaMalloc((void**)&fundamental_device,9*sizeof(float)));
  CudaSafeCall(cudaMemcpy(fundamental_device,fundamental,9*sizeof(float),cudaMemcpyHostToDevice));

  clock_t timer = clock();

  if(seedDistances == nullptr){
    matchFeaturesConstrained<<<grid, block>>>(query->id, queryFeatures->numElements, queryFeatures->device,
    target->id, targetFeatures->numElements, targetFeatures->device, matches->device, epsilon, fundamental_device,this->absoluteThreshold);
  }
  else if(seedDistances->numElements != queryFeatures->numElements){
    std::cerr<<"ERROR: seedDistances should have come from matching a seed image to queryFeatures"<<std::endl;
    exit(-1);
  }
  else{
    if(seedDistances->fore != gpu) seedDistances->setMemoryState(gpu);
    matchFeaturesConstrained<<<grid, block>>>(query->id, queryFeatures->numElements, queryFeatures->device,
    target->id, targetFeatures->numElements, targetFeatures->device, matches->device, epsilon, fundamental_device,seedDistances->device,
    this->relativeThreshold,this->absoluteThreshold);
  }
  cudaDeviceSynchronize();
  CudaCheckError();

  CudaSafeCall(cudaFree(fundamental_device));

  this->validateMatches(matches);

  printf("done in %f seconds.\n\n",((float) clock() -  timer)/CLOCKS_PER_SEC);

  if(origin[0] != queryFeatures->state) queryFeatures->setMemoryState(origin[0]);
  if(origin[1] != targetFeatures->state) targetFeatures->setMemoryState(origin[1]);

  return matches;
}


template<typename T>
ssrlcv::Unity<ssrlcv::FeatureMatch<T>>* ssrlcv::MatchFactory<T>::generateFeatureMatches(ssrlcv::Image* query, ssrlcv::Unity<ssrlcv::Feature<T>>* queryFeatures,
ssrlcv::Image* target, ssrlcv::Unity<ssrlcv::Feature<T>>* targetFeatures, Unity<float>* seedDistances){

  MemoryState origin[2] = {queryFeatures->state, targetFeatures->state};

  if(queryFeatures->fore == cpu) queryFeatures->setMemoryState(gpu);
  if(targetFeatures->fore == cpu) targetFeatures->setMemoryState(gpu);

  unsigned int numPossibleMatches = queryFeatures->numElements;

  FeatureMatch<T>* matches_device = nullptr;
  CudaSafeCall(cudaMalloc((void**)&matches_device, numPossibleMatches*sizeof(FeatureMatch<T>)));

  Unity<FeatureMatch<T>>* matches = new Unity<FeatureMatch<T>>(matches_device, numPossibleMatches, gpu);

  dim3 grid = {1,1,1};
  dim3 block = {192,1,1};//may need to make a device query for largest block size
  void (*fp)(unsigned int, unsigned long,
    Feature<T>*, unsigned int, unsigned long,
    Feature<T>*, FeatureMatch<T>*, float) = &matchFeaturesBruteForce;
  getGrid(matches->numElements,grid,fp);

  clock_t timer = clock();

  if(seedDistances == nullptr){
    matchFeaturesBruteForce<<<grid, block>>>(query->id, queryFeatures->numElements, queryFeatures->device,
    target->id, targetFeatures->numElements, targetFeatures->device, matches->device,this->absoluteThreshold);
  }
  else if(seedDistances->numElements != queryFeatures->numElements){
    std::cerr<<"ERROR: seedDistances should have come from matching a seed image to queryFeatures"<<std::endl;
    exit(-1);
  }
  else{
    if(seedDistances->fore != gpu) seedDistances->setMemoryState(gpu);
    matchFeaturesBruteForce<<<grid, block>>>(query->id, queryFeatures->numElements, queryFeatures->device,
    target->id, targetFeatures->numElements, targetFeatures->device, matches->device,seedDistances->device,this->relativeThreshold,this->absoluteThreshold);
  }

  cudaDeviceSynchronize();
  CudaCheckError();

  this->validateMatches(matches);

  printf("done in %f seconds.\n\n",((float) clock() -  timer)/CLOCKS_PER_SEC);

  if(origin[0] != queryFeatures->state) queryFeatures->setMemoryState(origin[0]);
  if(origin[1] != targetFeatures->state) targetFeatures->setMemoryState(origin[1]);

  return matches;
}
template<typename T>
ssrlcv::Unity<ssrlcv::FeatureMatch<T>>* ssrlcv::MatchFactory<T>::generateFeatureMatchesConstrained(ssrlcv::Image* query, ssrlcv::Unity<ssrlcv::Feature<T>>* queryFeatures,
ssrlcv::Image* target, ssrlcv::Unity<ssrlcv::Feature<T>>* targetFeatures, float epsilon, float fundamental[3][3], Unity<float>* seedDistances){

  MemoryState origin[2] = {queryFeatures->state, targetFeatures->state};

  if(queryFeatures->fore == cpu) queryFeatures->setMemoryState(gpu);
  if(targetFeatures->fore == cpu) targetFeatures->setMemoryState(gpu);

  unsigned int numPossibleMatches = queryFeatures->numElements;

  FeatureMatch<T>* matches_device = nullptr;
  CudaSafeCall(cudaMalloc((void**)&matches_device, numPossibleMatches*sizeof(FeatureMatch<T>)));

  Unity<FeatureMatch<T>>* matches = new Unity<FeatureMatch<T>>(matches_device, numPossibleMatches, gpu);

  dim3 grid = {1,1,1};
  dim3 block = {192,1,1};//may need to make a device query for largest block size
  void (*fp)(unsigned int, unsigned long,
    Feature<T>*, unsigned int, unsigned long,
    Feature<T>*, FeatureMatch<T>*, float, float*, float) = &matchFeaturesConstrained;
  getGrid(matches->numElements,grid,fp);

  float* fundamental_device = nullptr;
  CudaSafeCall(cudaMalloc((void**)&fundamental_device,9*sizeof(float)));
  CudaSafeCall(cudaMemcpy(fundamental_device,fundamental,9*sizeof(float),cudaMemcpyHostToDevice));

  clock_t timer = clock();

  if(seedDistances == nullptr){
    matchFeaturesConstrained<<<grid, block>>>(query->id, queryFeatures->numElements, queryFeatures->device,
    target->id, targetFeatures->numElements, targetFeatures->device, matches->device, epsilon, fundamental_device,this->absoluteThreshold);
  }
  else if(seedDistances->numElements != queryFeatures->numElements){
    std::cerr<<"ERROR: seedDistances should have come from matching a seed image to queryFeatures"<<std::endl;
    exit(-1);
  }
  else{
    if(seedDistances->fore != gpu) seedDistances->setMemoryState(gpu);
    matchFeaturesConstrained<<<grid, block>>>(query->id, queryFeatures->numElements, queryFeatures->device,
    target->id, targetFeatures->numElements, targetFeatures->device, matches->device, epsilon, fundamental_device,seedDistances->device,
    this->relativeThreshold,this->absoluteThreshold);
  }
  cudaDeviceSynchronize();
  CudaCheckError();

  CudaSafeCall(cudaFree(fundamental_device));

  this->validateMatches(matches);

  printf("done in %f seconds.\n\n",((float) clock() -  timer)/CLOCKS_PER_SEC);

  if(origin[0] != queryFeatures->state) queryFeatures->setMemoryState(origin[0]);
  if(origin[1] != targetFeatures->state) targetFeatures->setMemoryState(origin[1]);

  return matches;

}

ssrlcv::Unity<ssrlcv::Match>* ssrlcv::generateDiparityMatches(uint2 querySize, Unity<unsigned char>* queryPixels, uint2 targetSize, Unity<unsigned char>* targetPixels, 
  float fundamental[3][3], unsigned int maxDisparity,unsigned int windowSize, Direction direction){
  if(direction != right && direction != left && direction != undefined){
    std::cerr<<"ERROR: unsupported search direction for disparity matching"<<std::endl;
    exit(-1);
  }
  if(maxDisparity > querySize.x){
    std::cerr<<"Max disparity cannot be larger than image size"<<std::endl;
    exit(-1);
  }
  printf(
    "running disparity matching on parallel images \n\timage[0] = %lux%lu\n\timage[1] = %lux%lu\n\tmaxDisparity = %d\n\twindow size = %lux%lu\n",
    querySize.x,querySize.y,targetSize.x,targetSize.y,maxDisparity,windowSize,windowSize
  );

  if(windowSize == 0 || windowSize % 2 == 0 || windowSize > 31){
    std::cerr<<"ERROR window size for disparity matching must be greater than 0, less than 31 and odd"<<std::endl;
    exit(-1);
  }

  MemoryState origin[2] = {queryPixels->state, targetPixels->state};

  if(queryPixels->fore == cpu) queryPixels->setMemoryState(gpu);
  if(targetPixels->fore == cpu) targetPixels->setMemoryState(gpu);
  
  uint2 minimizedSize = {querySize.x-windowSize-1,querySize.y-windowSize-1};

  unsigned int numPossibleMatches = minimizedSize.x*minimizedSize.y;

  Match* matches_device = nullptr;
  CudaSafeCall(cudaMalloc((void**)&matches_device, numPossibleMatches*sizeof(Match)));

  Unity<Match>* matches = new Unity<Match>(matches_device, numPossibleMatches, gpu);

  dim3 grid = {1,1,1};
  dim3 block = {windowSize,windowSize,1};
  getGrid(numPossibleMatches,grid,disparityMatching);

  bool parallel = true;
  for(int x = 0; x < 3 && parallel; ++x){
    for(int y = 0; y < 3; ++y){
      if((x == 2 && y == 1 && fundamental[y][x] == -1.0f) || (x == 1 && y == 2 && fundamental[y][x] == 1.0f)) continue;
      if(fundamental[y][x] != 0.0f){
        parallel = false;
        break;
      }
    }
  }

  clock_t timer = clock();

  if(!parallel){
    float* fundamental_device = nullptr;
    CudaSafeCall(cudaMalloc((void**)&fundamental_device,9*sizeof(float)));
    CudaSafeCall(cudaMemcpy(fundamental_device,fundamental,9*sizeof(float),cudaMemcpyHostToDevice));
    disparityMatching<<<grid, block>>>(querySize,queryPixels->device,targetSize,targetPixels->device,fundamental_device,matches->device,maxDisparity,direction);
    CudaSafeCall(cudaFree(fundamental_device));
  }
  else{
    disparityScanMatching<<<grid,block>>>(querySize,queryPixels->device,targetSize,targetPixels->device,matches->device,maxDisparity,direction);
  }
  
  cudaDeviceSynchronize();
  CudaCheckError();  
  printf("done in %f seconds.\n\n",((float) clock() -  timer)/CLOCKS_PER_SEC);

  if(origin[0] != queryPixels->state) queryPixels->setMemoryState(origin[0]);
  if(origin[1] != targetPixels->state) targetPixels->setMemoryState(origin[1]);

  thrust::device_ptr<Match> needsValidating(matches->device);
  thrust::device_ptr<Match> new_end = thrust::remove_if(needsValidating,needsValidating+matches->numElements,validate());
  cudaDeviceSynchronize();
  CudaCheckError();
  int numMatchesLeft = new_end - needsValidating;
  if(numMatchesLeft == 0){
    std::cout<<"No valid matches found"<<std::endl;
    delete matches;
    matches = nullptr;
  }
  else{
    printf("%d valid matches found out of %d original matches\n",numMatchesLeft,matches->numElements);
    Match* validatedMatches_device = nullptr;
    CudaSafeCall(cudaMalloc((void**)&validatedMatches_device,numMatchesLeft*sizeof(Match)));
    CudaSafeCall(cudaMemcpy(validatedMatches_device,matches->device,numMatchesLeft*sizeof(Match),cudaMemcpyDeviceToDevice));
    matches->setData(validatedMatches_device,numMatchesLeft,gpu);
  }

  return matches;
}


void ssrlcv::writeMatchFile(Unity<Match>* matches, std::string pathToFile, bool binary){
  MemoryState origin = matches->state;
  if(origin == gpu) matches->transferMemoryTo(cpu);
  if(binary){
    std::ofstream matchstream(pathToFile,std::ios_base::binary);
    if(matchstream.is_open()){
      for(int i = 0; i < matches->numElements; ++i){
        matchstream.write((char*)&matches->host[i].keyPoints[0].loc,2*sizeof(float));
        matchstream.write((char*)&matches->host[i].keyPoints[1].loc,2*sizeof(float));
      }
    }
    else{
      std::cerr<<"ERROR: cannot write "<<pathToFile<<std::endl;
    }
    matchstream.close();
  }
  else{
    std::ofstream matchstream(pathToFile);
    if(matchstream.is_open()){
      std::string line;
      for(int i = 0; i < matches->numElements; ++i){
        line = std::to_string(matches->host[i].keyPoints[0].loc.x) + ",";
        line += std::to_string(matches->host[i].keyPoints[0].loc.y) + ",";
        line += std::to_string(matches->host[i].keyPoints[1].loc.x) + ",";
        line += std::to_string(matches->host[i].keyPoints[1].loc.y) + "\n";
        matchstream << line;
      }
      matchstream.close();
  }
  else{
    std::cerr<<"ERROR: cannot write match files"<<std::endl;
    exit(-1);
  }
  }
  
  std::cout<<pathToFile<<" has been written"<<std::endl;
  if(origin == gpu) matches->setMemoryState(gpu);
}
//NOTE currently only capable of reading in pairwise match files
ssrlcv::Unity<ssrlcv::Match>* ssrlcv::readMatchFile(std::string pathToFile){
  std::ifstream matchstream(pathToFile);
  std::vector<Match> match_vec;
  if(matchstream.is_open()){
    std::string line;
    std::string item;
    getline(matchstream,line);//calibration parameters
    while(getline(matchstream,line)){
      std::istringstream s(line);
      Match match = Match();
      match.keyPoints[0].parentId = 0;
      match.keyPoints[1].parentId = 1;
      getline(s,item,',');
      match.keyPoints[0].loc.x = std::stof(item);
      getline(s,item,',');
      match.keyPoints[0].loc.y = std::stof(item);
      getline(s,item,',');
      match.keyPoints[1].loc.x = std::stof(item);
      getline(s,item,',');
      match.keyPoints[1].loc.y = std::stof(item);
      match_vec.push_back(match);
    }
  }
  std::cout<<match_vec.size()<<" matches have been read."<<std::endl;
  Unity<Match>* matches = new Unity<Match>(nullptr,match_vec.size(),cpu);
  std::memcpy(matches->host,&match_vec[0],match_vec.size()*sizeof(Match));
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

__host__ __device__ __forceinline__ float ssrlcv::sum(const float3 &a){
  return a.x + a.y + a.z;
}
__host__ __device__ __forceinline__ float ssrlcv::square(const float &a){
  return a*a;
}
__device__ __forceinline__ float ssrlcv::atomicMinFloat (float * addr, float value) {
  float old;
  old = (value >= 0) ? __int_as_float(atomicMin((int *)addr, __float_as_int(value))) :
    __uint_as_float(atomicMax((unsigned int *)addr, __float_as_uint(value)));
  return old;
}
__host__ __device__ __forceinline__ float ssrlcv::findSubPixelContributer(const float2 &loc, const int &width){
  return ((loc.y - 12)*(width - 24)) + (loc.x - 12);
}

//TODO change currentDist to type D

/*
matching
*/
//base matching kernels

//base matching kernels
//TODO block this out
__global__ void ssrlcv::disparityMatching(uint2 querySize, unsigned char* pixelsQuery, uint2 targetSize, unsigned char* pixelsTarget, float* fundamental, Match* matches, unsigned int maxDisparity, Direction direction){
  unsigned long blockId = blockIdx.y * gridDim.x + blockIdx.x;
  uint2 minimizedSize = {querySize.x-blockDim.x-1,querySize.y-blockDim.x-1};
  if(blockId < minimizedSize.x*minimizedSize.y){
    int2 loc = {blockId%minimizedSize.x + (blockDim.x/2),blockId/minimizedSize.x + (blockDim.y/2)};
    int2 threadLoc = {threadIdx.x - (blockDim.x/2),threadIdx.y - (blockDim.y/2)};
    __shared__ int3 matchInfo;
    __shared__ int currentDist;
    __shared__ float3 epipolar;
    __shared__ int2 searchLoc;
    __shared__ int stop;
    stop = maxDisparity; 
    int stride = 1;
    if(threadIdx.x + threadIdx.y == 0){
      matchInfo = {-1,-1,INT_MAX};
      currentDist = 0;
      epipolar.x = (fundamental[0]*loc.x) + (fundamental[1]*loc.y) + fundamental[2];
      epipolar.y = (fundamental[3]*loc.x) + (fundamental[4]*loc.y) + fundamental[5];
      epipolar.z = (fundamental[6]*loc.x) + (fundamental[7]*loc.y) + fundamental[8];
      if(direction == right){
        searchLoc.x = loc.x;
        stop -= querySize.x - ((int)maxDisparity + loc.x);
      }
      else if(direction == left){
        stride = -1;
        searchLoc.x = loc.x;
        stop += loc.x - (int)maxDisparity;
      }
      else{
        searchLoc.x = loc.x - ((int)maxDisparity/2);
        if(searchLoc.x < 0){
          searchLoc.x = 0;
        }
      }
      searchLoc.y = (int)floor(-1*((epipolar.x*searchLoc.x) + epipolar.z)/epipolar.y);
    }  
    __syncthreads();

    int threadPixel = pixelsQuery[(loc.y + threadLoc.y)*querySize.x + loc.x + threadLoc.x];
    for(int i = 0; i < stop; ++i){
      atomicAdd(&currentDist,abs(threadPixel-(int)pixelsTarget[(searchLoc.y+threadLoc.y)*targetSize.x + searchLoc.x + threadLoc.x]));
      __syncthreads();
      if(threadIdx.x + threadIdx.y == 0){
        if(currentDist < matchInfo.z){
          matchInfo = {searchLoc.x,searchLoc.y,currentDist};
        }
        searchLoc.x+=stride;
        searchLoc.y = (int)floor(-1*((epipolar.x*searchLoc.x) + epipolar.z)/epipolar.y);
        currentDist = 0;
      } 
      __syncthreads();
    }

    Match match;
    if(matchInfo.x == -1){
      match.invalid = true;
    }
    else{
      match.invalid = false;
      match.keyPoints[0].loc = {(float)loc.x + threadLoc.x,(float)loc.y + threadLoc.y};
      match.keyPoints[1].loc = {(float)matchInfo.x + threadLoc.x,(float)loc.y + threadLoc.y};
      match.keyPoints[0].parentId = 0;
      match.keyPoints[1].parentId = 1;
    }
    matches[(loc.y+threadLoc.y)*minimizedSize.x + loc.x + threadLoc.x] = match;
  }
}
__global__ void ssrlcv::disparityScanMatching(uint2 querySize, unsigned char* pixelsQuery, uint2 targetSize, unsigned char* pixelsTarget, Match* matches, unsigned int maxDisparity, Direction direction){
  unsigned long blockId = blockIdx.y * gridDim.x + blockIdx.x;
  uint2 minimizedSize = {querySize.x-blockDim.x-1,querySize.y-blockDim.x-1};
  if(blockId < minimizedSize.x*minimizedSize.y){
    int2 loc = {blockId%minimizedSize.x + (blockDim.x/2), blockId/minimizedSize.x + (blockDim.y/2)};
    int2 threadLoc = {threadIdx.x - (blockDim.x/2),threadIdx.y - (blockDim.y/2)};
    __shared__ int2 matchInfo;
    __shared__ int currentDist;
    __shared__ int searchX;
    __shared__ int stop;
    stop = maxDisparity;
    int stride = 1;
    if(threadIdx.x + threadIdx.y == 0){
      matchInfo = {-1,INT_MAX};
      currentDist = 0;
      if(direction == right){
        searchX = loc.x;
        if(stop + loc.x > targetSize.x){
          stop = targetSize.x - loc.x;
        }
      }
      else if(direction == left){
        stride = -1;
        searchX = loc.x;
        if(loc.x - stop < 0){
          stop = loc.x;
        }
      }
      else{
        searchX = loc.x - ((int)maxDisparity/2);
        if(searchX < 0){
          searchX = 0;
        }
      }
    }
    __syncthreads();
    int threadPixel = pixelsQuery[(loc.y + threadLoc.y)*querySize.x + loc.x + threadLoc.x];
    int indexHelper = (loc.y + threadLoc.y)*targetSize.x;
    for(int i = 0; i < stop; ++i){
      atomicAdd(&currentDist,abs(threadPixel - (int)pixelsTarget[indexHelper + searchX + threadLoc.x]));
      __syncthreads();
      if(threadIdx.x + threadIdx.y == 0){
        if(currentDist < matchInfo.y){
          matchInfo = {searchX,currentDist};
        }
        searchX+=stride;
        currentDist = 0;
      }
      __syncthreads();
    }
    
    Match match;
    if(matchInfo.x == -1){
      match.invalid = true;
    }
    else{
      match.invalid = false;
      match.keyPoints[0].loc = {(float)loc.x + threadLoc.x,(float)loc.y + threadLoc.y};
      match.keyPoints[1].loc = {(float)matchInfo.x + threadLoc.x,(float)loc.y + threadLoc.y};
      match.keyPoints[0].parentId = 0;
      match.keyPoints[1].parentId = 1;
    }
    matches[(loc.y+threadLoc.y)*minimizedSize.x + loc.x + threadLoc.x] = match;
  }
}


template<typename T>
__global__ void ssrlcv::getSeedMatchDistances(unsigned long numFeaturesQuery, Feature<T>* featuresQuery, unsigned long numSeedFeatures,
Feature<T>* seedFeatures, float* matchDistances){
  unsigned long blockId = blockIdx.y * gridDim.x + blockIdx.x;
  if(blockId < numFeaturesQuery){
    Feature<T> feature = featuresQuery[blockId];
    __shared__ float localDist[192];
    localDist[threadIdx.x] = FLT_MAX;
    __syncthreads();
    float currentDist = 0.0f;
    unsigned long numSeedFeatures_reg = numSeedFeatures;
    for(int f = threadIdx.x; f < numSeedFeatures_reg; f += 192){
      currentDist = feature.descriptor.distProtocol(seedFeatures[f].descriptor,localDist[threadIdx.x]);
      if(localDist[threadIdx.x] > currentDist){
        localDist[threadIdx.x] = currentDist;
      }
    }
    __syncthreads();
    if(threadIdx.x != 0) return;
    currentDist = FLT_MAX;
    for(int i = 0; i < 192; ++i){
      if(currentDist > localDist[i]){
        currentDist = localDist[i];
      }
    }
    matchDistances[blockId] = currentDist;
  }
}

template<typename T>
__global__ void ssrlcv::matchFeaturesBruteForce(unsigned int queryImageID, unsigned long numFeaturesQuery,
ssrlcv::Feature<T>* featuresQuery, unsigned int targetImageID, unsigned long numFeaturesTarget,
ssrlcv::Feature<T>* featuresTarget, Match* matches, float absoluteThreshold){
  unsigned long blockId = blockIdx.y * gridDim.x + blockIdx.x;
  if(blockId < numFeaturesQuery){
    Feature<T> feature = featuresQuery[blockId];
    __shared__ int localMatch[192];
    __shared__ float localDist[192];
    localMatch[threadIdx.x] = -1;
    localDist[threadIdx.x] = absoluteThreshold;
    __syncthreads();
    float currentDist = 0.0f;
    unsigned long numFeaturesTarget_register = numFeaturesTarget;
    for(int f = threadIdx.x; f < numFeaturesTarget_register; f += 192){
      currentDist = feature.descriptor.distProtocol(featuresTarget[f].descriptor,localDist[threadIdx.x]);
      if(localDist[threadIdx.x] > currentDist){
        localDist[threadIdx.x] = currentDist;
        localMatch[threadIdx.x] = f;
      }
    }
    __syncthreads();
    if(threadIdx.x != 0) return;
    currentDist = absoluteThreshold;
    int matchIndex = -1;
    for(int i = 0; i < 192; ++i){
      if(currentDist > localDist[i]){
        currentDist = localDist[i];
        matchIndex = localMatch[i];
      }
    }
    Match match;
    if(currentDist > absoluteThreshold){
      match.invalid = true;
    }
    else{
      match.invalid = false;
      match.keyPoints[0].loc = feature.loc;
      match.keyPoints[1].loc = featuresTarget[matchIndex].loc;
      match.keyPoints[0].parentId = queryImageID;
      match.keyPoints[1].parentId = targetImageID;
    }
    matches[blockId] = match;
  }
}
template<typename T>
__global__ void ssrlcv::matchFeaturesConstrained(unsigned int queryImageID, unsigned long numFeaturesQuery,
ssrlcv::Feature<T>* featuresQuery, unsigned int targetImageID, unsigned long numFeaturesTarget,
ssrlcv::Feature<T>* featuresTarget, Match* matches, float epsilon, float* fundamental, float absoluteThreshold){
  unsigned long blockId = blockIdx.y * gridDim.x + blockIdx.x;
  if(blockId < numFeaturesQuery){
    Feature<T> feature = featuresQuery[blockId];
    __shared__ int localMatch[192];
    __shared__ float localDist[192];
    localMatch[threadIdx.x] = -1;
    localDist[threadIdx.x] = absoluteThreshold;
    __syncthreads();
    float currentDist = 0.0f;
    unsigned long numFeaturesTarget_register = numFeaturesTarget;
    float3 epipolar = {0.0f,0.0f,0.0f};
    epipolar.x = (fundamental[0]*feature.loc.x) + (fundamental[1]*feature.loc.y) + fundamental[2];
    epipolar.y = (fundamental[3]*feature.loc.x) + (fundamental[4]*feature.loc.y) + fundamental[5];
    epipolar.z = (fundamental[6]*feature.loc.x) + (fundamental[7]*feature.loc.y) + fundamental[8];

    float p = 0.0f;

    Feature<T> currentFeature;
    float regEpsilon = epsilon;

    for(int f = threadIdx.x; f < numFeaturesTarget_register; f += 192){

      currentFeature = featuresTarget[f];
      //ax + by + c = 0
      p = -1*((epipolar.x*currentFeature.loc.x) + epipolar.z)/epipolar.y;
      if(abs(currentFeature.loc.y - p) > regEpsilon) continue;
      currentDist = feature.descriptor.distProtocol(currentFeature.descriptor,localDist[threadIdx.x]);
      if(localDist[threadIdx.x] > currentDist){
        localDist[threadIdx.x] = currentDist;
        localMatch[threadIdx.x] = f;
      }
    }
    __syncthreads();
    if(threadIdx.x != 0) return;
    currentDist = absoluteThreshold;
    int matchIndex = -1;
    for(int i = 0; i < 192; ++i){
      if(currentDist > localDist[i]){
        currentDist = localDist[i];
        matchIndex = localMatch[i];
      }
    }
    Match match;
    if(currentDist > absoluteThreshold){
      match.invalid = true;
    }
    else{
      match.invalid = false;
      match.keyPoints[0].loc = feature.loc;
      match.keyPoints[1].loc = featuresTarget[matchIndex].loc;
      match.keyPoints[0].parentId = queryImageID;
      match.keyPoints[1].parentId = targetImageID;
    }
    matches[blockId] = match;
  }
}
template<typename T>
__global__ void ssrlcv::matchFeaturesBruteForce(unsigned int queryImageID, unsigned long numFeaturesQuery,
ssrlcv::Feature<T>* featuresQuery, unsigned int targetImageID, unsigned long numFeaturesTarget,
ssrlcv::Feature<T>* featuresTarget, Match* matches, float* seedDistances, float relativeThreshold, float absoluteThreshold){
  unsigned long blockId = blockIdx.y * gridDim.x + blockIdx.x;
  if(blockId < numFeaturesQuery){
    Feature<T> feature = featuresQuery[blockId];
    __shared__ int localMatch[192];
    __shared__ float localDist[192];
    localMatch[threadIdx.x] = -1;
    float nearestSeed = seedDistances[blockId];
    localDist[threadIdx.x] = absoluteThreshold;
    __syncthreads();
    float currentDist = 0.0f;
    unsigned long numFeaturesTarget_register = numFeaturesTarget;
    for(int f = threadIdx.x; f < numFeaturesTarget_register; f += 192){
      currentDist = feature.descriptor.distProtocol(featuresTarget[f].descriptor,localDist[threadIdx.x]);
      if(localDist[threadIdx.x] > currentDist){
        localDist[threadIdx.x] = currentDist;
        localMatch[threadIdx.x] = f;
      }
    }
    __syncthreads();
    if(threadIdx.x != 0) return;
    currentDist = absoluteThreshold;
    int matchIndex = -1;
    for(int i = 0; i < 192; ++i){
      if(currentDist > localDist[i]){
        currentDist = localDist[i];
        matchIndex = localMatch[i];
      }
    }
    Match match;
    if(currentDist > absoluteThreshold || matchIndex == -1){
      match.invalid = true;
    }
    else{
      if(currentDist/nearestSeed > relativeThreshold){
        match.invalid = true;
      }
      else{
        match.invalid = false;
        match.keyPoints[0].loc = feature.loc;
        match.keyPoints[1].loc = featuresTarget[matchIndex].loc;
        match.keyPoints[0].parentId = queryImageID;
        match.keyPoints[1].parentId = targetImageID;
      }
    }
    matches[blockId] = match;
  }
}
template<typename T>
__global__ void ssrlcv::matchFeaturesConstrained(unsigned int queryImageID, unsigned long numFeaturesQuery,
ssrlcv::Feature<T>* featuresQuery, unsigned int targetImageID, unsigned long numFeaturesTarget,
ssrlcv::Feature<T>* featuresTarget, Match* matches, float epsilon, float* fundamental, float* seedDistances, 
float relativeThreshold, float absoluteThreshold){
  unsigned long blockId = blockIdx.y * gridDim.x + blockIdx.x;
  if(blockId < numFeaturesQuery){
    Feature<T> feature = featuresQuery[blockId];
    __shared__ int localMatch[192];
    __shared__ float localDist[192];
    localMatch[threadIdx.x] = -1;
    float nearestSeed = seedDistances[blockId];
    localDist[threadIdx.x] = absoluteThreshold;
    __syncthreads();
    float currentDist = 0.0f;
    unsigned long numFeaturesTarget_register = numFeaturesTarget;
    float3 epipolar = {0.0f,0.0f,0.0f};
    epipolar.x = (fundamental[0]*feature.loc.x) + (fundamental[1]*feature.loc.y) + fundamental[2];
    epipolar.y = (fundamental[3]*feature.loc.x) + (fundamental[4]*feature.loc.y) + fundamental[5];
    epipolar.z = (fundamental[6]*feature.loc.x) + (fundamental[7]*feature.loc.y) + fundamental[8];

    float p = 0.0f;

    Feature<T> currentFeature;
    float regEpsilon = epsilon;

    for(int f = threadIdx.x; f < numFeaturesTarget_register; f += 192){

      currentFeature = featuresTarget[f];
      //ax + by + c = 0
      p = -1*((epipolar.x*currentFeature.loc.x) + epipolar.z)/epipolar.y;
      if(abs(currentFeature.loc.y - p) > regEpsilon) continue;
      currentDist = feature.descriptor.distProtocol(currentFeature.descriptor,localDist[threadIdx.x]);
      if(localDist[threadIdx.x] > currentDist){
        localDist[threadIdx.x] = currentDist;
        localMatch[threadIdx.x] = f;
      }
    }
    __syncthreads();
    if(threadIdx.x != 0) return;
    currentDist = absoluteThreshold;
    int matchIndex = -1;
    for(int i = 0; i < 192; ++i){
      if(currentDist > localDist[i]){
        currentDist = localDist[i];
        matchIndex = localMatch[i];
      }
    }
    Match match;
    if(currentDist > absoluteThreshold || matchIndex == -1){
      match.invalid = true;
    }
    else{
      if(currentDist/nearestSeed > relativeThreshold){
        match.invalid = true;
      }
      else{
        match.invalid = false;
        match.keyPoints[0].loc = feature.loc;
        match.keyPoints[1].loc = featuresTarget[matchIndex].loc;
        match.keyPoints[0].parentId = queryImageID;
        match.keyPoints[1].parentId = targetImageID;
      }
    }
    matches[blockId] = match;
  }
}


template<typename T>
__global__ void ssrlcv::matchFeaturesBruteForce(unsigned int queryImageID, unsigned long numFeaturesQuery,
ssrlcv::Feature<T>* featuresQuery, unsigned int targetImageID, unsigned long numFeaturesTarget,
ssrlcv::Feature<T>* featuresTarget, DMatch* matches, float absoluteThreshold){
  unsigned long blockId = blockIdx.y * gridDim.x + blockIdx.x;
  if(blockId < numFeaturesQuery){
    Feature<T> feature = featuresQuery[blockId];
    __shared__ int localMatch[192];
    __shared__ float localDist[192];
    localMatch[threadIdx.x] = -1;
    localDist[threadIdx.x] = absoluteThreshold;
    __syncthreads();
    float currentDist = 0.0f;
    unsigned long numFeaturesTarget_register = numFeaturesTarget;
    for(int f = threadIdx.x; f < numFeaturesTarget_register; f += 192){
      currentDist = feature.descriptor.distProtocol(featuresTarget[f].descriptor,localDist[threadIdx.x]);
      if(localDist[threadIdx.x] > currentDist){
        localDist[threadIdx.x] = currentDist;
        localMatch[threadIdx.x] = f;
      }
    }
    __syncthreads();
    if(threadIdx.x != 0) return;
    currentDist = absoluteThreshold;
    int matchIndex = -1;
    for(int i = 0; i < 192; ++i){
      if(currentDist > localDist[i]){
        currentDist = localDist[i];
        matchIndex = localMatch[i];
      }
    }
    DMatch match;
    match.distance = currentDist;
    if(match.distance > absoluteThreshold){
      match.invalid = true;
    }
    else{
      match.invalid = false;
      match.keyPoints[0].loc = feature.loc;
      match.keyPoints[1].loc = featuresTarget[matchIndex].loc;
      match.keyPoints[0].parentId = queryImageID;
      match.keyPoints[1].parentId = targetImageID;
    }
    matches[blockId] = match;
  }
}
template<typename T>
__global__ void ssrlcv::matchFeaturesConstrained(unsigned int queryImageID, unsigned long numFeaturesQuery,
ssrlcv::Feature<T>* featuresQuery, unsigned int targetImageID, unsigned long numFeaturesTarget,
ssrlcv::Feature<T>* featuresTarget, DMatch* matches, float epsilon, float* fundamental, float absoluteThreshold){
  unsigned long blockId = blockIdx.y * gridDim.x + blockIdx.x;
  if(blockId < numFeaturesQuery){
    Feature<T> feature = featuresQuery[blockId];
    __shared__ int localMatch[192];
    __shared__ float localDist[192];
    localMatch[threadIdx.x] = -1;
    localDist[threadIdx.x] = absoluteThreshold;
    __syncthreads();
    float currentDist = 0.0f;
    unsigned long numFeaturesTarget_register = numFeaturesTarget;
    float3 epipolar = {0.0f,0.0f,0.0f};
    epipolar.x = (fundamental[0]*feature.loc.x) + (fundamental[1]*feature.loc.y) + fundamental[2];
    epipolar.y = (fundamental[3]*feature.loc.x) + (fundamental[4]*feature.loc.y) + fundamental[5];
    epipolar.z = (fundamental[6]*feature.loc.x) + (fundamental[7]*feature.loc.y) + fundamental[8];

    float p = 0.0f;

    Feature<T> currentFeature;
    float regEpsilon = epsilon;

    for(int f = threadIdx.x; f < numFeaturesTarget_register; f += 192){

      currentFeature = featuresTarget[f];
      //ax + by + c = 0
      p = -1*((epipolar.x*currentFeature.loc.x) + epipolar.z)/epipolar.y;
      if(abs(currentFeature.loc.y - p) > regEpsilon) continue;
      currentDist = feature.descriptor.distProtocol(currentFeature.descriptor,localDist[threadIdx.x]);
      if(localDist[threadIdx.x] > currentDist){
        localDist[threadIdx.x] = currentDist;
        localMatch[threadIdx.x] = f;
      }
    }
    __syncthreads();
    if(threadIdx.x != 0) return;
    currentDist = absoluteThreshold;
    int matchIndex = -1;
    for(int i = 0; i < 192; ++i){
      if(currentDist > localDist[i]){
        currentDist = localDist[i];
        matchIndex = localMatch[i];
      }
    }
    DMatch match;
    match.distance = currentDist;
    if(match.distance > absoluteThreshold){
      match.invalid = true;
    }
    else{
      match.invalid = false;
      match.keyPoints[0].loc = feature.loc;
      match.keyPoints[1].loc = featuresTarget[matchIndex].loc;
      match.keyPoints[0].parentId = queryImageID;
      match.keyPoints[1].parentId = targetImageID;
    }
    matches[blockId] = match;
  }
}
template<typename T>
__global__ void ssrlcv::matchFeaturesBruteForce(unsigned int queryImageID, unsigned long numFeaturesQuery,
ssrlcv::Feature<T>* featuresQuery, unsigned int targetImageID, unsigned long numFeaturesTarget,
ssrlcv::Feature<T>* featuresTarget, DMatch* matches, 
float* seedDistances, float relativeThreshold, float absoluteThreshold){
  unsigned long blockId = blockIdx.y * gridDim.x + blockIdx.x;
  if(blockId < numFeaturesQuery){
    Feature<T> feature = featuresQuery[blockId];
    __shared__ int localMatch[192];
    __shared__ float localDist[192];
    localMatch[threadIdx.x] = -1;
    float nearestSeed = seedDistances[blockId];
    localDist[threadIdx.x] = absoluteThreshold;
    __syncthreads();
    float currentDist = 0.0f;
    unsigned long numFeaturesTarget_register = numFeaturesTarget;
    for(int f = threadIdx.x; f < numFeaturesTarget_register; f += 192){
      currentDist = feature.descriptor.distProtocol(featuresTarget[f].descriptor,localDist[threadIdx.x]);
      if(localDist[threadIdx.x] > currentDist){
        localDist[threadIdx.x] = currentDist;
        localMatch[threadIdx.x] = f;
      }
    }
    __syncthreads();
    if(threadIdx.x != 0) return;
    currentDist = absoluteThreshold;
    int matchIndex = -1;
    for(int i = 0; i < 192; ++i){
      if(currentDist > localDist[i]){
        currentDist = localDist[i];
        matchIndex = localMatch[i];
      }
    }
    DMatch match;
    match.distance = currentDist;
    if(match.distance > absoluteThreshold || matchIndex == -1){
      match.invalid = true;
    }
    else{
      if(match.distance/nearestSeed > relativeThreshold*relativeThreshold){
        match.invalid = true;
      }
      else{
        match.invalid = false;
        match.keyPoints[0].loc = feature.loc;
        match.keyPoints[1].loc = featuresTarget[matchIndex].loc;
        match.keyPoints[0].parentId = queryImageID;
        match.keyPoints[1].parentId = targetImageID;
      }
    }
    matches[blockId] = match;
  }
}
template<typename T>
__global__ void ssrlcv::matchFeaturesConstrained(unsigned int queryImageID, unsigned long numFeaturesQuery,
ssrlcv::Feature<T>* featuresQuery, unsigned int targetImageID, unsigned long numFeaturesTarget,
ssrlcv::Feature<T>* featuresTarget, DMatch* matches, float epsilon, float* fundamental, 
float* seedDistances, float relativeThreshold, float absoluteThreshold){
  unsigned long blockId = blockIdx.y * gridDim.x + blockIdx.x;
  if(blockId < numFeaturesQuery){
    Feature<T> feature = featuresQuery[blockId];
    __shared__ int localMatch[192];
    __shared__ float localDist[192];
    localMatch[threadIdx.x] = -1;
    float nearestSeed = seedDistances[blockId];
    localDist[threadIdx.x] = absoluteThreshold;
    __syncthreads();
    float currentDist = 0.0f;
    unsigned long numFeaturesTarget_register = numFeaturesTarget;
    float3 epipolar = {0.0f,0.0f,0.0f};
    epipolar.x = (fundamental[0]*feature.loc.x) + (fundamental[1]*feature.loc.y) + fundamental[2];
    epipolar.y = (fundamental[3]*feature.loc.x) + (fundamental[4]*feature.loc.y) + fundamental[5];
    epipolar.z = (fundamental[6]*feature.loc.x) + (fundamental[7]*feature.loc.y) + fundamental[8];

    float p = 0.0f;

    Feature<T> currentFeature;
    float regEpsilon = epsilon;

    for(int f = threadIdx.x; f < numFeaturesTarget_register; f += 192){

      currentFeature = featuresTarget[f];
      //ax + by + c = 0
      p = -1*((epipolar.x*currentFeature.loc.x) + epipolar.z)/epipolar.y;
      if(abs(currentFeature.loc.y - p) > regEpsilon) continue;
      currentDist = feature.descriptor.distProtocol(currentFeature.descriptor,localDist[threadIdx.x]);
      if(localDist[threadIdx.x] > currentDist){
        localDist[threadIdx.x] = currentDist;
        localMatch[threadIdx.x] = f;
      }
    }
    __syncthreads();
    if(threadIdx.x != 0) return;
    currentDist = absoluteThreshold;
    int matchIndex = -1;
    for(int i = 0; i < 192; ++i){
      if(currentDist > localDist[i]){
        currentDist = localDist[i];
        matchIndex = localMatch[i];
      }
    }
    DMatch match;
    match.distance = currentDist;
    if(match.distance > absoluteThreshold || matchIndex == -1){
      match.invalid = true;
    }
    else{
      if(match.distance/nearestSeed > relativeThreshold*relativeThreshold){
        match.invalid = true;
      }
      else{
        match.invalid = false;
        match.keyPoints[0].loc = feature.loc;
        match.keyPoints[1].loc = featuresTarget[matchIndex].loc;
        match.keyPoints[0].parentId = queryImageID;
        match.keyPoints[1].parentId = targetImageID;
      }
    }
    matches[blockId] = match;
  }
}

template<typename T>
__global__ void ssrlcv::matchFeaturesBruteForce(unsigned int queryImageID, unsigned long numFeaturesQuery,
ssrlcv::Feature<T>* featuresQuery, unsigned int targetImageID, unsigned long numFeaturesTarget,
ssrlcv::Feature<T>* featuresTarget, ssrlcv::FeatureMatch<T>* matches, float absoluteThreshold){
  unsigned long blockId = blockIdx.y * gridDim.x + blockIdx.x;
  if(blockId < numFeaturesQuery){
    Feature<T> feature = featuresQuery[blockId];
    __shared__ int localMatch[192];
    __shared__ float localDist[192];
    localMatch[threadIdx.x] = -1;
    localDist[threadIdx.x] = absoluteThreshold;
    __syncthreads();
    float currentDist = 0.0f;
    unsigned long numFeaturesTarget_register = numFeaturesTarget;
    for(int f = threadIdx.x; f < numFeaturesTarget_register; f += 192){
      currentDist = feature.descriptor.distProtocol(featuresTarget[f].descriptor,localDist[threadIdx.x]);
      if(localDist[threadIdx.x] > currentDist){
        localDist[threadIdx.x] = currentDist;
        localMatch[threadIdx.x] = f;
      }
    }
    __syncthreads();
    if(threadIdx.x != 0) return;
    currentDist = absoluteThreshold;
    int matchIndex = -1;
    for(int i = 0; i < 192; ++i){
      if(currentDist > localDist[i]){
        currentDist = localDist[i];
        matchIndex = localMatch[i];
      }
    }
    FeatureMatch<T> match;    
    match.distance = currentDist;
    if(match.distance > absoluteThreshold){
      match.invalid = true;
    }
    else{
      match.invalid = false;
      match.descriptors[0] = feature.descriptor;
      match.descriptors[1] = featuresTarget[matchIndex].descriptor;
      match.keyPoints[0].loc = feature.loc;
      match.keyPoints[1].loc = featuresTarget[matchIndex].loc;
      match.keyPoints[0].parentId = queryImageID;
      match.keyPoints[1].parentId = targetImageID;
    }
    matches[blockId] = match;
  }
}
template<typename T>
__global__ void ssrlcv::matchFeaturesConstrained(unsigned int queryImageID, unsigned long numFeaturesQuery,
ssrlcv::Feature<T>* featuresQuery, unsigned int targetImageID, unsigned long numFeaturesTarget,
ssrlcv::Feature<T>* featuresTarget, ssrlcv::FeatureMatch<T>* matches, float epsilon, float* fundamental, float absoluteThreshold){
  unsigned long blockId = blockIdx.y * gridDim.x + blockIdx.x;
  if(blockId < numFeaturesQuery){
    Feature<T> feature = featuresQuery[blockId];
    __shared__ int localMatch[192];
    __shared__ float localDist[192];
    localMatch[threadIdx.x] = -1;
    localDist[threadIdx.x] = absoluteThreshold;
    __syncthreads();
    float currentDist = 0.0f;
    unsigned long numFeaturesTarget_register = numFeaturesTarget;
    float3 epipolar = {0.0f,0.0f,0.0f};
    epipolar.x = (fundamental[0]*feature.loc.x) + (fundamental[1]*feature.loc.y) + fundamental[2];
    epipolar.y = (fundamental[3]*feature.loc.x) + (fundamental[4]*feature.loc.y) + fundamental[5];
    epipolar.z = (fundamental[6]*feature.loc.x) + (fundamental[7]*feature.loc.y) + fundamental[8];

    float p = 0.0f;

    Feature<T> currentFeature;
    float regEpsilon = epsilon;

    for(int f = threadIdx.x; f < numFeaturesTarget_register; f += 192){

      currentFeature = featuresTarget[f];
      //ax + by + c = 0
      p = -1*((epipolar.x*currentFeature.loc.x) + epipolar.z)/epipolar.y;
      if(abs(currentFeature.loc.y - p) > regEpsilon) continue;
      currentDist = feature.descriptor.distProtocol(currentFeature.descriptor,localDist[threadIdx.x]);
      if(localDist[threadIdx.x] > currentDist){
        localDist[threadIdx.x] = currentDist;
        localMatch[threadIdx.x] = f;
      }
    }
    __syncthreads();
    if(threadIdx.x != 0) return;
    currentDist = absoluteThreshold;
    int matchIndex = -1;
    for(int i = 0; i < 192; ++i){
      if(currentDist > localDist[i]){
        currentDist = localDist[i];
        matchIndex = localMatch[i];
      }
    }
    FeatureMatch<T> match;    
    match.distance = currentDist;
    if(match.distance > absoluteThreshold){
      match.invalid = true;
    }
    else{
      match.invalid = false;
      match.descriptors[0] = feature.descriptor;
      match.descriptors[1] = featuresTarget[matchIndex].descriptor;
      match.keyPoints[0].loc = feature.loc;
      match.keyPoints[1].loc = featuresTarget[matchIndex].loc;
      match.keyPoints[0].parentId = queryImageID;
      match.keyPoints[1].parentId = targetImageID;
    }
    matches[blockId] = match;
  }
}
template<typename T>
__global__ void ssrlcv::matchFeaturesBruteForce(unsigned int queryImageID, unsigned long numFeaturesQuery,
ssrlcv::Feature<T>* featuresQuery, unsigned int targetImageID, unsigned long numFeaturesTarget,
ssrlcv::Feature<T>* featuresTarget, ssrlcv::FeatureMatch<T>* matches,
float* seedDistances, float relativeThreshold, float absoluteThreshold){
  unsigned long blockId = blockIdx.y * gridDim.x + blockIdx.x;
  if(blockId < numFeaturesQuery){
    Feature<T> feature = featuresQuery[blockId];
    __shared__ int localMatch[192];
    __shared__ float localDist[192];
    localMatch[threadIdx.x] = -1;
    float nearestSeed = seedDistances[blockId];
    localDist[threadIdx.x] = absoluteThreshold;
    __syncthreads();
    float currentDist = 0.0f;
    unsigned long numFeaturesTarget_register = numFeaturesTarget;
    for(int f = threadIdx.x; f < numFeaturesTarget_register; f += 192){
      currentDist = feature.descriptor.distProtocol(featuresTarget[f].descriptor,localDist[threadIdx.x]);
      if(localDist[threadIdx.x] > currentDist){
        localDist[threadIdx.x] = currentDist;
        localMatch[threadIdx.x] = f;
      }
    }
    __syncthreads();
    if(threadIdx.x != 0) return;
    currentDist = absoluteThreshold;
    int matchIndex = -1;
    for(int i = 0; i < 192; ++i){
      if(currentDist > localDist[i]){
        currentDist = localDist[i];
        matchIndex = localMatch[i];
      }
    }
    FeatureMatch<T> match;    
    match.distance = currentDist;
    if(match.distance > absoluteThreshold || matchIndex == -1){
      match.invalid = true;
    }
    else{
      if(match.distance/nearestSeed > relativeThreshold*relativeThreshold){
        match.invalid = true;
      }
      else{
        match.invalid = false;
        match.descriptors[0] = feature.descriptor;
        match.descriptors[1] = featuresTarget[matchIndex].descriptor;
        match.keyPoints[0].loc = feature.loc;
        match.keyPoints[1].loc = featuresTarget[matchIndex].loc;
        match.keyPoints[0].parentId = queryImageID;
        match.keyPoints[1].parentId = targetImageID;
      }
    }
    matches[blockId] = match;
  }
}
template<typename T>
__global__ void ssrlcv::matchFeaturesConstrained(unsigned int queryImageID, unsigned long numFeaturesQuery,
ssrlcv::Feature<T>* featuresQuery, unsigned int targetImageID, unsigned long numFeaturesTarget,
ssrlcv::Feature<T>* featuresTarget, ssrlcv::FeatureMatch<T>* matches, float epsilon, float* fundamental,
float* seedDistances, float relativeThreshold, float absoluteThreshold){
  unsigned long blockId = blockIdx.y * gridDim.x + blockIdx.x;
  if(blockId < numFeaturesQuery){
    Feature<T> feature = featuresQuery[blockId];
    __shared__ int localMatch[192];
    __shared__ float localDist[192];
    localMatch[threadIdx.x] = -1;
    float nearestSeed = seedDistances[blockId];
    localDist[threadIdx.x] = absoluteThreshold;
    __syncthreads();
    float currentDist = 0.0f;
    unsigned long numFeaturesTarget_register = numFeaturesTarget;
    float3 epipolar = {0.0f,0.0f,0.0f};
    epipolar.x = (fundamental[0]*feature.loc.x) + (fundamental[1]*feature.loc.y) + fundamental[2];
    epipolar.y = (fundamental[3]*feature.loc.x) + (fundamental[4]*feature.loc.y) + fundamental[5];
    epipolar.z = (fundamental[6]*feature.loc.x) + (fundamental[7]*feature.loc.y) + fundamental[8];

    float p = 0.0f;

    Feature<T> currentFeature;
    float regEpsilon = epsilon;

    for(int f = threadIdx.x; f < numFeaturesTarget_register; f += 192){

      currentFeature = featuresTarget[f];
      //ax + by + c = 0
      p = -1*((epipolar.x*currentFeature.loc.x) + epipolar.z)/epipolar.y;
      if(abs(currentFeature.loc.y - p) > regEpsilon) continue;
      currentDist = feature.descriptor.distProtocol(currentFeature.descriptor,localDist[threadIdx.x]);
      if(localDist[threadIdx.x] > currentDist){
        localDist[threadIdx.x] = currentDist;
        localMatch[threadIdx.x] = f;
      }
    }
    __syncthreads();
    if(threadIdx.x != 0) return;
    currentDist = absoluteThreshold;
    int matchIndex = -1;
    for(int i = 0; i < 192; ++i){
      if(currentDist > localDist[i]){
        currentDist = localDist[i];
        matchIndex = localMatch[i];
      }
    }
    FeatureMatch<T> match;    
    match.distance = currentDist;
    if(match.distance > absoluteThreshold || matchIndex == -1){
      match.invalid = true;
    }
    else{
      if(match.distance/nearestSeed > relativeThreshold*relativeThreshold){
        match.invalid = true;
      }
      else{
        match.invalid = false;
        match.descriptors[0] = feature.descriptor;
        match.descriptors[1] = featuresTarget[matchIndex].descriptor;
        match.keyPoints[0].loc = feature.loc;
        match.keyPoints[1].loc = featuresTarget[matchIndex].loc;
        match.keyPoints[0].parentId = queryImageID;
        match.keyPoints[1].parentId = targetImageID;
      }
    }
    matches[blockId] = match;
  }
}

//utility kernels
__global__ void ssrlcv::convertMatchToRaw(unsigned long numMatches, ssrlcv::Match* rawMatches, ssrlcv::DMatch* matches){
  unsigned long globalID = (blockIdx.y* gridDim.x+ blockIdx.x)*blockDim.x + threadIdx.x;
  if(globalID < numMatches){
    rawMatches[globalID] = Match(matches[globalID]);
  }
}
template<typename T>
__global__ void ssrlcv::convertMatchToRaw(unsigned long numMatches, ssrlcv::Match* rawMatches, ssrlcv::FeatureMatch<T>* matches){
  unsigned long globalID = (blockIdx.y* gridDim.x+ blockIdx.x)*blockDim.x + threadIdx.x;
  if(globalID < numMatches){
    rawMatches[globalID] = Match(matches[globalID]);
  }
}
