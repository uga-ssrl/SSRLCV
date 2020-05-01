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
void ssrlcv::MatchFactory<T>::validateMatches(ssrlcv::Unity<uint2_pair>* matches){
  MemoryState origin = matches->getMemoryState();
  if(origin != gpu) matches->setMemoryState(gpu);
  
  thrust::device_ptr<uint2_pair> needsValidating(matches->device);
  thrust::device_ptr<uint2_pair> new_end = thrust::remove_if(needsValidating,needsValidating+matches->size(),validate());
  cudaDeviceSynchronize();
  CudaCheckError();
  int numMatchesLeft = new_end - needsValidating;
  if(numMatchesLeft == 0){
    std::cout<<"No valid matches found"<<"\n";
    delete matches;
    matches = nullptr;
    return;
  }
  

  printf("%d valid matches found out of %lu original matches\n",numMatchesLeft,matches->size());

  uint2_pair* validatedMatches_device = nullptr;
  CudaSafeCall(cudaMalloc((void**)&validatedMatches_device,numMatchesLeft*sizeof(uint2_pair)));
  CudaSafeCall(cudaMemcpy(validatedMatches_device,matches->device,numMatchesLeft*sizeof(uint2_pair),cudaMemcpyDeviceToDevice));

  matches->setData(validatedMatches_device,numMatchesLeft,gpu);

  if(origin != gpu) matches->setMemoryState(origin);
}
template<typename T>
void ssrlcv::MatchFactory<T>::validateMatches(ssrlcv::Unity<ssrlcv::Match>* matches){
  MemoryState origin = matches->getMemoryState();
  if(origin != gpu) matches->setMemoryState(gpu);
  
  thrust::device_ptr<Match> needsValidating(matches->device);
  thrust::device_ptr<Match> new_end = thrust::remove_if(needsValidating,needsValidating+matches->size(),validate());
  cudaDeviceSynchronize();
  CudaCheckError();
  int numMatchesLeft = new_end - needsValidating;
  if(numMatchesLeft == 0){
    std::cout<<"No valid matches found"<<"\n";
    delete matches;
    matches = nullptr;
    return;
  }
  

  printf("%d valid matches found out of %lu original matches\n",numMatchesLeft,matches->size());

  Match* validatedMatches_device = nullptr;
  CudaSafeCall(cudaMalloc((void**)&validatedMatches_device,numMatchesLeft*sizeof(Match)));
  CudaSafeCall(cudaMemcpy(validatedMatches_device,matches->device,numMatchesLeft*sizeof(Match),cudaMemcpyDeviceToDevice));

  matches->setData(validatedMatches_device,numMatchesLeft,gpu);

  if(origin != gpu) matches->setMemoryState(origin);
}
template<typename T>
void ssrlcv::MatchFactory<T>::validateMatches(ssrlcv::Unity<ssrlcv::DMatch>* matches){
  MemoryState origin = matches->getMemoryState();
  if(origin != gpu) matches->setMemoryState(gpu);
  

  thrust::device_ptr<DMatch> needsValidating(matches->device);
  thrust::device_ptr<DMatch> new_end = thrust::remove_if(needsValidating,needsValidating+matches->size(),validate());
  cudaDeviceSynchronize();
  CudaCheckError();
  int numMatchesLeft = new_end - needsValidating;
  if(numMatchesLeft == 0){
    std::cout<<"No valid matches found"<<"\n";
    delete matches;
    matches = nullptr;
    return;
  }
  

  printf("%d valid matches found out of %lu original matches\n",numMatchesLeft,matches->size());

  DMatch* validatedMatches_device = nullptr;
  CudaSafeCall(cudaMalloc((void**)&validatedMatches_device,numMatchesLeft*sizeof(DMatch)));
  CudaSafeCall(cudaMemcpy(validatedMatches_device,matches->device,numMatchesLeft*sizeof(DMatch),cudaMemcpyDeviceToDevice));

  matches->setData(validatedMatches_device,numMatchesLeft,gpu);

  if(origin != gpu) matches->setMemoryState(origin);
}
template<typename T>
void ssrlcv::MatchFactory<T>::validateMatches(ssrlcv::Unity<ssrlcv::FeatureMatch<T>>* matches){
  MemoryState origin = matches->getMemoryState();
  if(origin != gpu) matches->setMemoryState(gpu);


  thrust::device_ptr<FeatureMatch<T>> needsValidating(matches->device);
  thrust::device_ptr<FeatureMatch<T>> new_end = thrust::remove_if(needsValidating,needsValidating+matches->size(),validate());
  cudaDeviceSynchronize();
  CudaCheckError();
  int numMatchesLeft = new_end - needsValidating;
  if(numMatchesLeft == 0){
    std::cout<<"No valid matches found"<<"\n";
    delete matches;
    matches = nullptr;
    return;
  }
  

  printf("%d valid matches found out of %lu original matches\n",numMatchesLeft,matches->size());

  FeatureMatch<T>* validatedMatches_device = nullptr;
  CudaSafeCall(cudaMalloc((void**)&validatedMatches_device,numMatchesLeft*sizeof(FeatureMatch<T>)));
  CudaSafeCall(cudaMemcpy(validatedMatches_device,matches->device,numMatchesLeft*sizeof(FeatureMatch<T>),cudaMemcpyDeviceToDevice));

  matches->setData(validatedMatches_device,numMatchesLeft,gpu);

  if(origin != gpu) matches->setMemoryState(origin);

}
template<typename T>
void ssrlcv::MatchFactory<T>::refineMatches(ssrlcv::Unity<ssrlcv::DMatch>* matches, float threshold){
  if(threshold == 0.0f){
    std::cout<<"ERROR illegal value used for threshold: 0.0"<<"\n";
    exit(-1);
  }
  MemoryState origin = matches->getMemoryState();
  if(origin != gpu) matches->setMemoryState(gpu);


  thrust::device_ptr<DMatch> needsCompacting(matches->device);
  thrust::device_ptr<DMatch> end = thrust::remove_if(needsCompacting, needsCompacting + matches->size(), match_dist_thresholder(threshold));
  unsigned int numElementsBelowThreshold = end - needsCompacting;
  if(numElementsBelowThreshold == 0){
    delete matches;
    matches = nullptr;
    return;
  }

  printf("%lu matches have been refined to %u matches using a cutoff of %f\n",matches->size(),numElementsBelowThreshold,threshold);

  DMatch* compactedMatches_device = nullptr;
  CudaSafeCall(cudaMalloc((void**)&compactedMatches_device,numElementsBelowThreshold*sizeof(DMatch)));
  CudaSafeCall(cudaMemcpy(compactedMatches_device,matches->device,numElementsBelowThreshold*sizeof(DMatch),cudaMemcpyDeviceToDevice));

  matches->setData(compactedMatches_device,numElementsBelowThreshold,gpu);

  if(origin != gpu) matches->setMemoryState(origin);
}
template<typename T>
void ssrlcv::MatchFactory<T>::refineMatches(ssrlcv::Unity<ssrlcv::FeatureMatch<T>>* matches, float threshold){
  if(threshold == 0.0f){
    std::cout<<"ERROR illegal value used for cutoff ratio: 0.0"<<"\n";
    exit(-1);
  }
  MemoryState origin = matches->getMemoryState();
  if(origin != gpu) matches->setMemoryState(gpu);

  thrust::device_ptr<FeatureMatch<T>> needsCompacting(matches->device);
  thrust::device_ptr<FeatureMatch<T>> end = thrust::remove_if(needsCompacting, needsCompacting + matches->size(), match_dist_thresholder(threshold));
  unsigned int numElementsBelowThreshold = end - needsCompacting;
  if(numElementsBelowThreshold == 0){
    delete matches;
    matches = nullptr;
    return;
  }

  printf("%lu matches have been refined to %u matches using a cutoff of %f\n",matches->size(),numElementsBelowThreshold,threshold);

  FeatureMatch<T>* compactedMatches_device = nullptr;
  CudaSafeCall(cudaMalloc((void**)&compactedMatches_device,numElementsBelowThreshold*sizeof(FeatureMatch<T>)));
  CudaSafeCall(cudaMemcpy(compactedMatches_device,matches->device,numElementsBelowThreshold*sizeof(FeatureMatch<T>),cudaMemcpyDeviceToDevice));

  matches->setData(compactedMatches_device,numElementsBelowThreshold,gpu);

  if(origin != gpu) matches->setMemoryState(origin);
}
template<typename T>
void ssrlcv::MatchFactory<T>::sortMatches(Unity<DMatch>* matches){
  if(matches->getFore() == gpu || matches->getFore() == both){
    thrust::device_ptr<DMatch> toSort(matches->device);
    thrust::sort(toSort, toSort + matches->size(),match_dist_comparator());
    matches->setFore(gpu);
    if(matches->getMemoryState() == both) matches->transferMemoryTo(cpu);
  }
  else if(matches->getFore() == cpu){
    unsigned long len = matches->size();
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
    if(matches->getMemoryState() == both) matches->transferMemoryTo(gpu);
  }
  else{
    logger.err<<"ERROR cannot perform sortMatches with matches->getMemoryState() = "<<std::to_string(matches->getMemoryState())<<"\n";
    exit(-1);
  }
}
template<typename T>
void ssrlcv::MatchFactory<T>::sortMatches(Unity<FeatureMatch<T>>* matches){
  if(matches->getFore() == gpu || matches->getFore() == both){
    thrust::device_ptr<FeatureMatch<T>> toSort(matches->device);
    thrust::sort(toSort, toSort + matches->size(),match_dist_comparator());
    matches->setFore(gpu);
    if(matches->getMemoryState() == both) matches->transferMemoryTo(cpu);
  }
  else if(matches->getFore() == cpu){
    unsigned long len = matches->size();
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
    if(matches->getMemoryState() == both) matches->transferMemoryTo(gpu);
  }
  else{
    logger.err<<"ERROR cannot perform sortMatches with matches->getMemoryState() = "<<std::to_string(matches->getMemoryState())<<"\n";
    exit(-1);
  }
}

template<typename T>
ssrlcv::Unity<ssrlcv::Match>* ssrlcv::MatchFactory<T>::getRawMatches(Unity<DMatch>* matches){
  if(matches->getMemoryState() == gpu || matches->getFore() == gpu){
    Match* rawMatches_device = nullptr;
    CudaSafeCall(cudaMalloc((void**)&rawMatches_device, matches->size()*sizeof(Match)));
    dim3 grid = {1,1,1};
    dim3 block = {1,1,1};
    void (*fp)(unsigned long, Match*, DMatch*) = &convertMatchToRaw;
    getFlatGridBlock(matches->size(),grid,block,fp);
    convertMatchToRaw<<<grid,block>>>(matches->size(),rawMatches_device,matches->device);
    cudaDeviceSynchronize();
    CudaCheckError();
    return new Unity<Match>(rawMatches_device,matches->size(),gpu);
  }
  else{
    Match* rawMatches_host = new Match[matches->size()];
    for(int i = 0; i < matches->size(); ++i){
      for(int f = 0; f < 2; ++f){
        rawMatches_host[i] = Match(matches->host[i]);
      }
    }
    return new Unity<Match>(rawMatches_host, matches->size(), cpu);
  }
}
template<typename T>
ssrlcv::Unity<ssrlcv::Match>* ssrlcv::MatchFactory<T>::getRawMatches(Unity<FeatureMatch<T>>* matches){
  if(matches->getMemoryState() == gpu || matches->getFore() == gpu){
    Match* rawMatches_device = nullptr;
    CudaSafeCall(cudaMalloc((void**)&rawMatches_device, matches->size()*sizeof(Match)));
    dim3 grid = {1,1,1};
    dim3 block = {1,1,1};
    void (*fp)(unsigned long, Match*, FeatureMatch<T>*) = &convertMatchToRaw;
    getFlatGridBlock(matches->size(),grid,block,fp);
    convertMatchToRaw<T><<<grid,block>>>(matches->size(),rawMatches_device,matches->device);
    cudaDeviceSynchronize();
    CudaCheckError();
    return new Unity<Match>(rawMatches_device,matches->size(),gpu);
  }
  else{
    Match* rawMatches_host = new Match[matches->size()];
    for(int i = 0; i < matches->size(); ++i){
      for(int f = 0; f < 2; ++f){
        rawMatches_host[i] = Match(matches->host[i]);
      }
    }
    return new Unity<Match>(rawMatches_host, matches->size(), cpu);
  }
}

template<typename T>
ssrlcv::Unity<float>* ssrlcv::MatchFactory<T>::getSeedDistances(Unity<Feature<T>>* features){
  MemoryState origin = features->getMemoryState();

  if(this->seedFeatures->getMemoryState() != gpu) this->seedFeatures->setMemoryState(gpu);
  if(origin != gpu) features->setMemoryState(gpu);

  unsigned int numPossibleMatches = features->size();

  Unity<float>* matchDistances = new Unity<float>(nullptr, numPossibleMatches,gpu);

  dim3 grid = {1,1,1};
  dim3 block = {32,1,1};//IMPROVE
  getGrid(matchDistances->size(),grid);

  clock_t timer = clock();

  getSeedMatchDistances<T><<<grid, block>>>(features->size(),features->device,this->seedFeatures->size(),
    this->seedFeatures->device,matchDistances->device);

  cudaDeviceSynchronize();
  CudaCheckError();

  printf("seed match distances computed in %f seconds.\n\n",((float) clock() -  timer)/CLOCKS_PER_SEC);

  if(origin != gpu) features->setMemoryState(origin);
  
  return matchDistances;
}

template<typename T>
ssrlcv::Unity<ssrlcv::Match>* ssrlcv::MatchFactory<T>::generateMatches(Image* query, Unity<Feature<T>>* queryFeatures, Image* target, Unity<Feature<T>>* targetFeatures, Unity<float>* seedDistances){
  MemoryState origin[2] = {queryFeatures->getMemoryState(), targetFeatures->getMemoryState()};

  if(origin[0] != gpu) queryFeatures->setMemoryState(gpu);
  if(origin[1] != gpu) targetFeatures->setMemoryState(gpu);

  unsigned int numPossibleMatches = queryFeatures->size();

  Match* matches_device = nullptr;
  CudaSafeCall(cudaMalloc((void**)&matches_device, numPossibleMatches*sizeof(Match)));
  Unity<Match>* matches = new Unity<Match>(matches_device, numPossibleMatches, gpu);

  dim3 grid = {1,1,1};
  dim3 block = {32,1,1};//IMPROVE
  getGrid(matches->size(),grid);

  clock_t timer = clock();

  if(seedDistances == nullptr){
    matchFeaturesBruteForce<T><<<grid, block>>>(query->id, queryFeatures->size(), queryFeatures->device,
    target->id, targetFeatures->size(), targetFeatures->device, matches->device,this->absoluteThreshold);
  }
  else if(seedDistances->size() != queryFeatures->size()){
    logger.err<<"ERROR: seedDistances should have come from matching a seed image to queryFeatures"<<"\n";
    exit(-1);
  }
  else{
    MemoryState seedOrigin = seedDistances->getMemoryState();
    if(seedOrigin != gpu) seedDistances->setMemoryState(gpu);
    matchFeaturesBruteForce<T><<<grid, block>>>(query->id, queryFeatures->size(), queryFeatures->device,
    target->id, targetFeatures->size(), targetFeatures->device, matches->device,seedDistances->device,
    this->relativeThreshold,this->absoluteThreshold);
    if(seedOrigin != gpu) seedDistances->setMemoryState(seedOrigin);
  }
  
  cudaDeviceSynchronize();
  CudaCheckError();

  this->validateMatches(matches);

  printf("done in %f seconds.\n\n",((float) clock() -  timer)/CLOCKS_PER_SEC);

  if(origin[0] != gpu) queryFeatures->setMemoryState(origin[0]);
  if(origin[1] != gpu) targetFeatures->setMemoryState(origin[1]);

  return matches;
}
template<typename T>
ssrlcv::Unity<ssrlcv::Match>* ssrlcv::MatchFactory<T>::generateMatchesConstrained(Image* query, Unity<Feature<T>>* queryFeatures, Image* target, Unity<Feature<T>>* targetFeatures, float epsilon, float fundamental[3][3], Unity<float>* seedDistances){
  MemoryState origin[2] = {queryFeatures->getMemoryState(), targetFeatures->getMemoryState()};

  if(origin[0] != gpu) queryFeatures->setMemoryState(gpu);
  if(origin[1] != gpu) targetFeatures->setMemoryState(gpu);

  unsigned int numPossibleMatches = queryFeatures->size();

  Match* matches_device = nullptr;
  CudaSafeCall(cudaMalloc((void**)&matches_device, numPossibleMatches*sizeof(Match)));

  Unity<Match>* matches = new Unity<Match>(matches_device, numPossibleMatches, gpu);

  dim3 grid = {1,1,1};
  dim3 block = {32,1,1};//IMPROVE
  getGrid(matches->size(),grid);

  float* fundamental_device = nullptr;
  CudaSafeCall(cudaMalloc((void**)&fundamental_device,9*sizeof(float)));
  CudaSafeCall(cudaMemcpy(fundamental_device,fundamental,9*sizeof(float),cudaMemcpyHostToDevice));

  clock_t timer = clock();

  if(seedDistances == nullptr){
    matchFeaturesConstrained<T><<<grid, block>>>(query->id, queryFeatures->size(), queryFeatures->device,
    target->id, targetFeatures->size(), targetFeatures->device, matches->device, epsilon,fundamental_device,this->absoluteThreshold);
  }
  else if(seedDistances->size() != queryFeatures->size()){
    logger.err<<"ERROR: seedDistances should have come from matching a seed image to queryFeatures"<<"\n";
    exit(-1);
  }
  else{
    MemoryState seedOrigin = seedDistances->getMemoryState();
    if(seedOrigin != gpu) seedDistances->setMemoryState(gpu);
    matchFeaturesConstrained<T><<<grid, block>>>(query->id, queryFeatures->size(), queryFeatures->device,
    target->id, targetFeatures->size(), targetFeatures->device, matches->device,epsilon,fundamental_device,seedDistances->device,
    this->relativeThreshold,this->absoluteThreshold);
    if(seedOrigin != gpu) seedDistances->setMemoryState(seedOrigin);
  }

  cudaDeviceSynchronize();
  CudaCheckError();

  CudaSafeCall(cudaFree(fundamental_device));

  this->validateMatches(matches);

  printf("done in %f seconds.\n\n",((float) clock() -  timer)/CLOCKS_PER_SEC);

  if(origin[0] != gpu) queryFeatures->setMemoryState(origin[0]);
  if(origin[1] != gpu) targetFeatures->setMemoryState(origin[1]);

  return matches;
}


template<typename T>
ssrlcv::Unity<ssrlcv::DMatch>*ssrlcv::MatchFactory<T>:: generateDistanceMatches(Image* query, Unity<Feature<T>>* queryFeatures, Image* target, Unity<Feature<T>>* targetFeatures, Unity<float>* seedDistances){
  MemoryState origin[2] = {queryFeatures->getMemoryState(), targetFeatures->getMemoryState()};

  if(origin[0] != gpu) queryFeatures->setMemoryState(gpu);
  if(origin[1] != gpu) targetFeatures->setMemoryState(gpu);

  unsigned int numPossibleMatches = queryFeatures->size();

  Unity<DMatch>* matches = new Unity<DMatch>(nullptr, numPossibleMatches, gpu);

  dim3 grid = {1,1,1};
  dim3 block = {32,1,1};//IMPROVE
  getGrid(matches->size(),grid);

  clock_t timer = clock();

  if(seedDistances == nullptr){
    matchFeaturesBruteForce<T><<<grid, block>>>(query->id, queryFeatures->size(), queryFeatures->device,
    target->id, targetFeatures->size(), targetFeatures->device, matches->device,this->absoluteThreshold);
  }
  else if(seedDistances->size() != queryFeatures->size()){
    logger.err<<"ERROR: seedDistances should have come from matching a seed image to queryFeatures"<<"\n";
    exit(-1);
  }
  else{
    MemoryState seedOrigin = seedDistances->getMemoryState();
    if(seedOrigin != gpu) seedDistances->setMemoryState(gpu);
    matchFeaturesBruteForce<T><<<grid, block>>>(query->id, queryFeatures->size(), queryFeatures->device,
    target->id, targetFeatures->size(), targetFeatures->device, matches->device,seedDistances->device,
    this->relativeThreshold,this->absoluteThreshold);
    if(seedOrigin != gpu) seedDistances->setMemoryState(seedOrigin);
  }
  cudaDeviceSynchronize();
  CudaCheckError();

  this->validateMatches(matches);

  printf("done in %f seconds.\n\n",((float) clock() -  timer)/CLOCKS_PER_SEC);

  if(origin[0] != gpu) queryFeatures->setMemoryState(origin[0]);
  if(origin[1] != gpu) targetFeatures->setMemoryState(origin[1]);

  return matches;
}
template<typename T>
ssrlcv::Unity<ssrlcv::DMatch>*ssrlcv::MatchFactory<T>:: generateDistanceMatchesConstrained(Image* query, Unity<Feature<T>>* queryFeatures, Image* target, Unity<Feature<T>>* targetFeatures, float epsilon, float fundamental[3][3], Unity<float>* seedDistances){
  MemoryState origin[2] = {queryFeatures->getMemoryState(), targetFeatures->getMemoryState()};

  if(origin[0] != gpu) queryFeatures->setMemoryState(gpu);
  if(origin[1] != gpu) targetFeatures->setMemoryState(gpu);

  unsigned int numPossibleMatches = queryFeatures->size();

  DMatch* matches_device = nullptr;
  CudaSafeCall(cudaMalloc((void**)&matches_device, numPossibleMatches*sizeof(DMatch)));

  Unity<DMatch>* matches = new Unity<DMatch>(matches_device, numPossibleMatches, gpu);

  dim3 grid = {1,1,1};
  dim3 block = {32,1,1};//IMPROVE
  getGrid(matches->size(),grid);

  float* fundamental_device = nullptr;
  CudaSafeCall(cudaMalloc((void**)&fundamental_device,9*sizeof(float)));
  CudaSafeCall(cudaMemcpy(fundamental_device,fundamental,9*sizeof(float),cudaMemcpyHostToDevice));

  clock_t timer = clock();

  if(seedDistances == nullptr){
    matchFeaturesConstrained<T><<<grid, block>>>(query->id, queryFeatures->size(), queryFeatures->device,
    target->id, targetFeatures->size(), targetFeatures->device, matches->device, epsilon, fundamental_device,this->absoluteThreshold);
  }
  else if(seedDistances->size() != queryFeatures->size()){
    logger.err<<"ERROR: seedDistances should have come from matching a seed image to queryFeatures"<<"\n";
    exit(-1);
  }
  else{
    MemoryState seedOrigin = seedDistances->getMemoryState();
    if(seedOrigin != gpu) seedDistances->setMemoryState(gpu);
    matchFeaturesConstrained<T><<<grid, block>>>(query->id, queryFeatures->size(), queryFeatures->device,
    target->id, targetFeatures->size(), targetFeatures->device, matches->device, epsilon, fundamental_device,seedDistances->device,
    this->relativeThreshold,this->absoluteThreshold);
    if(seedOrigin != gpu) seedDistances->setMemoryState(seedOrigin);
  }
  cudaDeviceSynchronize();
  CudaCheckError();

  CudaSafeCall(cudaFree(fundamental_device));

  this->validateMatches(matches);

  printf("done in %f seconds.\n\n",((float) clock() -  timer)/CLOCKS_PER_SEC);

  if(origin[0] != gpu) queryFeatures->setMemoryState(origin[0]);
  if(origin[1] != gpu) targetFeatures->setMemoryState(origin[1]);

  return matches;
}


template<typename T>
ssrlcv::Unity<ssrlcv::FeatureMatch<T>>* ssrlcv::MatchFactory<T>::generateFeatureMatches(ssrlcv::Image* query, ssrlcv::Unity<ssrlcv::Feature<T>>* queryFeatures,
ssrlcv::Image* target, ssrlcv::Unity<ssrlcv::Feature<T>>* targetFeatures, Unity<float>* seedDistances){

  MemoryState origin[2] = {queryFeatures->getMemoryState(), targetFeatures->getMemoryState()};
  
  if(origin[0] != gpu) queryFeatures->setMemoryState(gpu);
  if(origin[1] != gpu) targetFeatures->setMemoryState(gpu);

  unsigned int numPossibleMatches = queryFeatures->size();

  FeatureMatch<T>* matches_device = nullptr;
  CudaSafeCall(cudaMalloc((void**)&matches_device, numPossibleMatches*sizeof(FeatureMatch<T>)));

  Unity<FeatureMatch<T>>* matches = new Unity<FeatureMatch<T>>(matches_device, numPossibleMatches, gpu);

  dim3 grid = {1,1,1};
  dim3 block = {32,1,1};//IMPROVE
  getGrid(matches->size(),grid);

  clock_t timer = clock();

  if(seedDistances == nullptr){
    matchFeaturesBruteForce<T><<<grid, block>>>(query->id, queryFeatures->size(), queryFeatures->device,
    target->id, targetFeatures->size(), targetFeatures->device, matches->device,this->absoluteThreshold);
  }
  else if(seedDistances->size() != queryFeatures->size()){
    logger.err<<"ERROR: seedDistances should have come from matching a seed image to queryFeatures"<<"\n";
    exit(-1);
  }
  else{
    MemoryState seedOrigin = seedDistances->getMemoryState();
    if(seedOrigin != gpu) seedDistances->setMemoryState(gpu);
    matchFeaturesBruteForce<T><<<grid, block>>>(query->id, queryFeatures->size(), queryFeatures->device,
    target->id, targetFeatures->size(), targetFeatures->device, matches->device,seedDistances->device,
    this->relativeThreshold,this->absoluteThreshold);
    if(seedOrigin != gpu) seedDistances->setMemoryState(seedOrigin);
  }
    
  printf("done in %f seconds.\n\n",((float) clock() -  timer)/CLOCKS_PER_SEC);

  if(origin[0] != gpu) queryFeatures->setMemoryState(origin[0]);
  if(origin[1] != gpu) targetFeatures->setMemoryState(origin[1]);

  return matches;
}
template<typename T>
ssrlcv::Unity<ssrlcv::FeatureMatch<T>>* ssrlcv::MatchFactory<T>::generateFeatureMatchesConstrained(ssrlcv::Image* query, ssrlcv::Unity<ssrlcv::Feature<T>>* queryFeatures,
ssrlcv::Image* target, ssrlcv::Unity<ssrlcv::Feature<T>>* targetFeatures, float epsilon, float fundamental[3][3], Unity<float>* seedDistances){

  MemoryState origin[2] = {queryFeatures->getMemoryState(), targetFeatures->getMemoryState()};
  
  if(origin[0] != gpu) queryFeatures->setMemoryState(gpu);
  if(origin[1] != gpu) targetFeatures->setMemoryState(gpu);

  unsigned int numPossibleMatches = queryFeatures->size();

  FeatureMatch<T>* matches_device = nullptr;
  CudaSafeCall(cudaMalloc((void**)&matches_device, numPossibleMatches*sizeof(FeatureMatch<T>)));

  Unity<FeatureMatch<T>>* matches = new Unity<FeatureMatch<T>>(matches_device, numPossibleMatches, gpu);

  dim3 grid = {1,1,1};
  dim3 block = {32,1,1};//IMPROVE
  getGrid(matches->size(),grid);

  float* fundamental_device = nullptr;
  CudaSafeCall(cudaMalloc((void**)&fundamental_device,9*sizeof(float)));
  CudaSafeCall(cudaMemcpy(fundamental_device,fundamental,9*sizeof(float),cudaMemcpyHostToDevice));

  clock_t timer = clock();

  if(seedDistances == nullptr){
    matchFeaturesConstrained<T><<<grid, block>>>(query->id, queryFeatures->size(), queryFeatures->device,
    target->id, targetFeatures->size(), targetFeatures->device, matches->device, epsilon, fundamental_device,this->absoluteThreshold);
  }
  else if(seedDistances->size() != queryFeatures->size()){
    logger.err<<"ERROR: seedDistances should have come from matching a seed image to queryFeatures"<<"\n";
    exit(-1);
  }
  else{
    MemoryState seedOrigin = seedDistances->getMemoryState();
    if(seedOrigin != gpu) seedDistances->setMemoryState(gpu);
    matchFeaturesConstrained<T><<<grid, block>>>(query->id, queryFeatures->size(), queryFeatures->device,
    target->id, targetFeatures->size(), targetFeatures->device, matches->device, epsilon, fundamental_device,seedDistances->device,
    this->relativeThreshold,this->absoluteThreshold);
    if(seedOrigin != gpu) seedDistances->setMemoryState(seedOrigin);
  }
  cudaDeviceSynchronize();
  CudaCheckError();

  CudaSafeCall(cudaFree(fundamental_device));

  this->validateMatches(matches);

  printf("done in %f seconds.\n\n",((float) clock() -  timer)/CLOCKS_PER_SEC);

  if(origin[0] != gpu) queryFeatures->setMemoryState(origin[0]);
  if(origin[1] != gpu) targetFeatures->setMemoryState(origin[1]);

  return matches;

}



template<typename T>
ssrlcv::Unity<ssrlcv::uint2_pair>* ssrlcv::MatchFactory<T>::generateMatchesIndexOnly(Image* query, Unity<Feature<T>>* queryFeatures, Image* target, Unity<Feature<T>>* targetFeatures, Unity<float>* seedDistances){
  MemoryState origin[2] = {queryFeatures->getMemoryState(), targetFeatures->getMemoryState()};

  if(origin[0] != gpu) queryFeatures->setMemoryState(gpu);
  if(origin[1] != gpu) targetFeatures->setMemoryState(gpu);

  unsigned int numPossibleMatches = queryFeatures->size();

  uint2_pair* matches_device = nullptr;
  CudaSafeCall(cudaMalloc((void**)&matches_device, numPossibleMatches*sizeof(uint2_pair)));
  Unity<uint2_pair>* matches = new Unity<uint2_pair>(matches_device, numPossibleMatches, gpu);

  dim3 grid = {1,1,1};
  dim3 block = {32,1,1};//IMPROVE
  getGrid(matches->size(),grid);

  clock_t timer = clock();

  if(seedDistances == nullptr){
    matchFeaturesBruteForce<T><<<grid, block>>>(query->id, queryFeatures->size(), queryFeatures->device,
    target->id, targetFeatures->size(), targetFeatures->device, matches->device,this->absoluteThreshold);
  }
  else if(seedDistances->size() != queryFeatures->size()){
    logger.err<<"ERROR: seedDistances should have come from matching a seed image to queryFeatures"<<"\n";
    exit(-1);
  }
  else{
    MemoryState seedOrigin = seedDistances->getMemoryState();
    if(seedOrigin != gpu) seedDistances->setMemoryState(gpu);
    matchFeaturesBruteForce<T><<<grid, block>>>(query->id, queryFeatures->size(), queryFeatures->device,
    target->id, targetFeatures->size(), targetFeatures->device, matches->device,seedDistances->device,
    this->relativeThreshold,this->absoluteThreshold);
    if(seedOrigin != gpu) seedDistances->setMemoryState(seedOrigin);
  }
  
  cudaDeviceSynchronize();
  CudaCheckError();

  this->validateMatches(matches);

  printf("done in %f seconds.\n\n",((float) clock() -  timer)/CLOCKS_PER_SEC);

  if(origin[0] != gpu) queryFeatures->setMemoryState(origin[0]);
  if(origin[1] != gpu) targetFeatures->setMemoryState(origin[1]);

  return matches;
}
template<typename T>
ssrlcv::Unity<ssrlcv::uint2_pair>* ssrlcv::MatchFactory<T>::generateMatchesConstrainedIndexOnly(Image* query, Unity<Feature<T>>* queryFeatures, Image* target, Unity<Feature<T>>* targetFeatures, float epsilon, float fundamental[3][3], Unity<float>* seedDistances){
  MemoryState origin[2] = {queryFeatures->getMemoryState(), targetFeatures->getMemoryState()};

  if(origin[0] != gpu) queryFeatures->setMemoryState(gpu);
  if(origin[1] != gpu) targetFeatures->setMemoryState(gpu);

  unsigned int numPossibleMatches = queryFeatures->size();

  uint2_pair* matches_device = nullptr;
  CudaSafeCall(cudaMalloc((void**)&matches_device, numPossibleMatches*sizeof(uint2_pair)));

  Unity<uint2_pair>* matches = new Unity<uint2_pair>(matches_device, numPossibleMatches, gpu);

  dim3 grid = {1,1,1};
  dim3 block = {32,1,1};//IMPROVE
  getGrid(matches->size(),grid);

  float* fundamental_device = nullptr;
  CudaSafeCall(cudaMalloc((void**)&fundamental_device,9*sizeof(float)));
  CudaSafeCall(cudaMemcpy(fundamental_device,fundamental,9*sizeof(float),cudaMemcpyHostToDevice));

  clock_t timer = clock();

  if(seedDistances == nullptr){
    matchFeaturesConstrained<<<grid, block>>>(query->id, queryFeatures->size(), queryFeatures->device,
    target->id, targetFeatures->size(), targetFeatures->device, matches->device, epsilon,fundamental_device,this->absoluteThreshold);
  }
  else if(seedDistances->size() != queryFeatures->size()){
    logger.err<<"ERROR: seedDistances should have come from matching a seed image to queryFeatures"<<"\n";
    exit(-1);
  }
  else{
    MemoryState seedOrigin = seedDistances->getMemoryState();
    if(seedOrigin != gpu) seedDistances->setMemoryState(gpu);
    matchFeaturesConstrained<T><<<grid, block>>>(query->id, queryFeatures->size(), queryFeatures->device,
    target->id, targetFeatures->size(), targetFeatures->device, matches->device,epsilon,fundamental_device,seedDistances->device,
    this->relativeThreshold,this->absoluteThreshold);
    if(seedOrigin != gpu) seedDistances->setMemoryState(seedOrigin);
  }

  cudaDeviceSynchronize();
  CudaCheckError();

  CudaSafeCall(cudaFree(fundamental_device));

  this->validateMatches(matches);

  printf("done in %f seconds.\n\n",((float) clock() -  timer)/CLOCKS_PER_SEC);

  if(origin[0] != gpu) queryFeatures->setMemoryState(origin[0]);
  if(origin[1] != gpu) targetFeatures->setMemoryState(origin[1]);

  return matches;
}


template<typename T>
ssrlcv::MatchSet ssrlcv::MatchFactory<T>::generateMatchesExaustive(std::vector<ssrlcv::Image*> images, std::vector<ssrlcv::Unity<ssrlcv::Feature<T>>*> features, bool ordered, float estimatedOverlap){
  MatchSet matchSet;
  matchSet.keyPoints = nullptr;
  matchSet.matches = nullptr;
  if(estimatedOverlap == 0){
    logger.warn<<"WARNING: estimated overlap fraction of 0.0f requires unordered match interpolation"<<"\n";
  }
  std::vector<Image*>::iterator query = images.begin();
  std::vector<Image*>::iterator target = query + 1;
  typename std::vector<Unity<Feature<T>>*>::iterator features_query = features.begin();
  typename std::vector<Unity<Feature<T>>*>::iterator features_target = features_query + 1;
  std::vector<Unity<uint2_pair>*> matchIndices;
  Unity<float>* seedDistances = nullptr;
  unsigned long long totalMatches = 0;
  int i = 0;
  std::cout<<"matching images"<<"\n";
  for(int i = 0; query != images.end() - 1; ++query, ++features_query){
    if(this->seedFeatures != nullptr) seedDistances = this->getSeedDistances(*features_query);
    for(target = query + 1,features_target = features_query + 1; target != images.end(); ++target, ++features_target){
      if(ordered && estimatedOverlap > 0.0f && ++i*(1-estimatedOverlap) > 1.0f) continue; //based off linear images
      //now match
      matchIndices.push_back(this->generateMatchesIndexOnly(*query,*features_query,*target,*features_target,seedDistances));
      totalMatches += matchIndices[i++]->size();
    }
  }
  if(seedDistances != nullptr) delete seedDistances;
  if(totalMatches == 0){
    logger.err<<"There were no matches found in the set of images, likely due to unreasonable threshold"<<"\n";
    logger.err<<"exiting..."<<"\n";
    exit(0);
  }
  std::cout<<"prepping match interpolation on cpu"<<"\n";
  //required connections to make a match?
  std::vector<uint2>** adjacencyList = new std::vector<uint2>*[images.size() - 1];

  i = 0;
  adjacencyList[0] = new std::vector<uint2>[features[0]->size()];
  std::cout<<"building adjacency list"<<"\n";
  for(auto m = matchIndices.begin(); m != matchIndices.end(); ++m){
    Unity<uint2_pair>* currentMatches = *m;
    if(currentMatches->getMemoryState() != cpu) currentMatches->setMemoryState(cpu);
    if(currentMatches->host[0].a.x != i){
      i++;
      adjacencyList[i] = new std::vector<uint2>[features[i]->size()];
    }
    for(int p = 0; p < currentMatches->size(); ++p){
      uint2_pair* currentPair = &currentMatches->host[p];
      adjacencyList[currentPair->a.x][currentPair->a.y].push_back(currentPair->b); 
    }
    delete currentMatches;
  }
  MemoryState* origin = new MemoryState[images.size()];
  std::vector<std::vector<uint2>> multiMatch_vec;
  bool badMatch = false;
  std::cout<<"deriving matches from adjacency"<<"\n";
  for(i = 0; i < images.size() - 1; ++i){
    origin[i] = features[i]->getMemoryState();
    if(origin[i] != cpu) features[i]->setMemoryState(cpu);
    for(int f = 0; i < images.size() - 2 && f < features[i]->size(); ++f){
      std::vector<uint2>* adj = &adjacencyList[i][f];
      if(!adj->size()) continue;
      badMatch = false;
      std::vector<uint2>* prev_adj = adj;
      std::vector<uint2>* next_adj = nullptr;  
      while(true){
        if(prev_adj->begin()->x == images.size() - 1) break;
        next_adj = &adjacencyList[prev_adj->begin()->x][prev_adj->begin()->y];
        if(!next_adj->size()) break;
        std::vector<uint2> intersection;
        std::set_intersection(prev_adj->begin(),prev_adj->end(),next_adj->begin(),next_adj->end(),std::back_inserter(intersection));
        if(intersection.size() != next_adj->size()){
          badMatch = true;
          break;
        }
        else if(next_adj->size() == 1) break;
        else{
          prev_adj = next_adj;
        }
      } 
      if(badMatch) adj->clear();
      else{
        std::vector<uint2> match;
        match.push_back({(unsigned int)i,(unsigned int)f});
        match.insert(match.end(),adjacencyList[i][f].begin(),adjacencyList[i][f].end());
        multiMatch_vec.push_back(match);
        for(auto m = adj->begin(); m != adj->end() - 1; ++m){
          if(m->x == images.size() - 1) break;
          next_adj = &adjacencyList[m->x][m->y];
          next_adj->clear();
        }
      } 
    }
    delete[] adjacencyList[i];
  }
  delete[] adjacencyList;
  std::cout<<"total matches found in set = "<<multiMatch_vec.size()<<"\n";
  matchSet.matches = new Unity<MultiMatch>(nullptr,multiMatch_vec.size(),cpu);
  std::vector<KeyPoint> kp_vec;
  i = 0;
  int index = 0;
  for(auto m = multiMatch_vec.begin(); m != multiMatch_vec.end(); ++m){
    matchSet.matches->host[i++] = {(unsigned int)m->size(),index};
    index += m->size();
    for(auto kp = m->begin(); kp != m->end(); ++kp){
      kp_vec.push_back({(int)kp->x,features[kp->x]->host[kp->y].loc});
    }
  }
  matchSet.keyPoints = new Unity<KeyPoint>(nullptr,kp_vec.size(),gpu);
  CudaSafeCall(cudaMemcpy(matchSet.keyPoints->device,&kp_vec[0],kp_vec.size()*sizeof(KeyPoint),cudaMemcpyHostToDevice));
  for(int i = 0; i < images.size() - 1; ++i){
    if(origin[i] != cpu) features[i]->setMemoryState(origin[i]);
  }
  delete[] origin;
  //1:2,1:3,1:4,1:5,2:3,2:4,2:5,3:4,3:5,4:5
  return matchSet;

}


ssrlcv::Unity<ssrlcv::Match>* ssrlcv::generateDiparityMatches(uint2 querySize, Unity<unsigned char>* queryPixels, uint2 targetSize, Unity<unsigned char>* targetPixels, 
  float fundamental[3][3], unsigned int maxDisparity,unsigned int windowSize, Direction direction){
  if(direction != right && direction != left && direction != undefined){
    logger.err<<"ERROR: unsupported search direction for disparity matching"<<"\n";
    exit(-1);
  }
  if(maxDisparity > querySize.x){
    logger.err<<"Max disparity cannot be larger than image size"<<"\n";
    exit(-1);
  }
  printf(
    "running disparity matching on parallel images \n\timage[0] = %ux%u\n\timage[1] = %ux%u\n\tmaxDisparity = %u\n\twindow size = %ux%u\n",
    querySize.x,querySize.y,targetSize.x,targetSize.y,maxDisparity,windowSize,windowSize
  );

  if(windowSize == 0 || windowSize % 2 == 0 || windowSize > 31){
    logger.err<<"ERROR window size for disparity matching must be greater than 0, less than 31 and odd"<<"\n";
    exit(-1);
  }

  MemoryState origin[2] = {queryPixels->getMemoryState(), targetPixels->getMemoryState()};

  if(origin[0] != gpu) queryPixels->setMemoryState(gpu);
  if(origin[1] != gpu) targetPixels->setMemoryState(gpu);
  
  uint2 minimizedSize = {querySize.x-windowSize-1,querySize.y-windowSize-1};

  unsigned int numPossibleMatches = minimizedSize.x*minimizedSize.y;

  Match* matches_device = nullptr;
  CudaSafeCall(cudaMalloc((void**)&matches_device, numPossibleMatches*sizeof(Match)));

  Unity<Match>* matches = new Unity<Match>(matches_device, numPossibleMatches, gpu);

  dim3 grid = {1,1,1};
  dim3 block = {windowSize,windowSize,1};//NOTE some devices will not be able to handle large numbers here
  checkDims(grid,block);
  getGrid(numPossibleMatches,grid);

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

  if(origin[0] != gpu) queryPixels->setMemoryState(origin[0]);
  if(origin[1] != gpu) targetPixels->setMemoryState(origin[1]);

  thrust::device_ptr<Match> needsValidating(matches->device);
  thrust::device_ptr<Match> new_end = thrust::remove_if(needsValidating,needsValidating+matches->size(),validate());
  cudaDeviceSynchronize();
  CudaCheckError();
  int numMatchesLeft = new_end - needsValidating;
  if(numMatchesLeft == 0){
    std::cout<<"No valid matches found"<<"\n";
    delete matches;
    matches = nullptr;
  }
  else{
    printf("%d valid matches found out of %lu original matches\n",numMatchesLeft,matches->size());
    Match* validatedMatches_device = nullptr;
    CudaSafeCall(cudaMalloc((void**)&validatedMatches_device,numMatchesLeft*sizeof(Match)));
    CudaSafeCall(cudaMemcpy(validatedMatches_device,matches->device,numMatchesLeft*sizeof(Match),cudaMemcpyDeviceToDevice));
    matches->setData(validatedMatches_device,numMatchesLeft,gpu);
  }
  //returning based on origin of query pixels
  if(origin[0] != gpu) matches->setMemoryState(origin[0]);
  return matches;
}


void ssrlcv::writeMatchFile(Unity<Match>* matches, std::string pathToFile, bool binary){
  MemoryState origin = matches->getMemoryState();
  if(matches->getFore() == gpu) matches->transferMemoryTo(cpu);
  if(binary){
    std::ofstream matchstream(pathToFile,std::ios_base::binary);
    if(matchstream.is_open()){
      for(int i = 0; i < matches->size(); ++i){
        matchstream.write((char*)&matches->host[i].keyPoints[0].loc,2*sizeof(float));
        matchstream.write((char*)&matches->host[i].keyPoints[1].loc,2*sizeof(float));
      }
    }
    else{
      logger.err<<"ERROR: cannot write "<<pathToFile<<"\n";
    }
    matchstream.close();
  }
  else{
    std::ofstream matchstream(pathToFile);
    if(matchstream.is_open()){
      std::string line;
      for(int i = 0; i < matches->size(); ++i){
        line = std::to_string(matches->host[i].keyPoints[0].loc.x) + ",";
        line += std::to_string(matches->host[i].keyPoints[0].loc.y) + ",";
        line += std::to_string(matches->host[i].keyPoints[1].loc.x) + ",";
        line += std::to_string(matches->host[i].keyPoints[1].loc.y) + "\n";
        matchstream << line;
      }
      matchstream.close();
    }
    else{
      logger.err<<"ERROR: cannot write match files"<<"\n";
      exit(-1);
    }
  }
  std::cout<<pathToFile<<" has been written"<<"\n";
  if(origin != matches->getMemoryState()) matches->setMemoryState(origin);
}
void ssrlcv::writeMatchFile(MatchSet multiview_matches, std::string pathToFile, bool binary){
  Unity<MultiMatch>* matches = multiview_matches.matches;
  Unity<KeyPoint>* keyPoints = multiview_matches.keyPoints;
  MemoryState origin[2] = {matches->getMemoryState(),keyPoints->getMemoryState()};
  if(origin[0] != cpu) matches->setMemoryState(cpu);
  if(origin[1] != cpu) keyPoints->setMemoryState(cpu);

  std::ofstream matchstream(pathToFile);
  if(matchstream.is_open()){
    std::string line;
    for(int i = 0; i < matches->size(); ++i){
      line = std::to_string(matches->host[i].numKeyPoints) + ",";
      for(int kp = matches->host[i].index; kp < matches->host[i].index + matches->host[i].numKeyPoints; ++kp){
        line += std::to_string(keyPoints->host[kp].parentId) + ",";
        line += std::to_string(keyPoints->host[kp].loc.x) + ",";
        line += std::to_string(keyPoints->host[kp].loc.y) + ",";
      }
      line += "\n";
      matchstream << line;
    }
    matchstream.close();
  }
  else{
    logger.err<<"ERROR: cannot write match files"<<"\n";
    exit(-1);
  }
  
  std::cout<<pathToFile<<" has been written"<<"\n";
  if(origin[0] != cpu) matches->setMemoryState(origin[0]);
  if(origin[1] != cpu) keyPoints->setMemoryState(origin[1]);
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
  std::cout<<match_vec.size()<<" matches have been read."<<"\n";
  Unity<Match>* matches = new Unity<Match>(nullptr,match_vec.size(),cpu);
  std::memcpy(matches->host,&match_vec[0],match_vec.size()*sizeof(Match));
  return matches;
}


/*
CUDA implementations
*/

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

//TODO change currentDist to type D

/*
matching
*/
//base matching kernels

//base matching kernels
//TODO block this out
__global__ void ssrlcv::disparityMatching(uint2 querySize, unsigned char* pixelsQuery, uint2 targetSize, unsigned char* pixelsTarget, float* fundamental, Match* matches, unsigned int maxDisparity, Direction direction){
  unsigned int blockId = blockIdx.y * gridDim.x + blockIdx.x;
  uint2 minimizedSize = {querySize.x-blockDim.x-1,querySize.y-blockDim.x-1};
  if(blockId < minimizedSize.x*minimizedSize.y){
    uint2 loc = {blockId%minimizedSize.x + (blockDim.x/2),blockId/minimizedSize.x + (blockDim.y/2)};
    uint2 threadLoc = {threadIdx.x - (blockDim.x/2),threadIdx.y - (blockDim.y/2)};
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
        stop -= querySize.x - ((int)maxDisparity + (int)loc.x);
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
  unsigned int blockId = blockIdx.y * gridDim.x + blockIdx.x;
  uint2 minimizedSize = {querySize.x-blockDim.x-1,querySize.y-blockDim.x-1};
  if(blockId < minimizedSize.x*minimizedSize.y){
    uint2 loc = {blockId%minimizedSize.x + (blockDim.x/2), blockId/minimizedSize.x + (blockDim.y/2)};
    uint2 threadLoc = {threadIdx.x - (blockDim.x/2),threadIdx.y - (blockDim.y/2)};
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
        if((int)loc.x - stop < 0){
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
  unsigned int blockId = blockIdx.y * gridDim.x + blockIdx.x;
  if(blockId < numFeaturesQuery){
    Feature<T> feature = featuresQuery[blockId];
    __shared__ float localDist[32];
    localDist[threadIdx.x] = FLT_MAX;
    __syncthreads();
    float currentDist = 0.0f;
    unsigned long numSeedFeatures_reg = numSeedFeatures;
    for(int f = threadIdx.x; f < numSeedFeatures_reg; f += 32){
      currentDist = feature.descriptor.distProtocol(seedFeatures[f].descriptor,localDist[threadIdx.x]);
      if(localDist[threadIdx.x] > currentDist){
        localDist[threadIdx.x] = currentDist;
      }
    }
    __syncthreads();
    if(threadIdx.x != 0) return;
    currentDist = FLT_MAX;
    for(int i = 0; i < 32; ++i){
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
  unsigned int blockId = blockIdx.y * gridDim.x + blockIdx.x;
  if(blockId < numFeaturesQuery){
    Feature<T> feature = featuresQuery[blockId];
    __shared__ int localMatch[32];
    __shared__ float localDist[32];
    localMatch[threadIdx.x] = -1;
    localDist[threadIdx.x] = absoluteThreshold;
    __syncthreads();
    float currentDist = 0.0f;
    unsigned long numFeaturesTarget_register = numFeaturesTarget;
    for(int f = threadIdx.x; f < numFeaturesTarget_register; f += 32){
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
    for(int i = 0; i < 32; ++i){
      if(currentDist > localDist[i]){
        currentDist = localDist[i];
        matchIndex = localMatch[i];
      }
    }
    Match match;
    if(currentDist >= absoluteThreshold){
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
  unsigned int blockId = blockIdx.y * gridDim.x + blockIdx.x;
  if(blockId < numFeaturesQuery){
    Feature<T> feature = featuresQuery[blockId];
    __shared__ int localMatch[32];
    __shared__ float localDist[32];
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

    for(int f = threadIdx.x; f < numFeaturesTarget_register; f += 32){

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
    for(int i = 0; i < 32; ++i){
      if(currentDist > localDist[i]){
        currentDist = localDist[i];
        matchIndex = localMatch[i];
      }
    }
    Match match;
    if(currentDist >= absoluteThreshold){
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
  unsigned int blockId = blockIdx.y * gridDim.x + blockIdx.x;
  if(blockId < numFeaturesQuery){
    Feature<T> feature = featuresQuery[blockId];
    __shared__ int localMatch[32];
    __shared__ float localDist[32];
    localMatch[threadIdx.x] = -1;
    float nearestSeed = seedDistances[blockId];
    localDist[threadIdx.x] = absoluteThreshold;
    __syncthreads();
    float currentDist = 0.0f;
    unsigned long numFeaturesTarget_register = numFeaturesTarget;
    for(int f = threadIdx.x; f < numFeaturesTarget_register; f += 32){
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
    for(int i = 0; i < 32; ++i){
      if(currentDist > localDist[i]){
        currentDist = localDist[i];
        matchIndex = localMatch[i];
      }
    }
    Match match;
    if(currentDist >= absoluteThreshold || matchIndex == -1){
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
  unsigned int blockId = blockIdx.y * gridDim.x + blockIdx.x;
  if(blockId < numFeaturesQuery){
    Feature<T> feature = featuresQuery[blockId];
    __shared__ int localMatch[32];
    __shared__ float localDist[32];
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

    for(int f = threadIdx.x; f < numFeaturesTarget_register; f += 32){

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
    for(int i = 0; i < 32; ++i){
      if(currentDist > localDist[i]){
        currentDist = localDist[i];
        matchIndex = localMatch[i];
      }
    }
    Match match;
    if(currentDist >= absoluteThreshold || matchIndex == -1){
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
  unsigned int blockId = blockIdx.y * gridDim.x + blockIdx.x;
  if(blockId < numFeaturesQuery){
    Feature<T> feature = featuresQuery[blockId];
    __shared__ int localMatch[32];
    __shared__ float localDist[32];
    localMatch[threadIdx.x] = -1;
    localDist[threadIdx.x] = absoluteThreshold;
    __syncthreads();
    float currentDist = 0.0f;
    unsigned long numFeaturesTarget_register = numFeaturesTarget;
    for(int f = threadIdx.x; f < numFeaturesTarget_register; f += 32){
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
    for(int i = 0; i < 32; ++i){
      if(currentDist > localDist[i]){
        currentDist = localDist[i];
        matchIndex = localMatch[i];
      }
    }
    DMatch match;
    match.distance = currentDist;
    if(match.distance >= absoluteThreshold){
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
  unsigned int blockId = blockIdx.y * gridDim.x + blockIdx.x;
  if(blockId < numFeaturesQuery){
    Feature<T> feature = featuresQuery[blockId];
    __shared__ int localMatch[32];
    __shared__ float localDist[32];
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

    for(int f = threadIdx.x; f < numFeaturesTarget_register; f += 32){

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
    for(int i = 0; i < 32; ++i){
      if(currentDist > localDist[i]){
        currentDist = localDist[i];
        matchIndex = localMatch[i];
      }
    }
    DMatch match;
    match.distance = currentDist;
    if(match.distance >= absoluteThreshold){
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
  unsigned int blockId = blockIdx.y * gridDim.x + blockIdx.x;
  if(blockId < numFeaturesQuery){
    Feature<T> feature = featuresQuery[blockId];
    __shared__ int localMatch[32];
    __shared__ float localDist[32];
    localMatch[threadIdx.x] = -1;
    float nearestSeed = seedDistances[blockId];
    localDist[threadIdx.x] = absoluteThreshold;
    __syncthreads();
    float currentDist = 0.0f;
    unsigned long numFeaturesTarget_register = numFeaturesTarget;
    for(int f = threadIdx.x; f < numFeaturesTarget_register; f += 32){
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
    for(int i = 0; i < 32; ++i){
      if(currentDist > localDist[i]){
        currentDist = localDist[i];
        matchIndex = localMatch[i];
      }
    }
    DMatch match;
    match.distance = currentDist;
    if(match.distance >= absoluteThreshold || matchIndex == -1){
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
  unsigned int blockId = blockIdx.y * gridDim.x + blockIdx.x;
  if(blockId < numFeaturesQuery){
    Feature<T> feature = featuresQuery[blockId];
    __shared__ int localMatch[32];
    __shared__ float localDist[32];
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

    for(int f = threadIdx.x; f < numFeaturesTarget_register; f += 32){

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
    for(int i = 0; i < 32; ++i){
      if(currentDist > localDist[i]){
        currentDist = localDist[i];
        matchIndex = localMatch[i];
      }
    }
    DMatch match;
    match.distance = currentDist;
    if(match.distance >= absoluteThreshold || matchIndex == -1){
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
  unsigned int blockId = blockIdx.y * gridDim.x + blockIdx.x;
  if(blockId < numFeaturesQuery){
    Feature<T> feature = featuresQuery[blockId];
    __shared__ int localMatch[32];
    __shared__ float localDist[32];
    localMatch[threadIdx.x] = -1;
    localDist[threadIdx.x] = absoluteThreshold;
    __syncthreads();
    float currentDist = 0.0f;
    unsigned long numFeaturesTarget_register = numFeaturesTarget;
    for(int f = threadIdx.x; f < numFeaturesTarget_register; f += 32){
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
    for(int i = 0; i < 32; ++i){
      if(currentDist > localDist[i]){
        currentDist = localDist[i];
        matchIndex = localMatch[i];
      }
    }
    FeatureMatch<T> match;    
    match.distance = currentDist;
    if(match.distance >= absoluteThreshold){
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
  unsigned int blockId = blockIdx.y * gridDim.x + blockIdx.x;
  if(blockId < numFeaturesQuery){
    Feature<T> feature = featuresQuery[blockId];
    __shared__ int localMatch[32];
    __shared__ float localDist[32];
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

    for(int f = threadIdx.x; f < numFeaturesTarget_register; f += 32){

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
    for(int i = 0; i < 32; ++i){
      if(currentDist > localDist[i]){
        currentDist = localDist[i];
        matchIndex = localMatch[i];
      }
    }
    FeatureMatch<T> match;    
    match.distance = currentDist;
    if(match.distance >= absoluteThreshold){
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
  unsigned int blockId = blockIdx.y * gridDim.x + blockIdx.x;
  if(blockId < numFeaturesQuery){
    Feature<T> feature = featuresQuery[blockId];
    __shared__ int localMatch[32];
    __shared__ float localDist[32];
    localMatch[threadIdx.x] = -1;
    float nearestSeed = seedDistances[blockId];
    localDist[threadIdx.x] = absoluteThreshold;
    __syncthreads();
    float currentDist = 0.0f;
    unsigned long numFeaturesTarget_register = numFeaturesTarget;
    for(int f = threadIdx.x; f < numFeaturesTarget_register; f += 32){
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
    for(int i = 0; i < 32; ++i){
      if(currentDist > localDist[i]){
        currentDist = localDist[i];
        matchIndex = localMatch[i];
      }
    }
    FeatureMatch<T> match;    
    match.distance = currentDist;
    if(match.distance >= absoluteThreshold || matchIndex == -1){
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
  unsigned int blockId = blockIdx.y * gridDim.x + blockIdx.x;
  if(blockId < numFeaturesQuery){
    Feature<T> feature = featuresQuery[blockId];
    __shared__ int localMatch[32];
    __shared__ float localDist[32];
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

    for(int f = threadIdx.x; f < numFeaturesTarget_register; f += 32){

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
    for(int i = 0; i < 32; ++i){
      if(currentDist > localDist[i]){
        currentDist = localDist[i];
        matchIndex = localMatch[i];
      }
    }
    FeatureMatch<T> match;    
    match.distance = currentDist;
    if(match.distance >= absoluteThreshold || matchIndex == -1){
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
__global__ void ssrlcv::matchFeaturesBruteForce(unsigned int queryImageID, unsigned long numFeaturesQuery,
ssrlcv::Feature<T>* featuresQuery, unsigned int targetImageID, unsigned long numFeaturesTarget,
ssrlcv::Feature<T>* featuresTarget, uint2_pair* matches, float absoluteThreshold){
  unsigned int blockId = blockIdx.y * gridDim.x + blockIdx.x;
  if(blockId < numFeaturesQuery){
    Feature<T> feature = featuresQuery[blockId];
    __shared__ int localMatch[32];
    __shared__ float localDist[32];
    localMatch[threadIdx.x] = -1;
    localDist[threadIdx.x] = absoluteThreshold;
    __syncthreads();
    float currentDist = 0.0f;
    unsigned long numFeaturesTarget_register = numFeaturesTarget;
    for(int f = threadIdx.x; f < numFeaturesTarget_register; f += 32){
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
    for(int i = 0; i < 32; ++i){
      if(currentDist > localDist[i]){
        currentDist = localDist[i];
        matchIndex = localMatch[i];
      }
    }
    uint2_pair match;
    if(currentDist >= absoluteThreshold){
      match = {{queryImageID,blockId},{queryImageID,blockId}};
    }
    else{
      match = {{queryImageID,blockId},{targetImageID,(unsigned int)matchIndex}};
    }
    matches[blockId] = match;
  }
}
template<typename T>
__global__ void ssrlcv::matchFeaturesConstrained(unsigned int queryImageID, unsigned long numFeaturesQuery,
ssrlcv::Feature<T>* featuresQuery, unsigned int targetImageID, unsigned long numFeaturesTarget,
ssrlcv::Feature<T>* featuresTarget, uint2_pair* matches, float epsilon, float* fundamental, float absoluteThreshold){
  unsigned int blockId = blockIdx.y * gridDim.x + blockIdx.x;
  if(blockId < numFeaturesQuery){
    Feature<T> feature = featuresQuery[blockId];
    __shared__ int localMatch[32];
    __shared__ float localDist[32];
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

    for(int f = threadIdx.x; f < numFeaturesTarget_register; f += 32){

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
    for(int i = 0; i < 32; ++i){
      if(currentDist > localDist[i]){
        currentDist = localDist[i];
        matchIndex = localMatch[i];
      }
    }
    uint2_pair match;
    if(currentDist >= absoluteThreshold){
      match = {{queryImageID,blockId},{queryImageID,blockId}};
    }
    else{
      match = {{queryImageID,blockId},{targetImageID,(unsigned int)matchIndex}};
    }
    matches[blockId] = match;
  }
}
template<typename T>
__global__ void ssrlcv::matchFeaturesBruteForce(unsigned int queryImageID, unsigned long numFeaturesQuery,
ssrlcv::Feature<T>* featuresQuery, unsigned int targetImageID, unsigned long numFeaturesTarget,
ssrlcv::Feature<T>* featuresTarget, uint2_pair* matches, float* seedDistances, float relativeThreshold, float absoluteThreshold){
  unsigned int blockId = blockIdx.y * gridDim.x + blockIdx.x;
  if(blockId < numFeaturesQuery){
    Feature<T> feature = featuresQuery[blockId];
    __shared__ int localMatch[32];
    __shared__ float localDist[32];
    localMatch[threadIdx.x] = -1;
    float nearestSeed = seedDistances[blockId];
    localDist[threadIdx.x] = absoluteThreshold;
    __syncthreads();
    float currentDist = 0.0f;
    unsigned long numFeaturesTarget_register = numFeaturesTarget;
    for(int f = threadIdx.x; f < numFeaturesTarget_register; f += 32){
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
    for(int i = 0; i < 32; ++i){
      if(currentDist > localDist[i]){
        currentDist = localDist[i];
        matchIndex = localMatch[i];
      }
    }
    uint2_pair match;
    if(currentDist >= absoluteThreshold || matchIndex == -1){
      match = {{queryImageID,blockId},{queryImageID,blockId}};
    }
    else{
      if(currentDist/nearestSeed > relativeThreshold){
        match = {{queryImageID,blockId},{queryImageID,blockId}};
      }
      else{
      match = {{queryImageID,blockId},{targetImageID,(unsigned int)matchIndex}};
      }
    }
    matches[blockId] = match;
  }
}
template<typename T>
__global__ void ssrlcv::matchFeaturesConstrained(unsigned int queryImageID, unsigned long numFeaturesQuery,
ssrlcv::Feature<T>* featuresQuery, unsigned int targetImageID, unsigned long numFeaturesTarget,
ssrlcv::Feature<T>* featuresTarget, uint2_pair* matches, float epsilon, float* fundamental, float* seedDistances, 
float relativeThreshold, float absoluteThreshold){
  unsigned int blockId = blockIdx.y * gridDim.x + blockIdx.x;
  if(blockId < numFeaturesQuery){
    Feature<T> feature = featuresQuery[blockId];
    __shared__ int localMatch[32];
    __shared__ float localDist[32];
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

    for(int f = threadIdx.x; f < numFeaturesTarget_register; f += 32){

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
    for(int i = 0; i < 32; ++i){
      if(currentDist > localDist[i]){
        currentDist = localDist[i];
        matchIndex = localMatch[i];
      }
    }
    uint2_pair match;
    if(currentDist >= absoluteThreshold || matchIndex == -1){
      match = {{queryImageID,blockId},{queryImageID,blockId}};
    }
    else{
      if(currentDist/nearestSeed > relativeThreshold){
        match = {{queryImageID,blockId},{queryImageID,blockId}};
      }
      else{
        match = {{queryImageID,blockId},{targetImageID,(unsigned int)matchIndex}};
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
