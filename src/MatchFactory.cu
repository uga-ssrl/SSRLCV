#include "MatchFactory.cuh"


/**
 * forward declarations of MatchFactories with descriptors that have distProtocol() implemented
 */
template class ssrlcv::MatchFactory<ssrlcv::SIFT_Descriptor>;
template class ssrlcv::MatchFactory<ssrlcv::Window_3x3>;
template class ssrlcv::MatchFactory<ssrlcv::Window_9x9>;
template class ssrlcv::MatchFactory<ssrlcv::Window_15x15>;
template class ssrlcv::MatchFactory<ssrlcv::Window_25x25>;
template class ssrlcv::MatchFactory<ssrlcv::Window_31x31>;


template<typename T>
ssrlcv::MatchFactory<T>::MatchFactory(){
  this->relativeThreshold = 1.0f;
  this->absoluteThreshold = FLT_MAX;
  this->seedFeatures = nullptr;
}
template<typename T>
ssrlcv::MatchFactory<T>::MatchFactory(float relativeThreshold, float absoluteThreshold) :
relativeThreshold(relativeThreshold), absoluteThreshold(absoluteThreshold)
{
  this->seedFeatures = nullptr;
}
template<typename T>
void ssrlcv::MatchFactory<T>::setSeedFeatures(ssrlcv::ptr::value<Unity<Feature<T>>> seedFeatures){
  this->seedFeatures = seedFeatures;
}
template<typename T>
void ssrlcv::MatchFactory<T>::validateMatches(ssrlcv::ptr::value<ssrlcv::Unity<uint2_pair>> matches){
  MemoryState origin = matches->getMemoryState();
  if(origin != gpu) matches->setMemoryState(gpu);
  
  thrust::device_ptr<uint2_pair> needsValidating(matches->device.get());
  thrust::device_ptr<uint2_pair> new_end = thrust::remove_if(needsValidating,needsValidating+matches->size(),validate());
  cudaDeviceSynchronize();
  CudaCheckError();
  int numMatchesLeft = new_end - needsValidating;
  if(numMatchesLeft == 0){
    logger.info<<"No valid matches found";
    matches.clear();
    return;
  }
  

  logger.info.printf("%d valid matches found out of %lu original matches",numMatchesLeft,matches->size());

  ssrlcv::ptr::device<uint2_pair> validatedMatches_device(numMatchesLeft);
  CudaSafeCall(cudaMemcpy(validatedMatches_device.get(),matches->device.get(),numMatchesLeft*sizeof(uint2_pair),cudaMemcpyDeviceToDevice));

  matches->setData(validatedMatches_device,numMatchesLeft,gpu);

  if(origin != gpu) matches->setMemoryState(origin);
}
template<typename T>
void ssrlcv::MatchFactory<T>::validateMatches(ssrlcv::ptr::value<ssrlcv::Unity<ssrlcv::Match>> matches){
  MemoryState origin = matches->getMemoryState();
  if(origin != gpu) matches->setMemoryState(gpu);
  
  thrust::device_ptr<Match> needsValidating(matches->device.get());
  thrust::device_ptr<Match> new_end = thrust::remove_if(needsValidating,needsValidating+matches->size(),validate());
  cudaDeviceSynchronize();
  CudaCheckError();
  int numMatchesLeft = new_end - needsValidating;
  if(numMatchesLeft == 0){
    logger.info<<"No valid matches found";
    matches.clear();
    return;
  }
  

  logger.info.printf("%d valid matches found out of %lu original matches",numMatchesLeft,matches->size());

  ssrlcv::ptr::device<Match> validatedMatches_device(numMatchesLeft);
  CudaSafeCall(cudaMemcpy(validatedMatches_device.get(),matches->device.get(),numMatchesLeft*sizeof(Match),cudaMemcpyDeviceToDevice));

  matches->setData(validatedMatches_device,numMatchesLeft,gpu);

  if(origin != gpu) matches->setMemoryState(origin);
}
template<typename T>
void ssrlcv::MatchFactory<T>::validateMatches(ssrlcv::ptr::value<ssrlcv::Unity<ssrlcv::DMatch>> matches){
  MemoryState origin = matches->getMemoryState();
  if(origin != gpu) matches->setMemoryState(gpu);
  
  thrust::device_ptr<DMatch> needsValidating(matches->device.get());
  thrust::device_ptr<DMatch> new_end = thrust::remove_if(needsValidating,needsValidating+matches->size(),validate());
  cudaDeviceSynchronize();
  CudaCheckError();
  int numMatchesLeft = new_end - needsValidating;
  if(numMatchesLeft == 0){
    logger.info<<"No valid matches found";
    matches.clear();
    return;
  }
  

  logger.info.printf("%d valid matches found out of %lu original matches",numMatchesLeft,matches->size());

  ssrlcv::ptr::device<DMatch> validatedMatches_device(numMatchesLeft);
  CudaSafeCall(cudaMemcpy(validatedMatches_device.get(),matches->device.get(),numMatchesLeft*sizeof(DMatch),cudaMemcpyDeviceToDevice));

  matches->setData(validatedMatches_device,numMatchesLeft,gpu);

  if(origin != gpu) matches->setMemoryState(origin);
}
template<typename T>
void ssrlcv::MatchFactory<T>::validateMatches(ssrlcv::Unity<ssrlcv::FeatureMatch<T>>* matches){
  MemoryState origin = matches->getMemoryState();
  if(origin != gpu) matches->setMemoryState(gpu);

  FeatureMatch<T> *new_end = thrust::remove_if(matches->device.get(),matches->device.get()+matches->size(),validate());
  cudaDeviceSynchronize();
  CudaCheckError();
  int numMatchesLeft = new_end - matches->device.get();
  if(numMatchesLeft == 0){
    logger.info<<"No valid matches found";
    delete matches;
    matches = nullptr;
    return;
  }
  

  logger.info.printf("%d valid matches found out of %lu original matches",numMatchesLeft,matches->size());

  ssrlcv::ptr::device<FeatureMatch<T>> validatedMatches_device(numMatchesLeft);
  CudaSafeCall(cudaMemcpy(validatedMatches_device.get(),matches->device.get(),numMatchesLeft*sizeof(FeatureMatch<T>),cudaMemcpyDeviceToDevice));

  matches->setData(validatedMatches_device,numMatchesLeft,gpu);

  if(origin != gpu) matches->setMemoryState(origin);

}
template<typename T>
void ssrlcv::MatchFactory<T>::refineMatches(ssrlcv::ptr::value<ssrlcv::Unity<ssrlcv::DMatch>> matches, float threshold){
  if(threshold == 0.0f){
    logger.err<<"ERROR illegal value used for threshold: 0.0";
    exit(-1);
  }
  MemoryState origin = matches->getMemoryState();
  if(origin != gpu) matches->setMemoryState(gpu);


  thrust::device_ptr<DMatch> needsCompacting(matches->device.get());
  thrust::device_ptr<DMatch> end = thrust::remove_if(needsCompacting, needsCompacting + matches->size(), match_dist_thresholder(threshold));
  unsigned int numElementsBelowThreshold = end - needsCompacting;
  if(numElementsBelowThreshold == 0){
    matches.clear();
    return;
  }

  logger.info.printf("%lu matches have been refined to %u matches using a cutoff of %f",matches->size(),numElementsBelowThreshold,threshold);

  ssrlcv::ptr::device<DMatch> compactedMatches_device(numElementsBelowThreshold);
  CudaSafeCall(cudaMemcpy(compactedMatches_device.get(),matches->device.get(),numElementsBelowThreshold*sizeof(DMatch),cudaMemcpyDeviceToDevice));

  matches->setData(compactedMatches_device,numElementsBelowThreshold,gpu);

  if(origin != gpu) matches->setMemoryState(origin);
}
template<typename T>
void ssrlcv::MatchFactory<T>::refineMatches(ssrlcv::Unity<ssrlcv::FeatureMatch<T>>* matches, float threshold){
  if(threshold == 0.0f){
    logger.err<<"ERROR illegal value used for cutoff ratio: 0.0";
    exit(-1);
  }
  MemoryState origin = matches->getMemoryState();
  if(origin != gpu) matches->setMemoryState(gpu);

  thrust::device_ptr<FeatureMatch<T>> needsCompacting(matches->device.get());
  thrust::device_ptr<FeatureMatch<T>> end = thrust::remove_if(needsCompacting, needsCompacting + matches->size(), match_dist_thresholder(threshold));
  unsigned int numElementsBelowThreshold = end - needsCompacting;
  if(numElementsBelowThreshold == 0){
    delete matches;
    matches = nullptr;
    return;
  }

  logger.info.printf("%lu matches have been refined to %u matches using a cutoff of %f",matches->size(),numElementsBelowThreshold,threshold);

  ssrlcv::ptr::device<FeatureMatch<T>> compactedMatches_device(numElementsBelowThreshold);
  CudaSafeCall(cudaMemcpy(compactedMatches_device.get(),matches->device.get(),numElementsBelowThreshold*sizeof(FeatureMatch<T>),cudaMemcpyDeviceToDevice));

  matches->setData(compactedMatches_device,numElementsBelowThreshold,gpu);

  if(origin != gpu) matches->setMemoryState(origin);
}
template<typename T>
void ssrlcv::MatchFactory<T>::sortMatches(ssrlcv::ptr::value<ssrlcv::Unity<DMatch>> matches){
  if(matches->getFore() == gpu || matches->getFore() == both){
    thrust::device_ptr<DMatch> toSort(matches->device.get());
    thrust::sort(toSort, toSort + matches->size(),match_dist_comparator());
    matches->setFore(gpu);
    if(matches->getMemoryState() == both) matches->transferMemoryTo(cpu);
  }
  else if(matches->getFore() == cpu){
    unsigned long len = matches->size();
    // insertion sort
    // each match element is accessed with allMatches->host.get()[]
    unsigned long i = 0;
    unsigned long j = 0;
    ssrlcv::DMatch temp;
    while (i < len){
      j = i;
      while (j > 0 && matches->host.get()[j-1].distance > matches->host.get()[j].distance){
        temp = matches->host.get()[j];
        matches->host.get()[j] = matches->host.get()[j-1];
        matches->host.get()[j-1] = temp;
        j--;
      }
      i++;
    }
    if(matches->getMemoryState() == both) matches->transferMemoryTo(gpu);
  }
  else{
    logger.err<<"ERROR cannot perform sortMatches with matches->getMemoryState() = "<<std::to_string(matches->getMemoryState());
    exit(-1);
  }
}
template<typename T>
void ssrlcv::MatchFactory<T>::sortMatches(Unity<FeatureMatch<T>>* matches){
  if(matches->getFore() == gpu || matches->getFore() == both){
    thrust::sort(matches->device.get(), matches->device.get() + matches->size(),match_dist_comparator());
    matches->setFore(gpu);
    if(matches->getMemoryState() == both) matches->transferMemoryTo(cpu);
  }
  else if(matches->getFore() == cpu){
    unsigned long len = matches->size();
    // insertion sort
    // each match element is accessed with allMatches->host.get()[]
    unsigned long i = 0;
    unsigned long j = 0;
    ssrlcv::FeatureMatch<T> temp;
    while (i < len){
      j = i;
      while (j > 0 && matches->host.get()[j-1].distance > matches->host.get()[j].distance){
        temp = matches->host.get()[j];
        matches->host.get()[j] = matches->host.get()[j-1];
        matches->host.get()[j-1] = temp;
        j--;
      }
      i++;
    }
    if(matches->getMemoryState() == both) matches->transferMemoryTo(gpu);
  }
  else{
    logger.err<<"ERROR cannot perform sortMatches with matches->getMemoryState() = "<<std::to_string(matches->getMemoryState());
    exit(-1);
  }
}

template<typename T>
ssrlcv::ptr::value<ssrlcv::Unity<ssrlcv::Match>> ssrlcv::MatchFactory<T>::getRawMatches(ssrlcv::ptr::value<ssrlcv::Unity<DMatch>> matches){
  if(matches->getMemoryState() == gpu || matches->getFore() == gpu){
    ssrlcv::ptr::device<Match> rawMatches_device(matches->size());
    dim3 grid = {1,1,1};
    dim3 block = {1,1,1};
    void (*fp)(unsigned long, Match*, DMatch*) = &convertMatchToRaw;
    getFlatGridBlock(matches->size(),grid,block,fp);
    convertMatchToRaw<<<grid,block>>>(matches->size(),rawMatches_device.get(),matches->device.get());
    cudaDeviceSynchronize();
    CudaCheckError();
    return ssrlcv::ptr::value<Unity<Match>>(rawMatches_device,matches->size(),gpu);
  }
  else{
    ssrlcv::ptr::host<Match> rawMatches_host(matches->size());
    for(int i = 0; i < matches->size(); ++i){
      for(int f = 0; f < 2; ++f){
        rawMatches_host.get()[i] = Match(matches->host.get()[i]);
      }
    }
    return ssrlcv::ptr::value<Unity<Match>>(rawMatches_host, matches->size(), cpu);
  }
}
template<typename T>
ssrlcv::ptr::value<ssrlcv::Unity<ssrlcv::Match>> ssrlcv::MatchFactory<T>::getRawMatches(Unity<FeatureMatch<T>>* matches){
  if(matches->getMemoryState() == gpu || matches->getFore() == gpu){
    ssrlcv::ptr::device<Match> rawMatches_device(matches->size());
    dim3 grid = {1,1,1};
    dim3 block = {1,1,1};
    void (*fp)(unsigned long, Match*, FeatureMatch<T>*) = &convertMatchToRaw;
    getFlatGridBlock(matches->size(),grid,block,fp);
    convertMatchToRaw<T><<<grid,block>>>(matches->size(),rawMatches_device.get(),matches->device.get());
    cudaDeviceSynchronize();
    CudaCheckError();
    return ssrlcv::ptr::value<Unity<Match>>(rawMatches_device,matches->size(),gpu);
  }
  else{
    ssrlcv::ptr::host<Match> rawMatches_host(matches->size());
    for(int i = 0; i < matches->size(); ++i){
      for(int f = 0; f < 2; ++f){
        rawMatches_host.get()[i] = Match(matches->host.get()[i]);
      }
    }
    return ssrlcv::ptr::value<Unity<Match>>(rawMatches_host, matches->size(), cpu);
  }
}

/**
 * @brief This function computes the seed distances.
 * 
 * @tparam T 
 * @param features a pointer to a Unity data structure containing features of type T
 * @return a pointer to a Unity data structure containing the match distances
 */
template<typename T>
ssrlcv::ptr::value<ssrlcv::Unity<float>> ssrlcv::MatchFactory<T>::getSeedDistances(ssrlcv::ptr::value<Unity<Feature<T>>> features){
  MemoryState origin = features->getMemoryState();

  // set the memory state for seedFeatures and features to GPU
  if(this->seedFeatures->getMemoryState() != gpu) this->seedFeatures->setMemoryState(gpu);
  if(origin != gpu) features->setMemoryState(gpu);

  // the size of the feature vector is assigned to numPossibleMatches 
  unsigned int numPossibleMatches = features->size();

  ssrlcv::ptr::value<ssrlcv::Unity<float>> matchDistances = ssrlcv::ptr::value<ssrlcv::Unity<float>>(nullptr, numPossibleMatches,gpu);

  // initilize grid and block dimensions
  dim3 grid = {1,1,1};
  dim3 block = {32,1,1};//IMPROVE
  getGrid(matchDistances->size(),grid);

  clock_t timer = clock();

  // call the kernel function getSeedMatchDistances
  getSeedMatchDistances<T><<<grid, block>>>(features->size(),features->device.get(),this->seedFeatures->size(),
    this->seedFeatures->device.get(),matchDistances->device.get());

  cudaDeviceSynchronize();
  CudaCheckError();

  logger.info.printf("seed match distances computed in %f seconds.",((float) clock() -  timer)/CLOCKS_PER_SEC);

  if(origin != gpu) features->setMemoryState(origin);
  
  return matchDistances;
} // getSeedDistances

template<typename T>
ssrlcv::ptr::value<ssrlcv::Unity<ssrlcv::Match>> ssrlcv::MatchFactory<T>::generateMatches(ssrlcv::ptr::value<ssrlcv::Image> query, ssrlcv::ptr::value<Unity<Feature<T>>> queryFeatures, ssrlcv::ptr::value<ssrlcv::Image> target, ssrlcv::ptr::value<Unity<Feature<T>>> targetFeatures, ssrlcv::ptr::value<ssrlcv::Unity<float>> seedDistances){
  MemoryState origin[2] = {queryFeatures->getMemoryState(), targetFeatures->getMemoryState()};

  if(origin[0] != gpu) queryFeatures->setMemoryState(gpu);
  if(origin[1] != gpu) targetFeatures->setMemoryState(gpu);

  unsigned int numPossibleMatches = queryFeatures->size();

  ssrlcv::ptr::device<Match> matches_device( numPossibleMatches);
  ssrlcv::ptr::value<ssrlcv::Unity<Match>> matches = ssrlcv::ptr::value<ssrlcv::Unity<Match>>(matches_device, numPossibleMatches, gpu);

  dim3 grid = {1,1,1};
  dim3 block = {32,1,1};//IMPROVE
  getGrid(matches->size(),grid);

  clock_t timer = clock();

  if(seedDistances == nullptr){
    matchFeaturesBruteForce<T><<<grid, block>>>(query->id, queryFeatures->size(), queryFeatures->device.get(),
    target->id, targetFeatures->size(), targetFeatures->device.get(), matches->device.get(),this->absoluteThreshold);
  }
  else if(seedDistances->size() != queryFeatures->size()){
    logger.err<<"ERROR: seedDistances should have come from matching a seed image to queryFeatures";
    exit(-1);
  }
  else{
    MemoryState seedOrigin = seedDistances->getMemoryState();
    if(seedOrigin != gpu) seedDistances->setMemoryState(gpu);
    matchFeaturesBruteForce<T><<<grid, block>>>(query->id, queryFeatures->size(), queryFeatures->device.get(),
    target->id, targetFeatures->size(), targetFeatures->device.get(), matches->device.get(),seedDistances->device.get(),
    this->relativeThreshold,this->absoluteThreshold);
    if(seedOrigin != gpu) seedDistances->setMemoryState(seedOrigin);
  }
  
  cudaDeviceSynchronize();
  CudaCheckError();

  this->validateMatches(matches);

  logger.info.printf("done in %f seconds.",((float) clock() -  timer)/CLOCKS_PER_SEC);

  if(origin[0] != gpu) queryFeatures->setMemoryState(origin[0]);
  if(origin[1] != gpu) targetFeatures->setMemoryState(origin[1]);

  return matches;
}
template<typename T>
ssrlcv::ptr::value<ssrlcv::Unity<ssrlcv::Match>> ssrlcv::MatchFactory<T>::generateMatchesConstrained(ssrlcv::ptr::value<ssrlcv::Image> query, ssrlcv::ptr::value<Unity<Feature<T>>> queryFeatures, ssrlcv::ptr::value<ssrlcv::Image> target, ssrlcv::ptr::value<Unity<Feature<T>>> targetFeatures, float epsilon, float fundamental[3][3], ssrlcv::ptr::value<ssrlcv::Unity<float>> seedDistances){
  MemoryState origin[2] = {queryFeatures->getMemoryState(), targetFeatures->getMemoryState()};

  if(origin[0] != gpu) queryFeatures->setMemoryState(gpu);
  if(origin[1] != gpu) targetFeatures->setMemoryState(gpu);

  unsigned int numPossibleMatches = queryFeatures->size();

  ssrlcv::ptr::device<Match> matches_device( numPossibleMatches);

  ssrlcv::ptr::value<ssrlcv::Unity<Match>> matches = ssrlcv::ptr::value<ssrlcv::Unity<Match>>(matches_device, numPossibleMatches, gpu);

  dim3 grid = {1,1,1};
  dim3 block = {32,1,1};//IMPROVE
  getGrid(matches->size(),grid);

  ssrlcv::ptr::device<float> fundamental_device(9);
  CudaSafeCall(cudaMemcpy(fundamental_device.get(),fundamental,9*sizeof(float),cudaMemcpyHostToDevice));

  clock_t timer = clock();

  if(seedDistances == nullptr){
    matchFeaturesConstrained<T><<<grid, block>>>(query->id, queryFeatures->size(), queryFeatures->device.get(),
    target->id, targetFeatures->size(), targetFeatures->device.get(), matches->device.get(), epsilon,fundamental_device.get(),this->absoluteThreshold);
  }
  else if(seedDistances->size() != queryFeatures->size()){
    logger.err<<"ERROR: seedDistances should have come from matching a seed image to queryFeatures";
    exit(-1);
  }
  else{
    MemoryState seedOrigin = seedDistances->getMemoryState();
    if(seedOrigin != gpu) seedDistances->setMemoryState(gpu);
    matchFeaturesConstrained<T><<<grid, block>>>(query->id, queryFeatures->size(), queryFeatures->device.get(),
    target->id, targetFeatures->size(), targetFeatures->device.get(), matches->device.get(),epsilon,fundamental_device.get(),seedDistances->device.get(),
    this->relativeThreshold,this->absoluteThreshold);
    if(seedOrigin != gpu) seedDistances->setMemoryState(seedOrigin);
  }

  cudaDeviceSynchronize();
  CudaCheckError();

  this->validateMatches(matches);

  logger.info.printf("done in %f seconds.",((float) clock() -  timer)/CLOCKS_PER_SEC);

  if(origin[0] != gpu) queryFeatures->setMemoryState(origin[0]);
  if(origin[1] != gpu) targetFeatures->setMemoryState(origin[1]);

  return matches;
}


template<typename T>
ssrlcv::ptr::value<ssrlcv::Unity<ssrlcv::DMatch>> ssrlcv::MatchFactory<T>::generateDistanceMatches(ssrlcv::ptr::value<ssrlcv::Image> query, ssrlcv::ptr::value<Unity<Feature<T>>> queryFeatures, ssrlcv::ptr::value<ssrlcv::Image> target, ssrlcv::ptr::value<Unity<Feature<T>>> targetFeatures, ssrlcv::ptr::value<ssrlcv::Unity<float>> seedDistances){
  MemoryState origin[2] = {queryFeatures->getMemoryState(), targetFeatures->getMemoryState()};

  if(origin[0] != gpu) queryFeatures->setMemoryState(gpu);
  if(origin[1] != gpu) targetFeatures->setMemoryState(gpu);

  unsigned int numPossibleMatches = queryFeatures->size();

  ssrlcv::ptr::value<ssrlcv::Unity<DMatch>> matches = ssrlcv::ptr::value<ssrlcv::Unity<DMatch>>(nullptr, numPossibleMatches, gpu);

  dim3 grid = {1,1,1};
  dim3 block = {32,1,1};//IMPROVE
  getGrid(matches->size(),grid);

  clock_t timer = clock();

  if(seedDistances == nullptr){
    matchFeaturesBruteForce<T><<<grid, block>>>(query->id, queryFeatures->size(), queryFeatures->device.get(),
    target->id, targetFeatures->size(), targetFeatures->device.get(), matches->device.get(),this->absoluteThreshold);
  }
  else if(seedDistances->size() != queryFeatures->size()){
    logger.err<<"ERROR: seedDistances should have come from matching a seed image to queryFeatures";
    exit(-1);
  }
  else{
    MemoryState seedOrigin = seedDistances->getMemoryState();
    if(seedOrigin != gpu) seedDistances->setMemoryState(gpu);
    matchFeaturesBruteForce<T><<<grid, block>>>(query->id, queryFeatures->size(), queryFeatures->device.get(),
    target->id, targetFeatures->size(), targetFeatures->device.get(), matches->device.get(),seedDistances->device.get(),
    this->relativeThreshold,this->absoluteThreshold);
    if(seedOrigin != gpu) seedDistances->setMemoryState(seedOrigin);
  }
  cudaDeviceSynchronize();
  CudaCheckError();

  this->validateMatches(matches);

  logger.info.printf("done in %f seconds.",((float) clock() -  timer)/CLOCKS_PER_SEC);

  if(origin[0] != gpu) queryFeatures->setMemoryState(origin[0]);
  if(origin[1] != gpu) targetFeatures->setMemoryState(origin[1]);

  return matches;
} // generateDistanceMatches

template<typename T>
ssrlcv::ptr::value<ssrlcv::Unity<ssrlcv::DMatch>> ssrlcv::MatchFactory<T>::generateDistanceMatchesKDTree(ssrlcv::ptr::value<ssrlcv::Image> query, ssrlcv::ptr::value<ssrlcv::Unity<ssrlcv::Feature<ssrlcv::SIFT_Descriptor>>> queryFeatures, ssrlcv::ptr::value<ssrlcv::Image> target, ssrlcv::KDTree kdtree, ssrlcv::ptr::value<ssrlcv::Unity<float>> seedDistances) {

  // transfer query points to GPU
  MemoryState q_origin = queryFeatures->getMemoryState();
  if(q_origin != gpu) queryFeatures->setMemoryState(gpu);

  // transfer KD-Tree to GPU
  ssrlcv::ptr::device<ssrlcv::KDTree> d_kdtree(1);
  CudaSafeCall(cudaMemcpy(d_kdtree.get(),&kdtree,sizeof(kdtree),cudaMemcpyHostToDevice));
   
  // transfer KD-Tree nodes to GPU
  thrust::device_vector<typename KDTree::Node> d_nodes = kdtree.nodes;
  typename KDTree::Node* pd_nodes = thrust::raw_pointer_cast(d_nodes.data());

  // transfer KD-Tree points to GPU
  ssrlcv::ptr::value<ssrlcv::Unity<Feature<ssrlcv::SIFT_Descriptor>>> d_points = kdtree.points; 
  MemoryState t_origin = d_points->getMemoryState();
  if(t_origin != gpu) d_points->setMemoryState(gpu); 

  // array to hold the matched pairs
  unsigned int numPossibleMatches = queryFeatures->size();
  ssrlcv::ptr::value<ssrlcv::Unity<DMatch>> matches = ssrlcv::ptr::value<ssrlcv::Unity<DMatch>>(nullptr, numPossibleMatches, gpu);

  // grid and block initilization
  dim3 grid = {1,1,1};
  dim3 block = {1,1,1};
  void (*ptr)(unsigned int, unsigned long, Feature<ssrlcv::SIFT_Descriptor>*, unsigned int, KDTree*,
  typename KDTree::Node*, Feature<ssrlcv::SIFT_Descriptor>*, DMatch*, float) = &matchFeaturesKDTree;
  getFlatGridBlock(queryFeatures->size(), grid, block, ptr);

  clock_t timer = clock();
  
  if (seedDistances == nullptr) {
    matchFeaturesKDTree<<<grid, block>>>(query->id, queryFeatures->size(), queryFeatures->device.get(), 
    target->id, d_kdtree.get(), pd_nodes, d_points->device.get(), matches->device.get(), this->absoluteThreshold);
  } else if (seedDistances->size() != queryFeatures->size()) {
    logger.err<<"ERROR: seedDistances should have come from matching a seed image to queryFeatures"<<"\n";
    exit(-1);
  } else {
    MemoryState seedOrigin = seedDistances->getMemoryState();
    if(seedOrigin != gpu) seedDistances->setMemoryState(gpu);
    matchFeaturesKDTree<<<grid, block>>>(query->id, queryFeatures->size(), queryFeatures->device.get(), 
    target->id, d_kdtree.get(), pd_nodes, d_points->device.get(), matches->device.get(), seedDistances->device.get(),
    this->relativeThreshold, this->absoluteThreshold);
    if(seedOrigin != gpu) seedDistances->setMemoryState(seedOrigin); 
  }
  cudaDeviceSynchronize();
  CudaCheckError();

  this->validateMatches(matches);
  printf("\n\ndone in %f seconds.\n\n",((float) clock() -  timer)/CLOCKS_PER_SEC);
  if(q_origin != gpu) queryFeatures->setMemoryState(q_origin);
  if(t_origin != gpu) kdtree.points->setMemoryState(t_origin);

  return matches;
} // generateDistanceMatchesKDTree

template<typename T>
ssrlcv::ptr::value<ssrlcv::Unity<ssrlcv::DMatch>>ssrlcv::MatchFactory<T>:: generateDistanceMatchesConstrained(ssrlcv::ptr::value<ssrlcv::Image> query, ssrlcv::ptr::value<Unity<Feature<T>>> queryFeatures, ssrlcv::ptr::value<ssrlcv::Image> target, ssrlcv::ptr::value<Unity<Feature<T>>> targetFeatures, float epsilon, float fundamental[3][3], ssrlcv::ptr::value<ssrlcv::Unity<float>> seedDistances){
  MemoryState origin[2] = {queryFeatures->getMemoryState(), targetFeatures->getMemoryState()};

  if(origin[0] != gpu) queryFeatures->setMemoryState(gpu);
  if(origin[1] != gpu) targetFeatures->setMemoryState(gpu);

  unsigned int numPossibleMatches = queryFeatures->size();

  ssrlcv::ptr::device<DMatch> matches_device( numPossibleMatches);

  ssrlcv::ptr::value<ssrlcv::Unity<DMatch>> matches = ssrlcv::ptr::value<ssrlcv::Unity<DMatch>>(matches_device, numPossibleMatches, gpu);

  dim3 grid = {1,1,1};
  dim3 block = {32,1,1};//IMPROVE
  getGrid(matches->size(),grid);

  ssrlcv::ptr::device<float> fundamental_device(9);
  CudaSafeCall(cudaMemcpy(fundamental_device.get(),fundamental,9*sizeof(float),cudaMemcpyHostToDevice));

  clock_t timer = clock();

  if(seedDistances == nullptr){
    matchFeaturesConstrained<T><<<grid, block>>>(query->id, queryFeatures->size(), queryFeatures->device.get(),
    target->id, targetFeatures->size(), targetFeatures->device.get(), matches->device.get(), epsilon, fundamental_device.get(),this->absoluteThreshold);
  }
  else if(seedDistances->size() != queryFeatures->size()){
    logger.err<<"ERROR: seedDistances should have come from matching a seed image to queryFeatures";
    exit(-1);
  }
  else{
    MemoryState seedOrigin = seedDistances->getMemoryState();
    if(seedOrigin != gpu) seedDistances->setMemoryState(gpu);
    matchFeaturesConstrained<T><<<grid, block>>>(query->id, queryFeatures->size(), queryFeatures->device.get(),
    target->id, targetFeatures->size(), targetFeatures->device.get(), matches->device.get(), epsilon, fundamental_device.get(),seedDistances->device.get(),
    this->relativeThreshold,this->absoluteThreshold);
    if(seedOrigin != gpu) seedDistances->setMemoryState(seedOrigin);
  }
  cudaDeviceSynchronize();
  CudaCheckError();

  this->validateMatches(matches);

  logger.info.printf("done in %f seconds.",((float) clock() -  timer)/CLOCKS_PER_SEC);

  if(origin[0] != gpu) queryFeatures->setMemoryState(origin[0]);
  if(origin[1] != gpu) targetFeatures->setMemoryState(origin[1]);

  return matches;
} // generateDistanceMatchesConstrained


template<typename T>
ssrlcv::Unity<ssrlcv::FeatureMatch<T>>* ssrlcv::MatchFactory<T>::generateFeatureMatches(ssrlcv::ptr::value<ssrlcv::Image> query, ssrlcv::ptr::value<ssrlcv::Unity<ssrlcv::Feature<T>>> queryFeatures,
ssrlcv::ptr::value<ssrlcv::Image> target, ssrlcv::ptr::value<ssrlcv::Unity<ssrlcv::Feature<T>>> targetFeatures, ssrlcv::ptr::value<ssrlcv::Unity<float>> seedDistances){

  MemoryState origin[2] = {queryFeatures->getMemoryState(), targetFeatures->getMemoryState()};
  
  if(origin[0] != gpu) queryFeatures->setMemoryState(gpu);
  if(origin[1] != gpu) targetFeatures->setMemoryState(gpu);

  unsigned int numPossibleMatches = queryFeatures->size();

  ssrlcv::ptr::device<FeatureMatch<T>> matches_device( numPossibleMatches);

  Unity<FeatureMatch<T>>* matches = new Unity<FeatureMatch<T>>(matches_device, numPossibleMatches, gpu);

  dim3 grid = {1,1,1};
  dim3 block = {32,1,1};//IMPROVE
  getGrid(matches->size(),grid);

  clock_t timer = clock();

  if(seedDistances == nullptr){
    matchFeaturesBruteForce<T><<<grid, block>>>(query->id, queryFeatures->size(), queryFeatures->device.get(),
    target->id, targetFeatures->size(), targetFeatures->device.get(), matches->device.get(),this->absoluteThreshold);
  }
  else if(seedDistances->size() != queryFeatures->size()){
    logger.err<<"ERROR: seedDistances should have come from matching a seed image to queryFeatures";
    exit(-1);
  }
  else{
    MemoryState seedOrigin = seedDistances->getMemoryState();
    if(seedOrigin != gpu) seedDistances->setMemoryState(gpu);
    matchFeaturesBruteForce<T><<<grid, block>>>(query->id, queryFeatures->size(), queryFeatures->device.get(),
    target->id, targetFeatures->size(), targetFeatures->device.get(), matches->device.get(),seedDistances->device.get(),
    this->relativeThreshold,this->absoluteThreshold);
    if(seedOrigin != gpu) seedDistances->setMemoryState(seedOrigin);
  }
    
  logger.info.printf("done in %f seconds.",((float) clock() -  timer)/CLOCKS_PER_SEC);

  if(origin[0] != gpu) queryFeatures->setMemoryState(origin[0]);
  if(origin[1] != gpu) targetFeatures->setMemoryState(origin[1]);

  return matches;
}
template<typename T>
ssrlcv::Unity<ssrlcv::FeatureMatch<T>>* ssrlcv::MatchFactory<T>::generateFeatureMatchesConstrained(ssrlcv::ptr::value<ssrlcv::Image> query, ssrlcv::ptr::value<ssrlcv::Unity<ssrlcv::Feature<T>>> queryFeatures,
ssrlcv::ptr::value<ssrlcv::Image> target, ssrlcv::ptr::value<ssrlcv::Unity<ssrlcv::Feature<T>>> targetFeatures, float epsilon, float fundamental[3][3], ssrlcv::ptr::value<ssrlcv::Unity<float>> seedDistances){

  MemoryState origin[2] = {queryFeatures->getMemoryState(), targetFeatures->getMemoryState()};
  
  if(origin[0] != gpu) queryFeatures->setMemoryState(gpu);
  if(origin[1] != gpu) targetFeatures->setMemoryState(gpu);

  unsigned int numPossibleMatches = queryFeatures->size();

  ssrlcv::ptr::device<FeatureMatch<T>> matches_device( numPossibleMatches);

  Unity<FeatureMatch<T>>* matches = new Unity<FeatureMatch<T>>(matches_device, numPossibleMatches, gpu);

  dim3 grid = {1,1,1};
  dim3 block = {32,1,1};//IMPROVE
  getGrid(matches->size(),grid);

  ssrlcv::ptr::device<float> fundamental_device(9);
  CudaSafeCall(cudaMemcpy(fundamental_device.get(),fundamental,9*sizeof(float),cudaMemcpyHostToDevice));

  clock_t timer = clock();

  if(seedDistances == nullptr){
    matchFeaturesConstrained<T><<<grid, block>>>(query->id, queryFeatures->size(), queryFeatures->device.get(),
    target->id, targetFeatures->size(), targetFeatures->device.get(), matches->device.get(), epsilon, fundamental_device.get(),this->absoluteThreshold);
  }
  else if(seedDistances->size() != queryFeatures->size()){
    logger.err<<"ERROR: seedDistances should have come from matching a seed image to queryFeatures";
    exit(-1);
  }
  else{
    MemoryState seedOrigin = seedDistances->getMemoryState();
    if(seedOrigin != gpu) seedDistances->setMemoryState(gpu);
    matchFeaturesConstrained<T><<<grid, block>>>(query->id, queryFeatures->size(), queryFeatures->device.get(),
    target->id, targetFeatures->size(), targetFeatures->device.get(), matches->device.get(), epsilon, fundamental_device.get(),seedDistances->device.get(),
    this->relativeThreshold,this->absoluteThreshold);
    if(seedOrigin != gpu) seedDistances->setMemoryState(seedOrigin);
  }
  cudaDeviceSynchronize();
  CudaCheckError();

  this->validateMatches(matches);

  logger.info.printf("done in %f seconds.",((float) clock() -  timer)/CLOCKS_PER_SEC);

  if(origin[0] != gpu) queryFeatures->setMemoryState(origin[0]);
  if(origin[1] != gpu) targetFeatures->setMemoryState(origin[1]);

  return matches;

}



template<typename T>
ssrlcv::ptr::value<ssrlcv::Unity<ssrlcv::uint2_pair>> ssrlcv::MatchFactory<T>::generateMatchesIndexOnly(ssrlcv::ptr::value<ssrlcv::Image> query, ssrlcv::ptr::value<Unity<Feature<T>>> queryFeatures, ssrlcv::ptr::value<ssrlcv::Image> target, ssrlcv::ptr::value<Unity<Feature<T>>> targetFeatures, ssrlcv::ptr::value<ssrlcv::Unity<float>> seedDistances){
  MemoryState origin[2] = {queryFeatures->getMemoryState(), targetFeatures->getMemoryState()};

  if(origin[0] != gpu) queryFeatures->setMemoryState(gpu);
  if(origin[1] != gpu) targetFeatures->setMemoryState(gpu);

  unsigned int numPossibleMatches = queryFeatures->size();

  ssrlcv::ptr::device<uint2_pair> matches_device( numPossibleMatches);
  ssrlcv::ptr::value<ssrlcv::Unity<uint2_pair>> matches = ssrlcv::ptr::value<ssrlcv::Unity<uint2_pair>>(matches_device, numPossibleMatches, gpu);

  dim3 grid = {1,1,1};
  dim3 block = {32,1,1};//IMPROVE
  getGrid(matches->size(),grid);

  clock_t timer = clock();

  if(seedDistances == nullptr){
    matchFeaturesBruteForce<T><<<grid, block>>>(query->id, queryFeatures->size(), queryFeatures->device.get(),
    target->id, targetFeatures->size(), targetFeatures->device.get(), matches->device.get(),this->absoluteThreshold);
  }
  else if(seedDistances->size() != queryFeatures->size()){
    logger.err<<"ERROR: seedDistances should have come from matching a seed image to queryFeatures";
    exit(-1);
  }
  else{
    MemoryState seedOrigin = seedDistances->getMemoryState();
    if(seedOrigin != gpu) seedDistances->setMemoryState(gpu);
    matchFeaturesBruteForce<T><<<grid, block>>>(query->id, queryFeatures->size(), queryFeatures->device.get(),
    target->id, targetFeatures->size(), targetFeatures->device.get(), matches->device.get(),seedDistances->device.get(),
    this->relativeThreshold,this->absoluteThreshold);
    if(seedOrigin != gpu) seedDistances->setMemoryState(seedOrigin);
  }
  
  cudaDeviceSynchronize();
  CudaCheckError();

  this->validateMatches(matches);

  logger.info.printf("done in %f seconds.",((float) clock() -  timer)/CLOCKS_PER_SEC);

  if(origin[0] != gpu) queryFeatures->setMemoryState(origin[0]);
  if(origin[1] != gpu) targetFeatures->setMemoryState(origin[1]);

  return matches;
}
template<typename T>
ssrlcv::ptr::value<ssrlcv::Unity<ssrlcv::uint2_pair>> ssrlcv::MatchFactory<T>::generateMatchesConstrainedIndexOnly(ssrlcv::ptr::value<ssrlcv::Image> query, ssrlcv::ptr::value<Unity<Feature<T>>> queryFeatures, ssrlcv::ptr::value<ssrlcv::Image> target, ssrlcv::ptr::value<Unity<Feature<T>>> targetFeatures, float epsilon, float fundamental[3][3], ssrlcv::ptr::value<ssrlcv::Unity<float>> seedDistances){
  MemoryState origin[2] = {queryFeatures->getMemoryState(), targetFeatures->getMemoryState()};

  if(origin[0] != gpu) queryFeatures->setMemoryState(gpu);
  if(origin[1] != gpu) targetFeatures->setMemoryState(gpu);

  unsigned int numPossibleMatches = queryFeatures->size();

  ssrlcv::ptr::device<uint2_pair> matches_device( numPossibleMatches);

  ssrlcv::ptr::value<ssrlcv::Unity<uint2_pair>> matches = ssrlcv::ptr::value<ssrlcv::Unity<uint2_pair>>(matches_device, numPossibleMatches, gpu);

  dim3 grid = {1,1,1};
  dim3 block = {32,1,1};//IMPROVE
  getGrid(matches->size(),grid);

  ssrlcv::ptr::device<float> fundamental_device(9);
  CudaSafeCall(cudaMemcpy(fundamental_device.get(),fundamental,9*sizeof(float),cudaMemcpyHostToDevice));

  clock_t timer = clock();

  if(seedDistances == nullptr){
    matchFeaturesConstrained<<<grid, block>>>(query->id, queryFeatures->size(), queryFeatures->device.get(),
    target->id, targetFeatures->size(), targetFeatures->device.get(), matches->device.get(), epsilon,fundamental_device.get(),this->absoluteThreshold);
  }
  else if(seedDistances->size() != queryFeatures->size()){
    logger.err<<"ERROR: seedDistances should have come from matching a seed image to queryFeatures";
    exit(-1);
  }
  else{
    MemoryState seedOrigin = seedDistances->getMemoryState();
    if(seedOrigin != gpu) seedDistances->setMemoryState(gpu);
    matchFeaturesConstrained<T><<<grid, block>>>(query->id, queryFeatures->size(), queryFeatures->device.get(),
    target->id, targetFeatures->size(), targetFeatures->device.get(), matches->device.get(),epsilon,fundamental_device.get(),seedDistances->device.get(),
    this->relativeThreshold,this->absoluteThreshold);
    if(seedOrigin != gpu) seedDistances->setMemoryState(seedOrigin);
  }

  cudaDeviceSynchronize();
  CudaCheckError();

  this->validateMatches(matches);

  logger.info.printf("done in %f seconds.",((float) clock() -  timer)/CLOCKS_PER_SEC);

  if(origin[0] != gpu) queryFeatures->setMemoryState(origin[0]);
  if(origin[1] != gpu) targetFeatures->setMemoryState(origin[1]);

  return matches;
}


template<typename T>
ssrlcv::MatchSet ssrlcv::MatchFactory<T>::generateMatchesExaustive(std::vector<ssrlcv::ptr::value<ssrlcv::Image>> images, std::vector<ssrlcv::ptr::value<ssrlcv::Unity<ssrlcv::Feature<T>>>> features, bool ordered, float estimatedOverlap){
  MatchSet matchSet;
  matchSet.keyPoints = nullptr;
  matchSet.matches = nullptr;
  if(estimatedOverlap == 0){
    logger.warn<<"WARNING: estimated overlap fraction of 0.0f requires unordered match interpolation";
  }
  std::vector<ssrlcv::ptr::value<ssrlcv::Image>>::iterator query = images.begin();
  std::vector<ssrlcv::ptr::value<ssrlcv::Image>>::iterator target = query + 1;
  typename std::vector<ssrlcv::ptr::value<Unity<Feature<T>>>>::iterator features_query = features.begin();
  typename std::vector<ssrlcv::ptr::value<Unity<Feature<T>>>>::iterator features_target = features_query + 1;
  std::vector<ssrlcv::ptr::value<ssrlcv::Unity<uint2_pair>>> matchIndices;
  ssrlcv::ptr::value<ssrlcv::Unity<float>> seedDistances;
  unsigned long long totalMatches = 0;
  int i = 0;
  logger.info<<"matching images";
  for(int i = 0; query != images.end() - 1; ++query, ++features_query){
    if(this->seedFeatures != nullptr) seedDistances = this->getSeedDistances(*features_query);
    for(target = query + 1,features_target = features_query + 1; target != images.end(); ++target, ++features_target){
      if(ordered && estimatedOverlap > 0.0f && ++i*(1-estimatedOverlap) > 1.0f) continue; //based off linear images
      //now match
      matchIndices.push_back(this->generateMatchesIndexOnly(*query,*features_query,*target,*features_target,seedDistances));
      totalMatches += matchIndices[i++]->size();
    }
  }
  if(totalMatches == 0){
    logger.err<<"There were no matches found in the set of images, likely due to unreasonable threshold";
    logger.err<<"exiting...";
    exit(0);
  }
  logger.info<<"prepping match interpolation on cpu";
  //required connections to make a match?
  std::vector<uint2>** adjacencyList = new std::vector<uint2>*[images.size() - 1];

  i = 0;
  adjacencyList[0] = new std::vector<uint2>[features[0]->size()];
  logger.info<<"building adjacency list";
  for(auto m = matchIndices.begin(); m != matchIndices.end(); ++m){
    ssrlcv::ptr::value<ssrlcv::Unity<uint2_pair>> currentMatches = *m;
    if(currentMatches->getMemoryState() != cpu) currentMatches->setMemoryState(cpu);
    if(currentMatches->host.get()[0].a.x != i){
      i++;
      adjacencyList[i] = new std::vector<uint2>[features[i]->size()];
    }
    for(int p = 0; p < currentMatches->size(); ++p){
      uint2_pair* currentPair = &currentMatches->host.get()[p];
      adjacencyList[currentPair->a.x][currentPair->a.y].push_back(currentPair->b); 
    }
  }
  MemoryState* origin = new MemoryState[images.size()];
  std::vector<std::vector<uint2>> multiMatch_vec;
  bool badMatch = false;
  logger.info<<"deriving matches from adjacency";
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
  logger.info.printf("total matches found in set = %d", multiMatch_vec.size());
  matchSet.matches = ssrlcv::ptr::value<ssrlcv::Unity<MultiMatch>>(nullptr,multiMatch_vec.size(),cpu);
  std::vector<KeyPoint> kp_vec;
  i = 0;
  int index = 0;
  for(auto m = multiMatch_vec.begin(); m != multiMatch_vec.end(); ++m){
    matchSet.matches->host.get()[i++] = {(unsigned int)m->size(),index};
    index += m->size();
    for(auto kp = m->begin(); kp != m->end(); ++kp){
      kp_vec.push_back({(int)kp->x,features[kp->x]->host.get()[kp->y].loc});
    }
  }
  matchSet.keyPoints = ssrlcv::ptr::value<ssrlcv::Unity<KeyPoint>>(nullptr,kp_vec.size(),gpu);
  CudaSafeCall(cudaMemcpy(matchSet.keyPoints->device.get(),&kp_vec[0],kp_vec.size()*sizeof(KeyPoint),cudaMemcpyHostToDevice));
  for(int i = 0; i < images.size() - 1; ++i){
    if(origin[i] != cpu) features[i]->setMemoryState(origin[i]);
  }
  delete[] origin;
  //1:2,1:3,1:4,1:5,2:3,2:4,2:5,3:4,3:5,4:5
  return matchSet;

}


ssrlcv::ptr::value<ssrlcv::Unity<ssrlcv::Match>> ssrlcv::generateDiparityMatches(uint2 querySize, ssrlcv::ptr::value<ssrlcv::Unity<unsigned char>> queryPixels, uint2 targetSize, ssrlcv::ptr::value<ssrlcv::Unity<unsigned char>> targetPixels, 
  float fundamental[3][3], unsigned int maxDisparity,unsigned int windowSize, Direction direction){
  if(direction != right && direction != left && direction != undefined){
    logger.err<<"ERROR: unsupported search direction for disparity matching";
    exit(-1);
  }
  if(maxDisparity > querySize.x){
    logger.err<<"Max disparity cannot be larger than image size";
    exit(-1);
  }
  logger.info.printf("running disparity matching on parallel images");
  logger.info.printf("\timage[0] = %ux%u",querySize.x,querySize.y);
  logger.info.printf("\timage[1] = %ux%u",targetSize.x,targetSize.y);
  logger.info.printf("\tmaxDisparity = %u",maxDisparity);
  logger.info.printf("\twindow size = %ux%u",windowSize,windowSize);

  if(windowSize == 0 || windowSize % 2 == 0 || windowSize > 31){
    logger.err<<"ERROR window size for disparity matching must be greater than 0, less than 31 and odd";
    exit(-1);
  }

  MemoryState origin[2] = {queryPixels->getMemoryState(), targetPixels->getMemoryState()};

  if(origin[0] != gpu) queryPixels->setMemoryState(gpu);
  if(origin[1] != gpu) targetPixels->setMemoryState(gpu);
  
  uint2 minimizedSize = {querySize.x-windowSize-1,querySize.y-windowSize-1};

  unsigned int numPossibleMatches = minimizedSize.x*minimizedSize.y;

  ssrlcv::ptr::device<Match> matches_device( numPossibleMatches);

  ssrlcv::ptr::value<ssrlcv::Unity<Match>> matches = ssrlcv::ptr::value<ssrlcv::Unity<Match>>(matches_device, numPossibleMatches, gpu);

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
    ssrlcv::ptr::device<float> fundamental_device(9);
    CudaSafeCall(cudaMemcpy(fundamental_device.get(),fundamental,9*sizeof(float),cudaMemcpyHostToDevice));
    disparityMatching<<<grid, block>>>(querySize,queryPixels->device.get(),targetSize,targetPixels->device.get(),fundamental_device.get(),matches->device.get(),maxDisparity,direction);
  }
  else{
    disparityScanMatching<<<grid,block>>>(querySize,queryPixels->device.get(),targetSize,targetPixels->device.get(),matches->device.get(),maxDisparity,direction);
  }
  
  cudaDeviceSynchronize();
  CudaCheckError();  
  logger.info.printf("done in %f seconds.",((float) clock() -  timer)/CLOCKS_PER_SEC);

  if(origin[0] != gpu) queryPixels->setMemoryState(origin[0]);
  if(origin[1] != gpu) targetPixels->setMemoryState(origin[1]);

  thrust::device_ptr<Match> needsValidating(matches->device.get());
  thrust::device_ptr<Match> new_end = thrust::remove_if(needsValidating,needsValidating+matches->size(),validate());
  cudaDeviceSynchronize();
  CudaCheckError();
  int numMatchesLeft = new_end - needsValidating;
  if(numMatchesLeft == 0){
    logger.info<<"No valid matches found";
    matches.clear();
  }
  else{
    logger.info.printf("%d valid matches found out of %lu original matches",numMatchesLeft,matches->size());
    ssrlcv::ptr::device<Match> validatedMatches_device(numMatchesLeft);
    CudaSafeCall(cudaMemcpy(validatedMatches_device.get(),matches->device.get(),numMatchesLeft*sizeof(Match),cudaMemcpyDeviceToDevice));
    matches->setData(validatedMatches_device,numMatchesLeft,gpu);
  }
  //returning based on origin of query pixels
  if(origin[0] != gpu) matches->setMemoryState(origin[0]);
  return matches;
}


void ssrlcv::writeMatchFile(ssrlcv::ptr::value<ssrlcv::Unity<Match>> matches, std::string pathToFile, bool binary){
  MemoryState origin = matches->getMemoryState();
  if(matches->getFore() == gpu) matches->transferMemoryTo(cpu);
  if(binary){
    std::ofstream matchstream(pathToFile,std::ios_base::binary);
    if(matchstream.is_open()){
      for(int i = 0; i < matches->size(); ++i){
        matchstream.write((char*)&matches->host.get()[i].keyPoints[0].loc,2*sizeof(float));
        matchstream.write((char*)&matches->host.get()[i].keyPoints[1].loc,2*sizeof(float));
      }
    }
    else{
      logger.err<<"ERROR: cannot write "<<pathToFile;
    }
    matchstream.close();
  }
  else{
    std::ofstream matchstream(pathToFile);
    if(matchstream.is_open()){
      std::string line;
      for(int i = 0; i < matches->size(); ++i){
        line = std::to_string(matches->host.get()[i].keyPoints[0].loc.x) + ",";
        line += std::to_string(matches->host.get()[i].keyPoints[0].loc.y) + ",";
        line += std::to_string(matches->host.get()[i].keyPoints[1].loc.x) + ",";
        line += std::to_string(matches->host.get()[i].keyPoints[1].loc.y) + "\n";
        matchstream << line;
      }
      matchstream.close();
    }
    else{
      logger.err<<"ERROR: cannot write match files";
      exit(-1);
    }
  }
  logger.info << pathToFile + " has been written";
  if(origin != matches->getMemoryState()) matches->setMemoryState(origin);
}
void ssrlcv::writeMatchFile(MatchSet multiview_matches, std::string pathToFile, bool binary){
  ssrlcv::ptr::value<ssrlcv::Unity<MultiMatch>> matches = multiview_matches.matches;
  ssrlcv::ptr::value<ssrlcv::Unity<KeyPoint>> keyPoints = multiview_matches.keyPoints;
  MemoryState origin[2] = {matches->getMemoryState(),keyPoints->getMemoryState()};
  if(origin[0] != cpu) matches->setMemoryState(cpu);
  if(origin[1] != cpu) keyPoints->setMemoryState(cpu);

  std::ofstream matchstream(pathToFile);
  if(matchstream.is_open()){
    std::string line;
    for(int i = 0; i < matches->size(); ++i){
      line = std::to_string(matches->host.get()[i].numKeyPoints) + ",";
      for(int kp = matches->host.get()[i].index; kp < matches->host.get()[i].index + matches->host.get()[i].numKeyPoints; ++kp){
        line += std::to_string(keyPoints->host.get()[kp].parentId) + ",";
        line += std::to_string(keyPoints->host.get()[kp].loc.x) + ",";
        line += std::to_string(keyPoints->host.get()[kp].loc.y) + ",";
      }
      line += "\n";
      matchstream << line;
    }
    matchstream.close();
  }
  else{
    logger.err<<"ERROR: cannot write match files";
    exit(-1);
  }
  
  logger.info << pathToFile + " has been written";
  if(origin[0] != cpu) matches->setMemoryState(origin[0]);
  if(origin[1] != cpu) keyPoints->setMemoryState(origin[1]);
}

//NOTE currently only capable of reading in pairwise match files
ssrlcv::ptr::value<ssrlcv::Unity<ssrlcv::Match>> ssrlcv::readMatchFile(std::string pathToFile){
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
  logger.info.printf("%d matches have been read.", match_vec.size());
  ssrlcv::ptr::value<ssrlcv::Unity<Match>> matches = ssrlcv::ptr::value<ssrlcv::Unity<Match>>(nullptr,match_vec.size(),cpu);
  std::memcpy(matches->host.get(),&match_vec[0],match_vec.size()*sizeof(Match));
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

/*
  diparity matching kernels
*/

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


/*
  Matching kernels
*/

/**
 * @brief This function computes the seed match distances.
 * 
 * @tparam T 
 * @param numFeaturesQuery the number of features to query 
 * @param featuresQuery a ptr to the feature vector to be queried
 * @param numSeedFeatures the number of seed features
 * @param seedFeatures a ptr to the feature vector containing the seed features
 * @param matchDistances a ptr to the match distances
 * @return __global__ 
 */
template<typename T>
__global__ void ssrlcv::getSeedMatchDistances(unsigned long numFeaturesQuery, Feature<T>* featuresQuery, unsigned long numSeedFeatures,
Feature<T>* seedFeatures, float* matchDistances){
  // define the block ID
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
    } // for
    __syncthreads();
    if(threadIdx.x != 0) return;
    currentDist = FLT_MAX;
    for(int i = 0; i < 32; ++i){
      if(currentDist > localDist[i]){
        currentDist = localDist[i];
      }
    } // if
    matchDistances[blockId] = currentDist;
  } // if
} // getSeedMatchDistances


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
//template<typename T>
__global__ void ssrlcv::matchFeaturesKDTree(unsigned int queryImageID, unsigned long numFeaturesQuery, ssrlcv::Feature<ssrlcv::SIFT_Descriptor>* featuresQuery,
unsigned int targetImageID, ssrlcv::KDTree* kdtree, typename ssrlcv::KDTree::Node* nodes, ssrlcv::Feature<ssrlcv::SIFT_Descriptor>* featuresTree,
ssrlcv::DMatch* matches, float absoluteThreshold) {
  
  unsigned int globalThreadID = (blockIdx.y * gridDim.x + blockIdx.x) * blockDim.x + threadIdx.x; // 2D grid of 1D blocks 
  if (globalThreadID < numFeaturesQuery) { 
    Feature<ssrlcv::SIFT_Descriptor> feature = featuresQuery[globalThreadID]; 
    __syncthreads(); 
    DMatch match;
    int emax = 100; // at most, search 100 leaf nodes
    match = findNearest(kdtree, nodes, featuresTree, feature, queryImageID, targetImageID, emax, absoluteThreshold); // search function 
    __syncthreads();
    matches[globalThreadID] = match;
  } 

} // matchFeaturesKDTree

//template<typename T>
__global__ void ssrlcv::matchFeaturesKDTree(unsigned int queryImageID, unsigned long numFeaturesQuery, ssrlcv::Feature<ssrlcv::SIFT_Descriptor>* featuresQuery,
unsigned int targetImageID, ssrlcv::KDTree* kdtree, typename ssrlcv::KDTree::Node* nodes, ssrlcv::Feature<ssrlcv::SIFT_Descriptor>* featuresTree,
ssrlcv::DMatch* matches, float* seedDistances, float relativeThreshold, float absoluteThreshold) {
  
  unsigned int globalThreadID = (blockIdx.y * gridDim.x + blockIdx.x) * blockDim.x + threadIdx.x; // 2D grid of 1D blocks
  
  if (globalThreadID < numFeaturesQuery) { 
    Feature<ssrlcv::SIFT_Descriptor> feature = featuresQuery[globalThreadID];
    float nearestSeed = seedDistances[globalThreadID];
    __syncthreads();
    
    DMatch match;
    int emax = 100; // at most, search 100 leaf nodes
    match = findNearest(kdtree, nodes, featuresTree, feature, queryImageID, targetImageID, emax, relativeThreshold, absoluteThreshold, nearestSeed); 
    __syncthreads();
    matches[globalThreadID] = match;
  } 

} // matchFeaturesKDTree

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


/*
  utility kernels
*/
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
