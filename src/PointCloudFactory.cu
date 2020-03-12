
#include "PointCloudFactory.cuh"

ssrlcv::PointCloudFactory::PointCloudFactory(){

}

/**
* The CPU method that sets up the GPU enabled line generation, which stores lines
* and sets of lines as bundles
* @param matchSet a group of maches
* @param a group of images, used only for their stored camera parameters
*/
ssrlcv::BundleSet ssrlcv::PointCloudFactory::generateBundles(MatchSet* matchSet, std::vector<ssrlcv::Image*> images){


  Unity<Bundle>* bundles = new Unity<Bundle>(nullptr,matchSet->matches->size(),gpu);
  Unity<Bundle::Line>* lines = new Unity<Bundle::Line>(nullptr,matchSet->keyPoints->size(),gpu);

  // std::cout << "starting bundle generation ..." << std::endl;
  //MemoryState origin[2] = {matchSet->matches->getMemoryState(),matchSet->keyPoints->getMemoryState()};
  //if(origin[0] == cpu) matchSet->matches->transferMemoryTo(gpu);
  //if(origin[1] == cpu) matchSet->keyPoints->transferMemoryTo(gpu);
  // std::cout << "set the matches ... " << std::endl;
  matchSet->matches->transferMemoryTo(gpu);
  matchSet->keyPoints->transferMemoryTo(gpu);

  // the cameras
  size_t cam_bytes = images.size()*sizeof(ssrlcv::Image::Camera);
  // fill the cam boi
  ssrlcv::Image::Camera* h_cameras;
  h_cameras = (ssrlcv::Image::Camera*) malloc(cam_bytes);
  for(int i = 0; i < images.size(); i++){
    h_cameras[i] = images.at(i)->camera;
  }
  ssrlcv::Image::Camera* d_cameras;
  // std::cout << "set the cameras ... " << std::endl;
  CudaSafeCall(cudaMalloc(&d_cameras, cam_bytes));
  // copy the othe guy
  CudaSafeCall(cudaMemcpy(d_cameras, h_cameras, cam_bytes, cudaMemcpyHostToDevice));

  dim3 grid = {1,1,1};
  dim3 block = {1,1,1};
  getFlatGridBlock(bundles->size(),grid,block,generateBundle);

  //in this kernel fill lines and bundles from keyPoints and matches
  // std::cout << "Calling bundle generation kernel ..." << std::endl;
  generateBundle<<<grid, block>>>(bundles->size(),bundles->device, lines->device, matchSet->matches->device, matchSet->keyPoints->device, d_cameras);
  // std::cout << "Returned from bundle generation kernel ... \n" << std::endl;

  cudaDeviceSynchronize();
  CudaCheckError();

  // transfer and clear the match set information
  matchSet->matches->setFore(gpu);
  matchSet->keyPoints->setFore(gpu);
  matchSet->matches->transferMemoryTo(cpu);
  matchSet->keyPoints->transferMemoryTo(cpu);
  matchSet->matches->clear(gpu);
  matchSet->keyPoints->clear(gpu);

  // transfer and clear the cpu information
  bundles->transferMemoryTo(cpu);
  bundles->clear(gpu);
  lines->transferMemoryTo(cpu);
  lines->clear(gpu);

  BundleSet bundleSet = {lines,bundles};

  return bundleSet;
}

/**
* The CPU method that sets up the GPU enabled two view tringulation.
* @param bundleSet a set of lines and bundles that should be triangulated
* @param linearError is the total linear error of the triangulation, it is an analog for reprojection error
*/
ssrlcv::Unity<float3>* ssrlcv::PointCloudFactory::twoViewTriangulate(BundleSet bundleSet, float* linearError){

  // to total error cacluation is stored in this guy
  *linearError = 0;
  float* d_linearError;
  size_t eSize = sizeof(float);
  CudaSafeCall(cudaMalloc((void**) &d_linearError,eSize));
  CudaSafeCall(cudaMemcpy(d_linearError,linearError,eSize,cudaMemcpyHostToDevice));

  bundleSet.lines->transferMemoryTo(gpu);
  bundleSet.bundles->transferMemoryTo(gpu);

  // Unity<float3>* pointcloud = new Unity<float3>(nullptr,2*bundleSet.bundles->size(),gpu);
  Unity<float3>* pointcloud = new Unity<float3>(nullptr,bundleSet.bundles->size(),gpu);

  dim3 grid = {1,1,1};
  dim3 block = {1,1,1};
  void (*fp)(float*, unsigned long, Bundle::Line*, Bundle*, float3*) = &computeTwoViewTriangulate;
  getFlatGridBlock(bundleSet.bundles->size(),grid,block,fp);

  // std::cout << "Starting 2-view triangulation ..." << std::endl;
  computeTwoViewTriangulate<<<grid,block>>>(d_linearError,bundleSet.bundles->size(),bundleSet.lines->device,bundleSet.bundles->device,pointcloud->device);
  // std::cout << "2-view Triangulation done ... \n" << std::endl;

  cudaDeviceSynchronize();
  CudaCheckError();

  //transfer the poitns back to the CPU
  pointcloud->transferMemoryTo(cpu);
  pointcloud->clear(gpu);
  // clear the other boiz
  bundleSet.lines->clear(gpu);
  bundleSet.bundles->clear(gpu);
  // copy back the total error that occured
  CudaSafeCall(cudaMemcpy(linearError,d_linearError,eSize,cudaMemcpyDeviceToHost));
  cudaFree(d_linearError);

  return pointcloud;
}

/**
* The CPU method that sets up the GPU enabled two view tringulation.
* @param bundleSet a set of lines and bundles that should be triangulated
* @param the individual linear errors (for use in debugging and histogram)
* @param linearError is the total linear error of the triangulation, it is an analog for reprojection error
*/
ssrlcv::Unity<float3>* ssrlcv::PointCloudFactory::twoViewTriangulate(BundleSet bundleSet, Unity<float>* errors, float* linearError){

  // to total error cacluation is stored in this guy
  *linearError = 0;
  float* d_linearError;
  size_t eSize = sizeof(float);
  CudaSafeCall(cudaMalloc((void**) &d_linearError,eSize));
  CudaSafeCall(cudaMemcpy(d_linearError,linearError,eSize,cudaMemcpyHostToDevice));

  bundleSet.lines->transferMemoryTo(gpu);
  bundleSet.bundles->transferMemoryTo(gpu);
  errors->transferMemoryTo(gpu);

  // Unity<float3>* pointcloud = new Unity<float3>(nullptr,2*bundleSet.bundles->size(),gpu);
  Unity<float3>* pointcloud = new Unity<float3>(nullptr,bundleSet.bundles->size(),gpu);

  dim3 grid = {1,1,1};
  dim3 block = {1,1,1};
  void (*fp)(float*, float*, unsigned long, Bundle::Line*, Bundle*, float3*) = &computeTwoViewTriangulate;
  getFlatGridBlock(bundleSet.bundles->size(),grid,block,fp);

  // std::cout << "Starting 2-view triangulation ..." << std::endl;
  computeTwoViewTriangulate<<<grid,block>>>(d_linearError,errors->device,bundleSet.bundles->size(),bundleSet.lines->device,bundleSet.bundles->device,pointcloud->device);
  // std::cout << "2-view Triangulation done ... \n" << std::endl;

  cudaDeviceSynchronize();
  CudaCheckError();

  // transfer the poitns back to the CPU
  pointcloud->transferMemoryTo(cpu);
  pointcloud->clear(gpu);
  // transfer the individual linear errors back to the CPU
  errors->transferMemoryTo(cpu);
  errors->clear(gpu);
  // clear the other boiz
  bundleSet.lines->clear(gpu);
  bundleSet.bundles->clear(gpu);
  // copy back the total error that occured
  CudaSafeCall(cudaMemcpy(linearError,d_linearError,eSize,cudaMemcpyDeviceToHost));
  cudaFree(d_linearError);

  return pointcloud;
}

/**
* The CPU method that sets up the GPU enabled two view tringulation.
* @param bundleSet a set of lines and bundles that should be triangulated
* @param the individual linear errors (for use in debugging and histogram)
* @param linearError is the total linear error of the triangulation, it is an analog for reprojection error
* @param linearErrorCutoff is a value that all linear errors should be less than. points with larger errors are discarded.
*/
ssrlcv::Unity<float3>* ssrlcv::PointCloudFactory::twoViewTriangulate(BundleSet bundleSet, Unity<float>* errors, float* linearError, float* linearErrorCutoff){

  // to total error cacluation is stored in this guy
  *linearError = 0;
  float* d_linearError;
  size_t eSize = sizeof(float);
  CudaSafeCall(cudaMalloc((void**) &d_linearError,eSize));
  CudaSafeCall(cudaMemcpy(d_linearError,linearError,eSize,cudaMemcpyHostToDevice));
  // the cutoff boi
  // *linearErrorCutoff = 10000.0;
  float* d_linearErrorCutoff;
  size_t cutSize = sizeof(float);
  CudaSafeCall(cudaMalloc((void**) &d_linearErrorCutoff,cutSize));
  CudaSafeCall(cudaMemcpy(d_linearErrorCutoff,linearErrorCutoff,cutSize,cudaMemcpyHostToDevice));

  bundleSet.lines->transferMemoryTo(gpu);
  bundleSet.bundles->transferMemoryTo(gpu);
  errors->transferMemoryTo(gpu);

  // Unity<float3>* pointcloud = new Unity<float3>(nullptr,2*bundleSet.bundles->size(),gpu);
  Unity<float3>* pointcloud = new Unity<float3>(nullptr,bundleSet.bundles->size(),gpu);

  dim3 grid = {1,1,1};
  dim3 block = {1,1,1};
  void (*fp)(float*, float*, float*, unsigned long, Bundle::Line*, Bundle*, float3*) = &computeTwoViewTriangulate;
  getFlatGridBlock(bundleSet.bundles->size(),grid,block,fp);

  // std::cout << "Starting 2-view triangulation ..." << std::endl;
  computeTwoViewTriangulate<<<grid,block>>>(d_linearError,d_linearErrorCutoff,errors->device,bundleSet.bundles->size(),bundleSet.lines->device,bundleSet.bundles->device,pointcloud->device);
  // std::cout << "2-view Triangulation done ... \n" << std::endl;

  // tell which data is most up to date
  bundleSet.lines->setFore(gpu);
  bundleSet.bundles->setFore(gpu);
  errors->setFore(gpu);

  cudaDeviceSynchronize();
  CudaCheckError();

  // return to find invalid bundles
  bundleSet.lines->transferMemoryTo(cpu);
  bundleSet.bundles->transferMemoryTo(cpu);
  bundleSet.lines->clear(gpu);
  bundleSet.bundles->clear(gpu);

  // transfer the poitns back to the CPU
  pointcloud->transferMemoryTo(cpu);
  pointcloud->clear(gpu);
  // transfer the individual linear errors back to the CPU
  errors->transferMemoryTo(cpu);
  errors->clear(gpu);

  // copy back the total error that occured
  CudaSafeCall(cudaMemcpy(linearError,d_linearError,eSize,cudaMemcpyDeviceToHost));
  cudaFree(d_linearError);
  // free the cutoff, it's not needed on the cpu again tho
  cudaFree(d_linearErrorCutoff);

  return pointcloud;
}

/**
* The CPU method that sets up the GPU enabled two view tringulation.
* This method uses the extra bit in the float3 data structure as a "filter" bit which can be used to remove bad points
* @param bundleSet a set of lines and bundles that should be triangulated
* @param the individual linear errors (for use in debugging and histogram)
* @param linearError is the total linear error of the triangulation, it is an analog for reprojection error
* @param linearErrorCutoff is a value that all linear errors should be less than. points with larger errors are discarded.
*/
ssrlcv::Unity<float3_b>* ssrlcv::PointCloudFactory::twoViewTriangulate_b(BundleSet bundleSet, Unity<float>* errors, float* linearError, float* linearErrorCutoff){

    // to total error cacluation is stored in this guy
    *linearError = 0;
    float* d_linearError;
    size_t eSize = sizeof(float);
    CudaSafeCall(cudaMalloc((void**) &d_linearError,eSize));
    CudaSafeCall(cudaMemcpy(d_linearError,linearError,eSize,cudaMemcpyHostToDevice));
    // the cutoff boi
    // *linearErrorCutoff = 10000.0;
    float* d_linearErrorCutoff;
    size_t cutSize = sizeof(float);
    CudaSafeCall(cudaMalloc((void**) &d_linearErrorCutoff,cutSize));
    CudaSafeCall(cudaMemcpy(d_linearErrorCutoff,linearErrorCutoff,cutSize,cudaMemcpyHostToDevice));

    bundleSet.lines->transferMemoryTo(gpu);
    bundleSet.bundles->transferMemoryTo(gpu);
    errors->transferMemoryTo(gpu);

    Unity<float3_b>* pointcloud_b = new Unity<float3_b>(nullptr,bundleSet.bundles->size(),gpu);

    dim3 grid = {1,1,1};
    dim3 block = {1,1,1};
    getFlatGridBlock(bundleSet.bundles->size(),grid,block,computeTwoViewTriangulate_b);

    // std::cout << "Starting 2-view triangulation ..." << std::endl;
    computeTwoViewTriangulate_b<<<grid,block>>>(d_linearError,d_linearErrorCutoff,errors->device,bundleSet.bundles->size(),bundleSet.lines->device,bundleSet.bundles->device,pointcloud_b->device);
    // std::cout << "2-view Triangulation done ... \n" << std::endl;

    pointcloud_b->setFore(gpu);
    errors->setFore(gpu);

    cudaDeviceSynchronize();
    CudaCheckError();

    // transfer the poitns back to the CPU
    pointcloud_b->transferMemoryTo(cpu);
    pointcloud_b->clear(gpu);
    // transfer the individual linear errors back to the CPU
    errors->transferMemoryTo(cpu);
    errors->clear(gpu);
    // temp

    std::cout << std::endl;
    // end temp
    // clear the other boiz
    bundleSet.lines->clear(gpu);
    bundleSet.bundles->clear(gpu);
    // copy back the total error that occured
    CudaSafeCall(cudaMemcpy(linearError,d_linearError,eSize,cudaMemcpyDeviceToHost));
    cudaFree(d_linearError);
    // free the cutoff, it's not needed on the cpu again tho
    cudaFree(d_linearErrorCutoff);

    return pointcloud_b;
}

//TODO put this in the right place, in the same order as the header file
ssrlcv::Unity<float3>* ssrlcv::PointCloudFactory::nViewTriangulate(BundleSet bundleSet){
  bundleSet.lines->transferMemoryTo(gpu);
  bundleSet.bundles->transferMemoryTo(gpu);

  Unity<float3>* pointcloud = new Unity<float3>(nullptr,bundleSet.bundles->size(),gpu);

  dim3 grid = {1,1,1};
  dim3 block = {1,1,1};
  getFlatGridBlock(bundleSet.bundles->size(),grid,block,generateBundle);

  /*
  std::cout << "Starting n-view triangulation ..." << std::endl;
  computeNViewTriangulate<<<grid,block>>>(bundleSet.bundles->size(),bundleSet.lines->device,bundleSet.bundles->device,pointcloud->device);
  std::cout << "n-view Triangulation done ... \n" << std::endl;
  */

  pointcloud->transferMemoryTo(cpu);
  pointcloud->clear(gpu);
  bundleSet.lines->clear(gpu);
  bundleSet.bundles->clear(gpu);

  return pointcloud;
}

/**
 * Same method as two view triangulation, but all that is desired fro this method is a calculation of the linearError
 * @param bundleSet a set of lines and bundles that should be triangulated
 * @param linearError is the total linear error of the triangulation, it is an analog for reprojection error
 * @param linearErrorCutoff is a value that all linear errors should be less than. points with larger errors are discarded.
 */
void ssrlcv::PointCloudFactory::voidTwoViewTriangulate(BundleSet bundleSet, float* linearError, float* linearErrorCutoff){
  // to total error cacluation is stored in this guy
  *linearError = 0;
  float* d_linearError;
  size_t eSize = sizeof(float);
  CudaSafeCall(cudaMalloc((void**) &d_linearError,eSize));
  CudaSafeCall(cudaMemcpy(d_linearError,linearError,eSize,cudaMemcpyHostToDevice));
  // the cutoff boi
  // *linearErrorCutoff = 10000.0;
  float* d_linearErrorCutoff;
  size_t cutSize = sizeof(float);
  CudaSafeCall(cudaMalloc((void**) &d_linearErrorCutoff,cutSize));
  CudaSafeCall(cudaMemcpy(d_linearErrorCutoff,linearErrorCutoff,cutSize,cudaMemcpyHostToDevice));

  bundleSet.lines->transferMemoryTo(gpu);
  bundleSet.bundles->transferMemoryTo(gpu);

  // Unity<float3>* pointcloud = new Unity<float3>(nullptr,2*bundleSet.bundles->numElements,gpu);
  Unity<float3>* pointcloud = new Unity<float3>(nullptr,bundleSet.bundles->size(),gpu);

  dim3 grid = {1,1,1};
  dim3 block = {1,1,1};
  getFlatGridBlock(bundleSet.bundles->size(),grid,block,voidComputeTwoViewTriangulate);

  voidComputeTwoViewTriangulate<<<grid,block>>>(d_linearError,d_linearErrorCutoff,bundleSet.bundles->size(),bundleSet.lines->device,bundleSet.bundles->device);

  cudaDeviceSynchronize();
  CudaCheckError();

  // transfer the poitns back to the CPU
  pointcloud->transferMemoryTo(cpu);
  pointcloud->clear(gpu);
  // clear the other boiz
  bundleSet.lines->clear(gpu);
  bundleSet.bundles->clear(gpu);
  // copy back the total error that occured
  CudaSafeCall(cudaMemcpy(linearError,d_linearError,eSize,cudaMemcpyDeviceToHost));
  cudaFree(d_linearError);
  // free the cutoff, it's not needed on the cpu again tho
  cudaFree(d_linearErrorCutoff);

  return;
}

/**
* Preforms a Stereo Disparity with the correct scalar, calcualated form camera
* parameters
* @param matches0
* @param matches1
* @param points assumes this has been allocated prior to method call
* @param n the number of matches
* @param cameras a camera array of only 2 Image::Camera structs. This is used to
* dynamically calculate a scaling factor
*/
ssrlcv::Unity<float3>* ssrlcv::PointCloudFactory::stereo_disparity(Unity<Match>* matches, Image::Camera* cameras){

  float baseline = sqrtf( (cameras[0].cam_pos.x - cameras[1].cam_pos.x)*(cameras[0].cam_pos.x - cameras[1].cam_pos.x)
                        + (cameras[0].cam_pos.y - cameras[1].cam_pos.y)*(cameras[0].cam_pos.y - cameras[1].cam_pos.y)
                        + (cameras[0].cam_pos.z - cameras[1].cam_pos.z)*(cameras[0].cam_pos.z - cameras[1].cam_pos.z));
  float scale = (baseline * cameras[0].foc )/(cameras[0].dpix.x);

  std::cout << "Stereo Baseline: " << baseline << ", Stereo Scale Factor: " << scale <<  ", Inverted Stereo Scale Factor: " << (1.0/scale) << std::endl;

  MemoryState origin = matches->getMemoryState();
  if(origin == cpu) matches->transferMemoryTo(gpu);

  // depth points
  float3 *points_device = nullptr;

  cudaMalloc((void**) &points_device, matches->size()*sizeof(float3));

  //
  dim3 grid = {1,1,1};
  dim3 block = {1,1,1};
  void (*fp)(unsigned int, Match*, float3*, float) = &computeStereo;
  getFlatGridBlock(matches->size(),grid,block,fp);
  //
  computeStereo<<<grid, block>>>(matches->size(), matches->device, points_device, 8.0);
  // focal lenth / baseline

  // computeStereo<<<grid, block>>>(matches->size(), matches->device, points_device, 64.0);

  Unity<float3>* points = new Unity<float3>(points_device, matches->size(),gpu);
  if(origin == cpu) matches->setMemoryState(cpu);

  return points;
}

/**
* Preforms a Stereo Disparity, this SHOULD NOT BE THE DEFAULT as the scale is not
* dyamically calculated
* @param matches0
* @param matches1
* @param points assumes this has been allocated prior to method call
* @param n the number of matches
* @param scale the scale factor that is multiplied
*/
ssrlcv::Unity<float3>* ssrlcv::PointCloudFactory::stereo_disparity(Unity<Match>* matches, float scale){

  MemoryState origin = matches->getMemoryState();
  if(origin == cpu) matches->transferMemoryTo(gpu);

  // depth points
  float3 *points_device = nullptr;

  cudaMalloc((void**) &points_device, matches->size()*sizeof(float3));

  //
  dim3 grid = {1,1,1};
  dim3 block = {1,1,1};
  void (*fp)(unsigned int, Match*, float3*, float) = &computeStereo;
  getFlatGridBlock(matches->size(),grid,block,fp);
  //
  computeStereo<<<grid, block>>>(matches->size(), matches->device, points_device, scale);

  Unity<float3>* points = new Unity<float3>(points_device, matches->size(),gpu);
  if(origin == cpu) matches->setMemoryState(cpu);

  return points;
}

ssrlcv::Unity<float3>* ssrlcv::PointCloudFactory::stereo_disparity(Unity<Match>* matches, float foc, float baseline, float doffset){

  MemoryState origin = matches->getMemoryState();
  if(origin == cpu) matches->transferMemoryTo(gpu);


  Unity<float3>* points = new Unity<float3>(nullptr, matches->size(),gpu);
  //
  dim3 grid = {1,1,1};
  dim3 block = {1,1,1};
  void (*fp)(unsigned int, Match*, float3*, float, float, float) = &computeStereo;
  getFlatGridBlock(matches->size(),grid,block,fp);
  //
  computeStereo<<<grid, block>>>(matches->size(), matches->device, points->device, foc, baseline, doffset);

  if(origin == cpu) matches->setMemoryState(cpu);

  return points;
}

/**
 * A Naive bundle adjustment based on a two-view triangulation and a first order descrete gradient decent
 * @param matchSet a group of matches
 * @param a group of images, used only for their stored camera parameters
 * @return a bundle adjusted point cloud
 */
ssrlcv::Unity<float3>* ssrlcv::PointCloudFactory::BundleAdjustTwoView(ssrlcv::MatchSet* matchSet, std::vector<ssrlcv::Image*> images){

  // the initial linear error
  float* linearError = (float*) malloc(sizeof(float));
  *linearError = 100; // just something to satisfy the first if statment
  float* linearError_partial = (float*) malloc(sizeof(float));
  *linearError_partial = 0.0;
  // the cutoff
  float* linearErrorCutoff = (float*) malloc(sizeof(float));
  *linearErrorCutoff = 900100.0;
  // make the temporary image struct
  std::vector<ssrlcv::Image*> partials;
  partials.push_back(images[0]);
  partials.push_back(images[1]);
  // previous image information
  std::vector<ssrlcv::Image*> images_prev;
  images_prev.push_back(images[0]);
  images_prev.push_back(images[1]);
  // used to store the gradients in the camera
  std::vector<ssrlcv::Image::Camera> gradients;
  ssrlcv::Image::Camera gr_1 = ssrlcv::Image::Camera();
  ssrlcv::Image::Camera gr_2 = ssrlcv::Image::Camera();
  gradients.push_back(gr_1);
  gradients.push_back(gr_2);

  // used to store the gradients in the camera
  std::vector<ssrlcv::Image::Camera> gradients_prev;
  ssrlcv::Image::Camera gr_3 = ssrlcv::Image::Camera();
  ssrlcv::Image::Camera gr_4 = ssrlcv::Image::Camera();
  gradients_prev.push_back(gr_3);
  gradients_prev.push_back(gr_4);

  // the boiz in the loop
  ssrlcv::BundleSet bundleSet;
  ssrlcv::BundleSet bundleSet_partial;
  ssrlcv::MatchSet tempMatchSet;
  tempMatchSet.keyPoints = new ssrlcv::Unity<ssrlcv::KeyPoint>(nullptr,1,ssrlcv::cpu);
  tempMatchSet.matches   = new ssrlcv::Unity<ssrlcv::MultiMatch>(nullptr,1,ssrlcv::cpu);
  ssrlcv::Unity<float>*    errors;
  ssrlcv::Unity<float>*    errors_sample;
  ssrlcv::Unity<float3>*   points;
  ssrlcv::Unity<float3_b>* points_b;

  // for printing out data about iterations and shit later
  std::vector<float> errorTracker;

  // TODO make statistical filtering its own method
  /*
    // // the assumption is that choosing every 10 indexes is random enough
    // // https://en.wikipedia.org/wiki/Variance#Sample_variance
    // size_t sample_size = (int) (errors->size() - (errors->size()%10))/10; // make sure divisible by 10 always
    // errors_sample      = new ssrlcv::Unity<float>(nullptr,sample_size,ssrlcv::cpu);
    // float sample_sum = 0;
    // for (int k = 0; k < sample_size; k++){
    //   errors_sample->host[k] = errors->host[k*10];
    //   sample_sum += errors->host[k*10];
    // }
    // float sample_mean = sample_sum / errors_sample->size();
    // float squared_sum = 0;
    // for (int k = 0; k < sample_size; k++){
    //   squared_sum += (errors_sample->host[k] - sample_mean)*(errors_sample->host[k] - sample_mean);
    // }
    // float variance = squared_sum / errors_sample->size();
    // // std::cout << "Sample variance: " << variance << std::endl;
    // std::cout << "\tSigma Calculated As: " << sqrtf(variance) << std::endl;
    // std::cout << "\tLinear Error Cutoff Adjusted To: " << 5 * sqrtf(variance) << std::endl;
    // *linearErrorCutoff = 5 * sqrtf(variance);
    //
    // // only do this once
    // if (i == 1) ssrlcv::writeCSV(errors_sample->host, (int) errors_sample->size(), "filteredIndividualLinearErrors" + std::to_string(i));
    //
    // // recalculate with new cutoff
    // points = twoViewTriangulate(bundleSet, errors, linearError, linearErrorCutoff);
    //
    // // CLEAR OUT THE DATA STRUCTURES
    // // count the number of bad bundles to be removed
    // int bad_bundles = 0;
    // for (int k = 0; k < bundleSet.bundles->size(); k++){
    //   if (bundleSet.bundles->host[k].invalid){
    //      bad_bundles++;
    //   }
    // }
    // if (bad_bundles) std::cout << "Detected " << bad_bundles << " bad bundles to remove" << std::endl;
    // // Need to generated and adjustment match set
    // // make a temporary match set
    // delete tempMatchSet.keyPoints;
    // delete tempMatchSet.matches;
    // tempMatchSet.keyPoints = new ssrlcv::Unity<ssrlcv::KeyPoint>(nullptr,matchSet->matches->size()*2,ssrlcv::cpu);
    // tempMatchSet.matches   = new ssrlcv::Unity<ssrlcv::MultiMatch>(nullptr,matchSet->matches->size(),ssrlcv::cpu);
    // // fill in the boiz
    // for (int k = 0; k < tempMatchSet.keyPoints->size(); k++){
    //   tempMatchSet.keyPoints->host[k] = matchSet->keyPoints->host[k];
    // }
    // for (int k = 0; k < tempMatchSet.matches->size(); k++){
    //   tempMatchSet.matches->host[k] = matchSet->matches->host[k];
    // }
    // // resize the standard matchSet
    // size_t new_kp_size = 2*(matchSet->matches->size() - bad_bundles);
    // size_t new_mt_size = matchSet->matches->size() - bad_bundles;
    // delete matchSet->keyPoints;
    // delete matchSet->matches;
    // matchSet->keyPoints = new ssrlcv::Unity<ssrlcv::KeyPoint>(nullptr,new_kp_size,ssrlcv::cpu);
    // matchSet->matches   = new ssrlcv::Unity<ssrlcv::MultiMatch>(nullptr,new_mt_size,ssrlcv::cpu);
    // // this is much easier because of the 2 view assumption
    // // there are the same number of lines as there are are keypoints and the same number of bundles as there are matches
    // int k_adjust = 0;
    // // if (bad_bundles){
    // for (int k = 0; k < bundleSet.bundles->size(); k++){
    // 	if (!bundleSet.bundles->host[k].invalid){
    // 	  matchSet->keyPoints->host[2*k_adjust]     = tempMatchSet.keyPoints->host[2*k];
    // 	  matchSet->keyPoints->host[2*k_adjust + 1] = tempMatchSet.keyPoints->host[2*k + 1];
    // 	  // matchSet->matches->host[k_adjust]         = tempMatchSet.matches->host[k];
    //     matchSet->matches->host[k_adjust]         = {2,2*k_adjust};
    // 	  k_adjust++;
    // 	}
    // }
    //
    // // only do this once for now
    // if (i == 1)
    //
    // if (false){
    //   std::cout << "\tsize of bundles:       " << bundleSet.bundles->size() << std::endl;
    //   std::cout << "\tgood bundles:          " << bundleSet.bundles->size() - bad_bundles << std::endl;
    //   std::cout << "\tsize of old matches:   " << tempMatchSet.matches->size() << std::endl;
    //   std::cout << "\tsize of new matches:   " << matchSet->matches->size() << std::endl;
    //   std::cout << "\tk_adjust:              " << k_adjust << std::endl;
    //   std::cout << "\tsize of old keyPoints: " << tempMatchSet.keyPoints->size() << std::endl;
    //   std::cout << "\tsize of new keyPoints: " << matchSet->keyPoints->size() << std::endl;
    // }
  */

  // https://en.wikipedia.org/wiki/Gradient_descent#Description
  int i = 1;
  float min_step = 0.000001;
  float h_rot = min_step;
  float h_pos = min_step;
  float h_foc = min_step;
  float h_fov = min_step;
  // the stepsize along the gradient
  float step  = min_step;
  while(i < 5){
  // while(*linearError > (100000.0)*matchSet->matches->numElements){
    // generate the bundle set
    bundleSet = generateBundles(matchSet,images);

    // do an initial triangulation
    errors = new ssrlcv::Unity<float>(nullptr,matchSet->matches->size(),ssrlcv::cpu);

    // for filtering
    points = twoViewTriangulate(bundleSet, errors, linearError, linearErrorCutoff);

    // do this only once
    if (i == 1) ssrlcv::writePLY("out/rawPoints.ply",points);
    // write some errors for debug
    // for now only do this once
    if (i == 1) ssrlcv::writeCSV(errors->host, (int) errors->size(), "individualLinearErrors" + std::to_string(i));

    // clear uneeded memory
    delete bundleSet.lines;
    delete bundleSet.bundles;
    delete errors;
    //delete errors_sample;
    // delete points;

    // a nice printout for the humans
    std::cout << std::fixed;
    std::cout << std::setprecision(16);
    // std::cout << "==============================================================" << std::endl;
    std::cout << "\n[itr: " << i << "] linear error: " << *linearError << std::endl;
    errorTracker.push_back(*linearError);

    //
    // Calculate the new gradient
    //
    for (int j = 0; j < partials.size(); j++){

      // rotations

      // partial for x rotation
      partials[j]->camera.cam_rot.x = images[j]->camera.cam_rot.x + h_rot;
      // get the error
      bundleSet_partial = generateBundles(matchSet,partials);
      // std::cout << "BundleSet Size: " << bundleSet_partial.bundles->numElements << std::endl;
      voidTwoViewTriangulate(bundleSet_partial, linearError_partial, linearErrorCutoff);
      delete bundleSet_partial.lines;
      delete bundleSet_partial.bundles;
      gradients[j].cam_rot.x = (*linearError - *linearError_partial) / (h_rot);
      // reset
      partials[j]->camera.cam_rot.x = images[j]->camera.cam_rot.x;

      // partial for y rotation
      partials[j]->camera.cam_rot.y = images[j]->camera.cam_rot.y + h_rot;
      // get the error
      bundleSet_partial = generateBundles(matchSet,partials);
      voidTwoViewTriangulate(bundleSet_partial, linearError_partial, linearErrorCutoff);
      delete bundleSet_partial.lines;
      delete bundleSet_partial.bundles;
      gradients[j].cam_rot.y = (*linearError - *linearError_partial) / (h_rot);
      // reset
      partials[j]->camera.cam_rot.y = images[j]->camera.cam_rot.y;

      // partial for z rotation
      partials[j]->camera.cam_rot.z = images[j]->camera.cam_rot.z + h_rot;
      // get the error
      bundleSet_partial = generateBundles(matchSet,partials);
      voidTwoViewTriangulate(bundleSet_partial, linearError_partial, linearErrorCutoff);
      delete bundleSet_partial.lines;
      delete bundleSet_partial.bundles;
      gradients[j].cam_rot.z = (*linearError - *linearError_partial) / (h_rot);
      // reset
      partials[j]->camera.cam_rot.z = images[j]->camera.cam_rot.z;

      // positions

      // partial for x position
      partials[j]->camera.cam_pos.x = images[j]->camera.cam_pos.x + h_pos;
      // get the error
      bundleSet_partial = generateBundles(matchSet,partials);
      voidTwoViewTriangulate(bundleSet_partial, linearError_partial, linearErrorCutoff);
      delete bundleSet_partial.lines;
      delete bundleSet_partial.bundles;
      gradients[j].cam_pos.x = (*linearError - *linearError_partial) / (h_pos);
      // reset
      partials[j]->camera.cam_pos.x = images[j]->camera.cam_pos.x;

      // partial for y position
      partials[j]->camera.cam_pos.y = images[j]->camera.cam_rot.y + h_pos;
      // get the error
      bundleSet_partial = generateBundles(matchSet,partials);
      voidTwoViewTriangulate(bundleSet_partial, linearError_partial, linearErrorCutoff);
      delete bundleSet_partial.lines;
      delete bundleSet_partial.bundles;
      gradients[j].cam_pos.y = (*linearError - *linearError_partial) / (h_pos);
      // reset
      partials[j]->camera.cam_pos.y = images[j]->camera.cam_pos.y;

      // partial for z position
      partials[j]->camera.cam_pos.z = images[j]->camera.cam_pos.z + h_pos;
      // get the error
      bundleSet_partial = generateBundles(matchSet,partials);
      voidTwoViewTriangulate(bundleSet_partial, linearError_partial, linearErrorCutoff);
      delete bundleSet_partial.lines;
      delete bundleSet_partial.bundles;
      gradients[j].cam_pos.z = (*linearError - *linearError_partial) / (h_pos);
      // reset
      partials[j]->camera.cam_pos.z = images[j]->camera.cam_pos.z;

      // focal length

      // partial for focal length
      partials[j]->camera.foc = images[j]->camera.foc + h_foc;
      // update dpix
      partials[j]->camera.dpix.x = (partials[j]->camera.foc * tanf(partials[j]->camera.fov.x / 2.0f)) / (partials[j]->camera.size.x / 2.0f );
      partials[j]->camera.dpix.y = partials[j]->camera.dpix.x;
      // get the error
      bundleSet_partial = generateBundles(matchSet,partials);
      voidTwoViewTriangulate(bundleSet_partial, linearError_partial, linearErrorCutoff);
      delete bundleSet_partial.lines;
      delete bundleSet_partial.bundles;
      gradients[j].foc = (*linearError - *linearError_partial) / (h_foc);
      // reset
      partials[j]->camera.foc = images[j]->camera.foc;
      partials[j]->camera.dpix = images[j]->camera.dpix;

      // field of view

      // partial for x field of view
      partials[j]->camera.fov.x = images[j]->camera.fov.x + h_fov;
      // update dpix
      partials[j]->camera.dpix.x = (partials[j]->camera.foc * tanf(partials[j]->camera.fov.x / 2.0f)) / (partials[j]->camera.size.x / 2.0f );
      partials[j]->camera.dpix.y = partials[j]->camera.dpix.x;
      // get the error
      bundleSet_partial = generateBundles(matchSet,partials);
      voidTwoViewTriangulate(bundleSet_partial, linearError_partial, linearErrorCutoff);
      delete bundleSet_partial.lines;
      delete bundleSet_partial.bundles;
      gradients[j].fov.x = (*linearError - *linearError_partial) / (h_fov);
      // reset
      partials[j]->camera.fov.x = images[j]->camera.fov.x;
      partials[j]->camera.dpix = images[j]->camera.dpix;

      // partial for y field of view
      partials[j]->camera.fov.y = images[j]->camera.fov.y + h_fov;
      // update dpix
      partials[j]->camera.dpix.x = (partials[j]->camera.foc * tanf(partials[j]->camera.fov.x / 2.0f)) / (partials[j]->camera.size.x / 2.0f );
      partials[j]->camera.dpix.y = partials[j]->camera.dpix.x;
      // get the error
      bundleSet_partial = generateBundles(matchSet,partials);
      voidTwoViewTriangulate(bundleSet_partial, linearError_partial,linearErrorCutoff);
      delete bundleSet_partial.lines;
      delete bundleSet_partial.bundles;
      gradients[j].fov.y = (*linearError - *linearError_partial) / (h_fov);
      // reset
      partials[j]->camera.fov.y = images[j]->camera.fov.y;
      partials[j]->camera.dpix = images[j]->camera.dpix;

      // normalize the gradient now that all components are known
      float norm = 0.0f;
      norm += (gradients[j].cam_rot.x)*(gradients[j].cam_rot.x);
      norm += (gradients[j].cam_rot.y)*(gradients[j].cam_rot.y);
      norm += (gradients[j].cam_rot.z)*(gradients[j].cam_rot.z);
      norm += (gradients[j].cam_pos.x)*(gradients[j].cam_pos.x);
      norm += (gradients[j].cam_pos.y)*(gradients[j].cam_pos.y);
      norm += (gradients[j].cam_pos.z)*(gradients[j].cam_pos.z);
      norm += (gradients[j].foc)*(gradients[j].foc);
      norm += (gradients[j].fov.x)*(gradients[j].fov.x);
      norm += (gradients[j].fov.y)*(gradients[j].fov.y);
      norm = sqrtf(norm);
      gradients[j].cam_rot.x /= norm;
      gradients[j].cam_rot.y /= norm;
      gradients[j].cam_rot.z /= norm;
      gradients[j].cam_pos.x /= norm;
      gradients[j].cam_pos.y /= norm;
      gradients[j].cam_pos.z /= norm;
      gradients[j].foc /= norm;
      gradients[j].fov.x /= norm;
      gradients[j].fov.y /= norm;
    } // end generate gradients

    // std::cout << "Calculated gradients ..." << std::endl;

    if (false){
      for (int j = 0; j < partials.size(); j++){
        std::cout << std::fixed;
        std::cout << std::setprecision(12);
        std::cout << "===============================================" << std::endl;
        std::cout << "Gradient " << std::to_string(j+1) << " ::" << std::endl;
        std::cout << "rot: (";
        std::cout << gradients[j].cam_rot.x << ",";
        std::cout << gradients[j].cam_rot.y << ",";
        std::cout << gradients[j].cam_rot.z << ")\t" << std::endl;
        std::cout << "pos: (";
        std::cout << gradients[j].cam_pos.x << ",";
        std::cout << gradients[j].cam_pos.y << ",";
        std::cout << gradients[j].cam_pos.z << ")\t" << std::endl;
        std::cout << "foc: ";
        std::cout << gradients[j].foc << "\t";
        std::cout << "fov: (";
        std::cout << gradients[j].fov.x << ",";
        std::cout << gradients[j].fov.y << ")" << std::endl;
      }
    }

    //
    // calculate the step size if there was a step already taken
    //
    if (i > 1){
      // previous x_n-1
      float x_0[18] = {
        images_prev[0]->camera.cam_rot.x,
        images_prev[0]->camera.cam_rot.y,
        images_prev[0]->camera.cam_rot.z,
        images_prev[0]->camera.cam_pos.x,
        images_prev[0]->camera.cam_pos.y,
        images_prev[0]->camera.cam_pos.z,
        images_prev[0]->camera.foc,
        images_prev[0]->camera.fov.x,
        images_prev[0]->camera.fov.y,
        images_prev[1]->camera.cam_rot.x,
        images_prev[1]->camera.cam_rot.y,
        images_prev[1]->camera.cam_rot.z,
        images_prev[1]->camera.cam_pos.x,
        images_prev[1]->camera.cam_pos.y,
        images_prev[1]->camera.cam_pos.z,
        images_prev[1]->camera.foc,
        images_prev[1]->camera.fov.x,
        images_prev[1]->camera.fov.y
      };
      // current x_n
      float x_1[18] = {
        images[0]->camera.cam_rot.x,
        images[0]->camera.cam_rot.y,
        images[0]->camera.cam_rot.z,
        images[0]->camera.cam_pos.x,
        images[0]->camera.cam_pos.y,
        images[0]->camera.cam_pos.z,
        images[0]->camera.foc,
        images[0]->camera.fov.x,
        images[0]->camera.fov.y,
        images[1]->camera.cam_rot.x,
        images[1]->camera.cam_rot.y,
        images[1]->camera.cam_rot.z,
        images[1]->camera.cam_pos.x,
        images[1]->camera.cam_pos.y,
        images[1]->camera.cam_pos.z,
        images[1]->camera.foc,
        images[1]->camera.fov.x,
        images[1]->camera.fov.y
      };
      // previous gradient
      float g_0[18] = {
        gradients_prev[0].cam_rot.x,
        gradients_prev[0].cam_rot.y,
        gradients_prev[0].cam_rot.z,
        gradients_prev[0].cam_pos.x,
        gradients_prev[0].cam_pos.y,
        gradients_prev[0].cam_pos.z,
        gradients_prev[0].foc,
        gradients_prev[0].fov.x,
        gradients_prev[0].fov.y,
        gradients_prev[1].cam_rot.x,
        gradients_prev[1].cam_rot.y,
        gradients_prev[1].cam_rot.z,
        gradients_prev[1].cam_pos.x,
        gradients_prev[1].cam_pos.y,
        gradients_prev[1].cam_pos.z,
        gradients_prev[1].foc,
        gradients_prev[1].fov.x,
        gradients_prev[1].fov.y
      };
      // current gradient
      float g_1[18] = {
        gradients[0].cam_rot.x,
        gradients[0].cam_rot.y,
        gradients[0].cam_rot.z,
        gradients[0].cam_pos.x,
        gradients[0].cam_pos.y,
        gradients[0].cam_pos.z,
        gradients[0].foc,
        gradients[0].fov.x,
        gradients[0].fov.y,
        gradients[1].cam_rot.x,
        gradients[1].cam_rot.y,
        gradients[1].cam_rot.z,
        gradients[1].cam_pos.x,
        gradients[1].cam_pos.y,
        gradients[1].cam_pos.z,
        gradients[1].foc,
        gradients[1].fov.x,
        gradients[1].fov.y
      };

/*
      std::cout << "\tx_0: [ ";
      for (int k = 0; k < 18; k++){
        std::cout << x_0[k] << ", ";
      }
      std::cout << "] " << std::endl;

      std::cout << "\tx_1: [ ";
      for (int k = 0; k < 18; k++){
        std::cout << x_1[k] << ", ";
      }
      std::cout << "] " << std::endl;

      std::cout << "\tg_0: [ ";
      for (int k = 0; k < 18; k++){
        std::cout << g_0[k] << ", ";
      }
      std::cout << "] " << std::endl;

      std::cout << "\tg_1: [ ";
      for (int k = 0; k < 18; k++){
        std::cout << g_1[k] << ", ";
      }
      std::cout << "] " << std::endl;

      std::cout << "calculating step size ..." << std::endl;
      */

      float sub_x[18];
      //std::cout << "\t sub_x: [ ";
      for (int k = 0; k < 18; k++){
        sub_x[k] = x_1[k] - x_0[k];
        //std::cout << sub_x[k] << ", ";
      }
      //std::cout << " ]" << std::endl;

      float sub_g[18];
      float norm = 0.0f;
      //std::cout << "\t sub_g: [ ";
      for (int k = 0; k < 18; k++){
        sub_g[k] = g_1[k] - g_0[k];
        norm += sub_g[k] * sub_g[k];
        //std::cout << sub_g[k] << ", ";
      }
      norm = sqrtf(norm);
      //std::cout << " ]" << std::endl;
      //std::cout << "\t norm: " << norm << std::endl;

      float sub_g_norm[18];
      //std::cout << "\t sub_g_norm: [ ";
      for (int k = 0; k < 18; k++){
        sub_g_norm[k] /= norm;
        //std::cout << sub_g_norm[k] << ", ";
      }
      //std::cout << " ]" << std::endl;


      float denom = 0.0f;
      for (int k = 0; k < 18; k++){
        denom += sub_g_norm[k] * sub_g_norm[k];
      }
      //std::cout << "\t denom: " << denom << std::endl;

      //denom = sqrtf(denom);
      float numer = 0.0f;
      for (int k = 0; k < 18; k ++){
        numer += sub_x[k] * sub_g[k];
      }
      //std::cout << "\t numer: " << numer << std::endl;

      step = numer/denom;
      std::cout << "\tnew stepsize: " << step << std::endl;
    } // end stepsize calculation

    //->camera
    // take a step down the hill!
    //
    for (int j = 0; j < images.size(); j++){

      // set the previous here
      images_prev[j]->camera.cam_rot.x = images[j]->camera.cam_rot.x;
      images_prev[j]->camera.cam_rot.y = images[j]->camera.cam_rot.y;
      images_prev[j]->camera.cam_rot.z = images[j]->camera.cam_rot.z;
      images_prev[j]->camera.cam_pos.x = images[j]->camera.cam_pos.x;
      images_prev[j]->camera.cam_pos.y = images[j]->camera.cam_pos.y;
      images_prev[j]->camera.cam_pos.z = images[j]->camera.cam_pos.z;
      images_prev[j]->camera.foc       = images[j]->camera.foc;
      images_prev[j]->camera.fov.x     = images[j]->camera.fov.x;
      images_prev[j]->camera.fov.y     = images[j]->camera.fov.y;
      // update dpix
      images_prev[j]->camera.dpix.x = images[j]->camera.dpix.x;
      images_prev[j]->camera.dpix.y = images[j]->camera.dpix.y;

      // take the gradient step here

      images[j]->camera.cam_rot.x = images[j]->camera.cam_rot.x - step * gradients[j].cam_rot.x;
      images[j]->camera.cam_rot.y = images[j]->camera.cam_rot.y - step * gradients[j].cam_rot.y;
      images[j]->camera.cam_rot.z = images[j]->camera.cam_rot.z - step * gradients[j].cam_rot.z;
      images[j]->camera.cam_pos.x = images[j]->camera.cam_pos.x - step * gradients[j].cam_pos.x;
      images[j]->camera.cam_pos.y = images[j]->camera.cam_pos.y - step * gradients[j].cam_pos.y;
      images[j]->camera.cam_pos.z = images[j]->camera.cam_pos.z - step * gradients[j].cam_pos.z;
      images[j]->camera.foc       = images[j]->camera.foc       - step * gradients[j].foc      ;
      images[j]->camera.fov.x     = images[j]->camera.fov.x     - step * gradients[j].fov.x    ;
      images[j]->camera.fov.y     = images[j]->camera.fov.y     - step * gradients[j].fov.y    ;
      // up->cameradate dpix
      images[j]->camera.dpix.x = (images[j]->camera.foc * tanf(images[j]->camera.fov.x / 2.0f)) / (images[j]->camera.size.x / 2.0f );
      images[j]->camera.dpix.y = images[j]->camera.dpix.x;
    }

    // set the old gradients
    gradients_prev[0].cam_rot.x   = gradients[0].cam_rot.x;
    gradients_prev[0].cam_rot.y   = gradients[0].cam_rot.y;
    gradients_prev[0].cam_rot.z   = gradients[0].cam_rot.z;
    gradients_prev[0].cam_pos.x   = gradients[0].cam_pos.x;
    gradients_prev[0].cam_pos.y   = gradients[0].cam_pos.y;
    gradients_prev[0].cam_pos.z   = gradients[0].cam_pos.z;
    gradients_prev[0].foc         = gradients[0].foc;
    gradients_prev[0].fov.x       = gradients[0].fov.x;
    gradients_prev[0].fov.y       = gradients[0].fov.y;
    gradients_prev[1].cam_rot.x   = gradients[1].cam_rot.x;
    gradients_prev[1].cam_rot.y   = gradients[1].cam_rot.y;
    gradients_prev[1].cam_rot.z   = gradients[1].cam_rot.z;
    gradients_prev[1].cam_pos.x   = gradients[1].cam_pos.x;
    gradients_prev[1].cam_pos.y   = gradients[1].cam_pos.y;
    gradients_prev[1].cam_pos.z   = gradients[1].cam_pos.z;
    gradients_prev[1].foc         = gradients[1].foc;
    gradients_prev[1].fov.x       = gradients[1].fov.x;
    gradients_prev[1].fov.y       = gradients[1].fov.y;

/*
    std::cout << "gradient_prev: [";
    std::cout << ", " << gradients_prev[0].cam_rot.x;
    std::cout << ", " << gradients_prev[0].cam_rot.y;
    std::cout << ", " << gradients_prev[0].cam_rot.z;
    std::cout << ", " << gradients_prev[0].cam_pos.x;
    std::cout << ", " << gradients_prev[0].cam_pos.y;
    std::cout << ", " << gradients_prev[0].cam_pos.z;
    std::cout << ", " << gradients_prev[0].foc      ;
    std::cout << ", " << gradients_prev[0].fov.x    ;
    std::cout << ", " << gradients_prev[0].fov.y    ;
    std::cout << ", " << gradients_prev[1].cam_rot.x;
    std::cout << ", " << gradients_prev[1].cam_rot.y;
    std::cout << ", " << gradients_prev[1].cam_rot.z;
    std::cout << ", " << gradients_prev[1].cam_pos.x;
    std::cout << ", " << gradients_prev[1].cam_pos.y;
    std::cout << ", " << gradients_prev[1].cam_pos.z;
    std::cout << ", " << gradients_prev[1].foc      ;
    std::cout << ", " << gradients_prev[1].fov.x    ;
    std::cout << ", " << gradients_prev[1].fov.y    ;
    std::cout << " ]" << std::endl;

    std::cout << "images_prev: [";
    // set the previous here
    std::cout << ", " << images_prev[0]->camera.cam_rot.x;
    std::cout << ", " << images_prev[0]->camera.cam_rot.y;
    std::cout << ", " << images_prev[0]->camera.cam_rot.z;
    std::cout << ", " << images_prev[0]->camera.cam_pos.x;
    std::cout << ", " << images_prev[0]->camera.cam_pos.y;
    std::cout << ", " << images_prev[0]->camera.cam_pos.z;
    std::cout << ", " << images_prev[0]->camera.foc      ;
    std::cout << ", " << images_prev[0]->camera.fov.x    ;
    std::cout << ", " << images_prev[0]->camera.fov.y    ;
    std::cout << ", " << images_prev[0]->camera.dpix.x   ;
    std::cout << ", " << images_prev[0]->camera.dpix.y   ;
    std::cout << ", " << images_prev[1]->camera.cam_rot.x;
    std::cout << ", " << images_prev[1]->camera.cam_rot.y;
    std::cout << ", " << images_prev[1]->camera.cam_rot.z;
    std::cout << ", " << images_prev[1]->camera.cam_pos.x;
    std::cout << ", " << images_prev[1]->camera.cam_pos.y;
    std::cout << ", " << images_prev[1]->camera.cam_pos.z;
    std::cout << ", " << images_prev[1]->camera.foc      ;
    std::cout << ", " << images_prev[1]->camera.fov.x    ;
    std::cout << ", " << images_prev[1]->camera.fov.y    ;
    std::cout << ", " << images_prev[1]->camera.dpix.x   ;
    std::cout << ", " << images_prev[1]->camera.dpix.y   ;
    std::cout << " ]" << std::endl;
*/
    // yahboi
    i++;

  } // end main bundle adjustment loop

  // write linearError chagnes to a CSV
  writeCSV(errorTracker, "totalErrorOverIterations");
  std::cout << "Final Camera Parameters: " << std::endl;
  // take a step down the hill!
  for (int j = 0; j < images.size(); j++){
    std::cout << std::fixed;
    std::cout << std::setprecision(12);
    std::cout << "___________________________________" << std::endl;
    std::cout << "rot_x: " << images[j]->camera.cam_rot.x << "\t\t |" << std::endl;
    std::cout << "rot_y: " << images[j]->camera.cam_rot.y << "\t\t |" << std::endl;
    std::cout << "rot_z: " << images[j]->camera.cam_rot.z << "\t\t |" << std::endl;
    std::cout << "pos_x: " << images[j]->camera.cam_pos.x << "\t\t |" << std::endl;
    std::cout << "pos_y: " << images[j]->camera.cam_pos.y << "\t\t |" << std::endl;
    std::cout << "pos_z: " << images[j]->camera.cam_pos.z << "\t\t |" << std::endl;
    std::cout << "foc  : " << images[j]->camera.foc       << "\t\t |" << std::endl;
    std::cout << "fov_x: " << images[j]->camera.fov.x     << "\t\t |" << std::endl;
    std::cout << "fov_y: " << images[j]->camera.fov.y     << "\t\t |" << std::endl;
  }

  //goodbye
  return points;
}

/**
 * Saves a point cloud as a PLY while also saving cameras and projected points of those cameras
 * all as points in R3. Each is color coded RED for the cameras, GREEN for the point cloud, and
 * BLUE for the reprojected points.
 * @param pointCloud a Unity float3 that represents the point cloud itself
 * @param bundleSet is a BundleSet that contains lines and points to be drawn in front of the cameras
 * @param images a vector of images that contain value camera information
 */
void ssrlcv::PointCloudFactory::saveDebugCloud(Unity<float3>* pointCloud, BundleSet bundleSet, std::vector<ssrlcv::Image*> images){
  // main variables
  int size        = pointCloud->size() + bundleSet.lines->size() + images.size();
  int index       = 0; // the use of this just makes everything easier to read
  // test output of all the boiz
  struct colorPoint* cpoints = (colorPoint*)  malloc(size * sizeof(struct colorPoint));
  // fill in the camera points RED
  for (int i = 0; i < images.size(); i++){
    cpoints[index].x = images[i]->camera.cam_pos.x;
    cpoints[index].y = images[i]->camera.cam_pos.y;
    cpoints[index].z = images[i]->camera.cam_pos.z;
    cpoints[index].r = 255;
    cpoints[index].g = 0;
    cpoints[index].b = 0;
    index++;
  }
  // fill in the projected match locations in front of the cameras BLUE
  for (int i = 0; i < bundleSet.lines->size(); i++) {
    cpoints[index].x = bundleSet.lines->host[i].pnt.x + bundleSet.lines->host[i].vec.x;
    cpoints[index].y = bundleSet.lines->host[i].pnt.y + bundleSet.lines->host[i].vec.y;
    cpoints[index].z = bundleSet.lines->host[i].pnt.z + bundleSet.lines->host[i].vec.z;
    cpoints[index].r = 0;
    cpoints[index].g = 0;
    cpoints[index].b = 255;
    index++;
  }
  // fill in the point cloud GREEN
  for (int i = 0; i < pointCloud->size(); i++){
    cpoints[index].x = pointCloud->host[i].x; //
    cpoints[index].y = pointCloud->host[i].y;
    cpoints[index].z = pointCloud->host[i].z;
    cpoints[index].r = 0;
    cpoints[index].g = 255;
    cpoints[index].b = 0;
    index++;
  }
  // now save it
  ssrlcv::writePLY("debugCloud", cpoints, size); //
}

/**
 * Saves a point cloud as a PLY while also saving cameras and projected points of those cameras
 * all as points in R3. Each is color coded RED for the cameras, GREEN for the point cloud, and
 * BLUE for the reprojected points.
 * @param pointCloud a Unity float3 that represents the point cloud itself
 * @param bundleSet is a BundleSet that contains lines and points to be drawn in front of the cameras
 * @param images a vector of images that contain value camera information
 * @param fineName a filename for the debug cloud
 */
void ssrlcv::PointCloudFactory::saveDebugCloud(Unity<float3>* pointCloud, BundleSet bundleSet, std::vector<ssrlcv::Image*> images, std::string filename){
  // main variables
  int size        = pointCloud->size() + bundleSet.lines->size() + images.size();
  int index       = 0; // the use of this just makes everything easier to read
  // test output of all the boiz
  struct colorPoint* cpoints = (colorPoint*)  malloc(size * sizeof(struct colorPoint));
  // fill in the camera points RED
  for (int i = 0; i < images.size(); i++){
    cpoints[index].x = images[i]->camera.cam_pos.x;
    cpoints[index].y = images[i]->camera.cam_pos.y;
    cpoints[index].z = images[i]->camera.cam_pos.z;
    cpoints[index].r = 255;
    cpoints[index].g = 0;
    cpoints[index].b = 0;
    index++;
  }
  // fill in the projected match locations in front of the cameras BLUE
  for (int i = 0; i < bundleSet.lines->size(); i++) {
    cpoints[index].x = bundleSet.lines->host[i].pnt.x + bundleSet.lines->host[i].vec.x;
    cpoints[index].y = bundleSet.lines->host[i].pnt.y + bundleSet.lines->host[i].vec.y;
    cpoints[index].z = bundleSet.lines->host[i].pnt.z + bundleSet.lines->host[i].vec.z;
    cpoints[index].r = 0;
    cpoints[index].g = 0;
    cpoints[index].b = 255;
    index++;
  }
  // fill in the point cloud GREEN
  for (int i = 0; i < pointCloud->size(); i++){
    cpoints[index].x = pointCloud->host[i].x; //
    cpoints[index].y = pointCloud->host[i].y;
    cpoints[index].z = pointCloud->host[i].z;
    cpoints[index].r = 0;
    cpoints[index].g = 255;
    cpoints[index].b = 0;
    index++;
  }
  // now save it
  ssrlcv::writePLY(filename.c_str(), cpoints, size); //
}

/**
 * Saves a point cloud as a PLY while also saving cameras and projected points of those cameras
 * all as points in R3. Each is color coded RED for the cameras, GREEN for the point cloud, and
 * BLUE for the reprojected points.
 * @param pointCloud a Unity float3 that represents the point cloud itself
 * @param bundleSet is a BundleSet that contains lines and points to be drawn in front of the cameras
 * @param images a vector of images that contain value camera information
 * @param fineName a filename for the debug cloud
 */
void ssrlcv::PointCloudFactory::saveDebugCloud(Unity<float3>* pointCloud, BundleSet bundleSet, std::vector<ssrlcv::Image*> images, const char* filename){
  // main variables
  int size        = pointCloud->size() + bundleSet.lines->size() + images.size();
  int index       = 0; // the use of this just makes everything easier to read
  // test output of all the boiz
  struct colorPoint* cpoints = (colorPoint*)  malloc(size * sizeof(struct colorPoint));
  // fill in the camera points RED
  for (int i = 0; i < images.size(); i++){
    cpoints[index].x = images[i]->camera.cam_pos.x;
    cpoints[index].y = images[i]->camera.cam_pos.y;
    cpoints[index].z = images[i]->camera.cam_pos.z;
    cpoints[index].r = 255;
    cpoints[index].g = 0;
    cpoints[index].b = 0;
    index++;
  }
  // fill in the projected match locations in front of the cameras BLUE
  for (int i = 0; i < bundleSet.lines->size(); i++) {
    cpoints[index].x = bundleSet.lines->host[i].pnt.x + bundleSet.lines->host[i].vec.x;
    cpoints[index].y = bundleSet.lines->host[i].pnt.y + bundleSet.lines->host[i].vec.y;
    cpoints[index].z = bundleSet.lines->host[i].pnt.z + bundleSet.lines->host[i].vec.z;
    cpoints[index].r = 0;
    cpoints[index].g = 0;
    cpoints[index].b = 255;
    index++;
  }
  // fill in the point cloud GREEN
  for (int i = 0; i < pointCloud->size(); i++){
    cpoints[index].x = pointCloud->host[i].x; //
    cpoints[index].y = pointCloud->host[i].y;
    cpoints[index].z = pointCloud->host[i].z;
    cpoints[index].r = 0;
    cpoints[index].g = 255;
    cpoints[index].b = 0;
    index++;
  }
  // now save it
  ssrlcv::writePLY(filename, cpoints, size); //
}

/**
 * heat map
 */
uchar3 ssrlcv::heatMap(float value){
  uchar3 rgb;
  // float3 colorMap[5] = {
  //   {255.0f,0,0},
  //   {127.5f,127.5f,0},
  //   {0,255.0f,0.0f},
  //   {0,127.7f,127.5f},
  //   {0,0,255.0f},
  // };
  // float temp = colors->host[i];
  // colors->host[i] *= 5.0f;
  // colors->host[i] = floor(colors->host[i]);
  // if(colors->host[i] == 5.0f) colors->host[i] = 4.0f;
  // if(colors->host[i] == 0.0f) colors->host[i] = 1.0f;
  // rgb.x = (1-temp)*colorMap[(int)colors->host[i]-1].x + (temp*colorMap[(int)colors->host[i]].x);
  // rgb.y = (1-temp)*colorMap[(int)colors->host[i]-1].y + (temp*colorMap[(int)colors->host[i]].y);
  // rgb.z = (1-temp)*colorMap[(int)colors->host[i]-1].z + (temp*colorMap[(int)colors->host[i]].z);


  if(value <= 0.5f){
    value *= 2.0f;
    rgb.x = (unsigned char) 255*(1-value) + 0.5;
    rgb.y = (unsigned char) 255*value + 0.5;
    rgb.z = 0;
  }
  else{
    value = value*2.0f - 1;
    rgb.x = 0;
    rgb.y = (unsigned char) 255*(1-value) + 0.5;
    rgb.z = (unsigned char) 255*value + 0.5;
  }
  return rgb;
}

/**
 * write disparity image
 */
void ssrlcv::writeDisparityImage(Unity<float3>* points, unsigned int interpolationRadius, std::string pathToFile){
  MemoryState origin = points->getMemoryState();
  if(origin == gpu) points->transferMemoryTo(cpu);
  float3 min = {FLT_MAX,FLT_MAX,FLT_MAX};
  float3 max = {-FLT_MAX,-FLT_MAX,-FLT_MAX};
  for(int i = 0; i < points->size(); ++i){
    if(points->host[i].x < min.x) min.x = points->host[i].x;
    if(points->host[i].x > max.x) max.x = points->host[i].x;
    if(points->host[i].y < min.y) min.y = points->host[i].y;
    if(points->host[i].y > max.y) max.y = points->host[i].y;
    if(points->host[i].z < min.z) min.z = points->host[i].z;
    if(points->host[i].z > max.z) max.z = points->host[i].z;
  }
  uint2 imageDim = {(unsigned int)ceil(max.x-min.x)+1,(unsigned int)ceil(max.y-min.y)+1};
  unsigned char* disparityImage = new unsigned char[imageDim.x*imageDim.y*3];
  Unity<float>* colors = new Unity<float>(nullptr,imageDim.x*imageDim.y,cpu);
  for(int i = 0; i < imageDim.x*imageDim.y*3; ++i){
    disparityImage[i] = 0;
  }
  for(int i = 0; i < points->size(); ++i){
    float3 temp = points->host[i] - min;
    if(ceil(temp.x) != temp.x || ceil(temp.y) != temp.y){
      colors->host[((int)ceil(temp.y)*imageDim.x) + (int)ceil(temp.x)] += (1-ceil(temp.x)-temp.x)*(1-ceil(temp.y)-temp.y)*temp.z/(max.z-min.z);
      colors->host[((int)ceil(temp.y)*imageDim.x) + (int)floor(temp.x)] += (1-temp.x-floor(temp.x))*(1-ceil(temp.y)-temp.y)*temp.z/(max.z-min.z);
      colors->host[((int)floor(temp.y)*imageDim.x) + (int)ceil(temp.x)] += (1-ceil(temp.x)-temp.x)*(1-temp.y-floor(temp.y))*temp.z/(max.z-min.z);
      colors->host[((int)floor(temp.y)*imageDim.x) + (int)floor(temp.x)] += (1-temp.x-floor(temp.x))*(1-temp.y-floor(temp.y))*temp.z/(max.z-min.z);
    }
    else{
      colors->host[(int)temp.y*imageDim.x + (int)temp.x] += temp.z/(max.z-min.z);
    }
  }

  /*
  INTERPOLATE
  */
  if(interpolationRadius != 0){
    colors->setMemoryState(gpu);
    float* interpolated = nullptr;
    CudaSafeCall(cudaMalloc((void**)&interpolated,imageDim.x*imageDim.y*sizeof(float)));
    dim3 block = {1,1,1};
    dim3 grid = {1,1,1};
    getFlatGridBlock(imageDim.x*imageDim.y,grid,block,interpolateDepth);
    interpolateDepth<<<grid,block>>>(imageDim,interpolationRadius,colors->device,interpolated);
    cudaDeviceSynchronize();
    CudaCheckError();
    colors->setData(interpolated,colors->size(),gpu);
    colors->setMemoryState(cpu);
  }

  min.z = FLT_MAX;
  max.z = -FLT_MAX;
  for(int i = 0; i < imageDim.x*imageDim.y; ++i){
    if(min.z > colors->host[i]) min.z = colors->host[i];
    if(max.z < colors->host[i]) max.z = colors->host[i];
  }


  uchar3 rgb;
  for(int i = 0; i < imageDim.x*imageDim.y; ++i){
    colors->host[i] -= min.z;
    colors->host[i] /= (max.z-min.z);
    rgb = heatMap(colors->host[i]);
    disparityImage[i*3] = rgb.x;
    disparityImage[i*3 + 1] = rgb.y;
    disparityImage[i*3 + 2] = rgb.z;
  }
  delete colors;
  writePNG(pathToFile.c_str(),disparityImage,3,imageDim.x,imageDim.y);
  delete disparityImage;
}

// ==================================================================================================== //
//                                        device methods                                                //
// ==================================================================================================== //


__global__ void ssrlcv::generateBundle(unsigned int numBundles, Bundle* bundles, Bundle::Line* lines, MultiMatch* matches, KeyPoint* keyPoints, Image::Camera* cameras){
  unsigned long globalID = (blockIdx.y* gridDim.x+ blockIdx.x)*blockDim.x + threadIdx.x;
  if (globalID > numBundles - 1) return;
  MultiMatch match = matches[globalID];
  float3* kp = new float3[match.numKeyPoints]();
  int end =  (int) match.numKeyPoints + match.index;
  KeyPoint currentKP = {-1,{0.0f,0.0f}};
  bundles[globalID] = {match.numKeyPoints,match.index,false};
  for (int i = match.index, k= 0; i < end; i++,k++){
    // the current keypoint to transform
    currentKP = keyPoints[i];
    // set the dpix
    cameras[currentKP.parentId].dpix.x = (cameras[currentKP.parentId].foc * tanf(cameras[currentKP.parentId].fov.x / 2.0f)) / (cameras[currentKP.parentId].size.x / 2.0f );
    cameras[currentKP.parentId].dpix.y = cameras[currentKP.parentId].dpix.x; // assume square pixel for now
    // here we imagine the image plane is in the X Y plane AT a particular Z value, which is the focal length
    // begin movement to R3
    kp[k] = {
      cameras[currentKP.parentId].dpix.x * ((currentKP.loc.x) - (cameras[currentKP.parentId].size.x / 2.0f)),
      cameras[currentKP.parentId].dpix.y * ((currentKP.loc.y) - (cameras[currentKP.parentId].size.y / 2.0f)),
      cameras[currentKP.parentId].foc // this is the focal length
    }; // set the key point
    // rotate to correct orientation
    kp[k] = rotatePoint(kp[k], cameras[currentKP.parentId].cam_rot);
    // move to correct world coordinate
    kp[k].x = cameras[currentKP.parentId].cam_pos.x - (kp[k].x);
    kp[k].y = cameras[currentKP.parentId].cam_pos.y - (kp[k].y);
    kp[k].z = cameras[currentKP.parentId].cam_pos.z - (kp[k].z);
    // calculate the vector component of the line
    lines[i].vec = {
      cameras[currentKP.parentId].cam_pos.x - kp[k].x,
      cameras[currentKP.parentId].cam_pos.y - kp[k].y,
      cameras[currentKP.parentId].cam_pos.z - kp[k].z
    };
    // fill in the line values
    normalize(lines[i].vec);
    lines[i].pnt = cameras[currentKP.parentId].cam_pos;
    //printf("[%lu / %u] [i: %d] < %f , %f, %f > at ( %.12f, %.12f, %.12f ) \n", globalID,numBundles,i,lines[i].vec.x,lines[i].vec.y,lines[i].vec.z,lines[i].pnt.x,lines[i].pnt.y,lines[i].pnt.z);
  }
  delete[] kp;
}

__global__ void ssrlcv::computeStereo(unsigned int numMatches, Match* matches, float3* points, float scale){
  unsigned long globalID = (blockIdx.y* gridDim.x+ blockIdx.x)*blockDim.x + threadIdx.x;
  if (globalID < numMatches) {
    Match match = matches[globalID];
    float3 point = {match.keyPoints[0].loc.x,match.keyPoints[0].loc.y,0.0f};
    point.z = scale*sqrtf( dotProduct(match.keyPoints[0].loc-match.keyPoints[1].loc,match.keyPoints[0].loc-match.keyPoints[1].loc)) ;
    points[globalID] = point;
  }
}

__global__ void ssrlcv::computeStereo(unsigned int numMatches, Match* matches, float3* points, float foc, float baseLine, float doffset){
  unsigned long globalID = (blockIdx.y* gridDim.x+ blockIdx.x)*blockDim.x + threadIdx.x;
  if (globalID < numMatches) {
    Match match = matches[globalID];
    float3 point = {match.keyPoints[1].loc.x,match.keyPoints[1].loc.y,0.0f};
    //point.z = sqrtf(dotProduct(match.keyPoints[1].loc-match.keyPoints[0].loc,match.keyPoints[1].loc-match.keyPoints[0].loc));
    //with non parrallel or nonrecitified then replace .x - .x below with above
    point.z = foc*baseLine/(match.keyPoints[0].loc.x-match.keyPoints[1].loc.x+doffset);
    points[globalID] = point;
  }
}

__global__ void ssrlcv::interpolateDepth(uint2 disparityMapSize, int influenceRadius, float* disparities, float* interpolated){
  unsigned long globalID = (blockIdx.y* gridDim.x+ blockIdx.x)*blockDim.x + threadIdx.x;
  if(globalID < (disparityMapSize.x-(2*influenceRadius))*(disparityMapSize.y-(2*influenceRadius))){
    float disparity = disparities[globalID];
    int2 loc = {globalID%disparityMapSize.x + influenceRadius,globalID/disparityMapSize.x + influenceRadius};
    for(int y = loc.y - influenceRadius; y >= loc.y + influenceRadius; ++y){
      for(int x = loc.x - influenceRadius; x >= loc.x + influenceRadius; ++x){
        disparity += disparities[y*disparityMapSize.x + x]*(1 - abs((x-loc.x)/influenceRadius))*(1 - abs((y-loc.y)/influenceRadius));
      }
    }
    interpolated[globalID] = disparity;
  }
}


/**
* Does a trigulation with skew lines to find their closest intercetion.
* Generates a total LinearError, which is an analog for reprojection error
*/
__global__ void ssrlcv::computeTwoViewTriangulate(float* linearError, unsigned long pointnum, Bundle::Line* lines, Bundle* bundles, float3* pointcloud){
  // get ready to do the stuff local memory space
  // this will later be added back to a global memory space
  __shared__ float localSum;
  if (threadIdx.x == 0) localSum = 0;
  __syncthreads();

  // this method is from wikipedia, last seen janurary 2020
  // https://en.wikipedia.org/wiki/Skew_lines#Nearest_Points
  unsigned long globalID = (blockIdx.y* gridDim.x+ blockIdx.x)*blockDim.x + threadIdx.x;
  // if (globalID > (1)) return;
  if (globalID > (pointnum-1)) return;
  // we can assume that each line, so we don't need to get the numlines
  // ne guys are made just for easy of writing
  ssrlcv::Bundle::Line L1 = lines[bundles[globalID].index];
  ssrlcv::Bundle::Line L2 = lines[bundles[globalID].index+1];

  // calculate the normals
  float3 n2 = crossProduct(L2.vec,crossProduct(L1.vec,L2.vec));
  float3 n1 = crossProduct(L1.vec,crossProduct(L1.vec,L2.vec));

  // calculate the numerators
  float numer1 = dotProduct((L2.pnt - L1.pnt),n2);
  float numer2 = dotProduct((L1.pnt - L2.pnt),n1);

  // calculate the denominators
  float denom1 = dotProduct(L1.vec,n2);
  float denom2 = dotProduct(L2.vec,n1);

  // get the S points
  float3 s1 = L1.pnt + (numer1/denom1) * L1.vec;
  float3 s2 = L2.pnt + (numer2/denom2) * L2.vec;
  float3 point = (s1 + s2)/2.0;

  // fill in the value for the point cloud
  pointcloud[globalID] = point;

  // add the linaer errors locally within the block before
  float error = sqrtf((s1.x - s2.x)*(s1.x - s2.x) + (s1.y - s2.y)*(s1.y - s2.y) + (s1.z - s2.z)*(s1.z - s2.z));;
  // if(error != 0.0f) error = sqrtf(error);
  atomicAdd(&localSum,error);
  __syncthreads();
  if (!threadIdx.x) atomicAdd(linearError,localSum);
}

/**
* Does a trigulation with skew lines to find their closest intercetion.
* Generates a set of individual linear errors of debugging and analysis
* Generates a total LinearError, which is an analog for reprojection error
*/
__global__ void ssrlcv::computeTwoViewTriangulate(float* linearError, float* errors, unsigned long pointnum, Bundle::Line* lines, Bundle* bundles, float3* pointcloud){
  // get ready to do the stuff local memory space
  // this will later be added back to a global memory space
  __shared__ float localSum;
  if (threadIdx.x == 0) localSum = 0;
  __syncthreads();

  // this method is from wikipedia, last seen janurary 2020
  // https://en.wikipedia.org/wiki/Skew_lines#Nearest_Points
  unsigned long globalID = (blockIdx.y* gridDim.x+ blockIdx.x)*blockDim.x + threadIdx.x;
  // if (globalID > (1)) return;
  if (globalID > (pointnum-1)) return;
  // we can assume that each line, so we don't need to get the numlines
  // ne guys are made just for easy of writing
  ssrlcv::Bundle::Line L1 = lines[bundles[globalID].index];
  ssrlcv::Bundle::Line L2 = lines[bundles[globalID].index+1];

  // calculate the normals
  float3 n2 = crossProduct(L2.vec,crossProduct(L1.vec,L2.vec));
  float3 n1 = crossProduct(L1.vec,crossProduct(L1.vec,L2.vec));

  // calculate the numerators
  float numer1 = dotProduct((L2.pnt - L1.pnt),n2);
  float numer2 = dotProduct((L1.pnt - L2.pnt),n1);

  // calculate the denominators
  float denom1 = dotProduct(L1.vec,n2);
  float denom2 = dotProduct(L2.vec,n1);

  // get the S points
  float3 s1 = L1.pnt + (numer1/denom1) * L1.vec;
  float3 s2 = L2.pnt + (numer2/denom2) * L2.vec;
  float3 point = (s1 + s2)/2.0;

  // fill in the value for the point cloud
  pointcloud[globalID] = point;

  // add the linaer errors locally within the block before
  float error = sqrtf((s1.x - s2.x)*(s1.x - s2.x) + (s1.y - s2.y)*(s1.y - s2.y) + (s1.z - s2.z)*(s1.z - s2.z));
  // if(error != 0.0f) error = sqrtf(error);
  errors[globalID] = error;
  //int i_error = error;
  atomicAdd(&localSum,error);
  __syncthreads();
  if (!threadIdx.x) atomicAdd(linearError,localSum);
}

/**
* Does a trigulation with skew lines to find their closest intercetion.
* Generates a set of individual linear errors of debugging and analysis
* Generates a total LinearError, which is an analog for reprojection error
* If a point's linear error is larger than the cutoff it is not returned in the pointcloud
*/
__global__ void ssrlcv::computeTwoViewTriangulate(float* linearError, float* linearErrorCutoff, float* errors, unsigned long pointnum, Bundle::Line* lines, Bundle* bundles, float3* pointcloud){
  // get ready to do the stuff local memory space
  // this will later be added back to a global memory space
  __shared__ float localSum;
  if (threadIdx.x == 0) localSum = 0;
  __syncthreads();
  // this method is from wikipedia, last seen janurary 2020
  // https://en.wikipedia.org/wiki/Skew_lines#Nearest_Points
  unsigned long globalID = (blockIdx.y* gridDim.x+ blockIdx.x)*blockDim.x + threadIdx.x;
  // if (globalID > (1)) return;
  if (globalID > (pointnum-1)) return;
  // we can assume that each line, so we don't need to get the numlines
  // ne guys are made just for easy of writing
  ssrlcv::Bundle::Line L1 = lines[bundles[globalID].index];
  ssrlcv::Bundle::Line L2 = lines[bundles[globalID].index+1];

  float3 n = crossProduct(L1.vec,L2.vec);
  /*
  printf("Calculating n: (%.12f, %.12f, %.12f) x (%.12f, %.12f, %.12f)\n", L1.vec.x, L1.vec.y, L1.vec.z, L2.vec.x, L2.vec.y, L2.vec.z);
  printf("\tn (%.12f, %.12f, %.12f)\n", n.x, n.y, n.z);
 */

  // calculate the normals
  float3 n2 = crossProduct(L2.vec,crossProduct(L1.vec,L2.vec));
  float3 n1 = crossProduct(L1.vec,crossProduct(L1.vec,L2.vec));

  /*
  printf("Caclulating n1: (%.12f, %.12f, %.12f) x (%.12f, %.12f, %.12f)\n", L1.vec.x, L1.vec.y, L1.vec.z, n.x, n.y, n.z);
  printf("Caclulating n2: (%.12f, %.12f, %.12f) x (%.12f, %.12f, %.12f)\n", L2.vec.x, L2.vec.y, L2.vec.z, n.x, n.y, n.z);
  printf("\tn1 (%.12f, %.12f, %.12f)\n", n1.x, n1.y, n1.z);
  printf("\tn2 (%.12f, %.12f, %.12f)\n", n2.x, n2.y, n2.z);
  */

  // calculate the numerators
  float numer1 = dotProduct((L2.pnt - L1.pnt),n2);
  float numer2 = dotProduct((L1.pnt - L2.pnt),n1);

  /*
  printf("Caclulating numer1: [(%.12f, %.12f, %.12f) - (%.12f, %.12f, %.12f)] . (%.12f, %.12f, %.12f)\n", L2.pnt.x, L2.pnt.y, L2.pnt.z, L1.pnt.x, L1.pnt.y, L1.pnt.z, n2.x, n2.y, n2.z);
  printf("Caclulating numer2: [(%.12f, %.12f, %.12f) - (%.12f, %.12f, %.12f)] . (%.12f, %.12f, %.12f)\n", L1.pnt.x, L1.pnt.y, L1.pnt.z, L2.pnt.x, L2.pnt.y, L2.pnt.z, n1.x, n1.y, n1.z);
  printf("numer1 subtraction: (%.12f, %.12f, %.12f)\n", (L2.pnt - L1.pnt).x, (L2.pnt - L1.pnt).y, (L2.pnt - L1.pnt).z);
  printf("numer2 subtraction: (%.12f, %.12f, %.12f)\n", (L1.pnt - L2.pnt).x, (L1.pnt - L2.pnt).y, (L1.pnt - L2.pnt).z);
  printf("\tnumer1: %.12f\n", numer1);
  printf("\tnumer2: %.12f\n", numer2);
  */

  // calculate the denominators
  float denom1 = dotProduct(L1.vec,n2);
  float denom2 = dotProduct(L2.vec,n1);

  /*
  printf("Calculating denom1: (%.12f, %.12f, %.12f) . (%.12f, %.12f, %.12f)\n", L1.vec.x, L1.vec.y, L1.vec.z, n2.x, n2.y, n2.z);
  printf("Calculating denom2: (%.12f, %.12f, %.12f) . (%.12f, %.12f, %.12f)\n", L2.vec.x, L2.vec.y, L2.vec.z, n1.x, n1.y, n1.z);
  printf("\tdenom1: %.12f\n", denom1);
  printf("\tdenom2: %.12f\n", denom2);
  */

  // get the S points
  float3 s1 = L1.pnt + (numer1/denom1) * L1.vec;
  float3 s2 = L2.pnt + (numer2/denom2) * L2.vec;
  float3 point = (s1 + s2)/2.0;

  /*
  printf("S1 (%.12f, %.12f, %.12f)\n",s1.x,s1.y,s1.z);
  printf("S2 (%.12f, %.12f, %.12f)\n",s2.x,s2.y,s2.z);
  */

  // fill in the value for the point cloud
  pointcloud[globalID] = point;

  float error = sqrtf((s1.x - s2.x)*(s1.x - s2.x) + (s1.y - s2.y)*(s1.y - s2.y) + (s1.z - s2.z)*(s1.z - s2.z));

  errors[globalID] = error;
  // only add the errors that we like
  float i_error;

  if (error > *linearErrorCutoff) {
    i_error = error;
    // flag the bundle as bad!
    bundles[globalID].invalid = true;
  } else {
    i_error = error;
  }

  atomicAdd(&localSum,i_error);
  __syncthreads();
  if (!threadIdx.x) atomicAdd(linearError,localSum);
}

/**
* Does a trigulation with skew lines to find their closest intercetion.
* Generates a set of individual linear errors of debugging and analysis
* Generates a total LinearError, which is an analog for reprojection error
* If a point's linear error is larger than the cutoff it is not returned in the pointcloud
*/
__global__ void ssrlcv::computeTwoViewTriangulate_b(float* linearError, float* linearErrorCutoff, float* errors, unsigned long pointnum, Bundle::Line* lines, Bundle* bundles, float3_b* pointcloud_b){
  // get ready to do the stuff local memory space
  // this will later be added back to a global memory space
  __shared__ float localSum;
  if (threadIdx.x == 0) localSum = 0;
  __syncthreads();
  // this method is from wikipedia, last seen janurary 2020
  // https://en.wikipedia.org/wiki/Skew_lines#Nearest_Points
  unsigned long globalID = (blockIdx.y* gridDim.x+ blockIdx.x)*blockDim.x + threadIdx.x;
  // if (globalID > (1)) return;
  if (globalID > (pointnum-1)) return;
  // we can assume that each line, so we don't need to get the numlines
  // ne guys are made just for easy of writing
  ssrlcv::Bundle::Line L1 = lines[bundles[globalID].index];
  ssrlcv::Bundle::Line L2 = lines[bundles[globalID].index+1];

  // calculate the normals
  float3 n2 = crossProduct(L2.vec,crossProduct(L1.vec,L2.vec));
  float3 n1 = crossProduct(L1.vec,crossProduct(L1.vec,L2.vec));

  // calculate the numerators
  float numer1 = dotProduct((L2.pnt - L1.pnt),n2);
  float numer2 = dotProduct((L1.pnt - L2.pnt),n1);

  // calculate the denominators
  float denom1 = dotProduct(L1.vec,n2);
  float denom2 = dotProduct(L2.vec,n1);

  // get the S points
  float3 s1 = L1.pnt + (numer1/denom1) * L1.vec;
  float3 s2 = L2.pnt + (numer2/denom2) * L2.vec;
  float3 point = (s1 + s2)/2.0;

  // fill in the value for the point cloud
  float3_b point_b;
  point_b.x = point.x;
  point_b.y = point.y;
  point_b.z = point.z;
  point_b.invalid = false;
  pointcloud_b[globalID] = point_b;

  // Euclidean distance
  float error = (s1.x - s2.x)*(s1.x - s2.x) + (s1.y - s2.y)*(s1.y - s2.y) + (s1.z - s2.z)*(s1.z - s2.z);
  errors[globalID] = error;

  float i_error;
  if (error > *linearErrorCutoff) {
    pointcloud_b[globalID].invalid = true; // this means we should filter it out next time
    i_error = error;
  } else {
    i_error = error;
  }

  atomicAdd(&localSum,i_error);
  __syncthreads();
  if (!threadIdx.x) atomicAdd(linearError,localSum);
}

/**
* Does a trigulation with skew lines to find their closest intercetion.
* Generates a set of individual linear errors of debugging and analysis
* Generates a total LinearError, which is an analog for reprojection error
* this is purely used by the bundle adjustment to estimate errors and so on
*/
__global__ void ssrlcv::voidComputeTwoViewTriangulate(float* linearError, float* linearErrorCutoff, unsigned long pointnum, Bundle::Line* lines, Bundle* bundles){
  // get ready to do the stuff local memory space
  // this will later be added back to a global memory space
  __shared__ float localSum;
  if (threadIdx.x == 0) localSum = 0;
  __syncthreads();

  // this method is from wikipedia, last seen janurary 2020
  // https://en.wikipedia.org/wiki/Skew_lines#Nearest_Points
  unsigned long globalID = (blockIdx.y* gridDim.x+ blockIdx.x)*blockDim.x + threadIdx.x;
  // if (globalID > (1)) return;
  if (globalID > (pointnum-1)) return;
  // we can assume that each line, so we don't need to get the numlines
  // ne guys are made just for easy of writing
  ssrlcv::Bundle::Line L1 = lines[bundles[globalID].index];
  ssrlcv::Bundle::Line L2 = lines[bundles[globalID].index+1];

  // calculate the normals
  float3 n2 = crossProduct(L2.vec,crossProduct(L1.vec,L2.vec));
  float3 n1 = crossProduct(L1.vec,crossProduct(L1.vec,L2.vec));

  // calculate the numerators
  float numer1 = dotProduct((L2.pnt - L1.pnt),n2);
  float numer2 = dotProduct((L1.pnt - L2.pnt),n1);

  // calculate the denominators
  float denom1 = dotProduct(L1.vec,n2);
  float denom2 = dotProduct(L2.vec,n1);

  // get the S points
  float3 s1 = L1.pnt + (numer1/denom1) * L1.vec;
  float3 s2 = L2.pnt + (numer2/denom2) * L2.vec;
  float3 point = (s1 + s2)/2.0;

  // add the linear errors locally within the block before
  float error = sqrtf((s1.x - s2.x)*(s1.x - s2.x) + (s1.y - s2.y)*(s1.y - s2.y) + (s1.z - s2.z)*(s1.z - s2.z));
  //float error = dotProduct(s1,s2)*dotProduct(s1,s2);
  //if(error != 0.0f) error = sqrtf(error);
  // only add errors that we like
  float i_error;
  // filtering should only occur at the start of each adjustment step
  // TODO clean this up
  if (error > *linearErrorCutoff) {
    //point = {1.0f,1.0f,1.0f};
    //i_error = *linearErrorCutoff;
    i_error = error;
  } else {
    i_error = error;
  }
  atomicAdd(&localSum,i_error);
  __syncthreads();
  if (!threadIdx.x) atomicAdd(linearError,localSum);
}
