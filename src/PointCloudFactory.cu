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


  Unity<Bundle>* bundles = new Unity<Bundle>(nullptr,matchSet->matches->numElements,gpu);
  Unity<Bundle::Line>* lines = new Unity<Bundle::Line>(nullptr,matchSet->keyPoints->numElements,gpu);

  // std::cout << "starting bundle generation ..." << std::endl;
  MemoryState origin[2] = {matchSet->matches->getMemoryState(),matchSet->keyPoints->getMemoryState()};
  if(origin[0] == cpu) matchSet->matches->transferMemoryTo(gpu);
  if(origin[1] == cpu) matchSet->keyPoints->transferMemoryTo(gpu);
  // std::cout << "set the matches ... " << std::endl;
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
  getFlatGridBlock(bundles->numElements,grid,block,generateBundle);

  //in this kernel fill lines and bundles from keyPoints and matches
  // std::cout << "Calling bundle generation kernel ..." << std::endl;
  generateBundle<<<grid, block>>>(bundles->numElements,bundles->device, lines->device, matchSet->matches->device, matchSet->keyPoints->device, d_cameras);
  // std::cout << "Returned from bundle generation kernel ... \n" << std::endl;

  cudaDeviceSynchronize();
  CudaCheckError();

  // call the boi
  bundles->transferMemoryTo(cpu);
  bundles->clear(gpu);
  lines->transferMemoryTo(cpu);
  lines->clear(gpu);

  BundleSet bundleSet = {lines,bundles};

  if(origin[0] == cpu) matchSet->matches->setMemoryState(cpu);
  if(origin[1] == cpu) matchSet->keyPoints->setMemoryState(cpu);

  return bundleSet;
}

/**
* The CPU method that sets up the GPU enabled two view tringulation.
* @param bundleSet a set of lines and bundles that should be triangulated
* @param linearError is the total linear error of the triangulation, it is an analog for reprojection error
*/
ssrlcv::Unity<float3>* ssrlcv::PointCloudFactory::twoViewTriangulate(BundleSet bundleSet, unsigned long long int* linearError){

  // to total error cacluation is stored in this guy
  *linearError = 0;
  unsigned long long int* d_linearError;
  size_t eSize = sizeof(unsigned long long int);
  CudaSafeCall(cudaMalloc((void**) &d_linearError,eSize));
  CudaSafeCall(cudaMemcpy(d_linearError,linearError,eSize,cudaMemcpyHostToDevice));

  bundleSet.lines->transferMemoryTo(gpu);
  bundleSet.bundles->transferMemoryTo(gpu);

  // Unity<float3>* pointcloud = new Unity<float3>(nullptr,2*bundleSet.bundles->numElements,gpu);
  Unity<float3>* pointcloud = new Unity<float3>(nullptr,bundleSet.bundles->numElements,gpu);

  dim3 grid = {1,1,1};
  dim3 block = {1,1,1};
  getFlatGridBlock(bundleSet.bundles->numElements,grid,block,generateBundle);

  // std::cout << "Starting 2-view triangulation ..." << std::endl;
  computeTwoViewTriangulate<<<grid,block>>>(d_linearError,bundleSet.bundles->numElements,bundleSet.lines->device,bundleSet.bundles->device,pointcloud->device);
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
ssrlcv::Unity<float3>* ssrlcv::PointCloudFactory::twoViewTriangulate(BundleSet bundleSet, Unity<float>* errors, unsigned long long int* linearError){

  // to total error cacluation is stored in this guy
  *linearError = 0;
  unsigned long long int* d_linearError;
  size_t eSize = sizeof(unsigned long long int);
  CudaSafeCall(cudaMalloc((void**) &d_linearError,eSize));
  CudaSafeCall(cudaMemcpy(d_linearError,linearError,eSize,cudaMemcpyHostToDevice));

  bundleSet.lines->transferMemoryTo(gpu);
  bundleSet.bundles->transferMemoryTo(gpu);
  errors->transferMemoryTo(gpu);

  // Unity<float3>* pointcloud = new Unity<float3>(nullptr,2*bundleSet.bundles->numElements,gpu);
  Unity<float3>* pointcloud = new Unity<float3>(nullptr,bundleSet.bundles->numElements,gpu);

  dim3 grid = {1,1,1};
  dim3 block = {1,1,1};
  getFlatGridBlock(bundleSet.bundles->numElements,grid,block,generateBundle);

  // std::cout << "Starting 2-view triangulation ..." << std::endl;
  computeTwoViewTriangulate<<<grid,block>>>(d_linearError,errors->device,bundleSet.bundles->numElements,bundleSet.lines->device,bundleSet.bundles->device,pointcloud->device);
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
ssrlcv::Unity<float3>* ssrlcv::PointCloudFactory::twoViewTriangulate(BundleSet bundleSet, Unity<float>* errors, unsigned long long int* linearError, float* linearErrorCutoff){

  // to total error cacluation is stored in this guy
  *linearError = 0;
  unsigned long long int* d_linearError;
  size_t eSize = sizeof(unsigned long long int);
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

  // Unity<float3>* pointcloud = new Unity<float3>(nullptr,2*bundleSet.bundles->numElements,gpu);
  Unity<float3>* pointcloud = new Unity<float3>(nullptr,bundleSet.bundles->numElements,gpu);

  dim3 grid = {1,1,1};
  dim3 block = {1,1,1};
  getFlatGridBlock(bundleSet.bundles->numElements,grid,block,generateBundle);

  // std::cout << "Starting 2-view triangulation ..." << std::endl;
  computeTwoViewTriangulate<<<grid,block>>>(d_linearError,d_linearErrorCutoff,errors->device,bundleSet.bundles->numElements,bundleSet.lines->device,bundleSet.bundles->device,pointcloud->device);
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
  // free the cutoff, it's not needed on the cpu again tho
  cudaFree(d_linearErrorCutoff);

  return pointcloud;
}

/**
 * Same method as two view triangulation, but all that is desired fro this method is a calculation of the linearError
 * @param bundleSet a set of lines and bundles that should be triangulated
 * @param linearError is the total linear error of the triangulation, it is an analog for reprojection error
 * @param linearErrorCutoff is a value that all linear errors should be less than. points with larger errors are discarded.
 */
void ssrlcv::PointCloudFactory::voidTwoViewTriangulate(BundleSet bundleSet, unsigned long long int* linearError, float* linearErrorCutoff){
  // to total error cacluation is stored in this guy
  *linearError = 0;
  unsigned long long int* d_linearError;
  size_t eSize = sizeof(unsigned long long int);
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
  Unity<float3>* pointcloud = new Unity<float3>(nullptr,bundleSet.bundles->numElements,gpu);

  dim3 grid = {1,1,1};
  dim3 block = {1,1,1};
  getFlatGridBlock(bundleSet.bundles->numElements,grid,block,generateBundle);

  voidComputeTwoViewTriangulate<<<grid,block>>>(d_linearError,d_linearErrorCutoff,bundleSet.bundles->numElements,bundleSet.lines->device,bundleSet.bundles->device);

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

  cudaMalloc((void**) &points_device, matches->numElements*sizeof(float3));

  //
  dim3 grid = {1,1,1};
  dim3 block = {1,1,1};
  void (*fp)(unsigned int, Match*, float3*, float) = &computeStereo;
  getFlatGridBlock(matches->numElements,grid,block,fp);
  //
  computeStereo<<<grid, block>>>(matches->numElements, matches->device, points_device, 8.0);
  // focal lenth / baseline

  // computeStereo<<<grid, block>>>(matches->numElements, matches->device, points_device, 64.0);

  Unity<float3>* points = new Unity<float3>(points_device, matches->numElements,gpu);
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

  cudaMalloc((void**) &points_device, matches->numElements*sizeof(float3));

  //
  dim3 grid = {1,1,1};
  dim3 block = {1,1,1};
  void (*fp)(unsigned int, Match*, float3*, float) = &computeStereo;
  getFlatGridBlock(matches->numElements,grid,block,fp);
  //
  computeStereo<<<grid, block>>>(matches->numElements, matches->device, points_device, scale);

  Unity<float3>* points = new Unity<float3>(points_device, matches->numElements,gpu);
  if(origin == cpu) matches->setMemoryState(cpu);

  return points;
}

ssrlcv::Unity<float3>* ssrlcv::PointCloudFactory::stereo_disparity(Unity<Match>* matches, float foc, float baseline, float doffset){

  MemoryState origin = matches->getMemoryState();
  if(origin == cpu) matches->transferMemoryTo(gpu);


  Unity<float3>* points = new Unity<float3>(nullptr, matches->numElements,gpu);
  //
  dim3 grid = {1,1,1};
  dim3 block = {1,1,1};
  void (*fp)(unsigned int, Match*, float3*, float, float, float) = &computeStereo;
  getFlatGridBlock(matches->numElements,grid,block,fp);
  //
  computeStereo<<<grid, block>>>(matches->numElements, matches->device, points->device, foc, baseline, doffset);

  if(origin == cpu) matches->setMemoryState(cpu);

  return points;
}

/**
 * A Naive bundle adjustment based on a two-view triangulation and a first order descrete gradient decent
 * @param matchSet a group of matches
 * @param a group of images, used only for their stored camera parameters
 * @return a bundle adjusted point cloud
 */
ssrlcv::Unity<float3>* ssrlcv::PointCloudFactory::BundleAdjustTwoView(MatchSet* matchSet, std::vector<ssrlcv::Image*> images){

  // the initial linear error
  unsigned long long int* linearError = (unsigned long long int*) malloc(sizeof(unsigned long long int));
  *linearError = 100000; // just something to satisfy the first if statment
  unsigned long long int* linearError_partial = (unsigned long long int*) malloc(sizeof(unsigned long long int));
  // the cutoff
  // TODO this shold later remove points that are bad
  float* linearErrorCutoff = (float*) malloc(sizeof(float));
  // make the temporary image struct
  std::vector<ssrlcv::Image*> partials;
  partials.push_back(images[0]);
  partials.push_back(images[1]);
  // used to store the gradients in the camera
  std::vector<ssrlcv::Image::Camera> gradients;
  ssrlcv::Image::Camera g_1 = ssrlcv::Image::Camera();
  ssrlcv::Image::Camera g_2 = ssrlcv::Image::Camera();
  gradients.push_back(g_1);
  gradients.push_back(g_2);

  // the boiz in the loop
  ssrlcv::BundleSet bundleSet;
  ssrlcv::BundleSet bundleSet_partial;
  ssrlcv::Unity<float>* errors;
  ssrlcv::Unity<float3>* points;

  // for printing out data about iterations and shit later
  std::vector<unsigned long long int> errorTracker;

  int i = 1;
  while(*linearError > 9001){
    // generate the bundle set
    bundleSet = generateBundles(matchSet,images);
    // do an initial triangulation
    errors = new ssrlcv::Unity<float>(nullptr,matchSet->matches->numElements,ssrlcv::cpu);
    *linearErrorCutoff = 620.0;
    points = twoViewTriangulate(bundleSet, errors, linearError, linearErrorCutoff);
    // do this only once
    if (i == 1 ) ssrlcv::writePLY("out/rawPoints.ply",points);

    // write some errors for debug
    //writeCSV(errors->host, (int) errors->numElements, "individualLinearErrors" + std::to_string(i));

    // clear uneeded memory
    delete bundleSet.lines;
    delete bundleSet.bundles;
    errors->clear();
    // points->clear();

    // a nice printout for the humans
    std::cout << "[itr: " << i << "] linear error: " << *linearError << std::endl;
    errorTracker.push_back(*linearError);

    // what the step sizes should be tho:
    // this is only for the "sensitivity" in those component directions
    float h_rot = 0.0000001;
    float h_pos = 0.0000001;
    float h_foc = 0.0000001;
    float h_fov = 0.0000001;
    // the stepsize along the gradient
    float step  = 0.001;

    // calculate the descrete partial derivatives using forward difference
    for (int j = 0; j < partials.size(); j++){
      // rotations

      // partial for x rotation
      partials[j]->camera.cam_rot.x = images[j]->camera.cam_rot.x + h_rot;
      // get the error
      bundleSet_partial = generateBundles(matchSet,partials);
      voidTwoViewTriangulate(bundleSet_partial, linearError_partial, linearErrorCutoff);
      delete bundleSet_partial.lines;
      delete bundleSet_partial.bundles;
      gradients[j].cam_rot.x = ((float) *linearError + (float) *linearError_partial) / (h_rot);
      // reset
      partials[j]->camera.cam_rot.x = images[j]->camera.cam_rot.x;

      // partial for y rotation
      partials[j]->camera.cam_rot.y = images[j]->camera.cam_rot.y + h_rot;
      // get the error
      bundleSet_partial = generateBundles(matchSet,partials);
      voidTwoViewTriangulate(bundleSet_partial, linearError_partial, linearErrorCutoff);
      delete bundleSet_partial.lines;
      delete bundleSet_partial.bundles;
      gradients[j].cam_rot.y = ((float) *linearError + (float) *linearError_partial) / (h_rot);
      // reset
      partials[j]->camera.cam_rot.y = images[j]->camera.cam_rot.y;

      // partial for z rotation
      partials[j]->camera.cam_rot.z = images[j]->camera.cam_rot.z + h_rot;
      // get the error
      bundleSet_partial = generateBundles(matchSet,partials);
      voidTwoViewTriangulate(bundleSet_partial, linearError_partial, linearErrorCutoff);
      delete bundleSet_partial.lines;
      delete bundleSet_partial.bundles;
      gradients[j].cam_rot.z = ((float) *linearError + (float) *linearError_partial) / (h_rot);
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
      gradients[j].cam_pos.x = ((float) *linearError + (float) *linearError_partial) / (h_pos);
      // reset
      partials[j]->camera.cam_pos.x = images[j]->camera.cam_pos.x;

      // partial for y position
      partials[j]->camera.cam_pos.y = images[j]->camera.cam_rot.y + h_pos;
      // get the error
      bundleSet_partial = generateBundles(matchSet,partials);
      voidTwoViewTriangulate(bundleSet_partial, linearError_partial, linearErrorCutoff);
      delete bundleSet_partial.lines;
      delete bundleSet_partial.bundles;
      gradients[j].cam_pos.y = ((float) *linearError + (float) *linearError_partial) / (h_pos);
      // reset
      partials[j]->camera.cam_pos.y = images[j]->camera.cam_pos.y;

      // partial for z position
      partials[j]->camera.cam_pos.z = images[j]->camera.cam_pos.z + h_pos;
      // get the error
      bundleSet_partial = generateBundles(matchSet,partials);
      voidTwoViewTriangulate(bundleSet_partial, linearError_partial, linearErrorCutoff);
      delete bundleSet_partial.lines;
      delete bundleSet_partial.bundles;
      gradients[j].cam_pos.z = ((float) *linearError + (float) *linearError_partial) / (h_pos);
      // reset
      partials[j]->camera.cam_pos.z = images[j]->camera.cam_pos.z;

      // focal length

      // partial for focal length
      partials[j]->camera.foc = images[j]->camera.foc + h_foc;
      // get the error
      bundleSet_partial = generateBundles(matchSet,partials);
      voidTwoViewTriangulate(bundleSet_partial, linearError_partial, linearErrorCutoff);
      delete bundleSet_partial.lines;
      delete bundleSet_partial.bundles;
      gradients[j].foc = ((float) *linearError + (float) *linearError_partial) / (h_foc);
      // reset
      partials[j]->camera.foc = images[j]->camera.foc;

      // field of view

      // partial for x field of view
      partials[j]->camera.fov.x = images[j]->camera.fov.x + h_fov;
      // get the error
      bundleSet_partial = generateBundles(matchSet,partials);
      voidTwoViewTriangulate(bundleSet_partial, linearError_partial, linearErrorCutoff);
      delete bundleSet_partial.lines;
      delete bundleSet_partial.bundles;
      gradients[j].fov.x = ((float) *linearError + (float) *linearError_partial) / (h_fov);
      // reset
      partials[j]->camera.fov.x = images[j]->camera.fov.x;

      // partial for y field of view
      partials[j]->camera.fov.y = images[j]->camera.fov.y + h_fov;
      // get the error
      bundleSet_partial = generateBundles(matchSet,partials);
      voidTwoViewTriangulate(bundleSet_partial, linearError_partial,linearErrorCutoff);
      delete bundleSet_partial.lines;
      delete bundleSet_partial.bundles;
      gradients[j].fov.y = ((float) *linearError + (float) *linearError_partial) / (h_fov);
      // reset
      partials[j]->camera.fov.y = images[0]->camera.fov.y;

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

      // ya boi
      i++;
    }

    if (false){
      for (int j = 0; j < partials.size(); j++){
        std::cout << "Gradient " << std::to_string(j+1) << " | rot: (";
        std::cout << gradients[j].cam_rot.x << ",";
        std::cout << gradients[j].cam_rot.y << ",";
        std::cout << gradients[j].cam_rot.z << ")\t";
        std::cout << "pos: (";
        std::cout << gradients[j].cam_pos.x << ",";
        std::cout << gradients[j].cam_pos.y << ",";
        std::cout << gradients[j].cam_pos.z << ")\t";
        std::cout << "foc: ";
        std::cout << gradients[j].foc << "\t";
        std::cout << "fov: (";
        std::cout << gradients[j].fov.x << ",";
        std::cout << gradients[j].fov.y << ")" << std::endl;
      }
    }

    // take a step down the hill!
    for (int j = 0; j < images.size(); j++){
      images[j]->camera.cam_rot.x -= step * gradients[j].cam_rot.x;
      images[j]->camera.cam_rot.y -= step * gradients[j].cam_rot.y;
      images[j]->camera.cam_rot.z -= step * gradients[j].cam_rot.z;
      images[j]->camera.cam_pos.x -= step * gradients[j].cam_pos.x;
      images[j]->camera.cam_pos.y -= step * gradients[j].cam_pos.y;
      images[j]->camera.cam_pos.z -= step * gradients[j].cam_pos.z;
      images[j]->camera.foc       -= step * gradients[j].foc;
      images[j]->camera.fov.x     -= step * gradients[j].fov.x;
      images[j]->camera.fov.y     -= step * gradients[j].fov.y;
    }


  }

  // write linearError chagnes to a CSV
  writeCSV(errorTracker, "totalErrorOverIterations");
  //goodbye
  return points;
}

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

void ssrlcv::writeDisparityImage(Unity<float3>* points, unsigned int interpolationRadius, std::string pathToFile){
  MemoryState origin = points->getMemoryState();
  if(origin == gpu) points->transferMemoryTo(cpu);
  float3 min = {FLT_MAX,FLT_MAX,FLT_MAX};
  float3 max = {-FLT_MAX,-FLT_MAX,-FLT_MAX};
  for(int i = 0; i < points->numElements; ++i){
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
  for(int i = 0; i < points->numElements; ++i){
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
    colors->setData(interpolated,colors->numElements,gpu);
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
  bundles[globalID] = {match.numKeyPoints,match.index};
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
    //printf("[%lu / %u] [i: %d] < %f , %f, %f > at ( %f, %f, %f ) \n", globalID,numBundles,i,lines[i].vec.x,lines[i].vec.y,lines[i].vec.z,lines[i].pnt.x,lines[i].pnt.y,lines[i].pnt.z);
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
__global__ void ssrlcv::computeTwoViewTriangulate(unsigned long long int* linearError, unsigned long pointnum, Bundle::Line* lines, Bundle* bundles, float3* pointcloud){
  // get ready to do the stuff local memory space
  // this will later be added back to a global memory space
  __shared__ unsigned long long int localSum;
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
  int error = sqrtf(dotProduct(s1,s2));
  atomicAdd(&localSum,error);
  __syncthreads();
  if (!threadIdx.x) atomicAdd(linearError,localSum);
}

/**
* Does a trigulation with skew lines to find their closest intercetion.
* Generates a set of individual linear errors of debugging and analysis
* Generates a total LinearError, which is an analog for reprojection error
*/
__global__ void ssrlcv::computeTwoViewTriangulate(unsigned long long int* linearError, float* errors, unsigned long pointnum, Bundle::Line* lines, Bundle* bundles, float3* pointcloud){
  // get ready to do the stuff local memory space
  // this will later be added back to a global memory space
  __shared__ unsigned long long int localSum;
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
  float error = sqrtf(dotProduct(s1,s2));
  errors[globalID] = error;
  int i_error = (int) error;
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
__global__ void ssrlcv::computeTwoViewTriangulate(unsigned long long int* linearError, float* linearErrorCutoff, float* errors, unsigned long pointnum, Bundle::Line* lines, Bundle* bundles, float3* pointcloud){
  // get ready to do the stuff local memory space
  // this will later be added back to a global memory space
  __shared__ unsigned long long int localSum;
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

  // add the linear errors locally within the block before
  float error = sqrtf(dotProduct(s1,s2));
  errors[globalID] = error;
  if (error > *linearErrorCutoff) pointcloud[globalID] = {NULL,NULL,NULL};
  int i_error = (int) error;
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
__global__ void ssrlcv::voidComputeTwoViewTriangulate(unsigned long long int* linearError, float* linearErrorCutoff, unsigned long pointnum, Bundle::Line* lines, Bundle* bundles){
  // get ready to do the stuff local memory space
  // this will later be added back to a global memory space
  __shared__ unsigned long long int localSum;
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
  float error = sqrtf(dotProduct(s1,s2));
  if (error > *linearErrorCutoff) point = {NULL,NULL,NULL};
  int i_error = (int) error;
  atomicAdd(&localSum,i_error);
  __syncthreads();
  if (!threadIdx.x) atomicAdd(linearError,localSum);
}





















































// yee
