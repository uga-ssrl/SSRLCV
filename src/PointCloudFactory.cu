
#include "PointCloudFactory.cuh"

ssrlcv::PointCloudFactory::PointCloudFactory(){

}

// =============================================================================================================
//
// Stereo Disparity Methods
//
// ==================================================================================z===========================

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

/**
 * TODO
 */
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

// =============================================================================================================
//
// 2 View Methods
//
// =============================================================================================================

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

/**
 * Same method as two view triangulation, but all that is desired fro this method is a calculation of the linearError
 * @param bundleSet a set of lines and bundles that should be triangulated
 * @param linearError is the total linear error of the triangulation, it is an analog for reprojection error
 */
void ssrlcv::PointCloudFactory::voidTwoViewTriangulate(BundleSet bundleSet, float* linearError){
  // to total error cacluation is stored in this guy
  *linearError = 0;
  float* d_linearError;
  size_t eSize = sizeof(float);
  CudaSafeCall(cudaMalloc((void**) &d_linearError,eSize));
  CudaSafeCall(cudaMemcpy(d_linearError,linearError,eSize,cudaMemcpyHostToDevice));

  bundleSet.lines->transferMemoryTo(gpu);
  bundleSet.bundles->transferMemoryTo(gpu);

  dim3 grid = {1,1,1};
  dim3 block = {1,1,1};
  void (*fp)(float*, unsigned long, Bundle::Line*, Bundle*) = &voidComputeTwoViewTriangulate;
  getFlatGridBlock(bundleSet.bundles->size(),grid,block,fp);

  voidComputeTwoViewTriangulate<<<grid,block>>>(d_linearError,bundleSet.bundles->size(),bundleSet.lines->device,bundleSet.bundles->device);

  cudaDeviceSynchronize();
  CudaCheckError();

  // clear the other boiz
  bundleSet.lines->clear(gpu);
  bundleSet.bundles->clear(gpu);
  // copy back the total error that occured
  CudaSafeCall(cudaMemcpy(linearError,d_linearError,eSize,cudaMemcpyDeviceToHost));
  cudaFree(d_linearError);

  return;
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

  dim3 grid = {1,1,1};
  dim3 block = {1,1,1};
  void (*fp)(float*, float*, unsigned long, Bundle::Line*, Bundle*) = &voidComputeTwoViewTriangulate;
  getFlatGridBlock(bundleSet.bundles->size(),grid,block,fp);

  voidComputeTwoViewTriangulate<<<grid,block>>>(d_linearError,d_linearErrorCutoff,bundleSet.bundles->size(),bundleSet.lines->device,bundleSet.bundles->device);

  cudaDeviceSynchronize();
  CudaCheckError();

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

// =============================================================================================================
//
// N View Methods
//
// =============================================================================================================

/**
 * The CPU method that sets up the GPU enabled n view triangulation.
 * @param bundleSet a set of lines and bundles to be triangulated
 */
ssrlcv::Unity<float3>* ssrlcv::PointCloudFactory::nViewTriangulate(BundleSet bundleSet){
  bundleSet.lines->transferMemoryTo(gpu);
  bundleSet.bundles->transferMemoryTo(gpu);

  Unity<float3>* pointcloud = new Unity<float3>(nullptr,bundleSet.bundles->size(),gpu);

  dim3 grid = {1,1,1};
  dim3 block = {1,1,1};
  void (*fp)(unsigned long, Bundle::Line*, Bundle*, float3*) = &computeNViewTriangulate;
  getFlatGridBlock(bundleSet.bundles->size(),grid,block,fp);


  std::cout << "Starting n-view triangulation ..." << std::endl;
  computeNViewTriangulate<<<grid,block>>>(bundleSet.bundles->size(),bundleSet.lines->device,bundleSet.bundles->device,pointcloud->device);
  std::cout << "n-view Triangulation done ... \n" << std::endl;

  pointcloud->setFore(gpu);
  bundleSet.lines->setFore(gpu);
  bundleSet.bundles->setFore(gpu);

  pointcloud->transferMemoryTo(cpu);
  pointcloud->clear(gpu);
  bundleSet.lines->clear(gpu);
  bundleSet.bundles->clear(gpu);

  return pointcloud;
}

/**
 * The CPU method that sets up the GPU enabled n view triangulation.
 * @param bundleSet a set of lines and bundles to be triangulated
 * @param
 */
ssrlcv::Unity<float3>* ssrlcv::PointCloudFactory::nViewTriangulate(BundleSet bundleSet, float* angularError){

  // make the error guys
  *angularError = 0;
  float* d_angularError;
  size_t eSize = sizeof(float);
  CudaSafeCall(cudaMalloc((void**) &d_angularError,eSize));
  CudaSafeCall(cudaMemcpy(d_angularError,angularError,eSize,cudaMemcpyHostToDevice));

  bundleSet.lines->transferMemoryTo(gpu);
  bundleSet.bundles->transferMemoryTo(gpu);

  Unity<float3>* pointcloud = new Unity<float3>(nullptr,bundleSet.bundles->size(),gpu);

  dim3 grid = {1,1,1};
  dim3 block = {1,1,1};
  void (*fp)(float*, unsigned long, Bundle::Line*, Bundle*, float3*) = &computeNViewTriangulate;
  getFlatGridBlock(bundleSet.bundles->size(),grid,block,fp);


  std::cout << "Starting n-view triangulation ..." << std::endl;
  computeNViewTriangulate<<<grid,block>>>(d_angularError,bundleSet.bundles->size(),bundleSet.lines->device,bundleSet.bundles->device,pointcloud->device);
  std::cout << "n-view Triangulation done ... \n" << std::endl;

  pointcloud->setFore(gpu);
  bundleSet.lines->setFore(gpu);
  bundleSet.bundles->setFore(gpu);

  pointcloud->transferMemoryTo(cpu);
  pointcloud->clear(gpu);
  bundleSet.lines->clear(gpu);
  bundleSet.bundles->clear(gpu);

  // copy back the total error that occured
  CudaSafeCall(cudaMemcpy(angularError,d_angularError,eSize,cudaMemcpyDeviceToHost));
  cudaFree(d_angularError);

  return pointcloud;
}

/**
 * The CPU method that sets up the GPU enabled n view triangulation.
 * @param bundleSet a set of lines and bundles to be triangulated
 * @param errors the individual angular errors per point
 * @param angularError the total diff between vectors
 */
ssrlcv::Unity<float3>* ssrlcv::PointCloudFactory::nViewTriangulate(BundleSet bundleSet, Unity<float>* errors, float* angularError){
  // make the error guys
  *angularError = 0;
  float* d_angularError;
  size_t eSize = sizeof(float);
  CudaSafeCall(cudaMalloc((void**) &d_angularError,eSize));
  CudaSafeCall(cudaMemcpy(d_angularError,angularError,eSize,cudaMemcpyHostToDevice));

  bundleSet.lines->transferMemoryTo(gpu);
  bundleSet.bundles->transferMemoryTo(gpu);
  errors->transferMemoryTo(gpu);

  Unity<float3>* pointcloud = new Unity<float3>(nullptr,bundleSet.bundles->size(),gpu);

  dim3 grid = {1,1,1};
  dim3 block = {1,1,1};
  void (*fp)(float*, unsigned long, Bundle::Line*, Bundle*, float3*) = &computeNViewTriangulate;
  getFlatGridBlock(bundleSet.bundles->size(),grid,block,fp);


  std::cout << "Starting n-view triangulation ..." << std::endl;
  computeNViewTriangulate<<<grid,block>>>(d_angularError,errors->device,bundleSet.bundles->size(),bundleSet.lines->device,bundleSet.bundles->device,pointcloud->device);
  std::cout << "n-view Triangulation done ... \n" << std::endl;

  //
  pointcloud->setFore(gpu);
  bundleSet.lines->setFore(gpu);
  bundleSet.bundles->setFore(gpu);
  // transfer the individual linear errors back to the CPU
  errors->setFore(gpu);
  errors->transferMemoryTo(cpu);
  errors->clear(gpu);
  //
  pointcloud->transferMemoryTo(cpu);
  pointcloud->clear(gpu);
  bundleSet.lines->clear(gpu);
  bundleSet.bundles->clear(gpu);

  // copy back the total error that occured
  CudaSafeCall(cudaMemcpy(angularError,d_angularError,eSize,cudaMemcpyDeviceToHost));
  cudaFree(d_angularError);

  return pointcloud;
}

/**
 * The CPU method that sets up the GPU enabled n view triangulation.
 * @param bundleSet a set of lines and bundles to be triangulated
 * @param errors the individual angular errors per point
 * @param angularError the total diff between vectors
 * @param angularErrorCutoff any point past the angular error cutoff will be tagged as invalid
 */
ssrlcv::Unity<float3>* ssrlcv::PointCloudFactory::nViewTriangulate(BundleSet bundleSet, Unity<float>* errors, float* angularError, float* angularErrorCutoff){
  // make the error guys
  *angularError = 0;
  float* d_angularError;
  size_t eSize = sizeof(float);
  CudaSafeCall(cudaMalloc((void**) &d_angularError,eSize));
  CudaSafeCall(cudaMemcpy(d_angularError,angularError,eSize,cudaMemcpyHostToDevice));
  float* d_angularErrorCutoff;
  size_t cutSize = sizeof(float);
  CudaSafeCall(cudaMalloc((void**) &d_angularErrorCutoff,cutSize));
  CudaSafeCall(cudaMemcpy(d_angularErrorCutoff, angularErrorCutoff,cutSize,cudaMemcpyHostToDevice));

  bundleSet.lines->transferMemoryTo(gpu);
  bundleSet.bundles->transferMemoryTo(gpu);
  errors->transferMemoryTo(gpu);

  Unity<float3>* pointcloud = new Unity<float3>(nullptr,bundleSet.bundles->size(),gpu);

  dim3 grid = {1,1,1};
  dim3 block = {1,1,1};
  void (*fp)(float*, unsigned long, Bundle::Line*, Bundle*, float3*) = &computeNViewTriangulate;
  getFlatGridBlock(bundleSet.bundles->size(),grid,block,fp);


  std::cout << "Starting n-view triangulation ..." << std::endl;
  computeNViewTriangulate<<<grid,block>>>(d_angularError,d_angularErrorCutoff,errors->device,bundleSet.bundles->size(),bundleSet.lines->device,bundleSet.bundles->device,pointcloud->device);
  std::cout << "n-view Triangulation done ... \n" << std::endl;

  bundleSet.lines->setFore(gpu);
  bundleSet.bundles->setFore(gpu);
  bundleSet.lines->transferMemoryTo(cpu);
  bundleSet.bundles->transferMemoryTo(cpu);
  bundleSet.lines->clear(gpu);
  bundleSet.bundles->clear(gpu);
  // transfer the individual linear errors back to the CPU
  errors->setFore(gpu);
  errors->transferMemoryTo(cpu);
  errors->clear(gpu);
  //
  pointcloud->setFore(gpu);
  pointcloud->transferMemoryTo(cpu);
  pointcloud->clear(gpu);

  // copy back the total error that occured
  CudaSafeCall(cudaMemcpy(angularError,d_angularError,eSize,cudaMemcpyDeviceToHost));
  cudaFree(d_angularError);
  CudaSafeCall(cudaMemcpy(angularErrorCutoff,d_angularErrorCutoff,eSize,cudaMemcpyDeviceToHost));
  cudaFree(d_angularErrorCutoff);

  return pointcloud;
}

// =============================================================================================================
//
// Bundle Adjustment Methods
//
// =============================================================================================================

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
 * A Naive bundle adjustment based on a two-view triangulation and a first order descrete gradient decent
 * @param matchSet a group of matches
 * @param a group of images, used only for their stored camera parameters
 * @return a bundle adjusted point cloud
 */
ssrlcv::Unity<float3>* ssrlcv::PointCloudFactory::BundleAdjustTwoView(ssrlcv::MatchSet* matchSet, std::vector<ssrlcv::Image*> images, unsigned int iterations){

  // local variabels for function
  ssrlcv::Unity<float3>* points;
  ssrlcv::BundleSet bundleTemp;

  // gradients are stored in the image structs, but only the camera is modified
  // this containts the gradient
  std::vector<ssrlcv::Image*> gradient;
  for (int i = 0; i < images.size(); i++){
    ssrlcv::Image* grad = new ssrlcv::Image();
    gradient.push_back(grad);
    // initially, set all of the gradients to 0
    gradient[i]->id = i;
    gradient[i]->camera.size = {0,0};
    gradient[i]->camera.cam_pos = {0.0, 0.0, 0.0};
    gradient[i]->camera.cam_rot = {0.0, 0.0, 0.0};
    gradient[i]->camera.fov = {0.0,0.0};
    gradient[i]->camera.foc = 0.0;
  }
  // this temp vector is only used for the +/- h steps when calculating the gradients
  std::vector<ssrlcv::Image*> temp;
  for (int i = 0; i < images.size(); i++){
    temp.push_back(images[i]); // fill in the initial images
  }

  float* initialError  = (float*) malloc(sizeof(float)); // this stays constant per iteration
  float* gradientError = (float*) malloc(sizeof(float)); // this chaneges per iteration

  // This is for error tracking and printing later
  std::vector<float> errorTracker;

  unsigned int max_iterations = iterations;
  bool local_debug = false;
  bool const_step = true;
  // TODO hangle linear stepsize differently than radial stepsize
  float gamma    = 0.0001;// the initial stepsize
  float h_linear = 0.001; // gradient difference
  float h_radial = 0.0000001;

  struct CamAdjust2 x0;
  struct CamAdjust2 x1;
  struct CamAdjust2 g0;
  struct CamAdjust2 g1;
  struct CamAdjust2 adjustment;

  // begin iterative gradient decent
  for (int i = 0; i < max_iterations; i++){
    // the intialError from the cost function, or the f(x)
    bundleTemp = generateBundles(matchSet,images);
    points = twoViewTriangulate(bundleTemp, initialError);
    // free up Memory
    delete bundleTemp.bundles;
    delete bundleTemp.lines;

    // calculate all of the graients with central difference
    // https://v8doc.sas.com/sashtml/ormp/chap5/sect28.htm
    // we are only testing position and orientation gradients

    //
    // X Position Gradients
    //
    for (int j = 0; j < images.size(); j++){
      // ----> Forward
      *gradientError = 0.0f;
      temp[j]->camera.cam_pos.x = images[j]->camera.cam_pos.x; // reset for forwards
      temp[j]->camera.cam_pos.x += h_linear;
      bundleTemp = generateBundles(matchSet,temp); // get the bundles for the new temp images
      twoViewTriangulate(bundleTemp, gradientError);
      float forward = *gradientError;
      // free up Memory
      delete bundleTemp.bundles;
      delete bundleTemp.lines;
      // <---- Backwards
      *gradientError = 0.0f;
      temp[j]->camera.cam_pos.x = images[j]->camera.cam_pos.x; // reset for backwards
      temp[j]->camera.cam_pos.x -= h_linear;
      bundleTemp = generateBundles(matchSet,temp); // get the bundles for the new temp images
      twoViewTriangulate(bundleTemp, gradientError);
      float backwards = *gradientError;
      // free up Memory
      delete bundleTemp.bundles;
      delete bundleTemp.lines;
      // calculate the gradient with central difference
      gradient[j]->camera.cam_pos.x = ( forward - backwards ) / ( 2*h_linear );
    }

    //
    // Y Postition Gradients
    //
    for (int j = 0; j < images.size(); j++){
      // ----> Forward
      *gradientError = 0.0f;
      temp[j]->camera.cam_pos.y = images[j]->camera.cam_pos.y; // reset for forwards
      temp[j]->camera.cam_pos.y += h_linear;
      bundleTemp = generateBundles(matchSet,temp); // get the bundles for the new temp images
      twoViewTriangulate(bundleTemp, gradientError);
      float forward = *gradientError;
      // free up Memory
      delete bundleTemp.bundles;
      delete bundleTemp.lines;
      // <---- Backwards
      *gradientError = 0.0f;
      temp[j]->camera.cam_pos.y = images[j]->camera.cam_pos.y; // reset for backwards
      temp[j]->camera.cam_pos.y -= h_linear;
      bundleTemp = generateBundles(matchSet,temp); // get the bundles for the new temp images
      twoViewTriangulate(bundleTemp, gradientError);
      float backwards = *gradientError;
      // free up Memory
      delete bundleTemp.bundles;
      delete bundleTemp.lines;
      // calculate the gradient with central difference
      gradient[j]->camera.cam_pos.y = ( forward - backwards ) / ( 2*h_linear );
    }

    //
    // Z Postition Gradients
    //
    for (int j = 0; j < images.size(); j++){
      // ----> Forward
      *gradientError = 0.0f;
      temp[j]->camera.cam_pos.z = images[j]->camera.cam_pos.z; // reset for forwards
      temp[j]->camera.cam_pos.z += h_linear;
      bundleTemp = generateBundles(matchSet,temp); // get the bundles for the new temp images
      twoViewTriangulate(bundleTemp, gradientError);
      float forward = *gradientError;
      // free up Memory
      delete bundleTemp.bundles;
      delete bundleTemp.lines;
      // <---- Backwards
      *gradientError = 0.0f;
      temp[j]->camera.cam_pos.z = images[j]->camera.cam_pos.z; // reset for backwards
      temp[j]->camera.cam_pos.z -= h_linear;
      bundleTemp = generateBundles(matchSet,temp); // get the bundles for the new temp images
      twoViewTriangulate(bundleTemp, gradientError);
      float backwards = *gradientError;
      // free up Memory
      delete bundleTemp.bundles;
      delete bundleTemp.lines;
      // calculate the gradient with central difference
      gradient[j]->camera.cam_pos.z = ( forward - backwards ) / ( 2*h_linear );
    }

    //
    // Rotation x^ Gradient
    //
    for (int j = 0; j < images.size(); j++){
      // ----> Forward
      *gradientError = 0.0f;
      temp[j]->camera.cam_rot.x = images[j]->camera.cam_rot.x; // reset for forwards
      temp[j]->camera.cam_rot.x += h_radial;
      bundleTemp = generateBundles(matchSet,temp); // get the bundles for the new temp images
      twoViewTriangulate(bundleTemp, gradientError);
      float forward = *gradientError;
      // free up Memory
      delete bundleTemp.bundles;
      delete bundleTemp.lines;
      // <---- Backwards
      *gradientError = 0.0f;
      temp[j]->camera.cam_rot.x = images[j]->camera.cam_rot.x; // reset for backwards
      temp[j]->camera.cam_rot.x -= h_radial;
      bundleTemp = generateBundles(matchSet,temp); // get the bundles for the new temp images
      twoViewTriangulate(bundleTemp, gradientError);
      float backwards = *gradientError;
      // free up Memory
      delete bundleTemp.bundles;
      delete bundleTemp.lines;
      // calculate the gradient with central difference
      gradient[j]->camera.cam_rot.x = ( forward - backwards ) / ( 2*h_radial );
      if (gradient[j]->camera.cam_rot.x > (2*PI)){
        gradient[j]->camera.cam_rot.x -= floor((gradient[j]->camera.cam_rot.x/(2*PI)))*(2*PI);
      }
    }

    //
    // Rotation y^ Gradient
    //
    for (int j = 0; j < images.size(); j++){
      // ----> Forward
      *gradientError = 0.0f;
      temp[j]->camera.cam_rot.y = images[j]->camera.cam_rot.y; // reset for forwards
      temp[j]->camera.cam_rot.y += h_radial;
      bundleTemp = generateBundles(matchSet,temp); // get the bundles for the new temp images
      twoViewTriangulate(bundleTemp, gradientError);
      float forward = *gradientError;
      // free up Memory
      delete bundleTemp.bundles;
      delete bundleTemp.lines;
      // <---- Backwards
      *gradientError = 0.0f;
      temp[j]->camera.cam_rot.y = images[j]->camera.cam_rot.y; // reset for backwards
      temp[j]->camera.cam_rot.y -= h_radial;
      bundleTemp = generateBundles(matchSet,temp); // get the bundles for the new temp images
      twoViewTriangulate(bundleTemp, gradientError);
      float backwards = *gradientError;
      // free up Memory
      delete bundleTemp.bundles;
      delete bundleTemp.lines;
      // calculate the gradient with central difference
      gradient[j]->camera.cam_rot.y = ( forward - backwards ) / ( 2*h_radial );
      // adjust to be within bounds if needed
      if (gradient[j]->camera.cam_rot.y > (2*PI)){
        gradient[j]->camera.cam_rot.y -= floor((gradient[j]->camera.cam_rot.y/(2*PI)))*(2*PI);
      }
    }

    //
    // Rotation z^ Gradient
    //
    for (int j = 0; j < images.size(); j++){
      // ----> Forward
      *gradientError = 0.0f;
      temp[j]->camera.cam_rot.z = images[j]->camera.cam_rot.z; // reset for forwards
      temp[j]->camera.cam_rot.z += h_radial;
      bundleTemp = generateBundles(matchSet,temp); // get the bundles for the new temp images
      twoViewTriangulate(bundleTemp, gradientError);
      float forward = *gradientError;
      // free up Memory
      delete bundleTemp.bundles;
      delete bundleTemp.lines;
      // <---- Backwards
      *gradientError = 0.0f;
      temp[j]->camera.cam_rot.z = images[j]->camera.cam_rot.z; // reset for backwards
      temp[j]->camera.cam_rot.z -= h_radial;
      bundleTemp = generateBundles(matchSet,temp); // get the bundles for the new temp images
      twoViewTriangulate(bundleTemp, gradientError);
      float backwards = *gradientError;
      // free up Memory
      delete bundleTemp.bundles;
      delete bundleTemp.lines;
      // calculate the gradient with central difference
      gradient[j]->camera.cam_rot.z = ( forward - backwards ) / ( 2*h_radial );
      if (gradient[j]->camera.cam_rot.z > (2*PI)){
        gradient[j]->camera.cam_rot.z -= floor((gradient[j]->camera.cam_rot.z/(2*PI)))*(2*PI);
      }
    }

    // print of the gradients if debugging
    if (local_debug){
      std::cout << "\t gradient calculated as: " << std::endl;
      for (int j = 0; j < images.size(); j++) {
        std::cout << "\t\t     id : " << std::setprecision(12) << j << std::endl;
        std::cout << "\t\t size x [ " << std::setprecision(12) << gradient[j]->camera.size.x << " ]" << std::endl;
        std::cout << "\t\t size y [ " << std::setprecision(12) << gradient[j]->camera.size.y << " ]" << std::endl;
        std::cout << "\t\t  pos x [ " << std::setprecision(12) << gradient[j]->camera.cam_pos.x << " ]" << std::endl;
        std::cout << "\t\t  pos y [ " << std::setprecision(12) << gradient[j]->camera.cam_pos.y << " ]" << std::endl;
        std::cout << "\t\t  pos z [ " << std::setprecision(12) << gradient[j]->camera.cam_pos.z << " ]" << std::endl;
        std::cout << "\t\t  rot x [ " << std::setprecision(12) << gradient[j]->camera.cam_rot.x << " ]" << std::endl;
        std::cout << "\t\t  rot y [ " << std::setprecision(12) << gradient[j]->camera.cam_rot.y << " ]" << std::endl;
        std::cout << "\t\t  rot z [ " << std::setprecision(12) << gradient[j]->camera.cam_rot.z << " ]" << std::endl;
        std::cout << "\t\t  fov x [ " << std::setprecision(12) << gradient[j]->camera.fov.x << " ]" << std::endl;
        std::cout << "\t\t  fov y [ " << std::setprecision(12) << gradient[j]->camera.fov.y << " ]" << std::endl;
        std::cout << "\t\t    foc [ " << std::setprecision(12) << gradient[j]->camera.foc << " ]" << std::endl;
      }
    }

    // fill in the previous step's params
    x0.cam_pos0 = images[0]->camera.cam_pos;
    x0.cam_rot0 = images[0]->camera.cam_rot;
    x0.cam_pos1 = images[1]->camera.cam_pos;
    x0.cam_rot1 = images[1]->camera.cam_rot;

    // calculating the real adjustment
    adjustment.cam_pos0 = gamma * gradient[0]->camera.cam_pos;
    adjustment.cam_rot0 = gamma * gradient[0]->camera.cam_rot;
    adjustment.cam_pos1 = gamma * gradient[1]->camera.cam_pos;
    adjustment.cam_rot1 = gamma * gradient[1]->camera.cam_rot;

    // print the adjustment
    // TODO remove this later
    if (local_debug){
      std::cout << "\t adjustment calculated as: " << std::endl;
      std::cout << "\t\t  0" << std::endl;
      std::cout << "\t\t  pos x [ " << std::setprecision(12) << adjustment.cam_pos0.x << " ]" << std::endl;
      std::cout << "\t\t  pos y [ " << std::setprecision(12) << adjustment.cam_pos0.y << " ]" << std::endl;
      std::cout << "\t\t  pos z [ " << std::setprecision(12) << adjustment.cam_pos0.z << " ]" << std::endl;
      std::cout << "\t\t  rot x [ " << std::setprecision(12) << adjustment.cam_rot0.x << " ]" << std::endl;
      std::cout << "\t\t  rot y [ " << std::setprecision(12) << adjustment.cam_rot0.y << " ]" << std::endl;
      std::cout << "\t\t  rot z [ " << std::setprecision(12) << adjustment.cam_rot0.z << " ]" << std::endl;
      std::cout << "\t\t  1" << std::endl;
      std::cout << "\t\t  pos x [ " << std::setprecision(12) << adjustment.cam_pos1.x << " ]" << std::endl;
      std::cout << "\t\t  pos y [ " << std::setprecision(12) << adjustment.cam_pos1.y << " ]" << std::endl;
      std::cout << "\t\t  pos z [ " << std::setprecision(12) << adjustment.cam_pos1.z << " ]" << std::endl;
      std::cout << "\t\t  rot x [ " << std::setprecision(12) << adjustment.cam_rot1.x << " ]" << std::endl;
      std::cout << "\t\t  rot y [ " << std::setprecision(12) << adjustment.cam_rot1.y << " ]" << std::endl;
      std::cout << "\t\t  rot z [ " << std::setprecision(12) << adjustment.cam_rot1.z << " ]" << std::endl;
    }

    // take a step along along the gradient with a magnitude of gamma
    images[0]->camera.cam_pos = images[0]->camera.cam_pos - adjustment.cam_pos0;
    // images[0]->camera.cam_rot = images[0]->camera.cam_rot - adjustment.cam_rot0;
    images[1]->camera.cam_pos = images[1]->camera.cam_pos - adjustment.cam_pos1;
    // images[1]->camera.cam_rot = images[1]->camera.cam_rot - adjustment.cam_rot1;

    // fill in the new iteration's params
    x1.cam_pos0 = images[0]->camera.cam_pos;
    x1.cam_rot0 = images[0]->camera.cam_rot;
    x1.cam_pos1 = images[1]->camera.cam_pos;
    x1.cam_rot1 = images[1]->camera.cam_rot;

    // store the gradient
    if (i > 0){
      g0 = g1;
      // -- set the old iteration
      g1.cam_pos0 = gradient[0]->camera.cam_pos;
      g1.cam_rot0 = gradient[0]->camera.cam_rot;
      g1.cam_pos1 = gradient[1]->camera.cam_pos;
      g1.cam_rot1 = gradient[1]->camera.cam_rot;
    } else {
      g1.cam_pos0 = gradient[0]->camera.cam_pos;
      g1.cam_rot0 = gradient[0]->camera.cam_rot;
      g1.cam_pos1 = gradient[1]->camera.cam_pos;
      g1.cam_rot1 = gradient[1]->camera.cam_rot;
    }

    // calculate new gamma
    if(i > 0 && !const_step){
      struct CamAdjust2 xtemp;
      xtemp.cam_pos0 = x1.cam_pos0 - x0.cam_pos0;
      xtemp.cam_rot0 = x1.cam_rot0 - x0.cam_rot0;
      xtemp.cam_pos1 = x1.cam_pos1 - x0.cam_pos1;
      xtemp.cam_rot1 = x1.cam_rot1 - x0.cam_rot1;
      struct CamAdjust2 gtemp;
      gtemp.cam_pos0 = g1.cam_pos0 - g0.cam_pos0;
      gtemp.cam_rot0 = g1.cam_rot0 - g0.cam_rot0;
      gtemp.cam_pos1 = g1.cam_pos1 - g0.cam_pos1;
      gtemp.cam_rot1 = g1.cam_rot1 - g0.cam_rot1;
      float numer  = (xtemp.cam_pos0.x * gtemp.cam_pos0.x) + (xtemp.cam_pos0.y * gtemp.cam_pos0.y) + (xtemp.cam_pos0.z * gtemp.cam_pos0.z);
            // numer += (xtemp.cam_rot0.x * gtemp.cam_rot0.x) + (xtemp.cam_rot0.y * gtemp.cam_rot0.y) + (xtemp.cam_rot0.z * gtemp.cam_rot0.z);
            numer += (xtemp.cam_pos1.x * gtemp.cam_pos1.x) + (xtemp.cam_pos1.y * gtemp.cam_pos1.y) + (xtemp.cam_pos1.z * gtemp.cam_pos1.z);
            // numer += (xtemp.cam_rot1.x * gtemp.cam_rot1.x) + (xtemp.cam_rot1.y * gtemp.cam_rot1.y) + (xtemp.cam_rot1.z * gtemp.cam_rot1.z);
      float denom  = (gtemp.cam_pos0.x * gtemp.cam_pos0.x) + (gtemp.cam_pos0.y * gtemp.cam_pos0.y) + (gtemp.cam_pos0.z * gtemp.cam_pos0.z);
            // denom += (gtemp.cam_rot0.x * gtemp.cam_rot0.x) + (gtemp.cam_rot0.y * gtemp.cam_rot0.y) + (gtemp.cam_rot0.z * gtemp.cam_rot0.z);
            denom += (gtemp.cam_pos1.x * gtemp.cam_pos1.x) + (gtemp.cam_pos1.y * gtemp.cam_pos1.y) + (gtemp.cam_pos1.z * gtemp.cam_pos1.z);
            // denom += (gtemp.cam_rot1.x * gtemp.cam_rot1.x) + (gtemp.cam_rot1.y * gtemp.cam_rot1.y) + (gtemp.cam_rot1.z * gtemp.cam_rot1.z);
            denom  = sqrtf(denom);
      gamma = abs(numer) / denom;
    }

    // print the new error after the step
    bundleTemp = generateBundles(matchSet,images);
    points = twoViewTriangulate(bundleTemp, initialError);
    std::cout << "[" << i << "]\t adjusted error: " << std::setprecision(32) << *initialError << std::endl;
    if (!const_step){
        std::cout << "\t\t new gamma: "    << std::setprecision(12)<< gamma << std::endl;
    }
    errorTracker.push_back(*initialError);
  } // end bundle adjustment loop

  // TODO only do if debugging
  // write linearError chagnes to a CSV
  writeCSV(errorTracker, "totalErrorOverIterations");

  // update the images that were passed in with the final image parameters
  for (int i = 0; i < images.size(); i++){
    images[i]->camera.cam_pos = temp[i]->camera.cam_pos; // Updates the positions
    // TODO update the orientations
  }

  // return the new points
  return points;
}


// =============================================================================================================
//
// Debug Methods
//
// =============================================================================================================

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
 * Saves a colored point cloud where the colors correspond do the linear errors from within the cloud.
 * @param matchSet a group of matches
 * @param images a group of images, used only for their stored camera parameters
 * @param filename the name of the file that should be saved
 */
void ssrlcv::PointCloudFactory::saveDebugLinearErrorCloud(ssrlcv::MatchSet* matchSet, std::vector<ssrlcv::Image*> images, const char* filename){
  // build the helpers to make the colors
  uchar3 colors[2000];
  float3 good = {108,255,221};
  float3 meh  = {251,215,134};
  float3 bad  = {247,121,125};
  float3 gr1  = (meh - good)/1000;
  float3 gr2  = (bad - meh )/1000;
  // initialize the gradient "mapping"
  float3 temp;
  std::cout << "building gradient" << std::endl;
  for (int i = 0; i < 2000; i++){
    if (i < 1000){
      temp = good + gr1*i;
      colors[i].x = (unsigned char) floor(temp.x);
      colors[i].y = (unsigned char) floor(temp.y);
      colors[i].z = (unsigned char) floor(temp.z);
    } else {
      temp = meh  + gr2*i;
      colors[i].x = (unsigned char) floor(temp.x);
      colors[i].y = (unsigned char) floor(temp.y);
      colors[i].z = (unsigned char) floor(temp.z);
    }
  }
  std::cout << "the boiz" << std::endl;
  float* linearError = (float*) malloc(sizeof(float));
  *linearError = 0.0; // just something to start
  float* linearErrorCutoff = (float*) malloc(sizeof(float));
  *linearErrorCutoff = 1000000.0; // just somethihng to start

  // the boiz
  ssrlcv::BundleSet      bundleSet;
  ssrlcv::Unity<float>*  errors;
  ssrlcv::Unity<float3>* points;

  // need bundles
  bundleSet = generateBundles(matchSet,images);
  // do an initial triangulation
  errors = new ssrlcv::Unity<float>(nullptr,matchSet->matches->size(),ssrlcv::cpu);
  struct colorPoint* cpoints = (colorPoint*)  malloc(matchSet->matches->size() * sizeof(struct colorPoint));

  std::cout << "attempting guy" << std::endl;
  if (images.size() == 2){
    //
    // 2-View Case
    //

    points = twoViewTriangulate(bundleSet, errors, linearError, linearErrorCutoff);
    float max = 0.0; // it would be nice to have a better way to get the max, but because this is only for debug idc
    for (int i = 0; i < errors->size(); i++){
      if (errors->host[i] > max){
        max = errors->host[i];
      }
    }
    std::cout << "found max: " << max << std::endl;
    // now fill in the color point locations
    for (int i = 0; i < points->size() - 1; i++){
      // i assume that the errors and the points will have the same indices
      cpoints[i].x = points->host[i].x; //
      cpoints[i].y = points->host[i].y;
      cpoints[i].z = points->host[i].z;
      int j = floor(errors->host[i] * (2000 / max));
      // std::cout << "j: " << j << "\t e: " << errors->host[i] << "\t ratio: " << (2000 / max) << "\t " << i << "/" << points->size() << std::endl;
      cpoints[i].r = colors[j].x;
      cpoints[i].g = colors[j].y;
      cpoints[i].b = colors[j].z;
    }

  } else {
    //
    // N-View Case
    //

    points = nViewTriangulate(bundleSet, errors, linearError, linearErrorCutoff);
    float max = 0.0; // it would be nice to have a better way to get the max, but because this is only for debug idc
    for (int i = 0; i < errors->size(); i++){
      if (errors->host[i] > max){
        max = errors->host[i];
      }
    }
    std::cout << "found max: " << max << std::endl;
    // now fill in the color point locations
    for (int i = 0; i < points->size() - 1; i++){
      // i assume that the errors and the points will have the same indices
      cpoints[i].x = points->host[i].x; //
      cpoints[i].y = points->host[i].y;
      cpoints[i].z = points->host[i].z;
      int j = floor(errors->host[i] * (2000 / max));
      // std::cout << "j: " << j << "\t e: " << errors->host[i] << "\t ratio: " << (2000 / max) << "\t " << i << "/" << points->size() << std::endl;
      cpoints[i].r = colors[j].x;
      cpoints[i].g = colors[j].y;
      cpoints[i].b = colors[j].z;
    }

  }

  // save the file
  ssrlcv::writePLY(filename, cpoints, matchSet->matches->size());
}

/**
 * Saves a colored point cloud where the colors correspond to the number of images matched in each color
 * @param matchSet a group of matches
 * @param images a group of images, used only for their stored camera parameters
 * @param filename the name of the file that should be saved
 */
void ssrlcv::PointCloudFactory::saveViewNumberCloud(ssrlcv::MatchSet* matchSet, std::vector<ssrlcv::Image*> images, const char* filename){
  // build the helpers to make the colors
  uchar3 colors[2000];
  float3 good = {108,255,221};
  float3 meh  = {251,215,134};
  float3 bad  = {247,121,125};
  float3 gr1  = (meh - good)/1000;
  float3 gr2  = (bad - meh )/1000;
  // initialize the gradient "mapping"
  float3 temp;
  std::cout << "building gradient" << std::endl;
  for (int i = 0; i < 2000; i++){
    if (i < 1000){
      temp = good + gr1*i;
      colors[i].x = (unsigned char) floor(temp.x);
      colors[i].y = (unsigned char) floor(temp.y);
      colors[i].z = (unsigned char) floor(temp.z);
    } else {
      temp = meh  + gr2*i;
      colors[i].x = (unsigned char) floor(temp.x);
      colors[i].y = (unsigned char) floor(temp.y);
      colors[i].z = (unsigned char) floor(temp.z);
    }
  }
  std::cout << "the boiz" << std::endl;
  float* linearError = (float*) malloc(sizeof(float));
  *linearError = 0.0; // just something to start
  float* linearErrorCutoff = (float*) malloc(sizeof(float));
  *linearErrorCutoff = 1000000.0; // just somethihng to start

  // the boiz
  ssrlcv::BundleSet      bundleSet;
  ssrlcv::Unity<float>*  errors;
  ssrlcv::Unity<float3>* points;

  // need bundles
  bundleSet = generateBundles(matchSet,images);
  // do an initial triangulation
  errors = new ssrlcv::Unity<float>(nullptr,matchSet->matches->size(),ssrlcv::cpu);
  struct colorPoint* cpoints = (colorPoint*)  malloc(matchSet->matches->size() * sizeof(struct colorPoint));

  std::cout << "attempting guy" << std::endl;



  //
  // N-View Case
  //

  points = nViewTriangulate(bundleSet, errors, linearError, linearErrorCutoff);
  float max = images.size();
  // now fill in the color point locations
  for (int i = 0; i < points->size() - 1; i++){
    // i assume that the errors and the points will have the same indices
    cpoints[i].x = points->host[i].x; //
    cpoints[i].y = points->host[i].y;
    cpoints[i].z = points->host[i].z;

    int j = floor(bundleSet.bundles->host[i].numLines * (2000 / max));
    // int j = floor(errors->host[i] * (2000 / max));
    // // std::cout << "j: " << j << "\t e: " << errors->host[i] << "\t ratio: " << (2000 / max) << "\t " << i << "/" << points->size() << std::endl;
    cpoints[i].r = colors[j].x;
    cpoints[i].g = colors[j].y;
    cpoints[i].b = colors[j].z;
  }



  // save the file
  ssrlcv::writePLY(filename, cpoints, matchSet->matches->size());
}

/**
 * Saves several CSV's which have (x,y) coordinates representing step the step from an intial condution and
 * the output error for that condition, this should be graphed
 * @param matchSet a group of matches
 * @param images a group of images, used only for their stored camera parameters
 * @param filename the name of the file that should be saved
 */
void ssrlcv::PointCloudFactory::generateSensitivityFunctions(ssrlcv::MatchSet* matchSet, std::vector<ssrlcv::Image*> images, std::string filename){

  // the bundle set that changes each iteration
  BundleSet bundleSet;

  // the ranges and step sizes
  float linearRange   = 10.0;     //   +/- linear Range
  float angularRange  = (PI);   //   +/- angular Range
  float deltaL = 0.01;          // linearRange stepsize
  float deltaA = 0.001;        // angular stpesize

  // the temp error to be stored
  float* currError = (float*)malloc(sizeof(float));
  float start;
  float end;

  // TRACKERS
  std::vector<float> trackerValue;
  std::vector<float> trackerError;

  // the camera to refrence when doing the sensitivity test
  int ref_cam = 0;

  // the temp cameras
  std::vector<ssrlcv::Image*> temp;
  for (int i = 0; i < images.size(); i++){
    temp.push_back(images[i]); // fill in the initial images
  }

  if (images.size() == 2){
    //
    // 2-View Case
    //

    std::cout << "WARNING!!!" << std::endl;
    std::cout << "WARNING: Starting an intesive debug feature, this should be disabled in production" << std::endl;
    std::cout << "WARNING: DISABLE GENERATE SENSITIVITY FUNCTIONS IN PRODUCTION!!" << std::endl;

    // TODO graph (deltas and value)

    //
    // DELTA X Linear
    //
    std::cout << "\tTesting x linear sensitivity ..." << std::endl;
    start = temp[ref_cam]->camera.cam_pos.x - linearRange;
    end   = temp[ref_cam]->camera.cam_pos.x + linearRange;
    temp[ref_cam]->camera.cam_pos.x = start;
    while (temp[ref_cam]->camera.cam_pos.x < end){
      bundleSet = generateBundles(matchSet,temp);
      voidTwoViewTriangulate(bundleSet, currError);
      trackerValue.push_back(temp[ref_cam]->camera.cam_pos.x);
      trackerError.push_back(*currError);
      temp[ref_cam]->camera.cam_pos.x += deltaL;
      // free up Memory
      delete bundleSet.bundles;
      delete bundleSet.lines;
    }
    // save the file
    writeCSV(trackerValue, trackerError, filename + "_DeltaXLinear");
    // reset
    temp[ref_cam]->camera.cam_pos.x = images[ref_cam]->camera.cam_pos.x;
    // free up memory
    trackerValue.clear();
    trackerError.clear();

    //
    // DELTA Y Linear
    //
    std::cout << "\tTesting y linear sensitivity ..." << std::endl;
    start = temp[ref_cam]->camera.cam_pos.y - linearRange;
    end   = temp[ref_cam]->camera.cam_pos.y + linearRange;
    temp[ref_cam]->camera.cam_pos.y = start;
    while (temp[ref_cam]->camera.cam_pos.y < end){
      bundleSet = generateBundles(matchSet,temp);
      voidTwoViewTriangulate(bundleSet, currError);
      trackerValue.push_back(temp[ref_cam]->camera.cam_pos.y);
      trackerError.push_back(*currError);
      temp[ref_cam]->camera.cam_pos.y += deltaL;
      // free up Memory
      delete bundleSet.bundles;
      delete bundleSet.lines;
    }
    // save the file
    writeCSV(trackerValue, trackerError, filename + "_DeltaYLinear");
    // reset
    temp[ref_cam]->camera.cam_pos.y = images[ref_cam]->camera.cam_pos.y;
    // free up memory
    trackerValue.clear();
    trackerError.clear();

    //
    // DELTA Z Linear
    //
    std::cout << "\tTesting z linear sensitivity ..." << std::endl;
    start = temp[ref_cam]->camera.cam_pos.z - linearRange;
    end   = temp[ref_cam]->camera.cam_pos.z + linearRange;
    temp[ref_cam]->camera.cam_pos.y = start;
    while (temp[ref_cam]->camera.cam_pos.z < end){
      bundleSet = generateBundles(matchSet,temp);
      voidTwoViewTriangulate(bundleSet, currError);
      trackerValue.push_back(temp[ref_cam]->camera.cam_pos.z);
      trackerError.push_back(*currError);
      temp[ref_cam]->camera.cam_pos.z += deltaL;
      // free up Memory
      delete bundleSet.bundles;
      delete bundleSet.lines;
    }
    // save the file
    writeCSV(trackerValue, trackerError, filename + "_DeltaZLinear");
    // reset
    temp[ref_cam]->camera.cam_pos.y = images[ref_cam]->camera.cam_pos.y;
    // free up memory
    trackerValue.clear();
    trackerError.clear();

    //
    // DELTA X Angular
    //
    std::cout << "\tTesting x angular sensitivity ..." << std::endl;
    start = temp[ref_cam]->camera.cam_rot.x - angularRange;
    end   = temp[ref_cam]->camera.cam_rot.x + angularRange;
    temp[ref_cam]->camera.cam_rot.x = start;
    while (temp[ref_cam]->camera.cam_rot.x < end){
      bundleSet = generateBundles(matchSet,temp);
      voidTwoViewTriangulate(bundleSet, currError);
      trackerValue.push_back(temp[ref_cam]->camera.cam_rot.x);
      trackerError.push_back(*currError);
      temp[ref_cam]->camera.cam_rot.x += deltaA;
      // free up Memory
      delete bundleSet.bundles;
      delete bundleSet.lines;
    }
    // save the file
    writeCSV(trackerValue, trackerError, filename + "_DeltaXAngular");
    // reset
    temp[ref_cam]->camera.cam_rot.x = images[ref_cam]->camera.cam_rot.x;
    // free up memory
    trackerValue.clear();
    trackerError.clear();

    //
    // DELTA Y Angular
    //
    std::cout << "\tTesting y angular sensitivity ..." << std::endl;
    start = temp[ref_cam]->camera.cam_rot.y - angularRange;
    end   = temp[ref_cam]->camera.cam_rot.y + angularRange;
    temp[ref_cam]->camera.cam_rot.y = start;
    while (temp[ref_cam]->camera.cam_rot.y < end){
      bundleSet = generateBundles(matchSet,temp);
      voidTwoViewTriangulate(bundleSet, currError);
      trackerValue.push_back(temp[ref_cam]->camera.cam_rot.y);
      trackerError.push_back(*currError);
      temp[ref_cam]->camera.cam_rot.y += deltaA;
      // free up Memory
      delete bundleSet.bundles;
      delete bundleSet.lines;
    }
    // save the file
    writeCSV(trackerValue, trackerError, filename + "_DeltaYAngular");
    // reset
    temp[ref_cam]->camera.cam_rot.y = images[ref_cam]->camera.cam_rot.y;
    // free up memory
    trackerValue.clear();
    trackerError.clear();

    //
    // DELTA Z Angular
    //
    std::cout << "\tTesting z angular sensitivity ..." << std::endl;
    start = temp[ref_cam]->camera.cam_rot.z - angularRange;
    end   = temp[ref_cam]->camera.cam_rot.z + angularRange;
    temp[ref_cam]->camera.cam_rot.z = start;
    while (temp[ref_cam]->camera.cam_rot.z < end){
      bundleSet = generateBundles(matchSet,temp);
      voidTwoViewTriangulate(bundleSet, currError);
      trackerValue.push_back(temp[ref_cam]->camera.cam_rot.z);
      trackerError.push_back(*currError);
      temp[ref_cam]->camera.cam_rot.z += deltaA;
      // free up Memory
      delete bundleSet.bundles;
      delete bundleSet.lines;
    }
    // save the file
    writeCSV(trackerValue, trackerError, filename + "_DeltaZAngular");
    // reset
    temp[ref_cam]->camera.cam_rot.z = images[ref_cam]->camera.cam_rot.z;
    // free up memory
    trackerValue.clear();
    trackerError.clear();

  } else {
    //
    // N-View Case
    //

    std::cerr << "ERROR: sensitivity generation not yet implemented for N-view" << std::endl;
    return;
  }



}

// =============================================================================================================
//
// Filtering Methods
//
// =============================================================================================================

/**
 * Deterministically filters, with the assumption that the data is guassian, statistical outliers of the pointcloud
 * set and returns a matchSet without such outliers. The method is deterministic by taking a uniformly spaced sample of points
 * within the matcheSet.
 * @param matchSet a group of matches
 * @param images a group of images, used only for their stored camera parameters
 * @param sigma is the variance to cutoff from
 * @param sampleSize represents a percentage and should be between 0.0 and 1.0
 */
void ssrlcv::PointCloudFactory::deterministicStatisticalFilter(ssrlcv::MatchSet* matchSet, std::vector<ssrlcv::Image*> images, float sigma, float sampleSize){
  if (sampleSize > 1.0 || sampleSize < 0.0) {
    std::cerr << "ERROR:  not statistical filtering possible with percentage greater than 1.0 or less than 0.0" << std::endl;
    return;
  }
  // find an integer skip that can be used
  int sampleJump = (int) (1/sampleSize); // need a constant jump

  // the initial linear error
  float* linearError = (float*) malloc(sizeof(float));
  *linearError = 0.0; // just something to start
  // the cutoff
  float* linearErrorCutoff = (float*) malloc(sizeof(float));
  *linearErrorCutoff = 0.0; // just somethihng to start

  // the boiz
  ssrlcv::BundleSet bundleSet;
  ssrlcv::MatchSet tempMatchSet;
  tempMatchSet.keyPoints = new ssrlcv::Unity<ssrlcv::KeyPoint>(nullptr,1,ssrlcv::cpu);
  tempMatchSet.matches   = new ssrlcv::Unity<ssrlcv::MultiMatch>(nullptr,1,ssrlcv::cpu);
  ssrlcv::Unity<float>*    errors;
  ssrlcv::Unity<float>*    errors_sample;
  ssrlcv::Unity<float3>*   points;

  // need bundles
  bundleSet = generateBundles(matchSet,images);
  // do an initial triangulation
  errors = new ssrlcv::Unity<float>(nullptr,matchSet->matches->size(),ssrlcv::cpu);

  std::cout << "Starting Determinstic Statistical Filter ..." << std::endl;

  // do an initial triangulate

  if (images.size() == 2){
    //
    // This is the 2-View case
    //

    points = twoViewTriangulate(bundleSet, errors, linearError, linearErrorCutoff);
  } else {
    //
    // This is the N-View case
    //

    points = nViewTriangulate(bundleSet, errors, linearError, linearErrorCutoff);
  }

  // the assumption is that choosing every ""stableJump"" indexes is random enough
  // https://en.wikipedia.org/wiki/Variance#Sample_variance
  size_t sample_size = (int) (errors->size() - (errors->size()%sampleJump))/sampleJump; // make sure divisible by the stableJump int always
  errors_sample      = new ssrlcv::Unity<float>(nullptr,sample_size,ssrlcv::cpu);
  float sample_sum   = 0;
  for (int k = 0; k < sample_size; k++){
    errors_sample->host[k] = errors->host[k*sampleJump];
    sample_sum += errors->host[k*sampleJump];
  }
  float sample_mean = sample_sum / errors_sample->size();
  std::cout << "\tSample Sum: " << std::setprecision(32) << sample_sum << std::endl;
  std::cout << "\tSample Mean: " << std::setprecision(32) << sample_mean << std::endl;
  float squared_sum = 0;
  for (int k = 0; k < sample_size; k++){
    squared_sum += (errors_sample->host[k] - sample_mean)*(errors_sample->host[k] - sample_mean);
  }
  float variance = squared_sum / errors_sample->size();
  std::cout << "\tSample variance: " << std::setprecision(32) << variance << std::endl;
  std::cout << "\tSigma Calculated As: " << std::setprecision(32) << sqrtf(variance) << std::endl;
  std::cout << "\tLinear Error Cutoff Adjusted To: " << std::setprecision(32) << sigma * sqrtf(variance) << std::endl;
  *linearErrorCutoff = sigma * sqrtf(variance);

  // do the two view version of this (easier for now)
  if (images.size() == 2){
    //
    // This is the 2-View case
    //

    // recalculate with new cutoff
    points = twoViewTriangulate(bundleSet, errors, linearError, linearErrorCutoff);

    // CLEAR OUT THE DATA STRUCTURES
    // count the number of bad bundles to be removed
    int bad_bundles = 0;
    for (int k = 0; k < bundleSet.bundles->size(); k++){
      if (bundleSet.bundles->host[k].invalid){
         bad_bundles++;
      }
    }
    if (bad_bundles) std::cout << "\tDetected " << bad_bundles << " bad bundles to remove" << std::endl;
    // Need to generated and adjustment match set
    // make a temporary match set
    delete tempMatchSet.keyPoints;
    delete tempMatchSet.matches;
    tempMatchSet.keyPoints = new ssrlcv::Unity<ssrlcv::KeyPoint>(nullptr,matchSet->matches->size()*2,ssrlcv::cpu);
    tempMatchSet.matches   = new ssrlcv::Unity<ssrlcv::MultiMatch>(nullptr,matchSet->matches->size(),ssrlcv::cpu);
    // fill in the boiz
    for (int k = 0; k < tempMatchSet.keyPoints->size(); k++){
      tempMatchSet.keyPoints->host[k] = matchSet->keyPoints->host[k];
    }
    for (int k = 0; k < tempMatchSet.matches->size(); k++){
      tempMatchSet.matches->host[k] = matchSet->matches->host[k];
    }
    if (!(matchSet->matches->size() - bad_bundles)){
      std::cerr << "ERROR: filtering is too aggressive, all points would be removed ..." << std::endl;
      return;
    }
    // resize the standard matchSet
    size_t new_kp_size = 2*(matchSet->matches->size() - bad_bundles);
    size_t new_mt_size = matchSet->matches->size() - bad_bundles;
    delete matchSet->keyPoints;
    delete matchSet->matches;
    matchSet->keyPoints = new ssrlcv::Unity<ssrlcv::KeyPoint>(nullptr,new_kp_size,ssrlcv::cpu);
    matchSet->matches   = new ssrlcv::Unity<ssrlcv::MultiMatch>(nullptr,new_mt_size,ssrlcv::cpu);
    // this is much easier because of the 2 view assumption
    // there are the same number of lines as there are are keypoints and the same number of bundles as there are matches
    int k_adjust = 0;
    // if (bad_bundles){
    for (int k = 0; k < bundleSet.bundles->size(); k++){
    	if (!bundleSet.bundles->host[k].invalid){
    	  matchSet->keyPoints->host[2*k_adjust]     = tempMatchSet.keyPoints->host[2*k];
    	  matchSet->keyPoints->host[2*k_adjust + 1] = tempMatchSet.keyPoints->host[2*k + 1];
        matchSet->matches->host[k_adjust]         = {2,2*k_adjust};
    	  k_adjust++;
    	}
    }
    if (bad_bundles) std::cout << "\tRemoved bad bundles" << std::endl;
  } else {
    //
    // This is the N-view case
    //

    // recalculate with new cutoff
    points = nViewTriangulate(bundleSet, errors, linearError, linearErrorCutoff);

    // CLEAR OUT THE DATA STRUCTURES
    // count the number of bad bundles to be removed
    int bad_bundles = 0;
    int bad_lines   = 0;
    for (int k = 0; k < bundleSet.bundles->size(); k++){
      if (bundleSet.bundles->host[k].invalid){
         bad_bundles++;
         bad_lines += bundleSet.bundles->host[k].numLines;
      }
    }
    if (bad_bundles) {
      std::cout << "\tDetected " << bad_bundles << " bundles to remove" << std::endl;
      std::cout << "\tDetected " << bad_lines << " lines to remove" << std::endl;
    } else {
      std::cout << "No points removed! all are less than " << linearErrorCutoff << std::endl;
      return;
    }
    // Need to generated and adjustment match set
    // make a temporary match set
    delete tempMatchSet.keyPoints;
    delete tempMatchSet.matches;
    tempMatchSet.keyPoints = new ssrlcv::Unity<ssrlcv::KeyPoint>(nullptr,matchSet->keyPoints->size(),ssrlcv::cpu);
    tempMatchSet.matches   = new ssrlcv::Unity<ssrlcv::MultiMatch>(nullptr,matchSet->matches->size(),ssrlcv::cpu);
    // fill in the boiz
    for (int k = 0; k < tempMatchSet.keyPoints->size(); k++){
      tempMatchSet.keyPoints->host[k] = matchSet->keyPoints->host[k];
    }
    for (int k = 0; k < tempMatchSet.matches->size(); k++){
      tempMatchSet.matches->host[k] = matchSet->matches->host[k];
    }
    if (!(matchSet->matches->size() - bad_bundles) || !(matchSet->keyPoints->size() - bad_lines)){
      std::cerr << "ERROR: filtering is too aggressive, all points would be removed ..." << std::endl;
      return;
    }
    // resize the standard matchSet
    size_t new_kp_size = matchSet->keyPoints->size() - bad_lines;
    size_t new_mt_size = matchSet->matches->size() - bad_bundles;
    delete matchSet->keyPoints;
    delete matchSet->matches;
    matchSet->keyPoints = new ssrlcv::Unity<ssrlcv::KeyPoint>(nullptr,new_kp_size,ssrlcv::cpu);
    matchSet->matches   = new ssrlcv::Unity<ssrlcv::MultiMatch>(nullptr,new_mt_size,ssrlcv::cpu);
    // this is much easier because of the 2 view assumption
    // there are the same number of lines as there are are keypoints and the same number of bundles as there are matches
    int k_adjust = 0;
    int k_lines  = 0;
    int k_bundle = 0;
    for (int k = 0; k < bundleSet.bundles->size(); k++){
      k_lines = bundleSet.bundles->host[k].numLines;
      if (!bundleSet.bundles->host[k].invalid){
        matchSet->matches->host[k_bundle] = {k_lines,k_adjust};
        for (int j = 0; j < k_lines; j++){
          matchSet->keyPoints->host[k_adjust + j] = tempMatchSet.keyPoints->host[k_adjust + j];
        }
        k_adjust += k_lines;
        k_bundle++;
      }
    }

    if (bad_bundles) std::cout << "\tRemoved bundles" << std::endl;

  }
}

/**
 * NonDeterministically filters, with the assumption that the data is guassian, statistical outliers of the pointcloud
 * set and returns a matchSet without such outliers. It is the same as the deterministicStatisticalFilter only samples
 * are chosen randomly rather than equally spaced.
 * @param matchSet a group of matches
 * @param images a group of images, used only for their stored camera parameters
 * @param sigma is the variance to cutoff from
 * @param sampleSize represents a percentage and should be between 0.0 and 1.0
 */
void ssrlcv::PointCloudFactory::nonDeterministicStatisticalFilter(ssrlcv::MatchSet* matchSet, std::vector<ssrlcv::Image*> images, float sigma, float sampleSize){
  if (sampleSize > 1.0 || sampleSize < 0.0) {
    std::cerr << "ERROR:  not statistical filtering possible with percentage greater than 1.0 or less than 0.0" << std::endl;
    return;
  }
  std::cout << "TODO" << std::endl;
}

/**
 * A filter that removes all points with a linear error greater than the cutoff. Modifies the matchSet that is pass thru
 * @param matchSet a group of matches
 * @param images a group of images, used only for their stored camera parameters
 * @param cutoff the float that no linear errors should be greater than
 */
void ssrlcv::PointCloudFactory::linearCutoffFilter(ssrlcv::MatchSet* matchSet, std::vector<ssrlcv::Image*> images, float cutoff){
  if (cutoff < 0.0){
    std::cerr << "ERROR: cutoff must be positive" << std::endl;
    return;
  }

  // the initial linear error
  float* linearError = (float*) malloc(sizeof(float));
  *linearError = 0.0; // just something to start
  // the cutoff
  float* linearErrorCutoff = (float*) malloc(sizeof(float));
  *linearErrorCutoff = cutoff; // just somethihng to start

  // the boiz
  ssrlcv::BundleSet        bundleSet;
  ssrlcv::Unity<float3>*   points;
  ssrlcv::Unity<float>*    errors;
  ssrlcv::MatchSet         tempMatchSet;
  tempMatchSet.keyPoints = new ssrlcv::Unity<ssrlcv::KeyPoint>(nullptr,1,ssrlcv::cpu);
  tempMatchSet.matches   = new ssrlcv::Unity<ssrlcv::MultiMatch>(nullptr,1,ssrlcv::cpu);

  // need bundles
  bundleSet = generateBundles(matchSet,images);

  errors = new ssrlcv::Unity<float>(nullptr,matchSet->matches->size(),ssrlcv::cpu);

  // do the two view version of this (easier for now)
  if (images.size() == 2){
    //
    // This is the 2-View case
    //

    // recalculate with new cutoff
    points = twoViewTriangulate(bundleSet, errors, linearError, linearErrorCutoff);

    // CLEAR OUT THE DATA STRUCTURES
    // count the number of bad bundles to be removed
    int bad_bundles = 0;
    for (int k = 0; k < bundleSet.bundles->size(); k++){
      if (bundleSet.bundles->host[k].invalid){
         bad_bundles++;
      }
    }
    if (bad_bundles) {
      std::cout << "\tDetected " << bad_bundles << " bundles to remove" << std::endl;
    } else {
      std::cout << "No points removed! all are less than " << cutoff << std::endl;
      return;
    }
    // Need to generated and adjustment match set
    // make a temporary match set
    delete tempMatchSet.keyPoints;
    delete tempMatchSet.matches;
    tempMatchSet.keyPoints = new ssrlcv::Unity<ssrlcv::KeyPoint>(nullptr,matchSet->matches->size()*2,ssrlcv::cpu);
    tempMatchSet.matches   = new ssrlcv::Unity<ssrlcv::MultiMatch>(nullptr,matchSet->matches->size(),ssrlcv::cpu);
    // fill in the boiz
    for (int k = 0; k < tempMatchSet.keyPoints->size(); k++){
      tempMatchSet.keyPoints->host[k] = matchSet->keyPoints->host[k];
    }
    for (int k = 0; k < tempMatchSet.matches->size(); k++){
      tempMatchSet.matches->host[k] = matchSet->matches->host[k];
    }
    if (!(matchSet->matches->size() - bad_bundles)){
      std::cerr << "ERROR: filtering is too aggressive, all points would be removed ..." << std::endl;
      return;
    }
    // resize the standard matchSet
    size_t new_kp_size = 2*(matchSet->matches->size() - bad_bundles);
    size_t new_mt_size = matchSet->matches->size() - bad_bundles;
    delete matchSet->keyPoints;
    delete matchSet->matches;
    matchSet->keyPoints = new ssrlcv::Unity<ssrlcv::KeyPoint>(nullptr,new_kp_size,ssrlcv::cpu);
    matchSet->matches   = new ssrlcv::Unity<ssrlcv::MultiMatch>(nullptr,new_mt_size,ssrlcv::cpu);
    // this is much easier because of the 2 view assumption
    // there are the same number of lines as there are are keypoints and the same number of bundles as there are matches
    int k_adjust = 0;
    for (int k = 0; k < bundleSet.bundles->size(); k++){
    	if (!bundleSet.bundles->host[k].invalid){
    	  matchSet->keyPoints->host[2*k_adjust]     = tempMatchSet.keyPoints->host[2*k];
    	  matchSet->keyPoints->host[2*k_adjust + 1] = tempMatchSet.keyPoints->host[2*k + 1];
        matchSet->matches->host[k_adjust]         = {2,2*k_adjust};
    	  k_adjust++;
    	}
    }
    if (bad_bundles) std::cout << "\tRemoved bundles" << std::endl;
  } else {
    //
    // This is the N-view case
    //

    // recalculate with new cutoff
    points = nViewTriangulate(bundleSet, errors, linearError, linearErrorCutoff);

    // CLEAR OUT THE DATA STRUCTURES
    // count the number of bad bundles to be removed
    int bad_bundles = 0;
    int bad_lines   = 0;
    for (int k = 0; k < bundleSet.bundles->size(); k++){
      if (bundleSet.bundles->host[k].invalid){
         bad_bundles++;
         bad_lines += bundleSet.bundles->host[k].numLines;
      }
    }
    if (bad_bundles) {
      std::cout << "\tDetected " << bad_bundles << " bundles to remove" << std::endl;
      std::cout << "\tDetected " << bad_lines << " lines to remove" << std::endl;
    } else {
      std::cout << "No points removed! all are less than " << cutoff << std::endl;
      return;
    }
    // Need to generated and adjustment match set
    // make a temporary match set
    delete tempMatchSet.keyPoints;
    delete tempMatchSet.matches;
    tempMatchSet.keyPoints = new ssrlcv::Unity<ssrlcv::KeyPoint>(nullptr,matchSet->keyPoints->size(),ssrlcv::cpu);
    tempMatchSet.matches   = new ssrlcv::Unity<ssrlcv::MultiMatch>(nullptr,matchSet->matches->size(),ssrlcv::cpu);
    // fill in the boiz
    for (int k = 0; k < tempMatchSet.keyPoints->size(); k++){
      tempMatchSet.keyPoints->host[k] = matchSet->keyPoints->host[k];
    }
    for (int k = 0; k < tempMatchSet.matches->size(); k++){
      tempMatchSet.matches->host[k] = matchSet->matches->host[k];
    }
    if (!(matchSet->matches->size() - bad_bundles) || !(matchSet->keyPoints->size() - bad_lines)){
      std::cerr << "ERROR: filtering is too aggressive, all points would be removed ..." << std::endl;
      return;
    }
    // resize the standard matchSet
    size_t new_kp_size = matchSet->keyPoints->size() - bad_lines;
    size_t new_mt_size = matchSet->matches->size() - bad_bundles;
    delete matchSet->keyPoints;
    delete matchSet->matches;
    matchSet->keyPoints = new ssrlcv::Unity<ssrlcv::KeyPoint>(nullptr,new_kp_size,ssrlcv::cpu);
    matchSet->matches   = new ssrlcv::Unity<ssrlcv::MultiMatch>(nullptr,new_mt_size,ssrlcv::cpu);
    // this is much easier because of the 2 view assumption
    // there are the same number of lines as there are are keypoints and the same number of bundles as there are matches
    int k_adjust = 0;
    int k_lines  = 0;
    int k_bundle = 0;
    for (int k = 0; k < bundleSet.bundles->size(); k++){
      k_lines = bundleSet.bundles->host[k].numLines;
    	if (!bundleSet.bundles->host[k].invalid){
        matchSet->matches->host[k_bundle] = {k_lines,k_adjust};
        for (int j = 0; j < k_lines; j++){
          matchSet->keyPoints->host[k_adjust + j] = tempMatchSet.keyPoints->host[k_adjust + j];
        }
    	  k_adjust += k_lines;
        k_bundle++;
    	}
    }

    if (bad_bundles) std::cout << "\tRemoved bundles" << std::endl;

  }
}

// =============================================================================

          // =============================================================================================================
          // =============================================================================================================
          // ==================================================================================================== //
          //                                        device methods                                                //
          // ==================================================================================================== //
          // =============================================================================================================
          // =============================================================================================================

// =============================================================================================================
//
// Bundle Adjustment Kernels
//
// =============================================================================================================

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

// =============================================================================================================
//
// Stereo Kernels
//
// =============================================================================================================

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

// =============================================================================================================
//
// 2 View Kernels
//
// =============================================================================================================

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
  bundles[globalID].invalid = false;
  // add the linaer errors locally within the block before
  float error = sqrtf((s1.x - s2.x)*(s1.x - s2.x) + (s1.y - s2.y)*(s1.y - s2.y) + (s1.z - s2.z)*(s1.z - s2.z));
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
  bundles[globalID].invalid = false;
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
    bundles[globalID].invalid = false;
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
__global__ void ssrlcv::voidComputeTwoViewTriangulate(float* linearError, unsigned long pointnum, Bundle::Line* lines, Bundle* bundles){
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
  i_error = error;
  bundles[globalID].invalid = false;
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
    bundles[globalID].invalid = true;
  } else {
    i_error = error;
    bundles[globalID].invalid = false;
  }
  atomicAdd(&localSum,i_error);
  __syncthreads();
  if (!threadIdx.x) atomicAdd(linearError,localSum);
}

// =============================================================================================================
//
// N View Kernels
//
// =============================================================================================================

/**
 *
 */
__global__ void ssrlcv::computeNViewTriangulate(unsigned long pointnum, Bundle::Line* lines, Bundle* bundles, float3* pointcloud){

  unsigned long globalID = (blockIdx.y* gridDim.x+ blockIdx.x)*blockDim.x + threadIdx.x;
  if (globalID > (pointnum-1)) return;

  //Initializing Variables
  float3 S [3];
  float3 C;
  S[0] = {0,0,0};
  S[1] = {0,0,0};
  S[2] = {0,0,0};
  C = {0,0,0};

  //Iterating through the Lines in a Bundle
  for(int i = bundles[globalID].index; i < bundles[globalID].index + bundles[globalID].numLines; i++){
    ssrlcv::Bundle::Line L1 = lines[i];
    float3 tmp [3];
    normalize(L1.vec);
    matrixProduct(L1.vec, tmp);
    //Subtracting the 3x3 Identity Matrix from tmp
    tmp[0].x -= 1;
    tmp[1].y -= 1;
    tmp[2].z -= 1;
    //Adding tmp to S
    S[0] = S[0] + tmp[0];
    S[1] = S[1] + tmp[1];
    S[2] = S[2] + tmp[2];
    //Adding tmp * pnt to C
    float3 vectmp;
    multiply(tmp, L1.pnt, vectmp);
    C = C + vectmp;
  }

  /**
   * If all of the directional vectors are skew and not parallel, then I think S is nonsingular.
   * However, I will look into this some more. This may have to use a pseudo-inverse matrix if that
   * is not the case.
   */
  float3 Inverse [3];
  if(inverse(S, Inverse)){
    float3 point;
    multiply(Inverse, C, point);
    pointcloud[globalID] = point;
  }


}

/**
 *
 */
__global__ void ssrlcv::computeNViewTriangulate(float* angularError, unsigned long pointnum, Bundle::Line* lines, Bundle* bundles, float3* pointcloud){

  // get ready to do the stuff local memory space
  // this will later be added back to a global memory space
  __shared__ float localSum;
  if (threadIdx.x == 0) localSum = 0;
  __syncthreads();

  unsigned long globalID = (blockIdx.y* gridDim.x+ blockIdx.x)*blockDim.x + threadIdx.x;
  if (globalID > (pointnum-1)) return;

  //Initializing Variables
  float3 S [3];
  float3 C;
  S[0] = {0,0,0};
  S[1] = {0,0,0};
  S[2] = {0,0,0};
  C = {0,0,0};

  //Iterating through the Lines in a Bundle
  for(int i = bundles[globalID].index; i < bundles[globalID].index + bundles[globalID].numLines; i++){
    ssrlcv::Bundle::Line L1 = lines[i];
    float3 tmp [3];
    normalize(L1.vec);
    matrixProduct(L1.vec, tmp);
    //Subtracting the 3x3 Identity Matrix from tmp
    tmp[0].x -= 1;
    tmp[1].y -= 1;
    tmp[2].z -= 1;
    //Adding tmp to S
    S[0] = S[0] + tmp[0];
    S[1] = S[1] + tmp[1];
    S[2] = S[2] + tmp[2];
    //Adding tmp * pnt to C
    float3 vectmp;
    multiply(tmp, L1.pnt, vectmp);
    C = C + vectmp;
  }

  /**
   * If all of the directional vectors are skew and not parallel, then I think S is nonsingular.
   * However, I will look into this some more. This may have to use a pseudo-inverse matrix if that
   * is not the case.
   */
  float3 Inverse [3];
  float3 point;
  if(inverse(S, Inverse)){
    multiply(Inverse, C, point);
    pointcloud[globalID] = point;
  }

  // float a_error = 0;
  // for(int i = bundles[globalID].index; i < bundles[globalID].index + bundles[globalID].numLines; i++){
  //   float3 v = lines[i].vec;
  //   normalize(v);
  //   // the refrence vector
  //   // we take the generated point and create a vector from it to the camera center
  //   float3 r = lines[i].pnt - point;
  //   normalize(r);
  //
  //   float3 er = v - r;
  //   a_error += sqrtf(er.x*er.x + er.y*er.y + er.z*er.z);
  // }

  float a_error = 0;
  for(int i = bundles[globalID].index; i < bundles[globalID].index + bundles[globalID].numLines; i++){
    float3 a = lines[i].vec;
    float3 b = lines[i].pnt - point;
    float3 c = crossProduct(b,a);
    // normalize(b);
    // normalize(c);
    float numer = magnitude(c);
    float denom = magnitude(b);
    a_error += numer / denom;
  }

  a_error /= (float) bundles[globalID].numLines;

  // after calculating local error add it all up
  atomicAdd(&localSum,a_error);
  __syncthreads();
  if (!threadIdx.x) atomicAdd(angularError,localSum);
}

/**
 *
 */
__global__ void ssrlcv::computeNViewTriangulate(float* angularError, float* errors, unsigned long pointnum, Bundle::Line* lines, Bundle* bundles, float3* pointcloud){

  // get ready to do the stuff local memory space
  // this will later be added back to a global memory space
  __shared__ float localSum;
  if (threadIdx.x == 0) localSum = 0;
  __syncthreads();

  unsigned long globalID = (blockIdx.y* gridDim.x+ blockIdx.x)*blockDim.x + threadIdx.x;
  if (globalID > (pointnum-1)) return;

  //Initializing Variables
  float3 S [3];
  float3 C;
  S[0] = {0,0,0};
  S[1] = {0,0,0};
  S[2] = {0,0,0};
  C = {0,0,0};

  //Iterating through the Lines in a Bundle
  for(int i = bundles[globalID].index; i < bundles[globalID].index + bundles[globalID].numLines; i++){
    ssrlcv::Bundle::Line L1 = lines[i];
    float3 tmp [3];
    normalize(L1.vec);
    matrixProduct(L1.vec, tmp);
    //Subtracting the 3x3 Identity Matrix from tmp
    tmp[0].x -= 1;
    tmp[1].y -= 1;
    tmp[2].z -= 1;
    //Adding tmp to S
    S[0] = S[0] + tmp[0];
    S[1] = S[1] + tmp[1];
    S[2] = S[2] + tmp[2];
    //Adding tmp * pnt to C
    float3 vectmp;
    multiply(tmp, L1.pnt, vectmp);
    C = C + vectmp;
  }

  /**
   * If all of the directional vectors are skew and not parallel, then I think S is nonsingular.
   * However, I will look into this some more. This may have to use a pseudo-inverse matrix if that
   * is not the case.
   */
  float3 Inverse [3];
  float3 point;
  if(inverse(S, Inverse)){
    multiply(Inverse, C, point);
    pointcloud[globalID] = point;
  }


  // float a_error = 0;
  // for(int i = bundles[globalID].index; i < bundles[globalID].index + bundles[globalID].numLines; i++){
  //   float3 v = lines[i].vec;
  //   normalize(v);
  //   // the refrence vector
  //   // we take the generated point and create a vector from it to the camera center
  //   float3 r = lines[i].pnt - point;
  //   normalize(r);
  //
  //   float3 er = v - r;
  //   a_error += sqrtf(er.x*er.x + er.y*er.y + er.z*er.z);
  // }

  float a_error = 0;
  for(int i = bundles[globalID].index; i < bundles[globalID].index + bundles[globalID].numLines; i++){
    float3 a = lines[i].vec;
    float3 b = lines[i].pnt - point;
    float3 c = crossProduct(b,a);
    // normalize(b);
    // normalize(c);
    float numer = magnitude(c);
    float denom = magnitude(b);
    a_error += numer / denom;
  }

  a_error /= (float) bundles[globalID].numLines;

  errors[globalID] = a_error;

  // after calculating local error add it all up
  atomicAdd(&localSum,a_error);
  __syncthreads();
  if (!threadIdx.x) atomicAdd(angularError,localSum);
}

/**
 *
 */
__global__ void ssrlcv::computeNViewTriangulate(float* angularError, float* angularErrorCutoff, float* errors, unsigned long pointnum, Bundle::Line* lines, Bundle* bundles, float3* pointcloud){

  // get ready to do the stuff local memory space
  // this will later be added back to a global memory space
  __shared__ float localSum;
  if (threadIdx.x == 0) localSum = 0;
  __syncthreads();

  unsigned long globalID = (blockIdx.y* gridDim.x+ blockIdx.x)*blockDim.x + threadIdx.x;
  if (globalID > (pointnum-1)) return;

  //Initializing Variables
  float3 S [3];
  float3 C;
  S[0] = {0,0,0};
  S[1] = {0,0,0};
  S[2] = {0,0,0};
  C = {0,0,0};

  //Iterating through the Lines in a Bundle
  for(int i = bundles[globalID].index; i < bundles[globalID].index + bundles[globalID].numLines; i++){
    ssrlcv::Bundle::Line L1 = lines[i];
    float3 tmp [3];
    normalize(L1.vec);
    matrixProduct(L1.vec, tmp);
    //Subtracting the 3x3 Identity Matrix from tmp
    tmp[0].x -= 1;
    tmp[1].y -= 1;
    tmp[2].z -= 1;
    //Adding tmp to S
    S[0] = S[0] + tmp[0];
    S[1] = S[1] + tmp[1];
    S[2] = S[2] + tmp[2];
    //Adding tmp * pnt to C
    float3 vectmp;
    multiply(tmp, L1.pnt, vectmp);
    C = C + vectmp;
  }

  /**
   * If all of the directional vectors are skew and not parallel, then I think S is nonsingular.
   * However, I will look into this some more. This may have to use a pseudo-inverse matrix if that
   * is not the case.
   */
  float3 Inverse [3];
  float3 point;
  if(inverse(S, Inverse)){
    multiply(Inverse, C, point);
    pointcloud[globalID] = point;
  }

  // float a_error = 0;
  // for(int i = bundles[globalID].index; i < bundles[globalID].index + bundles[globalID].numLines; i++){
  //   float3 v = lines[i].vec;
  //   normalize(v);
  //   // the refrence vector
  //   // we take the generated point and create a vector from it to the camera center
  //   float3 r = lines[i].pnt - point;
  //   normalize(r);
  //
  //   float3 er = v - r;
  //   a_error += sqrtf(er.x*er.x + er.y*er.y + er.z*er.z);
  // }

  float a_error = 0;
  for(int i = bundles[globalID].index; i < bundles[globalID].index + bundles[globalID].numLines; i++){
    float3 a = lines[i].vec;
    float3 b = lines[i].pnt - point;
    float3 c = crossProduct(b,a);
    // normalize(b);
    // normalize(c);
    float numer = magnitude(c);
    float denom = magnitude(b);
    a_error += numer / denom;
  }


  a_error /= (float) bundles[globalID].numLines;

  errors[globalID] = a_error;

  // filtering
  if (a_error > *angularErrorCutoff) {
    bundles[globalID].invalid = true;
  } else {
    bundles[globalID].invalid = false;
  }

  // after calculating local error add it all up
  atomicAdd(&localSum,a_error);
  __syncthreads();
  if (!threadIdx.x) atomicAdd(angularError,localSum);
}



























//
