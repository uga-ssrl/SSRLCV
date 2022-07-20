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
ssrlcv::ptr::value<ssrlcv::Unity<float3>> ssrlcv::PointCloudFactory::stereo_disparity(ssrlcv::ptr::value<ssrlcv::Unity<Match>> matches, Image::Camera* cameras){

  float baseline = sqrtf( (cameras[0].cam_pos.x - cameras[1].cam_pos.x)*(cameras[0].cam_pos.x - cameras[1].cam_pos.x)
                        + (cameras[0].cam_pos.y - cameras[1].cam_pos.y)*(cameras[0].cam_pos.y - cameras[1].cam_pos.y)
                        + (cameras[0].cam_pos.z - cameras[1].cam_pos.z)*(cameras[0].cam_pos.z - cameras[1].cam_pos.z));
  float scale = (baseline * cameras[0].foc )/(cameras[0].dpix.x);

  logger.info << "Stereo Baseline: " + std::to_string(baseline) + ", Stereo Scale Factor: " + std::to_string(scale) +  ", Inverted Stereo Scale Factor: " + std::to_string(1.0/scale);

  MemoryState origin = matches->getMemoryState();
  if(origin == cpu) matches->transferMemoryTo(gpu);

  // depth points
  ssrlcv::ptr::device<float3> points_device(matches->size());

  //
  dim3 grid = {1,1,1};
  dim3 block = {1,1,1};
  void (*fp)(unsigned int, Match*, float3*, float) = &computeStereo;
  getFlatGridBlock(matches->size(),grid,block,fp);
  //
  computeStereo<<<grid, block>>>(matches->size(), matches->device.get(), points_device.get(), 8.0);
  // focal lenth / baseline

  // computeStereo<<<grid, block>>>(matches->size(), matches->device.get(), points_device, 64.0);

  ssrlcv::ptr::value<ssrlcv::Unity<float3>> points = ssrlcv::ptr::value<ssrlcv::Unity<float3>>(points_device, matches->size(),gpu);
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
ssrlcv::ptr::value<ssrlcv::Unity<float3>> ssrlcv::PointCloudFactory::stereo_disparity(ssrlcv::ptr::value<ssrlcv::Unity<Match>> matches, float scale){

  MemoryState origin = matches->getMemoryState();
  if(origin == cpu) matches->transferMemoryTo(gpu);

  // depth points
  ssrlcv::ptr::device<float3> points_device(matches->size());

  //
  dim3 grid = {1,1,1};
  dim3 block = {1,1,1};
  void (*fp)(unsigned int, Match*, float3*, float) = &computeStereo;
  getFlatGridBlock(matches->size(),grid,block,fp);
  //
  computeStereo<<<grid, block>>>(matches->size(), matches->device.get(), points_device.get(), scale);

  ssrlcv::ptr::value<ssrlcv::Unity<float3>> points = ssrlcv::ptr::value<ssrlcv::Unity<float3>>(points_device, matches->size(),gpu);
  if(origin == cpu) matches->setMemoryState(cpu);

  return points;
}

/**
 * TODO
 */
ssrlcv::ptr::value<ssrlcv::Unity<float3>> ssrlcv::PointCloudFactory::stereo_disparity(ssrlcv::ptr::value<ssrlcv::Unity<Match>> matches, float foc, float baseline, float doffset){

  MemoryState origin = matches->getMemoryState();
  if(origin == cpu) matches->transferMemoryTo(gpu);


  ssrlcv::ptr::value<ssrlcv::Unity<float3>> points = ssrlcv::ptr::value<ssrlcv::Unity<float3>>(nullptr, matches->size(),gpu);
  //
  dim3 grid = {1,1,1};
  dim3 block = {1,1,1};
  void (*fp)(unsigned int, Match*, float3*, float, float, float) = &computeStereo;
  getFlatGridBlock(matches->size(),grid,block,fp);
  //
  computeStereo<<<grid, block>>>(matches->size(), matches->device.get(), points->device.get(), foc, baseline, doffset);

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
  // float temp = colors->host.get()[i];
  // colors->host.get()[i] *= 5.0f;
  // colors->host.get()[i] = floor(colors->host.get()[i]);
  // if(colors->host.get()[i] == 5.0f) colors->host.get()[i] = 4.0f;
  // if(colors->host.get()[i] == 0.0f) colors->host.get()[i] = 1.0f;
  // rgb.x = (1-temp)*colorMap[(int)colors->host.get()[i]-1].x + (temp*colorMap[(int)colors->host.get()[i]].x);
  // rgb.y = (1-temp)*colorMap[(int)colors->host.get()[i]-1].y + (temp*colorMap[(int)colors->host.get()[i]].y);
  // rgb.z = (1-temp)*colorMap[(int)colors->host.get()[i]-1].z + (temp*colorMap[(int)colors->host.get()[i]].z);


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
void ssrlcv::writeDisparityImage(ssrlcv::ptr::value<ssrlcv::Unity<float3>> points, unsigned int interpolationRadius, std::string pathToFile){
  MemoryState origin = points->getMemoryState();
  if(origin == gpu) points->transferMemoryTo(cpu);
  float3 min = {FLT_MAX,FLT_MAX,FLT_MAX};
  float3 max = {-FLT_MAX,-FLT_MAX,-FLT_MAX};
  for(int i = 0; i < points->size(); ++i){
    if(points->host.get()[i].x < min.x) min.x = points->host.get()[i].x;
    if(points->host.get()[i].x > max.x) max.x = points->host.get()[i].x;
    if(points->host.get()[i].y < min.y) min.y = points->host.get()[i].y;
    if(points->host.get()[i].y > max.y) max.y = points->host.get()[i].y;
    if(points->host.get()[i].z < min.z) min.z = points->host.get()[i].z;
    if(points->host.get()[i].z > max.z) max.z = points->host.get()[i].z;
  }
  uint2 imageDim = {(unsigned int)ceil(max.x-min.x)+1,(unsigned int)ceil(max.y-min.y)+1};
  unsigned char* disparityImage = new unsigned char[imageDim.x*imageDim.y*3];
  ssrlcv::ptr::value<ssrlcv::Unity<float>> colors = ssrlcv::ptr::value<ssrlcv::Unity<float>>(nullptr,imageDim.x*imageDim.y,cpu);
  for(int i = 0; i < imageDim.x*imageDim.y*3; ++i){
    disparityImage[i] = 0;
  }
  for(int i = 0; i < points->size(); ++i){
    float3 temp = points->host.get()[i] - min;
    if(ceil(temp.x) != temp.x || ceil(temp.y) != temp.y){
      colors->host.get()[((int)ceil(temp.y)*imageDim.x) + (int)ceil(temp.x)] += (1-ceil(temp.x)-temp.x)*(1-ceil(temp.y)-temp.y)*temp.z/(max.z-min.z);
      colors->host.get()[((int)ceil(temp.y)*imageDim.x) + (int)floor(temp.x)] += (1-temp.x-floor(temp.x))*(1-ceil(temp.y)-temp.y)*temp.z/(max.z-min.z);
      colors->host.get()[((int)floor(temp.y)*imageDim.x) + (int)ceil(temp.x)] += (1-ceil(temp.x)-temp.x)*(1-temp.y-floor(temp.y))*temp.z/(max.z-min.z);
      colors->host.get()[((int)floor(temp.y)*imageDim.x) + (int)floor(temp.x)] += (1-temp.x-floor(temp.x))*(1-temp.y-floor(temp.y))*temp.z/(max.z-min.z);
    }
    else{
      colors->host.get()[(int)temp.y*imageDim.x + (int)temp.x] += temp.z/(max.z-min.z);
    }
  }

  /*
  INTERPOLATE
  */
  if(interpolationRadius != 0){
    colors->setMemoryState(gpu);
    ssrlcv::ptr::device<float> interpolated(imageDim.x*imageDim.y);
    dim3 block = {1,1,1};
    dim3 grid = {1,1,1};
    getFlatGridBlock(imageDim.x*imageDim.y,grid,block,interpolateDepth);
    interpolateDepth<<<grid,block>>>(imageDim,interpolationRadius,colors->device.get(),interpolated.get());
    cudaDeviceSynchronize();
    CudaCheckError();
    colors->setData(interpolated,colors->size(),gpu);
    colors->setMemoryState(cpu);
  }

  min.z = FLT_MAX;
  max.z = -FLT_MAX;
  for(int i = 0; i < imageDim.x*imageDim.y; ++i){
    if(min.z > colors->host.get()[i]) min.z = colors->host.get()[i];
    if(max.z < colors->host.get()[i]) max.z = colors->host.get()[i];
  }


  uchar3 rgb;
  for(int i = 0; i < imageDim.x*imageDim.y; ++i){
    colors->host.get()[i] -= min.z;
    colors->host.get()[i] /= (max.z-min.z);
    rgb = heatMap(colors->host.get()[i]);
    disparityImage[i*3] = rgb.x;
    disparityImage[i*3 + 1] = rgb.y;
    disparityImage[i*3 + 2] = rgb.z;
  }
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
*/
ssrlcv::ptr::value<ssrlcv::Unity<float3>> ssrlcv::PointCloudFactory::twoViewTriangulate(BundleSet bundleSet){

  bundleSet.lines->transferMemoryTo(gpu);
  bundleSet.bundles->transferMemoryTo(gpu);

  ssrlcv::ptr::value<Unity<float3>> pointcloud = ssrlcv::ptr::value<Unity<float3>>(nullptr,bundleSet.bundles->size(),gpu);

  dim3 grid = {1,1,1};
  dim3 block = {1,1,1};
  void (*fp)(unsigned long, Bundle::Line*, Bundle*, float3*) = &computeTwoViewTriangulate;
  getFlatGridBlock(bundleSet.bundles->size(),grid,block,fp);

  computeTwoViewTriangulate<<<grid,block>>>(bundleSet.bundles->size(),bundleSet.lines->device.get(),bundleSet.bundles->device.get(),pointcloud->device.get());

  cudaDeviceSynchronize();
  CudaCheckError();

  //transfer the poitns back to the CPU
  pointcloud->transferMemoryTo(cpu);
  pointcloud->clear(gpu);
  // clear the other boiz
  bundleSet.lines->clear(gpu);
  bundleSet.bundles->clear(gpu);

  return pointcloud;
}

/**
* The CPU method that sets up the GPU enabled two view tringulation.
* @param bundleSet a set of lines and bundles that should be triangulated
* @param linearError is the total linear error of the triangulation, it is an analog for reprojection error
*/
ssrlcv::ptr::value<ssrlcv::Unity<float3>> ssrlcv::PointCloudFactory::twoViewTriangulate(BundleSet bundleSet, float* linearError){

  // to total error cacluation is stored in this guy
  *linearError = 0;
  float* d_linearError;
  size_t eSize = sizeof(float);
  CudaSafeCall(cudaMalloc((void**) &d_linearError,eSize));
  CudaSafeCall(cudaMemcpy(d_linearError,linearError,eSize,cudaMemcpyHostToDevice));

  bundleSet.lines->transferMemoryTo(gpu);
  bundleSet.bundles->transferMemoryTo(gpu);

  // ssrlcv::ptr::value<Unity<float3>> pointcloud = ssrlcv::ptr::value<Unity<float3>>(new Unity<float3>(nullptr,2*bundleSet.bundles->size(),gpu));
  ssrlcv::ptr::value<Unity<float3>> pointcloud = ssrlcv::ptr::value<Unity<float3>>(nullptr,bundleSet.bundles->size(),gpu);

  dim3 grid = {1,1,1};
  dim3 block = {1,1,1};
  void (*fp)(float*, unsigned long, Bundle::Line*, Bundle*, float3*) = &computeTwoViewTriangulate;
  getFlatGridBlock(bundleSet.bundles->size(),grid,block,fp);

  computeTwoViewTriangulate<<<grid,block>>>(d_linearError,bundleSet.bundles->size(),bundleSet.lines->device.get(),bundleSet.bundles->device.get(),pointcloud->device.get());

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
ssrlcv::ptr::value<ssrlcv::Unity<float3>> ssrlcv::PointCloudFactory::twoViewTriangulate(BundleSet bundleSet, ssrlcv::ptr::value<ssrlcv::Unity<float>> errors, float* linearError){

  // to total error cacluation is stored in this guy
  *linearError = 0;
  float* d_linearError;
  size_t eSize = sizeof(float);
  CudaSafeCall(cudaMalloc((void**) &d_linearError,eSize));
  CudaSafeCall(cudaMemcpy(d_linearError,linearError,eSize,cudaMemcpyHostToDevice));

  bundleSet.lines->transferMemoryTo(gpu);
  bundleSet.bundles->transferMemoryTo(gpu);
  errors->transferMemoryTo(gpu);

  // ssrlcv::ptr::value<Unity<float3>> pointcloud = ssrlcv::ptr::value<Unity<float3>>(new Unity<float3>(nullptr,2*bundleSet.bundles->size(),gpu));
  ssrlcv::ptr::value<Unity<float3>> pointcloud = ssrlcv::ptr::value<Unity<float3>>(nullptr,bundleSet.bundles->size(),gpu);

  dim3 grid = {1,1,1};
  dim3 block = {1,1,1};
  void (*fp)(float*, float*, unsigned long, Bundle::Line*, Bundle*, float3*) = &computeTwoViewTriangulate;
  getFlatGridBlock(bundleSet.bundles->size(),grid,block,fp);

  computeTwoViewTriangulate<<<grid,block>>>(d_linearError,errors->device.get(),bundleSet.bundles->size(),bundleSet.lines->device.get(),bundleSet.bundles->device.get(),pointcloud->device.get());

  cudaDeviceSynchronize();
  CudaCheckError();

  // transfer the poitns back to the CPU
  pointcloud->transferMemoryTo(cpu);
  pointcloud->clear(gpu);
  // transfer the individual linear errors back to the CPU
  errors->setFore(gpu);
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
ssrlcv::ptr::value<ssrlcv::Unity<float3>> ssrlcv::PointCloudFactory::twoViewTriangulate(BundleSet bundleSet, ssrlcv::ptr::value<ssrlcv::Unity<float>> errors, float* linearError, float* linearErrorCutoff){

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

  // ssrlcv::ptr::value<Unity<float3>> pointcloud = ssrlcv::ptr::value<Unity<float3>>(new Unity<float3>(nullptr,2*bundleSet.bundles->size(),gpu));
  ssrlcv::ptr::value<Unity<float3>> pointcloud = ssrlcv::ptr::value<Unity<float3>>(nullptr,bundleSet.bundles->size(),gpu);

  dim3 grid = {1,1,1};
  dim3 block = {1,1,1};
  void (*fp)(float*, float*, float*, unsigned long, Bundle::Line*, Bundle*, float3*) = &computeTwoViewTriangulate;
  getFlatGridBlock(bundleSet.bundles->size(),grid,block,fp);

  computeTwoViewTriangulate<<<grid,block>>>(d_linearError,d_linearErrorCutoff,errors->device.get(),bundleSet.bundles->size(),bundleSet.lines->device.get(),bundleSet.bundles->device.get(),pointcloud->device.get());

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
  errors->setFore(gpu);
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
ssrlcv::ptr::value<ssrlcv::Unity<float3_b>> ssrlcv::PointCloudFactory::twoViewTriangulate_b(BundleSet bundleSet, ssrlcv::ptr::value<ssrlcv::Unity<float>> errors, float* linearError, float* linearErrorCutoff){

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

    ssrlcv::ptr::value<ssrlcv::Unity<float3_b>> pointcloud_b = ssrlcv::ptr::value<ssrlcv::Unity<float3_b>>(nullptr,bundleSet.bundles->size(),gpu);

    dim3 grid = {1,1,1};
    dim3 block = {1,1,1};
    getFlatGridBlock(bundleSet.bundles->size(),grid,block,computeTwoViewTriangulate_b);

    computeTwoViewTriangulate_b<<<grid,block>>>(d_linearError,d_linearErrorCutoff,errors->device.get(),bundleSet.bundles->size(),bundleSet.lines->device.get(),bundleSet.bundles->device.get(),pointcloud_b->device.get());

    pointcloud_b->setFore(gpu);
    errors->setFore(gpu);

    cudaDeviceSynchronize();
    CudaCheckError();

    // transfer the poitns back to the CPU
    pointcloud_b->transferMemoryTo(cpu);
    pointcloud_b->clear(gpu);
    // transfer the individual linear errors back to the CPU
    errors->setFore(gpu);
    errors->transferMemoryTo(cpu);
    errors->clear(gpu);
    // temp

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

  voidComputeTwoViewTriangulate<<<grid,block>>>(d_linearError,bundleSet.bundles->size(),bundleSet.lines->device.get(),bundleSet.bundles->device.get());

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

  voidComputeTwoViewTriangulate<<<grid,block>>>(d_linearError,d_linearErrorCutoff,bundleSet.bundles->size(),bundleSet.lines->device.get(),bundleSet.bundles->device.get());

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
ssrlcv::ptr::value<ssrlcv::Unity<float3>> ssrlcv::PointCloudFactory::nViewTriangulate(BundleSet bundleSet){
  bundleSet.lines->transferMemoryTo(gpu);
  bundleSet.bundles->transferMemoryTo(gpu);

  ssrlcv::ptr::value<Unity<float3>> pointcloud = ssrlcv::ptr::value<Unity<float3>>(nullptr,bundleSet.bundles->size(),gpu);

  dim3 grid = {1,1,1};
  dim3 block = {1,1,1};
  void (*fp)(unsigned long, Bundle::Line*, Bundle*, float3*) = &computeNViewTriangulate;
  getFlatGridBlock(bundleSet.bundles->size(),grid,block,fp);


  logger.info << "Starting n-view triangulation ...";
  computeNViewTriangulate<<<grid,block>>>(bundleSet.bundles->size(),bundleSet.lines->device.get(),bundleSet.bundles->device.get(),pointcloud->device.get());
  logger.info << "n-view Triangulation done ...";

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
ssrlcv::ptr::value<ssrlcv::Unity<float3>> ssrlcv::PointCloudFactory::nViewTriangulate(BundleSet bundleSet, float* angularError){

  // make the error guys
  *angularError = 0;
  float* d_angularError;
  size_t eSize = sizeof(float);
  CudaSafeCall(cudaMalloc((void**) &d_angularError,eSize));
  CudaSafeCall(cudaMemcpy(d_angularError,angularError,eSize,cudaMemcpyHostToDevice));

  bundleSet.lines->transferMemoryTo(gpu);
  bundleSet.bundles->transferMemoryTo(gpu);

  ssrlcv::ptr::value<Unity<float3>> pointcloud = ssrlcv::ptr::value<Unity<float3>>(nullptr,bundleSet.bundles->size(),gpu);

  dim3 grid = {1,1,1};
  dim3 block = {1,1,1};
  void (*fp)(float*, unsigned long, Bundle::Line*, Bundle*, float3*) = &computeNViewTriangulate;
  getFlatGridBlock(bundleSet.bundles->size(),grid,block,fp);


  logger.info << "Starting n-view triangulation ...";
  computeNViewTriangulate<<<grid,block>>>(d_angularError,bundleSet.bundles->size(),bundleSet.lines->device.get(),bundleSet.bundles->device.get(),pointcloud->device.get());
  logger.info << "n-view Triangulation done ...";

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
ssrlcv::ptr::value<ssrlcv::Unity<float3>> ssrlcv::PointCloudFactory::nViewTriangulate(BundleSet bundleSet, ssrlcv::ptr::value<ssrlcv::Unity<float>> errors, float* angularError){
  // make the error guys
  *angularError = 0;
  float* d_angularError;
  size_t eSize = sizeof(float);
  CudaSafeCall(cudaMalloc((void**) &d_angularError,eSize));
  CudaSafeCall(cudaMemcpy(d_angularError,angularError,eSize,cudaMemcpyHostToDevice));

  bundleSet.lines->transferMemoryTo(gpu);
  bundleSet.bundles->transferMemoryTo(gpu);
  errors->transferMemoryTo(gpu);

  ssrlcv::ptr::value<Unity<float3>> pointcloud = ssrlcv::ptr::value<Unity<float3>>(nullptr,bundleSet.bundles->size(),gpu);

  dim3 grid = {1,1,1};
  dim3 block = {1,1,1};
  void (*fp)(float*, float*, unsigned long, Bundle::Line*, Bundle*, float3*) = &computeNViewTriangulate;
  getFlatGridBlock(bundleSet.bundles->size(),grid,block,fp);


  logger.info << "Starting n-view triangulation ...";
  computeNViewTriangulate<<<grid,block>>>(d_angularError,errors->device.get(),bundleSet.bundles->size(),bundleSet.lines->device.get(),bundleSet.bundles->device.get(),pointcloud->device.get());
  logger.info << "n-view Triangulation done ...";

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
ssrlcv::ptr::value<ssrlcv::Unity<float3>> ssrlcv::PointCloudFactory::nViewTriangulate(BundleSet bundleSet, ssrlcv::ptr::value<ssrlcv::Unity<float>> errors, float* angularError, float* angularErrorCutoff){
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

  ssrlcv::ptr::value<Unity<float3>> pointcloud = ssrlcv::ptr::value<Unity<float3>>(nullptr,bundleSet.bundles->size(),gpu);

  dim3 grid = {1,1,1};
  dim3 block = {1,1,1};
  // float* angularError, float* angularErrorCutoff, float* errors, unsigned long pointnum, Bundle::Line* lines, Bundle* bundles, float3* pointcloud
  void (*fp)(float*, float*, float*, unsigned long, Bundle::Line*, Bundle*, float3*) = &computeNViewTriangulate;
  getFlatGridBlock(bundleSet.bundles->size(),grid,block,fp);


  logger.info << "Starting n-view triangulation ...";
  computeNViewTriangulate<<<grid,block>>>(d_angularError,d_angularErrorCutoff,errors->device.get(),bundleSet.bundles->size(),bundleSet.lines->device.get(),bundleSet.bundles->device.get(),pointcloud->device.get());
  logger.info << "n-view Triangulation done ...";

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

/**
 * The CPU method that sets up the GPU enabled n view triangulation.
 * @param bundleSet a set of lines and bundles to be triangulated
 * @param errors the individual angular errors per point
 * @param angularError the total diff between vectors
 * @param lowCut generated points that have an error under this cutoff are marked invalid
 * @param highCut generated points that have an error over this cutoff are marked invalid
 */
ssrlcv::ptr::value<ssrlcv::Unity<float3>> ssrlcv::PointCloudFactory::nViewTriangulate(BundleSet bundleSet, ssrlcv::ptr::value<ssrlcv::Unity<float>> errors, float* angularError, float* lowCut, float* highCut){
  // make the error guys
  *angularError = 0;
  float* d_angularError;
  size_t eSize = sizeof(float);
  CudaSafeCall(cudaMalloc((void**) &d_angularError,eSize));
  CudaSafeCall(cudaMemcpy(d_angularError,angularError,eSize,cudaMemcpyHostToDevice));

  float* d_lowCut;
  size_t cutSize = sizeof(float);
  CudaSafeCall(cudaMalloc((void**) &d_lowCut,cutSize));
  CudaSafeCall(cudaMemcpy(d_lowCut, lowCut,cutSize,cudaMemcpyHostToDevice));

  float* d_highCut;
  CudaSafeCall(cudaMalloc((void**) &d_highCut,cutSize));
  CudaSafeCall(cudaMemcpy(d_highCut, highCut,cutSize,cudaMemcpyHostToDevice));

  bundleSet.lines->transferMemoryTo(gpu);
  bundleSet.bundles->transferMemoryTo(gpu);
  errors->transferMemoryTo(gpu);

  ssrlcv::ptr::value<Unity<float3>> pointcloud = ssrlcv::ptr::value<Unity<float3>>(nullptr,bundleSet.bundles->size(),gpu);

  dim3 grid = {1,1,1};
  dim3 block = {1,1,1};
  // float* angularError, float* lowCut, float* highCut, float* errors, unsigned long pointnum, Bundle::Line* lines, Bundle* bundles, float3* pointcloud)
  void (*fp)(float*, float*, float*, float*, unsigned long, Bundle::Line*, Bundle*, float3*) = &computeNViewTriangulate;
  getFlatGridBlock(bundleSet.bundles->size(),grid,block,fp);


  logger.info << "Starting n-view triangulation ...";
  computeNViewTriangulate<<<grid,block>>>(d_angularError,d_lowCut,d_highCut,errors->device.get(),bundleSet.bundles->size(),bundleSet.lines->device.get(),bundleSet.bundles->device.get(),pointcloud->device.get());
  logger.info << "n-view Triangulation done ...";

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
  CudaSafeCall(cudaMemcpy(lowCut,d_lowCut,eSize,cudaMemcpyDeviceToHost));
  cudaFree(d_lowCut);
  CudaSafeCall(cudaMemcpy(highCut,d_highCut,eSize,cudaMemcpyDeviceToHost));
  cudaFree(d_highCut);

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
* @param images a group of images, used only for their stored camera parameters
*/
ssrlcv::BundleSet ssrlcv::PointCloudFactory::generateBundles(MatchSet* matchSet, std::vector<ssrlcv::ptr::value<ssrlcv::Image>> images){

  bool local_debug   = false;
  bool local_verbose = true;

  ssrlcv::ptr::value<ssrlcv::Unity<Bundle>> bundles = ssrlcv::ptr::value<ssrlcv::Unity<Bundle>>(nullptr,matchSet->matches->size(),gpu);
  ssrlcv::ptr::value<ssrlcv::Unity<Bundle::Line>> lines = ssrlcv::ptr::value<ssrlcv::Unity<Bundle::Line>>(nullptr,matchSet->keyPoints->size(),gpu);

  matchSet->matches->transferMemoryTo(gpu);
  matchSet->keyPoints->transferMemoryTo(gpu);

  // currently completely separates pushbroom and standard projection

  if (!images.at(0)->isPushbroom) {

    //
    // standard projection case
    //

    if (local_debug || local_verbose) logger.info << "\t Generating standard projective bundles ... ";

    // the cameras
    size_t cam_bytes = images.size()*sizeof(ssrlcv::Image::Camera);
    // fill the cam boi
    ssrlcv::Image::Camera* h_cameras;
    h_cameras = (ssrlcv::Image::Camera*) malloc(cam_bytes);
    for(int i = 0; i < images.size(); i++){
      h_cameras[i] = images.at(i)->camera;
    }
    ssrlcv::Image::Camera* d_cameras;
    CudaSafeCall(cudaMalloc(&d_cameras, cam_bytes));
    // copy the othe guy
    CudaSafeCall(cudaMemcpy(d_cameras, h_cameras, cam_bytes, cudaMemcpyHostToDevice));

    free(h_cameras);

    dim3 grid = {1,1,1};
    dim3 block = {1,1,1};
    getFlatGridBlock(bundles->size(),grid,block,generateBundle);

    generateBundle<<<grid, block>>>(bundles->size(),bundles->device.get(), lines->device.get(), matchSet->matches->device.get(), matchSet->keyPoints->device.get(), d_cameras);

    CudaSafeCall(cudaFree(d_cameras));
  } else {

    //
    // pushbroom projection case
    //

    if (local_debug || local_verbose) logger.info << "\t Generating special pushbroom bundles ... ";

    // the cameras
    size_t push_bytes = images.size()*sizeof(ssrlcv::Image::PushbroomCamera);
    // fill the cam boi
    ssrlcv::Image::PushbroomCamera* h_pushbrooms;
    h_pushbrooms = (ssrlcv::Image::PushbroomCamera*) malloc(push_bytes);
    for(int i = 0; i < images.size(); i++){
      h_pushbrooms[i] = images.at(i)->pushbroom;
    }
    ssrlcv::Image::PushbroomCamera* d_pushbrooms;
    CudaSafeCall(cudaMalloc(&d_pushbrooms, push_bytes));
    // copy the othe guy
    CudaSafeCall(cudaMemcpy(d_pushbrooms, h_pushbrooms, push_bytes, cudaMemcpyHostToDevice));

    dim3 grid = {1,1,1};
    dim3 block = {1,1,1};
    getFlatGridBlock(bundles->size(),grid,block,generatePushbroomBundle);

    generatePushbroomBundle<<<grid, block>>>(bundles->size(),bundles->device.get(), lines->device.get(), matchSet->matches->device.get(), matchSet->keyPoints->device.get(), d_pushbrooms);

    CudaSafeCall(cudaFree(d_pushbrooms));
  }

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
* The CPU method that sets up the GPU enabled line generation, which stores lines
* and sets of lines as bundles
* @param matchSet a group of maches
* @param images a group of images, used only for their stored camera parameters
* @param params a unity of float's which store selected camera parameters for N many, this does not have to be completely full but each camera must have the same number of parameters. The expected order is X pos, Y pos, Z pos, X rot, Y rot, Z rot, fov X, fov Y, foc, dpix x, dpix y
*/
ssrlcv::BundleSet ssrlcv::PointCloudFactory::generateBundles(MatchSet* matchSet, std::vector<ssrlcv::ptr::value<ssrlcv::Image>> images, ssrlcv::ptr::value<ssrlcv::Unity<float>> params){

  bool local_debug = false;

  // update the cameras in the images with the params
  int params_per_cam = params->size() / images.size();
  ssrlcv::ptr::value<ssrlcv::Unity<float>> temp_params = ssrlcv::ptr::value<ssrlcv::Unity<float>>(nullptr,params_per_cam,cpu);
  std::vector<ssrlcv::ptr::value<ssrlcv::Image>> temp;
  for (int i = 0; i < images.size(); i++){
    // fill up the vector
    for (int j = 0; j < params_per_cam; j++){
      temp_params->host.get()[j] = params->host.get()[i*params_per_cam + j];
    }
    // store old and set new
    temp.push_back(images[i]);
    images[i]->setFloatVector(temp_params);
  }

  // should be something close to:
  // 0.000000000000 0.000000000000 -400.000000000000 0.000000000000 0.000000000000 0.000000000000 5.000000000000 72.459274291992 -391.923095703125 0.174532920122 0.000000000000 0.000000000000
  // for the cube 2 view test
  //debug
  if (local_debug){
    std::cout << std::endl;
    std::cout << "\t params per cam: " << params_per_cam << std::endl;
    ssrlcv::ptr::value<ssrlcv::Unity<float>> temp_params2;
    std::cout << "\t input params :" << std::endl;
    for (int i = 0; i < params->size(); i++){
      std::cout << params->host.get()[i] << " ";
    }
    std::cout << std::endl;
    std::cout << "\t images params: " << std::endl << "\t\t ";
    for (int i = 0; i < images.size(); i++){
      temp_params2 = images[i]->getFloatVector(params_per_cam);
      for (int j = 0; j < temp_params2->size(); j++) {
        std::cout << temp_params2->host.get()[j] << " ";
      }
      std::cout << std::endl << "\t\t ";
    }
    std::cout << "\t backup params: " << std::endl << "\t\t ";
    for (int i = 0; i < temp.size(); i++){
      temp_params2 = temp[i]->getFloatVector(params_per_cam);
      for (int j = 0; j < temp_params2->size(); j++) {
        std::cout << temp_params2->host.get()[j] << " ";
      }
      std::cout << std::endl << "\t\t ";
    }
  }

  ssrlcv::ptr::value<ssrlcv::Unity<Bundle>> bundles = ssrlcv::ptr::value<ssrlcv::Unity<Bundle>>(nullptr,matchSet->matches->size(),gpu);
  ssrlcv::ptr::value<ssrlcv::Unity<Bundle::Line>> lines = ssrlcv::ptr::value<ssrlcv::Unity<Bundle::Line>>(nullptr,matchSet->keyPoints->size(),gpu);

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
  CudaSafeCall(cudaMalloc(&d_cameras, cam_bytes));
  // copy the othe guy
  CudaSafeCall(cudaMemcpy(d_cameras, h_cameras, cam_bytes, cudaMemcpyHostToDevice));

  free(h_cameras);

  dim3 grid = {1,1,1};
  dim3 block = {1,1,1};
  getFlatGridBlock(bundles->size(),grid,block,generateBundle);

  //in this kernel fill lines and bundles from keyPoints and matches
  generateBundle<<<grid, block>>>(bundles->size(),bundles->device.get(), lines->device.get(), matchSet->matches->device.get(), matchSet->keyPoints->device.get(), d_cameras);

  CudaSafeCall(cudaFree(d_cameras));

  cudaDeviceSynchronize();
  CudaCheckError();

  // transfer and clear the match set information
  matchSet->matches->setFore(gpu);
  matchSet->keyPoints->setFore(gpu);
  matchSet->matches->transferMemoryTo(cpu);
  matchSet->keyPoints->transferMemoryTo(cpu);
  matchSet->matches->clear(gpu);
  matchSet->keyPoints->clear(gpu);

  // reset the images back to their original state
  for (int i = 0; i < images.size(); i++){
    images[i] = temp[i];
  }
  temp.clear();

  // transfer and clear the cpu information
  bundles->transferMemoryTo(cpu);
  bundles->clear(gpu);
  lines->transferMemoryTo(cpu);
  lines->clear(gpu);

  BundleSet bundleSet = {lines,bundles};

  return bundleSet;
}

/**
 * Caclulates the gradients for a given set of images and returns those gradients as a float array
 * @param matchSet a group of matches
 * @param a group of images, used only for their stored camera parameters
 * @param gradients the image gradients for the given inputs
 */
void ssrlcv::PointCloudFactory::calculateImageGradient(ssrlcv::MatchSet* matchSet, std::vector<ssrlcv::ptr::value<ssrlcv::Image>> images, ssrlcv::ptr::value<ssrlcv::Unity<float>> g){

  float h_linear = 0.00001;    // gradient difference
  float h_radial = 0.00001;  // graident diff

  float gradientError; // this chaneges per iteration

  ssrlcv::BundleSet bundleTemp;

  // gradients are stored in the image structs, but only the camera is modified
  // this containts the gradient
  std::vector<ssrlcv::ptr::value<ssrlcv::Image>> gradient;
  for (int i = 0; i < images.size(); i++){
    ssrlcv::ptr::value<ssrlcv::Image> grad;
    grad.construct();
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
  std::vector<ssrlcv::ptr::value<ssrlcv::Image>> temp;
  for (int i = 0; i < images.size(); i++){
    temp.push_back(images[i]); // fill in the initial images
  }

  // calculate all of the graients with central difference
  // https://v8doc.sas.com/sashtml/ormp/chap5/sect28.htm
  // we are only testing position and orientation gradients

  //
  // X Position Gradients
  //
  for (int j = 0; j < images.size(); j++){
    // ----> Forward
    gradientError = 0.0f;
    temp[j]->camera.cam_pos.x = images[j]->camera.cam_pos.x; // reset for forwards
    temp[j]->camera.cam_pos.x += h_linear;
    bundleTemp = generateBundles(matchSet,temp); // get the bundles for the new temp images
    voidTwoViewTriangulate(bundleTemp, &gradientError);
    float forward = gradientError;
    // <---- Backwards
    gradientError = 0.0f;
    temp[j]->camera.cam_pos.x = images[j]->camera.cam_pos.x; // reset for backwards
    temp[j]->camera.cam_pos.x -= h_linear;
    bundleTemp = generateBundles(matchSet,temp); // get the bundles for the new temp images
    voidTwoViewTriangulate(bundleTemp, &gradientError);
    float backwards = gradientError;
    // calculate the gradient with central difference
    gradient[j]->camera.cam_pos.x = ( forward - backwards ) / ( 2*h_linear );
  }

  //
  // Y Postition Gradients
  //
  for (int j = 0; j < images.size(); j++){
    // ----> Forward
    gradientError = 0.0f;
    temp[j]->camera.cam_pos.y = images[j]->camera.cam_pos.y; // reset for forwards
    temp[j]->camera.cam_pos.y += h_linear;
    bundleTemp = generateBundles(matchSet,temp); // get the bundles for the new temp images
    voidTwoViewTriangulate(bundleTemp, &gradientError);
    float forward = gradientError;
    // <---- Backwards
    gradientError = 0.0f;
    temp[j]->camera.cam_pos.y = images[j]->camera.cam_pos.y; // reset for backwards
    temp[j]->camera.cam_pos.y -= h_linear;
    bundleTemp = generateBundles(matchSet,temp); // get the bundles for the new temp images
    voidTwoViewTriangulate(bundleTemp, &gradientError);
    float backwards = gradientError;
    // calculate the gradient with central difference
    gradient[j]->camera.cam_pos.y = ( forward - backwards ) / ( 2*h_linear );
  }

  //
  // Z Postition Gradients
  //
  for (int j = 0; j < images.size(); j++){
    // ----> Forward
    gradientError = 0.0f;
    temp[j]->camera.cam_pos.z = images[j]->camera.cam_pos.z; // reset for forwards
    temp[j]->camera.cam_pos.z += h_linear;
    bundleTemp = generateBundles(matchSet,temp); // get the bundles for the new temp images
    voidTwoViewTriangulate(bundleTemp, &gradientError);
    float forward = gradientError;
    // <---- Backwards
    gradientError = 0.0f;
    temp[j]->camera.cam_pos.z = images[j]->camera.cam_pos.z; // reset for backwards
    temp[j]->camera.cam_pos.z -= h_linear;
    bundleTemp = generateBundles(matchSet,temp); // get the bundles for the new temp images
    voidTwoViewTriangulate(bundleTemp, &gradientError);
    float backwards = gradientError;
    // calculate the gradient with central difference
    gradient[j]->camera.cam_pos.z = ( forward - backwards ) / ( 2*h_linear );
  }

  //
  // Rotation x^ Gradient
  //
  for (int j = 0; j < images.size(); j++){
    // ----> Forward
    gradientError = 0.0f;
    temp[j]->camera.cam_rot.x = images[j]->camera.cam_rot.x; // reset for forwards
    temp[j]->camera.cam_rot.x += h_radial;
    bundleTemp = generateBundles(matchSet,temp); // get the bundles for the new temp images
    voidTwoViewTriangulate(bundleTemp, &gradientError);
    float forward = gradientError;
    // <---- Backwards
    gradientError = 0.0f;
    temp[j]->camera.cam_rot.x = images[j]->camera.cam_rot.x; // reset for backwards
    temp[j]->camera.cam_rot.x -= h_radial;
    bundleTemp = generateBundles(matchSet,temp); // get the bundles for the new temp images
    voidTwoViewTriangulate(bundleTemp, &gradientError);
    float backwards = gradientError;
    // calculate the gradient with central difference
    gradient[j]->camera.cam_rot.x = ( forward - backwards ) / ( 2*h_radial );
    /*if (gradient[j]->camera.cam_rot.x > (2*PI) || gradient[j]->camera.cam_rot.x < (-2*PI)){
      gradient[j]->camera.cam_rot.x -= floor((gradient[j]->camera.cam_rot.x/(2*PI)))*(2*PI);
    }*/
  }

  //
  // Rotation y^ Gradient
  //
  for (int j = 0; j < images.size(); j++){
    // ----> Forward
    gradientError = 0.0f;
    temp[j]->camera.cam_rot.y = images[j]->camera.cam_rot.y; // reset for forwards
    temp[j]->camera.cam_rot.y += h_radial;
    bundleTemp = generateBundles(matchSet,temp); // get the bundles for the new temp images
    voidTwoViewTriangulate(bundleTemp, &gradientError);
    float forward = gradientError;
    // <---- Backwards
    gradientError = 0.0f;
    temp[j]->camera.cam_rot.y = images[j]->camera.cam_rot.y; // reset for backwards
    temp[j]->camera.cam_rot.y -= h_radial;
    bundleTemp = generateBundles(matchSet,temp); // get the bundles for the new temp images
    voidTwoViewTriangulate(bundleTemp, &gradientError);
    float backwards = gradientError;
    // calculate the gradient with central difference
    gradient[j]->camera.cam_rot.y = ( forward - backwards ) / ( 2*h_radial );
    // adjust to be within bounds if needed
    /*if (gradient[j]->camera.cam_rot.y > (2*PI) || gradient[j]->camera.cam_rot.y < (-2*PI)){
      gradient[j]->camera.cam_rot.y -= floor((gradient[j]->camera.cam_rot.y/(2*PI)))*(2*PI);
    }*/
  }

  //
  // Rotation z^ Gradient
  //
  for (int j = 0; j < images.size(); j++){
    // ----> Forward
    gradientError = 0.0f;
    temp[j]->camera.cam_rot.z = images[j]->camera.cam_rot.z; // reset for forwards
    temp[j]->camera.cam_rot.z += h_radial;
    bundleTemp = generateBundles(matchSet,temp); // get the bundles for the new temp images
    voidTwoViewTriangulate(bundleTemp, &gradientError);
    float forward = gradientError;
    // <---- Backwards
    gradientError = 0.0f;
    temp[j]->camera.cam_rot.z = images[j]->camera.cam_rot.z; // reset for backwards
    temp[j]->camera.cam_rot.z -= h_radial;
    bundleTemp = generateBundles(matchSet,temp); // get the bundles for the new temp images
    voidTwoViewTriangulate(bundleTemp, &gradientError);
    float backwards = gradientError;
    // calculate the gradient with central difference
    gradient[j]->camera.cam_rot.z = ( forward - backwards ) / ( 2*h_radial );
    /*if (gradient[j]->camera.cam_rot.z > (2*PI) || gradient[j]->camera.cam_rot.z < (-2*PI)){
      gradient[j]->camera.cam_rot.z -= floor((gradient[j]->camera.cam_rot.z/(2*PI)))*(2*PI);
    }*/
  }

  int g_j = 0;
  for (int j = 0; j < images.size(); j++){
    g->host.get()[g_j    ] = gradient[j]->camera.cam_pos.x;
    g->host.get()[g_j + 1] = gradient[j]->camera.cam_pos.y;
    g->host.get()[g_j + 2] = gradient[j]->camera.cam_pos.z;
    g->host.get()[g_j + 3] = gradient[j]->camera.cam_rot.x;
    g->host.get()[g_j + 4] = gradient[j]->camera.cam_rot.y;
    g->host.get()[g_j + 5] = gradient[j]->camera.cam_rot.z;
    g_j += 6;
  }

  // free the temp memory
  temp.clear();
}

/**
 * Caclulates the hessian for a given set of images and returns those gradients as a float array
 * @param matchSet a group of matches
 * @param a group of images, used only for their stored camera parameters
 * @param h the image hessian for the given inputs
 */
void ssrlcv::PointCloudFactory::calculateImageHessian(MatchSet* matchSet, std::vector<ssrlcv::ptr::value<ssrlcv::Image>> images, ssrlcv::ptr::value<ssrlcv::Unity<float>> h){

  // stepsize and indexing related variables
  // TODO pass in this step vector
  //                  dX    dY    dZ     dX_r     dY_r     dZ_r
  float h_step[6] = {0.0001,0.0001,0.0001,0.00001,0.00001,0.00001}; // the step size vector
  int   h_i       = 0;               // the hessian location index

  // temp variables within the loops
  float A;
  float B;
  float C;
  float D;
  float E;
  float numer;
  float denom;

  float gradientError; // this chaneges per iteration

  ssrlcv::BundleSet bundleTemp;

  bool local_debug = false;

  // this temp vector is only used for the +/- h steps when calculating the gradients
  // convert the temp images to float vectors
  int num_params = 6; // 3 is just position, 6 position and orientation
  ssrlcv::ptr::value<ssrlcv::Unity<float>> params       = ssrlcv::ptr::value<ssrlcv::Unity<float>>(nullptr,(num_params * images.size()),ssrlcv::cpu);
  ssrlcv::ptr::value<ssrlcv::Unity<float>> params_reset = ssrlcv::ptr::value<ssrlcv::Unity<float>>(nullptr,(num_params * images.size()),ssrlcv::cpu);
  ssrlcv::ptr::value<ssrlcv::Unity<bool>> mfNaNs = ssrlcv::ptr::value<ssrlcv::Unity<bool>>(nullptr,h->size(),ssrlcv::cpu);
  std::vector<ssrlcv::ptr::value<ssrlcv::Image>> temp;
  for (int i = 0; i < images.size(); i++){
    temp.push_back(images[i]); // fill in the initial images
    ssrlcv::ptr::value<ssrlcv::Unity<float>> temp_params = temp[i]->getFloatVector(num_params);
    for (int j = 0; j < num_params; j++){
      params->host.get()[i*num_params + j] = temp_params->host.get()[j];
      params_reset->host.get()[i*num_params + j] = temp_params->host.get()[j];
    }
  }

  // calculates all of the hessians with central difference
  // https://v8doc.sas.com/sashtml/ormp/chap5/sect28.htm
  // we are only testing position and orientation gradients

  if (local_debug){
    std::cout << "\t Params: " << std::endl << "\t\t ";
    for (int i = 0; i < params->size(); i++) {
      std::cout << params->host.get()[i] << " ";
    }
    std::cout << std::endl;
    std::cout << "\t Params reset: " << std::endl << "\t\t ";
    for (int i = 0; i < params_reset->size(); i++) {
      std::cout << params_reset->host.get()[i] << " ";
    }
    std::cout << std::endl << "\t Hessian Size: " << params->size() * params->size() << std::endl;
  }

  for (int i = 0; i < params->size(); i++){
    for (int j = 0; j < params->size(); j++){

      A = 0.0f;
      B = 0.0f;
      C = 0.0f;
      D = 0.0f;
      E = 0.0f;
      gradientError = 0.0f;

      if (i == j){ // second derivative with respect to self

        // ----> First Function Evaluation, A
        params->host.get()[i] += 2.0*h_step[i%num_params];
        bundleTemp = generateBundles(matchSet,temp,params);
        voidTwoViewTriangulate(bundleTemp, &gradientError);
        A = gradientError;
        // reset params
        params->host.get()[i] = params_reset->host.get()[i];

        // ----> Second Function Evaluation, B
        params->host.get()[i] += h_step[i%num_params];
        bundleTemp = generateBundles(matchSet,temp,params);
        voidTwoViewTriangulate(bundleTemp, &gradientError);
        B = gradientError;
        // reset params
        params->host.get()[i] = params_reset->host.get()[i];

        // ----> Third Function Evaluation, C
        bundleTemp = generateBundles(matchSet,temp,params);
        voidTwoViewTriangulate(bundleTemp, &gradientError);
        C = gradientError;
        // reset params
        params->host.get()[i] = params_reset->host.get()[i];

        // ----> Forth Function Evaluation, D
        params->host.get()[i] -= h_step[i%num_params];
        bundleTemp = generateBundles(matchSet,temp,params);
        voidTwoViewTriangulate(bundleTemp, &gradientError);
        D = gradientError;
        // reset params
        params->host.get()[i] = params_reset->host.get()[i];

        // ----> Fifth Function Evaluation, E
        params->host.get()[i] -= 2.0*h_step[i%num_params];
        bundleTemp = generateBundles(matchSet,temp,params);
        voidTwoViewTriangulate(bundleTemp, &gradientError);
        E = gradientError;
        // reset params
        params->host.get()[i] = params_reset->host.get()[i];

        // calculate the result
        numer = -1.0*A + 16.0*B - 30.0*C + 16.0*D - 1.0*E;
        denom = 12.0 * h_step[i%num_params] * h_step[i%num_params];
        // update the hessian
        h->host.get()[h_i] = numer / denom;

        if (isnan(numer/denom) && local_debug){
          std::cout << "!!NaN found at: [" << i << "," << j << "] " << std::endl;
          std::cout << "\t " << A << "  " << B << "  " << C << "  " << D << "  " << E << std::endl;
          std::cout << "\t i mod " << num_params << " = " << i%num_params << "\t" << std::endl;
          std::cout << "\t j mod " << num_params << " = " << j%num_params << "\t" << std::endl;
          std::cout << "\t numer = " << numer << "\t denom = " << denom << std::endl;
          mfNaNs->host.get()[i*params->size() + j] = true;
        } else {
          mfNaNs->host.get()[i*params->size() + j] = false;
        }

      } else { // second derivative with respect to some over variable

        // ----> First Function Evaluation, A
        params->host.get()[i] += h_step[i%num_params];
        params->host.get()[j] += h_step[j%num_params];
        bundleTemp = generateBundles(matchSet,temp,params);
        voidTwoViewTriangulate(bundleTemp, &gradientError);
        A = gradientError;
        // reset params
        params->host.get()[i] = params_reset->host.get()[i];
        params->host.get()[j] = params_reset->host.get()[j];

        // ----> Second Function Evaluation, B
        params->host.get()[i] += h_step[i%num_params];
        params->host.get()[j] -= h_step[j%num_params];
        bundleTemp = generateBundles(matchSet,temp,params);
        voidTwoViewTriangulate(bundleTemp, &gradientError);
        B = gradientError;
        // reset params
        params->host.get()[i] = params_reset->host.get()[i];
        params->host.get()[j] = params_reset->host.get()[j];

        // ----> Third Function Evaluation, C
        params->host.get()[i] -= h_step[i%num_params];
        params->host.get()[j] += h_step[j%num_params];
        bundleTemp = generateBundles(matchSet,temp,params);
        voidTwoViewTriangulate(bundleTemp, &gradientError);
        C = gradientError;
        // reset params
        params->host.get()[i] = params_reset->host.get()[i];
        params->host.get()[j] = params_reset->host.get()[j];

        // ----> Forth Function Evaluation, D
        params->host.get()[i] -= h_step[i%num_params];
        params->host.get()[j] -= h_step[j%num_params];
        bundleTemp = generateBundles(matchSet,temp,params);
        voidTwoViewTriangulate(bundleTemp, &gradientError);
        D = gradientError;
        // reset params
        params->host.get()[i] = params_reset->host.get()[i];
        params->host.get()[j] = params_reset->host.get()[j];

        // calculate the result
        numer = A - B - C + D;
        denom = 4.0 * h_step[i%num_params] * h_step[j%num_params];
        // update the hessian
        h->host.get()[h_i] = numer / denom;

        if (isnan(numer/denom) && local_debug){
          std::cout << "!!NaN found at: [" << i << "," << j << "] " << std::endl;
          std::cout << "\t " << A << "  " << B << "  " << C << "  " << D << std::endl;
          std::cout << "\t i mod " << num_params << " = " << i%num_params << "\t" << std::endl;
          std::cout << "\t j mod " << num_params << " = " << j%num_params << "\t" << std::endl;
          std::cout << "\t numer = " << numer << "\t denom = " << denom << std::endl;
          mfNaNs->host.get()[i*params->size() + j] = true;
        } else {
          mfNaNs->host.get()[i*params->size() + j] = false;
        }

      }

      if (local_debug){
        // testing what is with respect to what
        if (!j) std::cout << std::endl <<  "\t\t " ;
        std::cout << "( " << h_i << " )" << "[ " << i << ", " << j << "] \t";
        if (i == j && i == (params->size() -1)) std::cout << std::endl;
      }

      // increment the hessian index
      h_i++;
    }
  }

  if (local_debug){
    std::cout << std::endl;
    for (int k = 0; k < mfNaNs->size(); k++){
      if (!(k%params->size())) std::cout << std::endl;
      if (mfNaNs->host.get()[k]){
        std::cout << " * ";
      } else {
        std::cout << "   ";
      }
    }
  }

  // cleanup memory
  temp.clear();
}

/**
* Calculates the inverse of the hessian h passed in by refrence
* @param h the image hessian
* @return inverse the inverse hessian
*/
ssrlcv::ptr::value<ssrlcv::Unity<float>> ssrlcv::PointCloudFactory::calculateImageHessianInverse(ssrlcv::ptr::value<ssrlcv::Unity<float>> hessian){

  // based off of cuSOLVER CUDA docs
  // see:
  // https://docs.nvidia.com/cuda/cusolver/index.html#svd-example1

  cusolverDnHandle_t  cusolverH       = NULL;
  cublasHandle_t      cublasH         = NULL;
  cublasStatus_t      cublas_status   = CUBLAS_STATUS_SUCCESS;
  cusolverStatus_t    cusolver_status = CUSOLVER_STATUS_SUCCESS;

  bool local_debug = false;

  const unsigned int N = sqrt(hessian->size());

  ssrlcv::ptr::value<ssrlcv::Unity<float>> A  = ssrlcv::ptr::value<ssrlcv::Unity<float>>(nullptr,hessian->size(),ssrlcv::cpu);
  ssrlcv::ptr::value<ssrlcv::Unity<float>> S  = ssrlcv::ptr::value<ssrlcv::Unity<float>>(nullptr,N              ,ssrlcv::gpu);
  ssrlcv::ptr::value<ssrlcv::Unity<float>> U  = ssrlcv::ptr::value<ssrlcv::Unity<float>>(nullptr,hessian->size(),ssrlcv::gpu);
  ssrlcv::ptr::value<ssrlcv::Unity<float>> VT = ssrlcv::ptr::value<ssrlcv::Unity<float>>(nullptr,hessian->size(),ssrlcv::gpu);

  // Put the hessian into matrix A in column major order
  if (local_debug) std::cout << std::endl << "\t H^T: " << std::endl;
  int index = 0;
  for (int i = 0; i < N; i++){
    if (local_debug) std::cout << std::endl << "\t\t ";
    for (int j = i; j < (N*N); j+=N){
      A->host.get()[index] = hessian->host.get()[j];
      if (local_debug) std::cout << std::fixed << std::setprecision(12) << hessian->host.get()[j] << "  ";
      index++;
    }
  }

  if (local_debug){
    std::cout << std::endl << "\t matrix A: " << std::endl;
    for (int i = 0; i < N; i++){
      std::cout << std::endl << "\t\t ";
      for (int j = i; j < (N*N); j+=N){
        std::cout << std::fixed << std::setprecision(12) << A->host.get()[j] << "  ";
      }
    }
  }

  // transfer hessian to device mem
  A->transferMemoryTo(gpu);

  // cuSOLVER and cuBLAS house keeping
  cusolver_status = cusolverDnCreate(&cusolverH);
  assert(CUSOLVER_STATUS_SUCCESS == cusolver_status);
  cublas_status = cublasCreate(&cublasH);
  assert(CUBLAS_STATUS_SUCCESS == cublas_status);

  // cuSOLVER SVD
  int lwork       = 0;
  int info_gpu    = 0;
  int* devInfo    = NULL;
  float *d_work   = NULL;
  float *d_rwork  = NULL;
  cusolver_status = cusolverDnDgesvd_bufferSize(cusolverH,N,N,&lwork);
  assert (cusolver_status == CUSOLVER_STATUS_SUCCESS);

  CudaSafeCall(cudaMalloc((void**)&d_work , sizeof(float)*lwork));
  CudaSafeCall(cudaMalloc ((void**)&devInfo, sizeof(int)));

  cusolver_status = cusolverDnSgesvd(cusolverH,'A','A',N,N,A->device.get(),N,S->device.get(),U->device.get(),N,VT->device.get(),N,d_work,lwork,d_rwork,devInfo);
  cudaDeviceSynchronize();

  if (cusolver_status != CUSOLVER_STATUS_SUCCESS){
    std::cerr << std::endl << "ERROR setting up cuSOLVER, error status: " << cusolver_status << std::endl;
    if (cusolver_status == CUSOLVER_STATUS_NOT_INITIALIZED) {
      std::cerr << "\t ERROR: CUSOLVER_STATUS_NOT_INITIALIZED" << std::endl;
    }
    if (cusolver_status == CUSOLVER_STATUS_INVALID_VALUE) {
      std::cerr << "\t ERROR: CUSOLVER_STATUS_INVALID_VALUE" << std::endl;
    }
    if (cusolver_status == CUSOLVER_STATUS_ARCH_MISMATCH) {
      std::cerr << "\t ERROR: CUSOLVER_STATUS_ARCH_MISMATCH" << std::endl;
    }
    if (cusolver_status == CUSOLVER_STATUS_INTERNAL_ERROR) {
      std::cerr << "\t ERROR: CUSOLVER_STATUS_INTERNAL_ERROR" << std::endl;
    }
    assert (cusolver_status == CUSOLVER_STATUS_SUCCESS);
    return nullptr;
  }

  // check the devInfo
  CudaSafeCall(cudaMemcpy(&info_gpu, devInfo, sizeof(int), cudaMemcpyDeviceToHost));
  if (info_gpu > 0){
    std::cerr << std::endl << "ERROR: " << info_gpu << " diagonals did not converge to zero " << std::endl;
    return nullptr;
  } else if (info_gpu < 0) {
    std::cerr << std::endl << "ERROR: parameter " << -1.0 * info_gpu << " in the SVD is wrong" << std::endl;
    return nullptr;
  } else {
    if (local_debug) std::cout << std::endl << "\t SVD compelete ..." << std::endl;
  }

  // move back the results
  S->transferMemoryTo(cpu);
  U->transferMemoryTo(cpu);
  VT->transferMemoryTo(cpu);
  S->clear(gpu);
  U->clear(gpu);
  VT->clear(gpu);

  // the following seciton is
  // only to be used for debugging s
  if (local_debug){
    std::cout << std::endl << "\t\t U = " << std::endl;
    index = 0;
    for (int i = 0; i < N; i++){
      std::cout << "\t\t ";
      for (int j = i; j < (N*N); j+=N){
        std::cout << std::fixed << std::setprecision(12) << U->host.get()[j] << " ";
        index++;
      }
      std::cout << std::endl;
    }

    std::cout << std::endl << "\t\t S = ";
    for (int i = 0; i < S->size(); i++){
      std::cout << std::fixed << std::setprecision(12) << S->host.get()[i] << " ";
    }
    std::cout << std::endl << "\t\t ";
    index = 0;
    for (int i = 0; i < N; i++){
      for (int j = 0; j < N; j++){
        if (i == j){
          std::cout << std::fixed << std::setprecision(12) << S->host.get()[index] << " ";
          index ++;
        } else {
          std::cout << std::fixed << std::setprecision(12) << 0.0f << " ";
        }
      }
      std::cout << std::endl << "\t\t ";
    }

    std::cout << std::endl << "\t\t VT = " << std::endl;
    index = 0;
    for (int i = 0; i < N; i++){
      std::cout << "\t\t ";
      for (int j = i; j < (N*N); j+=N){
        std::cout << std::fixed << std::setprecision(12) << VT->host.get()[j] << " ";
        index++;
      }
      std::cout << std::endl;
    }
  }


  // ~~~ now calculate the pseudo inverse ~~~
  // ( ͡° ͜ʖ ͡°)つ━━✫・*。
  // TODO cudafy the code below

  // transpose U
  for (int i = 0; i < U->size(); i++){
    int y = i % N;
    int x = i / N;
    if (y > x){ // swap!
      float temp = U->host.get()[i];
      U->host.get()[i] = U->host.get()[y*N + x];
      U->host.get()[y*N + x] = temp;
    }
  }

  // transpose V^T
  for (int i = 0; i < VT->size(); i++){
    int y = i % N;
    int x = i / N;
    if (y > x){ // swap!
      float temp = VT->host.get()[i];
      VT->host.get()[i] = VT->host.get()[y*N + x];
      VT->host.get()[y*N + x] = temp;
    }
  }

  // invert S
  for (int i = 0; i < S->size(); i++){
    // here I have arbitrarily chosen that the "noise" cutoff shoue be 0.0001
    // because having a value of 1/0.0001 = 10,000 seems absurd
    // and I was getting absolute crap before this
    // it might even be reasonable to make this even more strict
    if (S->host.get()[i] >= 0.0001){
      S->host.get()[i] = 1.0 / S->host.get()[i];
    }
  }

  // just for testing print of the transposes and inverse
  // only to be used for debugging s
  if (local_debug){
    std::cout << std::endl << "\t\t U^T = " << std::endl;
    index = 0;
    for (int i = 0; i < N; i++){
      std::cout << "\t\t ";
      for (int j = i; j < (N*N); j+=N){
        std::cout << std::fixed << std::setprecision(4) << U->host.get()[j] << " ";
        index++;
      }
      std::cout << std::endl;
    }

    std::cout << std::endl << "\t\t S^-1 = ";
    for (int i = 0; i < S->size(); i++){
      std::cout << std::fixed << std::setprecision(4) << S->host.get()[i] << " ";
    }
    std::cout << std::endl << "\t\t ";
    index = 0;
    for (int i = 0; i < N; i++){
      for (int j = 0; j < N; j++){
        if (i == j){
          std::cout << std::fixed << std::setprecision(4) << S->host.get()[index] << " ";
          index ++;
        } else {
          std::cout << "0.0000 ";
        }
      }
      std::cout << std::endl << "\t\t ";
    }

    std::cout << std::endl << "\t\t V = " << std::endl;
    index = 0;
    for (int i = 0; i < N; i++){
      std::cout << "\t\t ";
      for (int j = i; j < (N*N); j+=N){
        std::cout << std::fixed << std::setprecision(4) << VT->host.get()[j] << " ";
        index++;
      }
      std::cout << std::endl;
    }
  }

  // prep to multiply A^+ = V S^+ U^T
  S->transferMemoryTo(gpu);
  U->transferMemoryTo(gpu);
  VT->transferMemoryTo(gpu);
  // actually fill in a matrix for S
  ssrlcv::ptr::value<ssrlcv::Unity<float>> E = ssrlcv::ptr::value<ssrlcv::Unity<float>>(nullptr,hessian->size(),ssrlcv::cpu);
  index = 0;
  for (int i = 0; i < N; i++){
    for (int j = 0; j < N; j++){
      if (i == j){
        E->host.get()[i*N + j] = S->host.get()[index];
        index++;
      } else {
        E->host.get()[i*N + j] = 0.0f;
      }
    }
  }
  E->transferMemoryTo(gpu);
  ssrlcv::ptr::value<ssrlcv::Unity<float>> inverse = ssrlcv::ptr::value<ssrlcv::Unity<float>>(nullptr,hessian->size(),ssrlcv::cpu);
  for (int i = 0; i < inverse->size(); i++){
    inverse->host.get()[i] = 0.0f;
  }
  inverse->transferMemoryTo(gpu);
  ssrlcv::ptr::value<ssrlcv::Unity<float>> C = ssrlcv::ptr::value<ssrlcv::Unity<float>>(nullptr,hessian->size(),ssrlcv::cpu);
  for (int i = 0; i < C->size(); i++){
    C->host.get()[i] = 0.0f;
  }
  C->transferMemoryTo(gpu);
  float alpha = 1.0f;
  float beta  = 1.0f;

  // do the multiplication in cuBLAS
  cublas_status = cublasSgemm(cublasH,CUBLAS_OP_N,CUBLAS_OP_N,N,N,N,&alpha,VT->device.get(),N,E->device.get(),N,&beta,C->device.get(),N);
  cudaDeviceSynchronize();
  if (cublas_status != 0){
    std::cerr << std::endl << "ERROR: in cuBLAS multiply" << std::endl;
    return nullptr;
  }
  cublas_status = cublasSgemm(cublasH,CUBLAS_OP_N,CUBLAS_OP_N,N,N,N,&alpha,C->device.get(),N,U->device.get(),N,&beta,inverse->device.get(),N);
  cudaDeviceSynchronize();
  if (cublas_status != 0){
    std::cerr << std::endl << "ERROR: in cuBLAS multiply" << std::endl;
    return nullptr;
  }
  inverse->setFore(gpu);
  inverse->transferMemoryTo(cpu);
  inverse->clear(gpu);

  // to print of the inverse if debugging
  if (local_debug) {
    std::cout << std::endl << "\t\t Pseudo Inverse = " << std::endl;
    index = 0;
    for (int i = 0; i < N; i++){
      std::cout << "\t\t ";
      for (int j = i; j < (N*N); j+=N){
        std::cout << std::fixed << std::setprecision(4) << inverse->host.get()[j] << " ";
        index++;
      }
      std::cout << std::endl;
    }
  }

  cudaFree(devInfo);
  cudaFree(d_work);
  cudaFree(d_rwork);

  // end solver session
  cublasDestroy(cublasH);
  cusolverDnDestroy(cusolverH);

  return inverse;
}

/**
 * A bundle adjustment based on a two-view triangulation that includes a second order Hessian calculation and first order gradient caclulation
 * @param matchSet a group of matches
 * @param a group of images, used only for their stored camera parameters
 * @return a bundle adjusted point cloud
 */
ssrlcv::ptr::value<ssrlcv::Unity<float3>> ssrlcv::PointCloudFactory::BundleAdjustTwoView(ssrlcv::MatchSet* matchSet, std::vector<ssrlcv::ptr::value<ssrlcv::Image>> images, unsigned int iterations, const char * debugFilename){

  // local variabels for function
  ssrlcv::ptr::value<ssrlcv::Unity<float3>> points;
  ssrlcv::ptr::value<ssrlcv::Unity<float>> gradient;
  ssrlcv::ptr::value<ssrlcv::Unity<float>> hessian;
  ssrlcv::ptr::value<ssrlcv::Unity<float>> update;
  ssrlcv::ptr::value<ssrlcv::Unity<float>> inverse;
  ssrlcv::BundleSet bundleTemp;
  // This is for error tracking and also used for debug and
  std::vector<float> errorTracker;
  std::vector<float> alphaTracker;

  // allocate memory
  int num_params = 6; // 3 is just (x,y,z) and 6 incldue rotations in (x,y,z)
  float localError;   // this stays constant per iteration
  float initialError; // this is just used once to see if we're going down a bad path or not
  gradient = ssrlcv::ptr::value<ssrlcv::Unity<float>>(nullptr,(num_params * images.size()),ssrlcv::cpu);
  hessian  = ssrlcv::ptr::value<ssrlcv::Unity<float>>(nullptr,((num_params * images.size())*(num_params * images.size())),ssrlcv::cpu);
  update   = ssrlcv::ptr::value<ssrlcv::Unity<float>>(nullptr,(num_params * images.size()),ssrlcv::gpu); // used in state update

  // TODO these variables really should be passed thru
  // for debug
  bool local_debug    = false;
  bool local_verbose  = true;
  // const step
  bool second_order   = true;
  // enable or disable normalization
  bool do_normalization = false;
  // fixed_camera is used when
  bool fixed_camera = true; // <--- make all updates relative to the first camera and keeps that stationary

  // just to tell the user what is happening
  if (second_order && ( local_debug || local_verbose )){
    std::cout << "\t Bundle Adjustment is in Second Order Mode" << std::endl;
  } else {
    std::cout << "\t Bundle Adjustment is in First Order Mode" << std::endl;
  }

  // used to store the best params found in the optimization
  std::vector<ssrlcv::ptr::value<ssrlcv::Image>> bestParams;
  for (int i = 0; i < images.size(); i++){
    ssrlcv::ptr::value<ssrlcv::Image> t;
    t.construct();
    t->camera = images[i]->camera;
    bestParams.push_back(t); // fill in the initial images
  }
  std::vector<ssrlcv::ptr::value<ssrlcv::Image>> secondBestParams;
  for (int i = 0; i < images.size(); i++){
    ssrlcv::ptr::value<ssrlcv::Image> t;
    t.construct();
    t->camera = images[i]->camera;
    secondBestParams.push_back(t); // fill in the initial images
  }

  // scalars in the matrix multiplication
  // beta should always be 0.0
  // alpha can be changed to dampen stepsize
  // alpha should be between 0.0 - 1.0
  float alpha = 0.1f; // should less than 1
  float beta  = 0.0f;   // should always be 0
  int   inc   = 1;      // should always be 1

  // these are adjusted after a sequence point is reached
  float dist_step = 1.0; // starting distance magnitude
  float angle_mag = 1.0; // starting angle mag

  // each time the error is cut in half, so is the stepsize
  // float error_comp;
  float bestError;


  // does an initial computation of the starting error
  // NOTE print off new error
  bundleTemp = generateBundles(matchSet,images);
  voidTwoViewTriangulate(bundleTemp, &initialError);
  if (local_debug || local_verbose) std::cout << "[initial] \terror: " << initialError << std::endl;
  errorTracker.push_back(initialError); // saves the initial error
  bestError = initialError; // we want our future best erros to be less than the initial error

  // the Hessian is always square, N is one side of the hessian
  const unsigned int N = (unsigned int) sqrt(hessian->size());

  // cuBLAS housekeeping
  cublasHandle_t cublasH       = NULL;
  cublasStatus_t cublas_status = CUBLAS_STATUS_SUCCESS;
  cublas_status                = cublasCreate(&cublasH);
  assert(CUBLAS_STATUS_SUCCESS == cublas_status);

  // TEMP inversion test
  if (local_debug){
    ssrlcv::ptr::value<ssrlcv::Unity<float>> test = ssrlcv::ptr::value<ssrlcv::Unity<float>>(nullptr,9,ssrlcv::cpu);
    test->host.get()[0] = 1.0; //1.0;
    test->host.get()[1] = 2.0; //1.0;
    test->host.get()[2] = 3.0; //-1.0;
    test->host.get()[3] = 4.0; //-2.0;
    test->host.get()[4] = 5.0; //3.0;
    test->host.get()[5] = 6.0; //2.0;
    test->host.get()[6] = 7.0; //3.0;
    test->host.get()[7] = 8.0; //-3.0;
    test->host.get()[8] = 9.0; //3.0;
    std::cout << std::endl << std::endl << "\t\t test matrix = " << std::endl;
    for (int i = 0; i < 3; i++){
      std::cout << std::endl << "\t\t ";
      for (int j = 0; j < 3; j++){
        std::cout << std::fixed << std::setprecision(4) << test->host.get()[3*i + j] << "\t ";
      }

    }
    calculateImageHessianInverse(test);
  }

  // begin iterative gradient decent
  for (int i = 0; i < iterations; i++){

    // NOTE gradients calculated here
    if (local_debug || local_verbose) std::cout << "\tCalculating Gradient ..." << std::endl;
    calculateImageGradient(matchSet,images,gradient);

    // calculate second order stuff
    // if not in constant stepsize mode
    if (second_order){
      // TODO hessians calculated here
      if (local_debug || local_verbose) std::cout << "\tCalculating Hessian ..." << std::endl;
      calculateImageHessian(matchSet,images,hessian);

      if (local_debug){
        std::cout << "\t\t Hessian Size: " << hessian->size() << std::endl;
        for (int j = 0; j < 12; j++){
          std::cout << "\t\t";
          for (int k = 0; k < 12; k++){
            std::cout << " " << std::fixed << std::setprecision(4) << hessian->host.get()[12*j + k];
          }
          // std::cout << "  " << std::fixed << std::setprecision(4) << hessian->host.get()[j+1];
          // std::cout << "  " << std::fixed << std::setprecision(4) << hessian->host.get()[j+2];
          // std::cout << "  " << std::fixed << std::setprecision(4) << hessian->host.get()[j+3];
          // std::cout << "  " << std::fixed << std::setprecision(4) << hessian->host.get()[j+4];
          // std::cout << "  " << std::fixed << std::setprecision(4) << hessian->host.get()[j+5];
          std::cout << " " << std::endl;
        }
        std::cout << std::endl;
      }

      // invert hessian here
      if (local_debug || local_verbose) std::cout << "\tInverting Hessian ..." << std::endl;
      // see https://stackoverflow.com/questions/28794010/solving-dense-linear-systems-ax-b-with-cuda
      // see https://stackoverflow.com/questions/27094612/cublas-matrix-inversion-from-device
      // takes the pseudo inverse of the hessian
      inverse = calculateImageHessianInverse(hessian);
    }


    if (!second_order){
      //
      // First order mode
      //

      // only for testing
      if (local_debug) {
        // print the update
        std::cout << std::endl << "\t\t (update) pure gradient = " << std::endl;
        std::cout << "\t\t ";
        for (int j = 0; j < N; j++){
          std::cout << std::fixed << std::setprecision(8) << gradient->host.get()[j] << " ";
        }
        std::cout << std::endl;
      }

      // only for testing
      if (local_debug) {
        // print the update
        std::cout << std::endl << "\t\t (update) gradient scaled by alpha " << std::fixed << std::setprecision(8) << alpha << " = " << std::endl;
        std::cout << "\t\t ";
        for (int j = 0; j < N; j++){
          std::cout << std::fixed << std::setprecision(8) << alpha * gradient->host.get()[j] << " ";
        }
        std::cout << std::endl;
      }

      // prep to normalize the update
      // normalize the distance update, and set to specific distance
      float mag[2] = {1.0f, 1.0f};
      if (do_normalization){
        for (int j = 0; j < images.size(); j++){
          mag[j] = sqrtf((update->host.get()[6*j] * update->host.get()[6*j]) + (update->host.get()[6*j + 1] * update->host.get()[6*j + 1]) + (update->host.get()[6*j + 2] * update->host.get()[6*j + 2]));
        }
      }

      // normalize the rotation
      // move all algular update to within [-pi,pi]
      float ang[2] = {1.0f, 1.0f};
      if (do_normalization){
        for (int j = 0; j < images.size(); j++){
          ang[j] = sqrtf((update->host.get()[6*j + 3] * update->host.get()[6*j + 3]) + (update->host.get()[6*j + 4] * update->host.get()[6*j + 4]) + (update->host.get()[6*j + 5] * update->host.get()[6*j + 5]));
          if (ang[j] > angle_mag){
            angle_mag = ang[j];
            ang[j] = 1.0;
          }
        }
      }

      int g_j = 0;
      for (int j = 0; j < images.size(); j++){
        images[j]->camera.cam_pos.x = images[j]->camera.cam_pos.x - dist_step * (gradient->host.get()[g_j    ] / mag[j]);
        images[j]->camera.cam_pos.y = images[j]->camera.cam_pos.y - dist_step * (gradient->host.get()[g_j + 1] / mag[j]);
        images[j]->camera.cam_pos.z = images[j]->camera.cam_pos.z - dist_step * (gradient->host.get()[g_j + 2] / mag[j]);
        images[j]->camera.cam_rot.x = images[j]->camera.cam_rot.x - angle_mag * (gradient->host.get()[g_j + 3] / ang[j]);
        images[j]->camera.cam_rot.y = images[j]->camera.cam_rot.y - angle_mag * (gradient->host.get()[g_j + 4] / ang[j]);
        images[j]->camera.cam_rot.z = images[j]->camera.cam_rot.z - angle_mag * (gradient->host.get()[g_j + 5] / ang[j]);
        g_j += 6;
      }
    } else {
      //
      // Second order mode
      //

      if (local_debug){
        // print hessian
        std::cout << std::endl << "\t\t H^+ = " << std::endl;
        for (int j = 0; j < N; j++){
          std::cout << "\t\t ";
          for (int k = j; k < (N*N); k+=N){
            std::cout << std::fixed << std::setprecision(8) << inverse->host.get()[k] << " ";
          }
          std::cout << std::endl;
        }
        // print the gradient
        std::cout << std::endl << "\t\t g^T = " << std::endl;
        std::cout << "\t\t ";
        for (int j = 0; j < N; j++){
          std::cout << std::fixed << std::setprecision(8) << gradient->host.get()[j] << " ";
        }
        std::cout << std::endl;
      }

      gradient->transferMemoryTo(gpu);
      inverse->transferMemoryTo(gpu);

      // multiply
      cublas_status = cublasSgemv(cublasH, CUBLAS_OP_N, N, N, &alpha, inverse->device.get(), N, gradient->device.get(), inc, &beta, update->device.get(), inc);
      cudaDeviceSynchronize();

      // TODO maybe make a method for printing off exavt cuBLAS errors
      if (cublas_status != CUBLAS_STATUS_SUCCESS){
        std::cerr << std::endl << "ERROR: failure when multiplying inverse hessian and gradient for state update." << std::endl;
        std::cerr << "\t ERROR STATUS: multiplication failed with status: " << cublas_status << std::endl;
        if (cublas_status == CUBLAS_STATUS_NOT_INITIALIZED){
          std::cerr << "\t cuBLAS ERROR: CUBLAS_STATUS_NOT_INITIALIZED" << std::endl;
        }
        if (cublas_status == CUBLAS_STATUS_INVALID_VALUE){
          std::cerr << "\t cuBLAS ERROR: CUBLAS_STATUS_INVALID_VALUE" << std::endl;
        }
        if (cublas_status == CUBLAS_STATUS_ARCH_MISMATCH){
          std::cerr << "\t cuBLAS ERROR: CUBLAS_STATUS_ARCH_MISMATCH" << std::endl;
        }
        if (cublas_status == CUBLAS_STATUS_EXECUTION_FAILED){
          std::cerr << "\t cuBLAS ERROR: CUBLAS_STATUS_EXECUTION_FAILED" << std::endl;
        }
        return nullptr; // TODO have a safe shutdown of bundle adjustment given a failure
      }

      // temp cleanups
      gradient->transferMemoryTo(cpu);
      inverse->transferMemoryTo(cpu);
      // gradient->clear(gpu);
      // inverse->clear(gpu);

      // get the update
      //update->setFore(gpu);
      update->transferMemoryTo(cpu);

      // only for testing
      if (local_debug) {
        // print the update
        std::cout << std::endl << "\t\t (update) Delta X = " << std::endl;
        std::cout << "\t\t ";
        for (int j = 0; j < N; j++){
          std::cout << std::fixed << std::setprecision(8) << update->host.get()[j] << " ";
        }
        std::cout << std::endl;
      }

      // prep to normalize the update
      // normalize the distance update, and set to specific distance
      float mag[2] = {1.0f, 1.0f};
      if (do_normalization){
        for (int j = 0; j < images.size(); j++){
          mag[j] = sqrtf((update->host.get()[6*j] * update->host.get()[6*j]) + (update->host.get()[6*j + 1] * update->host.get()[6*j + 1]) + (update->host.get()[6*j + 2] * update->host.get()[6*j + 2]));
        }
      }

      // normalize the rotation
      // move all algular update to within [-pi,pi]
      float ang[2] = {1.0f, 1.0f};
      if (do_normalization){
        for (int j = 0; j < images.size(); j++){
          ang[j] = sqrtf((update->host.get()[6*j + 3] * update->host.get()[6*j + 3]) + (update->host.get()[6*j + 4] * update->host.get()[6*j + 4]) + (update->host.get()[6*j + 5] * update->host.get()[6*j + 5]));
          if (ang[j] > angle_mag){
            angle_mag = ang[j];
            ang[j] = 1.0;
          }
        }
      }

      // update
      int g_j = 0;
      for (int j = 0; j < images.size(); j++){
        if (!fixed_camera && j){ // <----- only modifiy the non 0 index cameras. All changes relative to camera 0
          images[j]->camera.cam_pos.x = images[j]->camera.cam_pos.x - dist_step * (update->host.get()[g_j    ] / mag[j]);
          images[j]->camera.cam_pos.y = images[j]->camera.cam_pos.y - dist_step * (update->host.get()[g_j + 1] / mag[j]);
          images[j]->camera.cam_pos.z = images[j]->camera.cam_pos.z - dist_step * (update->host.get()[g_j + 2] / mag[j]);
          images[j]->camera.cam_rot.x = images[j]->camera.cam_rot.x - angle_mag * (update->host.get()[g_j + 3] / ang[j]);
          images[j]->camera.cam_rot.y = images[j]->camera.cam_rot.y - angle_mag * (update->host.get()[g_j + 4] / ang[j]);
          images[j]->camera.cam_rot.z = images[j]->camera.cam_rot.z - angle_mag * (update->host.get()[g_j + 5] / ang[j]);
        }
        g_j += 6;
      }

    }

    // NOTE print off new error
    bundleTemp = generateBundles(matchSet,images);
    voidTwoViewTriangulate(bundleTemp, &localError);
    if (local_debug || local_verbose) std::cout << "[" << std::fixed << std::setprecision(12) << (i + 1) << "] \terror: " << localError << std::endl;

    // is this step better than the best?
    if (localError < bestError) {

      //
      // New best params found
      //


      bestError = localError;
      // the step improved the measured error
      for (int j = 0; j < bestParams.size(); j++){
        secondBestParams[j]->camera = bestParams[j]->camera;
        bestParams[j]->camera = images[j]->camera;
      }
      if (local_debug || local_verbose) std::cout << "\t New lowest value found: " << bestError << std::endl;

      // NOTE perhaps only scale down only after the frist step is taken?? maybe do it every time?
      if (i){
        float scale_down = errorTracker.back() / localError;
        // scale_down *= scale_down; // squared ratio
        // or
        // scale_down = pow(2.0f, scale_down);
        // or
        // scale_down = pow(scale_down, scale_down);
        alpha /= scale_down;
        if (local_debug || local_verbose) std::cout << "\t Alpha updated to: " << std::fixed << std::setprecision(24) << alpha << " with scale down: " << scale_down << std::endl;
        alphaTracker.push_back(alpha);
      }

      // add the newest error!
      errorTracker.push_back(localError);

    } else {

      //
      // Previous camera param was the best!!! oh noooo!
      // (╯°□°）╯︵ ┻━┻
      //

      if (local_debug || local_verbose) {
        std::cout << "\t BUNDLE ADJUSTMENT HAS FOUND LOCAL MINIMUM " << std::endl;
        std::cout << "\t\t Local minima found after: " << i << " iterations" << std::endl;
        std::cout << "\t\t Reduced error from: " << errorTracker[0] << " ---> " << errorTracker.back() << std::endl;
      }

      // reset the camera params to the best one's found
      for (int j = 0; j < images.size(); j++){
        images[j]->camera = bestParams[j]->camera;
      }

      if (!i){
        // this was the first time, so let's just cut alpha and try one more time
        alpha /= 100.0f;
      } else {
        // exit the optimization
        break;
      }
    } // end good / bad error check

    // try setting the iteration count??
    // &interations = &i;
  } // end bundle adjustment loop, i++

  // TODO only do if debugging
  // TODO add a flag that allows the user to test this
  // write linearError chagnes to a CSV
  if (local_debug || local_verbose) {
    std::string name1 = debugFilename;
    name1 += "Errors";
    std::string name2 = debugFilename;
    name2 += "Alphas";
    writeCSV(errorTracker, name1.c_str());
    writeCSV(alphaTracker, name2.c_str());
  }

  bundleTemp = generateBundles(matchSet,images);
  points = twoViewTriangulate(bundleTemp, &localError);

  cublasDestroy(cublasH);

  // cleanup memory
  // delete gradient;
  /*delete inverse;
  delete update;
  delete bundleTemp.bundles;
  delete bundleTemp.lines;*/

  // return the new points
  // bundleTemp = generateBundles(matchSet,images);
  return points;
}

/**
 * A bundle adjustment based on a N-view triangulation that includes a second order Hessian calculation and first order gradient caclulation
 * @param matchSet a group of matches
 * @param a group of images, used only for their stored camera parameters
 * @return a bundle adjusted point cloud
 */
ssrlcv::ptr::value<ssrlcv::Unity<float3>> ssrlcv::PointCloudFactory::BundleAdjustNView(ssrlcv::MatchSet* matchSet, std::vector<ssrlcv::ptr::value<ssrlcv::Image>> images, unsigned int interations){

  // TODO

  std::cerr << "ERROR: N view bundle adjustment not yet implemented" << std::endl;

  return nullptr;
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
void ssrlcv::PointCloudFactory::saveDebugCloud(ssrlcv::ptr::value<ssrlcv::Unity<float3>> pointCloud, BundleSet bundleSet, std::vector<ssrlcv::ptr::value<ssrlcv::Image>> images){
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
    cpoints[index].x = bundleSet.lines->host.get()[i].pnt.x + bundleSet.lines->host.get()[i].vec.x;
    cpoints[index].y = bundleSet.lines->host.get()[i].pnt.y + bundleSet.lines->host.get()[i].vec.y;
    cpoints[index].z = bundleSet.lines->host.get()[i].pnt.z + bundleSet.lines->host.get()[i].vec.z;
    cpoints[index].r = 0;
    cpoints[index].g = 0;
    cpoints[index].b = 255;
    index++;
  }
  // fill in the point cloud GREEN
  for (int i = 0; i < pointCloud->size(); i++){
    cpoints[index].x = pointCloud->host.get()[i].x; //
    cpoints[index].y = pointCloud->host.get()[i].y;
    cpoints[index].z = pointCloud->host.get()[i].z;
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
void ssrlcv::PointCloudFactory::saveDebugCloud(ssrlcv::ptr::value<ssrlcv::Unity<float3>> pointCloud, BundleSet bundleSet, std::vector<ssrlcv::ptr::value<ssrlcv::Image>> images, std::string filename){
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
    cpoints[index].x = bundleSet.lines->host.get()[i].pnt.x + bundleSet.lines->host.get()[i].vec.x;
    cpoints[index].y = bundleSet.lines->host.get()[i].pnt.y + bundleSet.lines->host.get()[i].vec.y;
    cpoints[index].z = bundleSet.lines->host.get()[i].pnt.z + bundleSet.lines->host.get()[i].vec.z;
    cpoints[index].r = 0;
    cpoints[index].g = 0;
    cpoints[index].b = 255;
    index++;
  }
  // fill in the point cloud GREEN
  for (int i = 0; i < pointCloud->size(); i++){
    cpoints[index].x = pointCloud->host.get()[i].x; //
    cpoints[index].y = pointCloud->host.get()[i].y;
    cpoints[index].z = pointCloud->host.get()[i].z;
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
void ssrlcv::PointCloudFactory::saveDebugCloud(ssrlcv::ptr::value<ssrlcv::Unity<float3>> pointCloud, BundleSet bundleSet, std::vector<ssrlcv::ptr::value<ssrlcv::Image>> images, const char* filename){
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
    cpoints[index].x = bundleSet.lines->host.get()[i].pnt.x + bundleSet.lines->host.get()[i].vec.x;
    cpoints[index].y = bundleSet.lines->host.get()[i].pnt.y + bundleSet.lines->host.get()[i].vec.y;
    cpoints[index].z = bundleSet.lines->host.get()[i].pnt.z + bundleSet.lines->host.get()[i].vec.z;
    cpoints[index].r = 0;
    cpoints[index].g = 0;
    cpoints[index].b = 255;
    index++;
  }
  // fill in the point cloud GREEN
  for (int i = 0; i < pointCloud->size(); i++){
    cpoints[index].x = pointCloud->host.get()[i].x; //
    cpoints[index].y = pointCloud->host.get()[i].y;
    cpoints[index].z = pointCloud->host.get()[i].z;
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
void ssrlcv::PointCloudFactory::saveDebugLinearErrorCloud(ssrlcv::MatchSet* matchSet, std::vector<ssrlcv::ptr::value<ssrlcv::Image>> images, const char* filename){
  // build the helpers to make the colors
  uchar3 colors[2000];
  float3 good = {108,255,221};
  float3 meh  = {251,215,134};
  float3 bad  = {247,121,125};
  float3 gr1  = (meh - good)/1000;
  float3 gr2  = (bad - meh )/1000;
  // initialize the gradient "mapping"
  float3 temp;
  // std::cout << "building gradient" << std::endl;
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
  // std::cout << "the boiz" << std::endl;
  float linearError;
  linearError = 0.0; // just something to start
  float linearErrorCutoff;
  linearErrorCutoff = 1000000.0; // just somethihng to start

  // the boiz
  ssrlcv::BundleSet      bundleSet;
  ssrlcv::ptr::value<ssrlcv::Unity<float>>  errors;
  ssrlcv::ptr::value<ssrlcv::Unity<float3>> points;

  // need bundles
  bundleSet = generateBundles(matchSet,images);
  // do an initial triangulation
  errors = ssrlcv::ptr::value<ssrlcv::Unity<float>>(nullptr,matchSet->matches->size(),ssrlcv::cpu);
  struct colorPoint* cpoints = (colorPoint*)  malloc(matchSet->matches->size() * sizeof(struct colorPoint));

  std::cout << "attempting guy" << std::endl;
  if (images.size() == 2){
    //
    // 2-View Case
    //

    points = twoViewTriangulate(bundleSet, errors, &linearError, &linearErrorCutoff);
    float max = 0.0; // it would be nice to have a better way to get the max, but because this is only for debug idc
    for (int i = 0; i < errors->size(); i++){
      if (errors->host.get()[i] > max){
        max = errors->host.get()[i];
      }
    }
    std::cout << "found max: " << max << std::endl;
    // now fill in the color point locations
    for (int i = 0; i < points->size(); i++){
      // i assume that the errors and the points will have the same indices
      cpoints[i].x = points->host.get()[i].x; //
      cpoints[i].y = points->host.get()[i].y;
      cpoints[i].z = points->host.get()[i].z;
      int j = floor(errors->host.get()[i] * (2000 / max));
      // std::cout << "j: " << j << "\t e: " << errors->host.get()[i] << "\t ratio: " << (2000 / max) << "\t " << i << "/" << points->size() << std::endl;
      cpoints[i].r = colors[j].x;
      cpoints[i].g = colors[j].y;
      cpoints[i].b = colors[j].z;
    }

  } else {
    //
    // N-View Case
    //

    points = nViewTriangulate(bundleSet, errors, &linearError, &linearErrorCutoff);
    float max = 0.0; // it would be nice to have a better way to get the max, but because this is only for debug idc
    for (int i = 0; i < errors->size(); i++){
      if (errors->host.get()[i] > max){
        max = errors->host.get()[i];
      }
    }
    std::cout << "found max: " << max << std::endl;
    // now fill in the color point locations
    for (int i = 0; i < points->size(); i++){
      // i assume that the errors and the points will have the same indices
      cpoints[i].x = points->host.get()[i].x; //
      cpoints[i].y = points->host.get()[i].y;
      cpoints[i].z = points->host.get()[i].z;
      int j = floor(errors->host.get()[i] * (2000 / max));
      // std::cout << "j: " << j << "\t e: " << errors->host.get()[i] << "\t ratio: " << (2000 / max) << "\t " << i << "/" << points->size() << std::endl;
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
void ssrlcv::PointCloudFactory::saveViewNumberCloud(ssrlcv::MatchSet* matchSet, std::vector<ssrlcv::ptr::value<ssrlcv::Image>> images, const char* filename){
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
  float linearError;
  linearError = 0.0; // just something to start
  float linearErrorCutoff;
  linearErrorCutoff = 1000000.0; // just somethihng to start

  // the boiz
  ssrlcv::BundleSet      bundleSet;
  ssrlcv::ptr::value<ssrlcv::Unity<float>>  errors;
  ssrlcv::ptr::value<ssrlcv::Unity<float3>> points;

  // need bundles
  bundleSet = generateBundles(matchSet,images);
  // do an initial triangulation
  errors = ssrlcv::ptr::value<ssrlcv::Unity<float>>(nullptr,matchSet->matches->size(),ssrlcv::cpu);
  struct colorPoint* cpoints = (colorPoint*)  malloc(matchSet->matches->size() * sizeof(struct colorPoint));

  std::cout << "attempting guy" << std::endl;



  //
  // N-View Case
  //

  points = nViewTriangulate(bundleSet, errors, &linearError, &linearErrorCutoff);
  float max = images.size();
  // now fill in the color point locations
  for (int i = 0; i < points->size() - 1; i++){
    // i assume that the errors and the points will have the same indices
    cpoints[i].x = points->host.get()[i].x; //
    cpoints[i].y = points->host.get()[i].y;
    cpoints[i].z = points->host.get()[i].z;

    int j = floor(bundleSet.bundles->host.get()[i].numLines * (2000 / max));
    // int j = floor(errors->host.get()[i] * (2000 / max));
    // // std::cout << "j: " << j << "\t e: " << errors->host.get()[i] << "\t ratio: " << (2000 / max) << "\t " << i << "/" << points->size() << std::endl;
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
void ssrlcv::PointCloudFactory::generateSensitivityFunctions(ssrlcv::MatchSet* matchSet, std::vector<ssrlcv::ptr::value<ssrlcv::Image>> images, std::string filename){

  // the bundle set that changes each iteration
  BundleSet bundleSet;

  // the ranges and step sizes
  float linearRange   = 4.0;     //   +/- linear Range
  float angularRange  = (PI);   //   +/- angular Range
  float deltaL = 0.01;          // linearRange stepsize
  float deltaA = 0.001;        // angular stpesize

  // the temp error to be stored
  float* currError = new float;
  float start;
  float end;

  // TRACKERS
  std::vector<float> trackerValue;
  std::vector<float> trackerError;

  // the camera to refrence when doing the sensitivity test
  int ref_cam = 0;

  // the temp cameras
  std::vector<ssrlcv::ptr::value<ssrlcv::Image>> temp;
  for (int i = 0; i < images.size(); i++){
    temp.push_back(images[i]); // fill in the initial images
  }

  if (images.size() == 2){
    //
    // 2-View Case
    //

    logger.warn << "WARNING: Starting an intesive debug feature, this should be disabled in production\n";
    logger.warn << "WARNING: DISABLE GENERATE SENSITIVITY FUNCTIONS IN PRODUCTION!!\n";

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

/**
* Saves the plane that was estimated to be the "primary" plane of the pointCloud
* this methods saves a plane which can be visualized as a mesh
* @param pointCloud the point cloud to visualize plane estimation from
* @param filename a string representing the filename that should be saved
* @param range a float representing 1/2 of a side in km, so +/- range is how but the plane will be
*/
void ssrlcv::PointCloudFactory::visualizePlaneEstimation(ssrlcv::ptr::value<ssrlcv::Unity<float3>> pointCloud, std::vector<ssrlcv::ptr::value<ssrlcv::Image>> images, const char* filename, float scale){

  // disable for local print statments
  bool local_debug    = false;
  bool local_verbose  = false;

  // extract the camera locations for normal estimation
  ssrlcv::ptr::value<ssrlcv::Unity<float3>> locations = ssrlcv::ptr::value<ssrlcv::Unity<float3>>(nullptr,images.size(),ssrlcv::cpu);
  for (int i = 0; i < images.size(); i++){
    locations->host.get()[i].x = images[i]->camera.cam_pos.x;
    locations->host.get()[i].y = images[i]->camera.cam_pos.y;
    locations->host.get()[i].z = images[i]->camera.cam_pos.z;
  }

  // create the octree
  Octree oct = Octree(pointCloud, 8, false);
  // caclulate the estimated plane normal
  ssrlcv::ptr::value<ssrlcv::Unity<float3>> normal = oct.computeAverageNormal(3, 10, images.size(), locations->host.get());

  if (local_debug || local_verbose) std::cout << "Estimated plane normal: (" << normal->host.get()[0].x << ", " << normal->host.get()[0].y << ", " << normal->host.get()[0].z << ")" << std::endl;

  // the averate point
  ssrlcv::ptr::value<ssrlcv::Unity<float3>> point = getAveragePoint(pointCloud);

  if (local_debug) {
    std::cout << "Average Point: ( " << point->host.get()[0].x << ", " << point->host.get()[0].y << ", " << point->host.get()[0].z << " )" << std::endl;
    std::cout << "Normal Vector: < " << normal->host.get()[0].x << ", " << normal->host.get()[0].y << ", " << normal->host.get()[0].z << " )" << std::endl;
    ssrlcv::writePLY("testNorm",point,normal);
  }

  // TODO perhaps find the location with the best density of points along the average normal

  // generate the example plane vertices
  // loop through x and y and caclulate z using the equation of the plane, see: https://en.wikipedia.org/wiki/Plane_(geometry)#Point-normal_form_and_general_form_of_the_equation_of_a_plane
  int step   = 40; // keep this evenly divisible
  int bounds = (int) ( (int) scale - ( (int) scale % step)); // does +/- at these bounds in x and y, needs to be divisible by step
  int index  = 0;
  ssrlcv::ptr::value<ssrlcv::Unity<float3>> vertices = ssrlcv::ptr::value<ssrlcv::Unity<float3>>(nullptr, (size_t) (2 * bounds / step)*(2 * bounds / step),ssrlcv::cpu);
  for (float x = - 1 * bounds; x < bounds; x += step){
    for (float y = - 1 * bounds; y < bounds; y += step){
      float z  = point->host.get()[0].z - ((normal->host.get()[0].x * ( (float) x - point->host.get()[0].x)) + (normal->host.get()[0].y * ( (float) y - point->host.get()[0].y))) / normal->host.get()[0].z;
      vertices->host.get()[index] = {x, y, z};
      index++;
    }
  }
  // generate the example faces
  // uses quadrilateral encoding
  int side    = (int) sqrt(vertices->size());
  int faceNum = 4 * (side - 1)*(side - 1); // number of vertex indices stored for faces
  index   = 0;
  ssrlcv::ptr::value<ssrlcv::Unity<int>> faces = ssrlcv::ptr::value<ssrlcv::Unity<int>>(nullptr, (size_t) faceNum ,ssrlcv::cpu);
  for (int x = 0; x < side; x++){
    for (int y = 0; y < side; y++){
      if (!(x == (side - 1) || y == (side - 1))) { // avoid the side "lines" of the plane we don't need to do those
        // curl around per the standard
        int location = (x * side) + y; // the refrence location
        faces->host.get()[index    ] = location    ; // top left
        faces->host.get()[index + 1] = location + 1; // top right
        faces->host.get()[index + 2] = location + side + 1; // bottom right
        faces->host.get()[index + 3] = location + side    ; // bottom left
        index += 4;
      }
    }
  }
  // save the output mesh
  ssrlcv::writePLY("estimatedPlane", vertices, faces, 4);
}

/**
* This function is used to test bundle adjustment by adding a bit of noise to the input data
* it saves an initial point cloud, final point cloud, and a CSV of errors over the iterations
* @param matchSet a group of matches
* @param a group of images, used only for their stored camera parameters
* @param iterations the max number of iterations bundle adjustment should do
* @param sigmas a list of float values representing noise to be added to orientaions and rotations
*/
ssrlcv::ptr::value<ssrlcv::Unity<float3>> ssrlcv::PointCloudFactory::testBundleAdjustmentTwoView(ssrlcv::MatchSet* matchSet, std::vector<ssrlcv::ptr::value<ssrlcv::Image>> images, unsigned int iterations, ssrlcv::ptr::value<ssrlcv::Unity<float>> noise){
  std::cout << "\t Running a 2 view bundle adjustment nosie test " << std::endl;

  bool local_debug = true;
  float linearError;

  // used to store the best params found in the optimization
  std::vector<ssrlcv::ptr::value<ssrlcv::Image>> noisey;
  for (int i = 0; i < images.size(); i++){
    ssrlcv::ptr::value<ssrlcv::Image> t;
    t.construct();
    t->camera = images[i]->camera;
    noisey.push_back(t); // fill in the initial images
  }

  // now add the noise to the
  if (noise->size() < 6) {
    std::cerr << "ERROR: noise array needs to have 6 elements!" << std::endl;
    return nullptr;
  } else {
    if (local_debug) std::cout << "Adding noise to image 1" << std::endl;
    noisey[1]->camera.cam_pos.x += noise->host.get()[0];
    noisey[1]->camera.cam_pos.y += noise->host.get()[1];
    noisey[1]->camera.cam_pos.z += noise->host.get()[2];
    noisey[1]->camera.cam_rot.x += noise->host.get()[3];
    noisey[1]->camera.cam_rot.y += noise->host.get()[4];
    noisey[1]->camera.cam_rot.z += noise->host.get()[5];
  }

  ssrlcv::ptr::value<Unity<float3>> points;
  BundleSet bundleSet;

  // save the noisey point cloud
  bundleSet = generateBundles(matchSet,noisey);
  points = twoViewTriangulate(bundleSet, &linearError);

  if (local_debug) std::cout << "Initial Error with noise: " << linearError << std::endl;

  // pre=BA noisey boi
  saveDebugCloud(points, bundleSet, noisey, "preBA");

  // attempt to coorect with BA!
  points = BundleAdjustTwoView(matchSet, noisey, iterations, "testingBA");
  saveDebugCloud(points, bundleSet, noisey, "postBA");

  ssrlcv::ptr::value<ssrlcv::Unity<float>> diff1 = images[0]->getExtrinsicDifference(images[1]->camera);
  ssrlcv::ptr::value<ssrlcv::Unity<float>> diff2 = noisey[0]->getExtrinsicDifference(noisey[1]->camera);

  std::cout << std::endl << "Goal:" << std::endl;
  for (int i = 0; i < diff1->size(); i++){
    std::cout << diff1->host.get()[i] << "  ";
  }
  std::cout << std::endl << "Result:" << std::endl;
  for (int i = 0; i < diff2->size(); i++){
    std::cout << diff2->host.get()[i] << "  ";
  }

  return points;
}

/**
* This function is used to test bundle adjustment by adding a bit of noise to the input data
* it saves an initial point cloud, final point cloud, and a CSV of errors over the iterations
* @param matchSet a group of matches
* @param a group of images, used only for their stored camera parameters
* @param iterations the max number of iterations bundle adjustment should do
* @param noise a list of sigma values to +/- from ranomly
* @param testNum the number of tests to perform
* @param sigma is the sigma to ranomize from the given noise params
*/
void ssrlcv::PointCloudFactory::testBundleAdjustmentTwoView(MatchSet* matchSet, std::vector<ssrlcv::ptr::value<ssrlcv::Image>> images, unsigned int iterations, ssrlcv::ptr::value<ssrlcv::Unity<float>> noise, int testNum){
  std::cout << "\t Running a 2 view bundle adjustment nosie test " << std::endl;

  bool local_debug = true;
  float linearError;

  // used to store the best params found in the optimization
  std::vector<ssrlcv::ptr::value<ssrlcv::Image>> noisey;
  for (int i = 0; i < images.size(); i++){
    ssrlcv::ptr::value<ssrlcv::Image> t;
    t.construct();
    t->camera = images[i]->camera;
    noisey.push_back(t); // fill in the initial images
  }

  // set the random boiz!
  std::default_random_engine generator;
  std::normal_distribution<float> pos_distribution_x(0.0,noise->host.get()[0]);
  std::normal_distribution<float> pos_distribution_y(0.0,noise->host.get()[1]);
  std::normal_distribution<float> pos_distribution_z(0.0,noise->host.get()[2]);
  std::normal_distribution<float> rot_distribution_rx(0.0,noise->host.get()[3]);
  std::normal_distribution<float> rot_distribution_ry(0.0,noise->host.get()[4]);
  std::normal_distribution<float> rot_distribution_rz(0.0,noise->host.get()[5]);

  // loop this boi
  for (int i = 0; i < testNum; i++) {
    std::cout << std::endl;
    std::cout << "====================================================================================" << std::endl;
    std::cout << "      Test: " << i << "/" << testNum << std::endl;
    std::cout << "====================================================================================" << std::endl;
    // now add the noise to the
    if (noise->size() < 6) {
      std::cerr << "ERROR: noise array needs to have 6 elements!" << std::endl;
      return;
    } else {
      if (local_debug) std::cout << "Adding noise to image 1" << std::endl;
      noisey[1]->camera.cam_pos.x += pos_distribution_x(generator);
      noisey[1]->camera.cam_pos.y += pos_distribution_y(generator);
      noisey[1]->camera.cam_pos.z += pos_distribution_z(generator);
      noisey[1]->camera.cam_rot.x += rot_distribution_rx(generator);
      noisey[1]->camera.cam_rot.y += rot_distribution_ry(generator);
      noisey[1]->camera.cam_rot.z += rot_distribution_rz(generator);
    }

    ssrlcv::ptr::value<Unity<float3>> points;
    BundleSet bundleSet;

    // save the noisey point cloud
    bundleSet = generateBundles(matchSet,noisey);
    points = twoViewTriangulate(bundleSet, &linearError);

    if (local_debug) std::cout << "Initial Error with noise: " << linearError << std::endl;

    // pre=BA noisey boi
    saveDebugCloud(points, bundleSet, noisey, "preBA");

    // attempt to coorect with BA!
    BundleAdjustTwoView(matchSet, noisey, iterations, std::to_string(i).c_str());
    saveDebugCloud(points, bundleSet, noisey, "postBA");

    ssrlcv::ptr::value<ssrlcv::Unity<float>> diff1 = images[0]->getExtrinsicDifference(images[1]->camera);
    ssrlcv::ptr::value<ssrlcv::Unity<float>> diff2 = noisey[0]->getExtrinsicDifference(noisey[1]->camera);

    std::cout << std::endl << "Goal:" << std::endl;
    for (int i = 0; i < diff1->size(); i++){
      std::cout << diff1->host.get()[i] << "  ";
    }
    std::cout << std::endl << "Result:" << std::endl;
    for (int i = 0; i < diff2->size(); i++){
      std::cout << diff2->host.get()[i] << "  ";
    }
    std::cout << std::endl << "Difference:" << std::endl;
    for (int i = 0; i < diff2->size(); i++){
      std::cout << abs( diff1->host.get()[i] - diff2->host.get()[i] ) << "  ";
    }



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
void ssrlcv::PointCloudFactory::deterministicStatisticalFilter(ssrlcv::MatchSet* matchSet, std::vector<ssrlcv::ptr::value<ssrlcv::Image>> images, float sigma, float sampleSize){
  if (sampleSize > 1.0 || sampleSize < 0.0) {
    std::cerr << "ERROR:  no statistical filtering possible with percentage greater than 1.0 or less than 0.0" << std::endl;
    return;
  }
  // find an integer skip that can be used
  int sampleJump = (int) (1/sampleSize); // need a constant jump

  // the initial linear error
  float linearError;
  linearError = 0.0; // just something to start
  // the cutoff
  float linearErrorCutoff;
  linearErrorCutoff = 0.0; // just somethihng to start

  // for the N view case:
  ssrlcv::ptr::device<float> lowCut(0.0);
  ssrlcv::ptr::device<float> highCut(0.0);

  // the boiz
  ssrlcv::BundleSet bundleSet;
  ssrlcv::MatchSet tempMatchSet;
  tempMatchSet.keyPoints = ssrlcv::ptr::value<ssrlcv::Unity<ssrlcv::KeyPoint>>(nullptr,1,ssrlcv::cpu);
  tempMatchSet.matches   = ssrlcv::ptr::value<ssrlcv::Unity<ssrlcv::MultiMatch>>(nullptr,1,ssrlcv::cpu);
  ssrlcv::ptr::value<ssrlcv::Unity<float>>    errors;
  ssrlcv::ptr::value<ssrlcv::Unity<float>>    errors_sample;
  ssrlcv::ptr::value<ssrlcv::Unity<float3>>   points;

  // need bundles
  bundleSet = generateBundles(matchSet,images);
  // do an initial triangulation
  errors = ssrlcv::ptr::value<ssrlcv::Unity<float>>(nullptr,matchSet->matches->size(),ssrlcv::cpu);

  std::cout << "Starting Determinstic Statistical Filter ..." << std::endl;

  // do an initial triangulate

  if (images.size() == 2){
    //
    // This is the 2-View case
    //

    points = twoViewTriangulate(bundleSet, errors, &linearError, &linearErrorCutoff);
  } else {
    //
    // This is the N-View case
    //

    points = nViewTriangulate(bundleSet, errors, &linearError, &linearErrorCutoff);
  }

  // the assumption is that choosing every ""stableJump"" indexes is random enough
  // https://en.wikipedia.org/wiki/Variance#Sample_variance
  size_t sample_size = (int) (errors->size() - (errors->size()%sampleJump))/sampleJump; // make sure divisible by the stableJump int always
  errors_sample      = ssrlcv::ptr::value<ssrlcv::Unity<float>>(nullptr,sample_size,ssrlcv::cpu);
  float sample_sum   = 0;
  for (int k = 0; k < sample_size; k++){
    errors_sample->host.get()[k] = errors->host.get()[k*sampleJump];
    sample_sum += errors->host.get()[k*sampleJump];
  }
  float sample_mean = sample_sum / errors_sample->size();
  std::cout << "\tSample Sum: " << std::setprecision(32) << sample_sum << std::endl;
  std::cout << "\tSample Mean: " << std::setprecision(32) << sample_mean << std::endl;
  float squared_sum = 0;
  for (int k = 0; k < sample_size; k++){
    squared_sum += (errors_sample->host.get()[k] - sample_mean)*(errors_sample->host.get()[k] - sample_mean);
  }
  float variance = squared_sum / errors_sample->size();

  //
  // 2-view has an optimum of 0, assumes 0 is the average
  //
  std::cout << "\tSample variance: " << std::setprecision(32) << variance << std::endl;
  std::cout << "\tSigma Calculated As: " << std::setprecision(32) << sqrtf(variance) << std::endl;
  std::cout << "\tLinear Error Cutoff Adjusted To: " << std::setprecision(32) << sigma * sqrtf(variance) << std::endl;
  linearErrorCutoff = sigma * sqrtf(variance);

  // do the two view version of this (easier for now)
  if (images.size() == 2){
    //
    // This is the 2-View case
    //

    // recalculate with new cutoff
    points = twoViewTriangulate(bundleSet, errors, &linearError, &linearErrorCutoff);

    // CLEAR OUT THE DATA STRUCTURES
    // count the number of bad bundles to be removed
    int bad_bundles = 0;
    for (int k = 0; k < bundleSet.bundles->size(); k++){
      if (bundleSet.bundles->host.get()[k].invalid){
         bad_bundles++;
      }
    }
    if (bad_bundles) std::cout << "\tDetected " << bad_bundles << " bad bundles to remove" << std::endl;
    // Need to generated and adjustment match set
    // make a temporary match set
    tempMatchSet.keyPoints = ssrlcv::ptr::value<ssrlcv::Unity<ssrlcv::KeyPoint>>(nullptr,matchSet->matches->size()*2,ssrlcv::cpu);
    tempMatchSet.matches   = ssrlcv::ptr::value<ssrlcv::Unity<ssrlcv::MultiMatch>>(nullptr,matchSet->matches->size(),ssrlcv::cpu);
    // fill in the boiz
    for (int k = 0; k < tempMatchSet.keyPoints->size(); k++){
      tempMatchSet.keyPoints->host.get()[k] = matchSet->keyPoints->host.get()[k];
    }
    for (int k = 0; k < tempMatchSet.matches->size(); k++){
      tempMatchSet.matches->host.get()[k] = matchSet->matches->host.get()[k];
    }
    if (!(matchSet->matches->size() - bad_bundles)){
      std::cerr << "ERROR: filtering is too aggressive, all points would be removed ..." << std::endl;
      return;
    }
    // resize the standard matchSet
    size_t new_kp_size = 2*(matchSet->matches->size() - bad_bundles);
    size_t new_mt_size = matchSet->matches->size() - bad_bundles;
    matchSet->keyPoints = ssrlcv::ptr::value<ssrlcv::Unity<ssrlcv::KeyPoint>>(nullptr,new_kp_size,ssrlcv::cpu);
    matchSet->matches   = ssrlcv::ptr::value<ssrlcv::Unity<ssrlcv::MultiMatch>>(nullptr,new_mt_size,ssrlcv::cpu);
    // this is much easier because of the 2 view assumption
    // there are the same number of lines as there are are keypoints and the same number of bundles as there are matches
    int k_adjust = 0;
    for (int k = 0; k < bundleSet.bundles->size(); k++){
    	if (!bundleSet.bundles->host.get()[k].invalid){
    	  matchSet->keyPoints->host.get()[2*k_adjust]     = tempMatchSet.keyPoints->host.get()[2*k];
    	  matchSet->keyPoints->host.get()[2*k_adjust + 1] = tempMatchSet.keyPoints->host.get()[2*k + 1];
        matchSet->matches->host.get()[k_adjust]         = {2,2*k_adjust};
    	  k_adjust++;
    	}
    }
    if (bad_bundles) std::cout << "\tRemoved bad bundles" << std::endl;
  } else {
    //
    // This is the N-view case
    //

    // recalculate with new cutoff
    points = nViewTriangulate(bundleSet, errors, &linearError, &linearErrorCutoff);

    // CLEAR OUT THE DATA STRUCTURES
    // count the number of bad bundles to be removed
    int bad_bundles = 0;
    int bad_lines   = 0;
    for (int k = 0; k < bundleSet.bundles->size(); k++){
      if (bundleSet.bundles->host.get()[k].invalid){
         bad_bundles++;
         bad_lines += bundleSet.bundles->host.get()[k].numLines;
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
    tempMatchSet.keyPoints = ssrlcv::ptr::value<ssrlcv::Unity<ssrlcv::KeyPoint>>(nullptr,matchSet->keyPoints->size(),ssrlcv::cpu);
    tempMatchSet.matches   = ssrlcv::ptr::value<ssrlcv::Unity<ssrlcv::MultiMatch>>(nullptr,matchSet->matches->size(),ssrlcv::cpu);
    // fill in the boiz
    for (int k = 0; k < tempMatchSet.keyPoints->size(); k++){
      tempMatchSet.keyPoints->host.get()[k] = matchSet->keyPoints->host.get()[k];
    }
    for (int k = 0; k < tempMatchSet.matches->size(); k++){
      tempMatchSet.matches->host.get()[k] = matchSet->matches->host.get()[k];
    }
    if (!(matchSet->matches->size() - bad_bundles) || !(matchSet->keyPoints->size() - bad_lines)){
      std::cerr << "ERROR: filtering is too aggressive, all points would be removed ..." << std::endl;
      return;
    }
    // resize the standard matchSet
    size_t new_kp_size = matchSet->keyPoints->size() - bad_lines;
    size_t new_mt_size = matchSet->matches->size() - bad_bundles;
    matchSet->keyPoints = ssrlcv::ptr::value<ssrlcv::Unity<ssrlcv::KeyPoint>>(nullptr,new_kp_size,ssrlcv::cpu);
    matchSet->matches   = ssrlcv::ptr::value<ssrlcv::Unity<ssrlcv::MultiMatch>>(nullptr,new_mt_size,ssrlcv::cpu);
    // this is harder to do with the N-view case
    unsigned int k_adjust = 0;
    unsigned int k_lines  = 0;
    int k_bundle = 0;
    int k_keypnt = 0;
    for (int k = 0; k < bundleSet.bundles->size(); k++){
      k_lines = bundleSet.bundles->host.get()[k].numLines;
      if (!bundleSet.bundles->host.get()[k].invalid){
        matchSet->matches->host.get()[k_bundle] = {k_lines,k_adjust};
        for (int j = 0; j < k_lines; j++){
          matchSet->keyPoints->host.get()[k_adjust + j] = tempMatchSet.keyPoints->host.get()[k_keypnt + j];
        }
        k_adjust += k_lines;
        k_bundle++;
      }
      k_keypnt += k_lines;
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
void ssrlcv::PointCloudFactory::nonDeterministicStatisticalFilter(ssrlcv::MatchSet* matchSet, std::vector<ssrlcv::ptr::value<ssrlcv::Image>> images, float sigma, float sampleSize){
  if (sampleSize > 1.0 || sampleSize < 0.0) {
    std::cerr << "ERROR:  no statistical filtering possible with percentage greater than 1.0 or less than 0.0" << std::endl;
    return;
  }

  // the initial linear error
  float linearError;
  linearError = 0.0; // just something to start
  // the cutoff
  float linearErrorCutoff;
  linearErrorCutoff = 0.0; // just somethihng to start

  // the boiz
  ssrlcv::BundleSet bundleSet;
  ssrlcv::MatchSet tempMatchSet;
  tempMatchSet.keyPoints = ssrlcv::ptr::value<ssrlcv::Unity<ssrlcv::KeyPoint>>(nullptr,1,ssrlcv::cpu);
  tempMatchSet.matches   = ssrlcv::ptr::value<ssrlcv::Unity<ssrlcv::MultiMatch>>(nullptr,1,ssrlcv::cpu);
  ssrlcv::ptr::value<ssrlcv::Unity<float>>    errors;
  ssrlcv::ptr::value<ssrlcv::Unity<float>>    errors_sample;
  ssrlcv::ptr::value<ssrlcv::Unity<float3>>   points;

  // need bundles
  bundleSet = generateBundles(matchSet,images);
  // do an initial triangulation
  errors = ssrlcv::ptr::value<ssrlcv::Unity<float>>(nullptr,matchSet->matches->size(),ssrlcv::cpu);

  std::cout << "Starting Determinstic Statistical Filter ..." << std::endl;

  // do an initial triangulate

  if (images.size() == 2){
    //
    // This is the 2-View case
    //

    points = twoViewTriangulate(bundleSet, errors, &linearError, &linearErrorCutoff);
  } else {
    //
    // This is the N-View case
    //

    points = nViewTriangulate(bundleSet, errors, &linearError, &linearErrorCutoff);
  }

  // https://en.wikipedia.org/wiki/Variance#Sample_variance

  size_t sample_size = (int) errors->size() * sampleSize;
  errors_sample      = ssrlcv::ptr::value<ssrlcv::Unity<float>>(nullptr,sample_size,ssrlcv::cpu);
  float sample_sum   = 0;
  // fill in indices to do a random shuffle
  // code snippet from: https://en.cppreference.com/w/cpp/algorithm/random_shuffle
  std::srand ( unsigned ( std::time(0) ) );
  std::vector<int> indexes;
  std::random_device rd;
  std::mt19937 g(rd());
  for (int i = 0; i < sample_size; i++) indexes.push_back(i);
  std::shuffle(indexes.begin(), indexes.end(), g);
  for (int i = 0; i < sample_size; i++){
    // set the random indices to the sample
    errors_sample->host.get()[i] = errors->host.get()[indexes[i]];
    sample_sum += errors->host.get()[indexes[i]];
  }
  indexes.clear();

  float sample_mean = sample_sum / errors_sample->size();
  std::cout << "\tSample Sum: " << std::setprecision(32) << sample_sum << std::endl;
  std::cout << "\tSample Mean: " << std::setprecision(32) << sample_mean << std::endl;
  float squared_sum = 0;
  for (int k = 0; k < sample_size; k++){
    squared_sum += (errors_sample->host.get()[k] - sample_mean)*(errors_sample->host.get()[k] - sample_mean);
  }
  float variance = squared_sum / errors_sample->size();
  std::cout << "\tSample variance: " << std::setprecision(32) << variance << std::endl;
  std::cout << "\tSigma Calculated As: " << std::setprecision(32) << sqrtf(variance) << std::endl;
  std::cout << "\tLinear Error Cutoff Adjusted To: " << std::setprecision(32) << sigma * sqrtf(variance) << std::endl;
  linearErrorCutoff = sigma * sqrtf(variance);

  // do the two view version of this (easier for now)
  if (images.size() == 2){
    //
    // This is the 2-View case
    //

    // recalculate with new cutoff
    points = twoViewTriangulate(bundleSet, errors, &linearError, &linearErrorCutoff);

    // CLEAR OUT THE DATA STRUCTURES
    // count the number of bad bundles to be removed
    int bad_bundles = 0;
    for (int k = 0; k < bundleSet.bundles->size(); k++){
      if (bundleSet.bundles->host.get()[k].invalid){
         bad_bundles++;
      }
    }
    if (bad_bundles) std::cout << "\tDetected " << bad_bundles << " bad bundles to remove" << std::endl;
    // Need to generated and adjustment match set
    // make a temporary match set
    tempMatchSet.keyPoints = ssrlcv::ptr::value<ssrlcv::Unity<ssrlcv::KeyPoint>>(nullptr,matchSet->matches->size()*2,ssrlcv::cpu);
    tempMatchSet.matches   = ssrlcv::ptr::value<ssrlcv::Unity<ssrlcv::MultiMatch>>(nullptr,matchSet->matches->size(),ssrlcv::cpu);
    // fill in the boiz
    for (int k = 0; k < tempMatchSet.keyPoints->size(); k++){
      tempMatchSet.keyPoints->host.get()[k] = matchSet->keyPoints->host.get()[k];
    }
    for (int k = 0; k < tempMatchSet.matches->size(); k++){
      tempMatchSet.matches->host.get()[k] = matchSet->matches->host.get()[k];
    }
    if (!(matchSet->matches->size() - bad_bundles)){
      std::cerr << "ERROR: filtering is too aggressive, all points would be removed ..." << std::endl;
      return;
    }
    // resize the standard matchSet
    size_t new_kp_size = 2*(matchSet->matches->size() - bad_bundles);
    size_t new_mt_size = matchSet->matches->size() - bad_bundles;
    matchSet->keyPoints = ssrlcv::ptr::value<ssrlcv::Unity<ssrlcv::KeyPoint>>(nullptr,new_kp_size,ssrlcv::cpu);
    matchSet->matches   = ssrlcv::ptr::value<ssrlcv::Unity<ssrlcv::MultiMatch>>(nullptr,new_mt_size,ssrlcv::cpu);
    // this is much easier because of the 2 view assumption
    // there are the same number of lines as there are are keypoints and the same number of bundles as there are matches
    int k_adjust = 0;
    // if (bad_bundles){
    for (int k = 0; k < bundleSet.bundles->size(); k++){
    	if (!bundleSet.bundles->host.get()[k].invalid){
    	  matchSet->keyPoints->host.get()[2*k_adjust]     = tempMatchSet.keyPoints->host.get()[2*k];
    	  matchSet->keyPoints->host.get()[2*k_adjust + 1] = tempMatchSet.keyPoints->host.get()[2*k + 1];
        matchSet->matches->host.get()[k_adjust]         = {2,2*k_adjust};
    	  k_adjust++;
    	}
    }
    if (bad_bundles) std::cout << "\tRemoved bad bundles" << std::endl;
  } else {
    //
    // This is the N-view case
    //

    // recalculate with new cutoff
    points = nViewTriangulate(bundleSet, errors, &linearError, &linearErrorCutoff);

    // CLEAR OUT THE DATA STRUCTURES
    // count the number of bad bundles to be removed
    int bad_bundles = 0;
    int bad_lines   = 0;
    for (int k = 0; k < bundleSet.bundles->size(); k++){
      if (bundleSet.bundles->host.get()[k].invalid){
         bad_bundles++;
         bad_lines += bundleSet.bundles->host.get()[k].numLines;
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
    tempMatchSet.keyPoints = ssrlcv::ptr::value<ssrlcv::Unity<ssrlcv::KeyPoint>>(nullptr,matchSet->keyPoints->size(),ssrlcv::cpu);
    tempMatchSet.matches   = ssrlcv::ptr::value<ssrlcv::Unity<ssrlcv::MultiMatch>>(nullptr,matchSet->matches->size(),ssrlcv::cpu);
    // fill in the boiz
    for (int k = 0; k < tempMatchSet.keyPoints->size(); k++){
      tempMatchSet.keyPoints->host.get()[k] = matchSet->keyPoints->host.get()[k];
    }
    for (int k = 0; k < tempMatchSet.matches->size(); k++){
      tempMatchSet.matches->host.get()[k] = matchSet->matches->host.get()[k];
    }
    if (!(matchSet->matches->size() - bad_bundles) || !(matchSet->keyPoints->size() - bad_lines)){
      std::cerr << "ERROR: filtering is too aggressive, all points would be removed ..." << std::endl;
      return;
    }
    // resize the standard matchSet
    size_t new_kp_size = matchSet->keyPoints->size() - bad_lines;
    size_t new_mt_size = matchSet->matches->size() - bad_bundles;
    matchSet->keyPoints = ssrlcv::ptr::value<ssrlcv::Unity<ssrlcv::KeyPoint>>(nullptr,new_kp_size,ssrlcv::cpu);
    matchSet->matches   = ssrlcv::ptr::value<ssrlcv::Unity<ssrlcv::MultiMatch>>(nullptr,new_mt_size,ssrlcv::cpu);
    // this is harder to do with the N-view case
    unsigned int k_adjust = 0;
    unsigned int k_lines  = 0;
    int k_bundle = 0;
    int k_keypnt = 0;
    for (int k = 0; k < bundleSet.bundles->size(); k++){
      k_lines = bundleSet.bundles->host.get()[k].numLines;
      if (!bundleSet.bundles->host.get()[k].invalid){
        matchSet->matches->host.get()[k_bundle] = {k_lines,k_adjust};
        for (int j = 0; j < k_lines; j++){
          matchSet->keyPoints->host.get()[k_adjust + j] = tempMatchSet.keyPoints->host.get()[k_keypnt + j];
        }
        k_adjust += k_lines;
        k_bundle++;
      }
      k_keypnt += k_lines;
    }

    if (bad_bundles) std::cout << "\tRemoved bundles" << std::endl;
  }
}

/**
 * A filter that removes all points with a linear error greater than the cutoff. Modifies the matchSet that is pass thru
 * @param matchSet a group of matches
 * @param images a group of images, used only for their stored camera parameters
 * @param cutoff the float that no linear errors should be greater than
 */
void ssrlcv::PointCloudFactory::linearCutoffFilter(ssrlcv::MatchSet* matchSet, std::vector<ssrlcv::ptr::value<ssrlcv::Image>> images, float cutoff){
  if (cutoff < 0.0){
    std::cerr << "ERROR: cutoff must be positive" << std::endl;
    return;
  }

  // the initial linear error
  float linearError;
  linearError = 0.0; // just something to start
  // the cutoff
  float linearErrorCutoff;
  linearErrorCutoff = cutoff; // just somethihng to start

  // the boiz
  ssrlcv::BundleSet        bundleSet;
  ssrlcv::ptr::value<ssrlcv::Unity<float3>>   points;
  ssrlcv::ptr::value<ssrlcv::Unity<float>>    errors;
  ssrlcv::MatchSet         tempMatchSet;

  // need bundles
  bundleSet = generateBundles(matchSet,images);

  errors = ssrlcv::ptr::value<ssrlcv::Unity<float>>(nullptr,matchSet->matches->size(),ssrlcv::cpu);

  // do the two view version of this (easier for now)
  if (images.size() == 2){
    //
    // This is the 2-View case
    //

    // recalculate with new cutoff
    points = twoViewTriangulate(bundleSet, errors, &linearError, &linearErrorCutoff);

    // CLEAR OUT THE DATA STRUCTURES
    // count the number of bad bundles to be removed
    int bad_bundles = 0;
    for (int k = 0; k < bundleSet.bundles->size(); k++){
      if (bundleSet.bundles->host.get()[k].invalid){
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
    tempMatchSet.keyPoints = ssrlcv::ptr::value<ssrlcv::Unity<ssrlcv::KeyPoint>>(nullptr,matchSet->matches->size()*2,ssrlcv::cpu);
    tempMatchSet.matches   = ssrlcv::ptr::value<ssrlcv::Unity<ssrlcv::MultiMatch>>(nullptr,matchSet->matches->size(),ssrlcv::cpu);
    // fill in the boiz
    for (int k = 0; k < tempMatchSet.keyPoints->size(); k++){
      tempMatchSet.keyPoints->host.get()[k] = matchSet->keyPoints->host.get()[k];
    }
    for (int k = 0; k < tempMatchSet.matches->size(); k++){
      tempMatchSet.matches->host.get()[k] = matchSet->matches->host.get()[k];
    }
    if (!(matchSet->matches->size() - bad_bundles)){
      std::cerr << "ERROR: filtering is too aggressive, all points would be removed ..." << std::endl;
      return;
    }
    // resize the standard matchSet
    size_t new_kp_size = 2*(matchSet->matches->size() - bad_bundles);
    size_t new_mt_size = matchSet->matches->size() - bad_bundles;
    matchSet->keyPoints = ssrlcv::ptr::value<ssrlcv::Unity<ssrlcv::KeyPoint>>(nullptr,new_kp_size,ssrlcv::cpu);
    matchSet->matches   = ssrlcv::ptr::value<ssrlcv::Unity<ssrlcv::MultiMatch>>(nullptr,new_mt_size,ssrlcv::cpu);
    // this is much easier because of the 2 view assumption
    // there are the same number of lines as there are are keypoints and the same number of bundles as there are matches
    int k_adjust = 0;
    for (int k = 0; k < bundleSet.bundles->size(); k++){
    	if (!bundleSet.bundles->host.get()[k].invalid){
    	  matchSet->keyPoints->host.get()[2*k_adjust]     = tempMatchSet.keyPoints->host.get()[2*k];
    	  matchSet->keyPoints->host.get()[2*k_adjust + 1] = tempMatchSet.keyPoints->host.get()[2*k + 1];
        matchSet->matches->host.get()[k_adjust]         = {2,2*k_adjust};
    	  k_adjust++;
    	}
    }
    if (bad_bundles) std::cout << "\tRemoved bundles" << std::endl;
  } else {
    //
    // This is the N-view case
    //

    // recalculate with new cutoff
    points = nViewTriangulate(bundleSet, errors, &linearError, &linearErrorCutoff);
    // CLEAR OUT THE DATA STRUCTURES
    // count the number of bad bundles to be removed
    int bad_bundles = 0;
    int bad_lines   = 0;
    for (int k = 0; k < bundleSet.bundles->size(); k++){
      if (bundleSet.bundles->host.get()[k].invalid){
         bad_bundles++;
         bad_lines += bundleSet.bundles->host.get()[k].numLines;
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
    tempMatchSet.keyPoints = ssrlcv::ptr::value<ssrlcv::Unity<ssrlcv::KeyPoint>>(nullptr,matchSet->keyPoints->size(),ssrlcv::cpu);
    tempMatchSet.matches   = ssrlcv::ptr::value<ssrlcv::Unity<ssrlcv::MultiMatch>>(nullptr,matchSet->matches->size(),ssrlcv::cpu);
    // fill in the boiz
    for (int k = 0; k < tempMatchSet.keyPoints->size(); k++){
      tempMatchSet.keyPoints->host.get()[k] = matchSet->keyPoints->host.get()[k];
    }
    for (int k = 0; k < tempMatchSet.matches->size(); k++){
      tempMatchSet.matches->host.get()[k] = matchSet->matches->host.get()[k];
    }
    if (!(matchSet->matches->size() - bad_bundles) || !(matchSet->keyPoints->size() - bad_lines)){
      std::cerr << "ERROR: filtering is too aggressive, all points would be removed ..." << std::endl;
      return;
    }
    // resize the standard matchSet
    size_t new_kp_size = matchSet->keyPoints->size() - bad_lines;
    size_t new_mt_size = matchSet->matches->size() - bad_bundles;
    matchSet->keyPoints = ssrlcv::ptr::value<ssrlcv::Unity<ssrlcv::KeyPoint>>(nullptr,new_kp_size,ssrlcv::cpu);
    matchSet->matches   = ssrlcv::ptr::value<ssrlcv::Unity<ssrlcv::MultiMatch>>(nullptr,new_mt_size,ssrlcv::cpu);
    // this is harder to do with the N-view case
    unsigned int k_adjust = 0;
    unsigned int k_lines  = 0;
    int k_bundle = 0;
    int k_keypnt = 0;
    for (int k = 0; k < bundleSet.bundles->size(); k++){
      k_lines = bundleSet.bundles->host.get()[k].numLines;
      if (!bundleSet.bundles->host.get()[k].invalid){
        matchSet->matches->host.get()[k_bundle] = {k_lines,k_adjust};
        for (int j = 0; j < k_lines; j++){
          matchSet->keyPoints->host.get()[k_adjust + j] = tempMatchSet.keyPoints->host.get()[k_keypnt + j];
        }
        k_adjust += k_lines;
        k_bundle++;
      }
      k_keypnt += k_lines;
    }

    if (bad_bundles) std::cout << "\tRemoved bundles" << std::endl;
  }
}

/**
 * This method estimates the plane the point cloud sits in and removes points that are outside of a certain
 * threashold distance from the plane. The bad locations are removed from the matchSet
 * @param matchSet a group of matches. this is altered and bad locations are removed from here
 * @param images a group of images, used only for their stored camera parameters
 * @param cutoff is a cutoff of +/- km distance from the plane, if the point cloud has been scaled then this should also be scaled
 */
void ssrlcv::PointCloudFactory::planarCutoffFilter(ssrlcv::MatchSet* matchSet, std::vector<ssrlcv::ptr::value<ssrlcv::Image>> images, float cutoff){
  // disable for local print statments
  bool local_debug    = true;
  bool local_verbose  = true;

  if (local_debug || local_verbose) std::cout << "Starting Planar Cutoff Filter ..." << std::endl;

  // prep for reconstructon
  ssrlcv::BundleSet bundleSet;
  ssrlcv::ptr::value<ssrlcv::Unity<float3>> points;
  // now filter from the generated plane

  bundleSet = generateBundles(matchSet,images);
  bundleSet.lines->transferMemoryTo(gpu);
  bundleSet.bundles->transferMemoryTo(gpu);

  // get an initial point cloud for plane estimation
  if (images.size() == 2){
    //
    // 2 view case
    //
    points = twoViewTriangulate(bundleSet);
  } else {
    //
    // N view case
    //
    points = nViewTriangulate(bundleSet);
  }

  bundleSet.lines->transferMemoryTo(cpu);
  bundleSet.bundles->transferMemoryTo(cpu);

  // estimate a plane from the generated point cloud

  // extract the camera locations for normal estimation
  ssrlcv::ptr::value<ssrlcv::Unity<float3>> locations = ssrlcv::ptr::value<ssrlcv::Unity<float3>>(nullptr,images.size(),ssrlcv::cpu);
  for (int i = 0; i < images.size(); i++){
    locations->host.get()[i].x = images[i]->camera.cam_pos.x;
    locations->host.get()[i].y = images[i]->camera.cam_pos.y;
    locations->host.get()[i].z = images[i]->camera.cam_pos.z;
  }

  // create the octree
  Octree oct = Octree(points, 8, false);
  // caclulate the estimated plane normal
  ssrlcv::ptr::value<Unity<float3>> normal = oct.computeAverageNormal(3, 10, images.size(), locations->host.get());
  // the averate point
  ssrlcv::ptr::value<Unity<float3>> averagePoint = getAveragePoint(points);

  if (local_debug) {
    std::cout << "Average Point: ( " << averagePoint->host.get()[0].x << ", " << averagePoint->host.get()[0].y << ", " << averagePoint->host.get()[0].z << " )" << std::endl;
    std::cout << "Normal Vector: < " << normal->host.get()[0].x << ", " << normal->host.get()[0].y << ", " << normal->host.get()[0].z << " )" << std::endl;
  }

  normal->transferMemoryTo(gpu);
  averagePoint->transferMemoryTo(gpu);

  bundleSet.lines->transferMemoryTo(gpu);
  bundleSet.bundles->transferMemoryTo(gpu);

  // move the cutoff to the device
  float* d_cutoff;
  CudaSafeCall(cudaMalloc((void**) &d_cutoff,sizeof(float)));
  CudaSafeCall(cudaMemcpy(d_cutoff,&cutoff,sizeof(float),cudaMemcpyHostToDevice));

  dim3 grid = {1,1,1};
  dim3 block = {1,1,1};

  if (images.size() == 2){
    //
    // 2 view case
    //

    void (*fp)(unsigned long, Bundle::Line*, Bundle*, float*, float3*, float3*) = &filterTwoViewFromEstimatedPlane;
    getFlatGridBlock(bundleSet.bundles->size(),grid,block,fp);

    // filterTwoViewFromEstimatedPlane(unsigned long pointnum, Bundle::Line* lines, Bundle* bundles, float* cutoff, float3* point, float3* vector)
    filterTwoViewFromEstimatedPlane<<<grid,block>>>(bundleSet.bundles->size(), bundleSet.lines->device.get(), bundleSet.bundles->device.get(), d_cutoff, averagePoint->device.get(), normal->device.get());

  } else {
    //
    // N view case
    //

    void (*fp)(unsigned long, Bundle::Line*, Bundle*, float*, float3*, float3*) = &filterNViewFromEstimatedPlane;
    getFlatGridBlock(bundleSet.bundles->size(),grid,block,fp);

    // filterNViewFromEstimatedPlane(unsigned long pointnum, Bundle::Line* lines, Bundle* bundles, float* cutoff, float3* point, float3* vector)
    filterNViewFromEstimatedPlane<<<grid,block>>>(bundleSet.bundles->size(), bundleSet.lines->device.get(), bundleSet.bundles->device.get(), d_cutoff, averagePoint->device.get(), normal->device.get());

  }

  cudaDeviceSynchronize();
  CudaCheckError();

  bundleSet.lines->setFore(gpu);
  bundleSet.bundles->setFore(gpu);
  bundleSet.lines->transferMemoryTo(cpu);
  bundleSet.bundles->transferMemoryTo(cpu);

  // to help when modifying the match set
  ssrlcv::MatchSet tempMatchSet;
  tempMatchSet.keyPoints = ssrlcv::ptr::value<ssrlcv::Unity<ssrlcv::KeyPoint>>(nullptr,1,ssrlcv::cpu);
  tempMatchSet.matches   = ssrlcv::ptr::value<ssrlcv::Unity<ssrlcv::MultiMatch>>(nullptr,1,ssrlcv::cpu);

  // CLEAR OUT THE DATA STRUCTURES
  // count the number of bad bundles to be removed
  int bad_bundles = 0;
  int bad_lines   = 0;
  for (int k = 0; k < bundleSet.bundles->size(); k++){
    if (bundleSet.bundles->host.get()[k].invalid){
       bad_bundles++;
       bad_lines += bundleSet.bundles->host.get()[k].numLines;
    }
  }
  if (bad_bundles) {
    std::cout << "\tDetected " << bad_bundles << " bundles to remove" << std::endl;
    std::cout << "\tDetected " << bad_lines << " lines to remove" << std::endl;
  } else {
    std::cout << "No bundles or liens to removed! all points are less than " << cutoff  << " km from estimated plane" << std::endl;
    return;
  }

  // Need to generated and adjustment match set
  // make a temporary match set
  tempMatchSet.keyPoints = ssrlcv::ptr::value<ssrlcv::Unity<ssrlcv::KeyPoint>>(nullptr,matchSet->keyPoints->size(),ssrlcv::cpu);
  tempMatchSet.matches   = ssrlcv::ptr::value<ssrlcv::Unity<ssrlcv::MultiMatch>>(nullptr,matchSet->matches->size(),ssrlcv::cpu);
  // fill in the boiz
  for (int k = 0; k < tempMatchSet.keyPoints->size(); k++){
    tempMatchSet.keyPoints->host.get()[k] = matchSet->keyPoints->host.get()[k];
  }
  for (int k = 0; k < tempMatchSet.matches->size(); k++){
    tempMatchSet.matches->host.get()[k] = matchSet->matches->host.get()[k];
  }
  if (!(matchSet->matches->size() - bad_bundles) || !(matchSet->keyPoints->size() - bad_lines)){
    std::cerr << "ERROR: filtering is too aggressive, all points would be removed ..." << std::endl;
    return;
  }
  // resize the standard matchSet
  size_t new_kp_size = matchSet->keyPoints->size() - bad_lines;
  size_t new_mt_size = matchSet->matches->size() - bad_bundles;
  matchSet->keyPoints = ssrlcv::ptr::value<ssrlcv::Unity<ssrlcv::KeyPoint>>(nullptr,new_kp_size,ssrlcv::cpu);
  matchSet->matches   = ssrlcv::ptr::value<ssrlcv::Unity<ssrlcv::MultiMatch>>(nullptr,new_mt_size,ssrlcv::cpu);
  // this is harder to do with the N-view case
  unsigned int k_adjust = 0;
  unsigned int k_lines  = 0;
  int k_bundle = 0;
  int k_keypnt = 0;
  for (int k = 0; k < bundleSet.bundles->size(); k++){
    k_lines = bundleSet.bundles->host.get()[k].numLines;
    if (!bundleSet.bundles->host.get()[k].invalid){
      matchSet->matches->host.get()[k_bundle] = {k_lines,k_adjust};
      for (int j = 0; j < k_lines; j++){
        matchSet->keyPoints->host.get()[k_adjust + j] = tempMatchSet.keyPoints->host.get()[k_keypnt + j];
      }
      k_adjust += k_lines;
      k_bundle++;
    }
    k_keypnt += k_lines;
  }

  if (bad_bundles) std::cout << "\tRemoved bundles" << std::endl;

  // clean up memory
  cudaFree(d_cutoff);

}

/**
 * removes bundles from the bundleSet that have been flagged as invalid and returns the reduced bundle set
 * @param bundleSet the bundleSet to reduce
 * @return bundleSet the new reduced bundleSet
 */
ssrlcv::BundleSet ssrlcv::PointCloudFactory::reduceBundleSet(BundleSet bundleSet){

  int bad_bundles = 0;
  int bad_lines   = 0;
  for (int k = 0; k < bundleSet.bundles->size(); k++){
    if (bundleSet.bundles->host.get()[k].invalid){
       bad_bundles++;
       bad_lines += bundleSet.bundles->host.get()[k].numLines;
    }
  }

  if (bad_bundles) {
    std::cout << "\tDetected " << bad_bundles << " bad bundles to remove and " << bad_lines << " lines ..." << std::endl;
  } else {
    std::cerr << "ERROR: no bad bundles detected, cannot reduce bundle set" << std::endl;
    return bundleSet;
  }

  // make a temp bundleset
  ssrlcv::ptr::value<ssrlcv::Unity<Bundle>> temp_bundles     = ssrlcv::ptr::value<ssrlcv::Unity<Bundle>>(nullptr,bundleSet.bundles->size(),cpu);
  ssrlcv::ptr::value<ssrlcv::Unity<Bundle::Line>> temp_lines = ssrlcv::ptr::value<ssrlcv::Unity<Bundle::Line>>(nullptr,bundleSet.lines->size(),cpu);

  // copy all over
  for (int i = 0; i < bundleSet.lines->size(); i++) {
    temp_lines->host.get()[i] = bundleSet.lines->host.get()[i];
  }
  for (int i = 0; i < bundleSet.bundles->size(); i++) {
    temp_bundles->host.get()[i] = bundleSet.bundles->host.get()[i];
  }

  // resize the standard matchSet
  size_t new_ln_size = bundleSet.lines->size() - bad_lines;
  size_t new_bd_size = bundleSet.bundles->size() - bad_bundles;
  bundleSet.lines   = ssrlcv::ptr::value<ssrlcv::Unity<Bundle::Line>>(nullptr,new_ln_size,ssrlcv::cpu);
  bundleSet.bundles = ssrlcv::ptr::value<ssrlcv::Unity<Bundle>>(nullptr,new_bd_size,ssrlcv::cpu);

  // fill the new bundle
  int k_adjust = 0;
  int k_bundle = 0;
  for (int k = 0; k < temp_bundles->size(); k++){
    if (!temp_bundles->host.get()[k].invalid){
      bundleSet.bundles->host.get()[k_bundle] = temp_bundles->host.get()[k_bundle];
      for (int i = temp_bundles->host.get()[k].index; i < temp_bundles->host.get()[k].index + temp_bundles->host.get()[k].numLines; i++ ) {
        bundleSet.lines->host.get()[k_adjust] = temp_lines->host.get()[i];
        k_adjust++;
      }
      k_bundle++;
    }
  }
  std::cout << "k_bundle: " << k_bundle << ", " << new_bd_size << std::endl;
  std::cout << "k_adjust: " << k_adjust << ", " << new_ln_size << std::endl;

  return bundleSet;
}

/**
 * reduces the input bundleset my the given statistical sigma value
 * @param bundleSet the bundleSet to reduce
 * @param sigma the statistical cutoff to reduce the bundleSet by
 * @return bundleSet the new reduced bundleSet
 */
ssrlcv::BundleSet ssrlcv::PointCloudFactory::reduceBundleSet(BundleSet bundleSet, float sigma){

  // TODO have an input for this
  int sampleJump = (int) (1/(0.1)); // need a constant jump

  // the initial linear error
  float linearError;
  linearError = 0.0;
  // the cutoff
  float linearErrorCutoff;
  linearErrorCutoff = 0.0;

  ssrlcv::ptr::value<ssrlcv::Unity<float>>  errors;
  ssrlcv::ptr::value<ssrlcv::Unity<float>>  errors_sample;
  ssrlcv::ptr::value<ssrlcv::Unity<float3>> points;

  // need bundles
  // bundleSet = generateBundles(matchSet,images);
  // do an initial triangulation
  errors = ssrlcv::ptr::value<ssrlcv::Unity<float>>(nullptr,bundleSet.bundles->size(),ssrlcv::cpu);

  std::cout << "Starting Bundle Reduciton Statistical Filter ..." << std::endl;

  points = nViewTriangulate(bundleSet, errors, &linearError, &linearErrorCutoff);

  // the assumption is that choosing every ""stableJump"" indexes is random enough
  // https://en.wikipedia.org/wiki/Variance#Sample_variance
  size_t sample_size = (int) (errors->size() - (errors->size()%sampleJump))/sampleJump; // make sure divisible by the stableJump int always
  errors_sample      = ssrlcv::ptr::value<ssrlcv::Unity<float>>(nullptr,sample_size,ssrlcv::cpu);
  float sample_sum   = 0;
  for (int k = 0; k < sample_size; k++){
    errors_sample->host.get()[k] = errors->host.get()[k*sampleJump];
    sample_sum += errors->host.get()[k*sampleJump];
  }
  float sample_mean = sample_sum / errors_sample->size();
  std::cout << "\tSample Sum: " << std::setprecision(32) << sample_sum << std::endl;
  std::cout << "\tSample Mean: " << std::setprecision(32) << sample_mean << std::endl;
  float squared_sum = 0;
  for (int k = 0; k < sample_size; k++){
    squared_sum += (errors_sample->host.get()[k] - sample_mean)*(errors_sample->host.get()[k] - sample_mean);
  }
  float variance = squared_sum / errors_sample->size();

  //
  // 2-view has an optimum of 0, assumes 0 is the average
  //
  std::cout << "\tSample variance: " << std::setprecision(32) << variance << std::endl;
  std::cout << "\tSigma Calculated As: " << std::setprecision(32) << sqrtf(variance) << std::endl;
  std::cout << "\tLinear Error Cutoff Adjusted To: " << std::setprecision(32) << sigma * sqrtf(variance) << std::endl;
  linearErrorCutoff = sigma * sqrtf(variance);

  // set the points to invalid
  points = nViewTriangulate(bundleSet, errors, &linearError, &linearErrorCutoff);

  int bad_bundles = 0;
  int bad_lines   = 0;
  for (int k = 0; k < bundleSet.bundles->size(); k++){
    if (bundleSet.bundles->host.get()[k].invalid){
       bad_bundles++;
       bad_lines += bundleSet.bundles->host.get()[k].numLines;
    }
  }

  if (bad_bundles) std::cout << "\tDetected " << bad_bundles << " bad bundles to remove and " << bad_lines << " lines ..." << std::endl;

  // make a temp bundleset
  ssrlcv::ptr::value<ssrlcv::Unity<Bundle>> temp_bundles     = ssrlcv::ptr::value<ssrlcv::Unity<Bundle>>(nullptr,bundleSet.bundles->size(),cpu);
  ssrlcv::ptr::value<ssrlcv::Unity<Bundle::Line>> temp_lines = ssrlcv::ptr::value<ssrlcv::Unity<Bundle::Line>>(nullptr,bundleSet.lines->size(),cpu);

  // copy all over
  for (int i = 0; i < bundleSet.lines->size(); i++) {
    temp_lines->host.get()[i] = bundleSet.lines->host.get()[i];
  }
  for (int i = 0; i < bundleSet.bundles->size(); i++) {
    temp_bundles->host.get()[i] = bundleSet.bundles->host.get()[i];
  }

  // resize the standard matchSet
  size_t new_ln_size = bundleSet.lines->size() - bad_lines;
  size_t new_bd_size = bundleSet.bundles->size() - bad_bundles;
  bundleSet.lines   = ssrlcv::ptr::value<ssrlcv::Unity<Bundle::Line>>(nullptr,new_ln_size,ssrlcv::cpu);
  bundleSet.bundles = ssrlcv::ptr::value<ssrlcv::Unity<Bundle>>(nullptr,new_bd_size,ssrlcv::cpu);

  // fill the new bundle
  int k_adjust = 0;
  int k_bundle = 0;
  for (int k = 0; k < temp_bundles->size(); k++){
    if (!temp_bundles->host.get()[k].invalid){
      bundleSet.bundles->host.get()[k_bundle] = temp_bundles->host.get()[k];
      // std::cout << "bundle set: " << bundleSet.bundles->host.get()[k]numLines << std::endl;
      // std::cout << "temp nundle: " << temp_bundles->host.get()[k].numLines << std::endl;
      for (int i = temp_bundles->host.get()[k].index; i < temp_bundles->host.get()[k].index + temp_bundles->host.get()[k].numLines; i++ ) {
        bundleSet.lines->host.get()[k_adjust] = temp_lines->host.get()[i];
        k_adjust++;
      }
      k_bundle++;
    }
  }
  std::cout << "k_bundle: " << k_bundle << ", " << new_bd_size << std::endl;
  std::cout << "k_adjust: " << k_adjust << ", " << new_ln_size << std::endl;

  //BundleSet newSet = {bundleSet.lines,bundleSet.bundles};
  // return newSet;
  return bundleSet;
}

// =============================================================================================================
//
// Bulk Point Cloud Alteration Methods
//
// =============================================================================================================

/**
* Scales every point in the point cloud by a given scalar s and passed back the point cloud by refrence from the input
* @param scale a float representing how much to scale up or down a point cloud
* @param points is the point cloud to be scaled by s, this value is directly altered
*/
void ssrlcv::PointCloudFactory::scalePointCloud(float scale, ssrlcv::ptr::value<ssrlcv::Unity<float3>> points){

  std::cout << "\t Scaling Point Cloud ..." << std::endl;

  // move to device
  float* d_scale;
  CudaSafeCall(cudaMalloc((void**) &d_scale, sizeof(float)));
  CudaSafeCall(cudaMemcpy(d_scale, &scale, sizeof(float), cudaMemcpyHostToDevice));

  points->transferMemoryTo(gpu);

  // call kernel
  dim3 grid = {1,1,1};
  dim3 block = {1,1,1};
  void (*fp)(float*, unsigned long, float3*) = &computeScalePointCloud;
  getFlatGridBlock(points->size(),grid,block,fp);

  computeScalePointCloud<<<grid,block>>>(d_scale,points->size(),points->device.get());

  cudaDeviceSynchronize();
  CudaCheckError();

  points->setFore(gpu);
  points->transferMemoryTo(cpu);
  points->clear(gpu);

  cudaFree(d_scale);

}

/**
* translates every point in the point cloud by a given vector t and passed back the point cloud by refrence from the input
* @param translate is a float3 representing how much to translate the point cloud in x,y,z
* @param points is the point cloud to be altered by t, this value is directly altered
*/
void ssrlcv::PointCloudFactory::translatePointCloud(float3 translate, ssrlcv::ptr::value<ssrlcv::Unity<float3>> points){

  std::cout << "\t Translating Point Cloud ..." << std::endl;

  ssrlcv::ptr::value<ssrlcv::Unity<float3>> d_translate = ssrlcv::ptr::value<ssrlcv::Unity<float3>>(nullptr,1,ssrlcv::cpu);
  d_translate->host.get()[0].x = translate.x;
  d_translate->host.get()[0].y = translate.y;
  d_translate->host.get()[0].z = translate.z;

  d_translate->transferMemoryTo(gpu);
  points->transferMemoryTo(gpu);

  // call kernel
  dim3 grid = {1,1,1};
  dim3 block = {1,1,1};
  void (*fp)(float3*, unsigned long, float3*) = &computeTranslatePointCloud;
  getFlatGridBlock(points->size(),grid,block,fp);

  computeTranslatePointCloud<<<grid,block>>>(d_translate->device.get(),points->size(),points->device.get());

  cudaDeviceSynchronize();
  CudaCheckError();

  points->setFore(gpu);
  points->transferMemoryTo(cpu);
  points->clear(gpu);

}

/**
* rotates every point in the point cloud by a given x,y,z axis rotation r and passed back the point cloud by refrence from the input
* always roates with respect to the origin
* @param rotate is a float3 representing an x,y,z axis rotation
* @param points is the point cloud to be altered by r, this value is directly altered
*/
void ssrlcv::PointCloudFactory::rotatePointCloud(float3 rotate, ssrlcv::ptr::value<ssrlcv::Unity<float3>> points) {

  std::cout << "\t Rotating Point Cloud ..." << std::endl;

  ssrlcv::ptr::value<ssrlcv::Unity<float3>> d_rotate = ssrlcv::ptr::value<ssrlcv::Unity<float3>>(nullptr,1,ssrlcv::cpu);
  d_rotate->host.get()[0].x = rotate.x;
  d_rotate->host.get()[0].y = rotate.y;
  d_rotate->host.get()[0].z = rotate.z;

  d_rotate->transferMemoryTo(gpu);
  points->transferMemoryTo(gpu);

  // call kernel
  dim3 grid = {1,1,1};
  dim3 block = {1,1,1};
  void (*fp)(float3*, unsigned long, float3*) = &computeRotatePointCloud;
  getFlatGridBlock(points->size(),grid,block,fp);

  computeRotatePointCloud<<<grid,block>>>(d_rotate->device.get(),points->size(),points->device.get());

  cudaDeviceSynchronize();
  CudaCheckError();

  points->setFore(gpu);
  points->transferMemoryTo(cpu);
  points->clear(gpu);

}

/**
 * A method which simply returns the average point in a point cloud
 * @param points a unity of float3 which contains the point cloud
 * @return average a single valued unity of float3 that is the aveage of the points in the point cloud
 */
ssrlcv::ptr::value<ssrlcv::Unity<float3>> ssrlcv::PointCloudFactory::getAveragePoint(ssrlcv::ptr::value<ssrlcv::Unity<float3>> points){

  std::cout << "\t Calculating average point ..." << std::endl;

  ssrlcv::ptr::value<Unity<float3>> average = ssrlcv::ptr::value<Unity<float3>>(nullptr,1,ssrlcv::gpu);

  points->transferMemoryTo(gpu);

  // call kernel
  dim3 grid = {1,1,1};
  dim3 block = {1,1,1}; // ssrlcv::computeAveragePoint(float3* average, unsigned long pointnum, float3* points){
  void (*fp)(float3*, unsigned long, float3*) = &computeAveragePoint;
  getFlatGridBlock(points->size(),grid,block,fp);

  computeAveragePoint<<<grid,block>>>(average->device.get(),points->size(),points->device.get());

  cudaDeviceSynchronize();
  CudaCheckError();

  points->transferMemoryTo(cpu);
  average->transferMemoryTo(cpu);
  points->clear(gpu);
  average->clear(gpu);

  return average;
}

// =============================================================================

            // =====================================================================================================//
            // =====================================================================================================//
            // ==================================================================================================== //
            //                                        device methods                                                //
            // ==================================================================================================== //
            // =====================================================================================================//
            // =====================================================================================================//

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

  for (int i = match.index, k = 0; i < end; i++,k++){
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

__global__ void ssrlcv::generatePushbroomBundle(unsigned int numBundles, Bundle* bundles, Bundle::Line* lines, MultiMatch* matches, KeyPoint* keyPoints, Image::PushbroomCamera* pushbrooms){
  unsigned long globalID = (blockIdx.y* gridDim.x+ blockIdx.x)*blockDim.x + threadIdx.x;
  if (globalID > numBundles - 1) return;

  MultiMatch match = matches[globalID];
  float3* kp = new float3[match.numKeyPoints]();
  int end =  (int) match.numKeyPoints + match.index;
  KeyPoint currentKP = {-1,{0.0f,0.0f}};
  bundles[globalID] = {match.numKeyPoints,match.index,false};

  // for now I am going to complete ignore the possibility that
  // the pushbroom image plane may be distorted in due to jitter
  // for HiRISE image info, see: https://hirise-pds.lpl.arizona.edu/PDS/DOCUMENT/HIRISE_RDR_SIS.PDF
  // an example file that I parsed is here: https://hirise-pds.lpl.arizona.edu/PDS/RDR/ESP/ORB_063700_063799/ESP_063752_1985/ESP_063752_1985_RED.LBL

  for (int i = match.index, k = 0; i < end; i++,k++){
    // the current keypoint to transform
    currentKP = keyPoints[i];
    // TODO check this is the real center point on the image
    float2 center = {(pushbrooms[currentKP.parentId].size.x / 2.0f), (pushbrooms[currentKP.parentId].size.y / 2.0f)}; // the image center
    // place the keypoint in the x y plane, scale it by dpix, and translate so that the center of the "image plane" is at the origin
    kp[k] = {
      pushbrooms[currentKP.parentId].dpix.x * ((currentKP.loc.x) - center.x),
      0.0f, //pushbrooms[currentKP.parentId].dpix.y * ((currentKP.loc.y) - center.y), // << try
      (-1.0f * pushbrooms[currentKP.parentId].foc)// 0.0f // pushbrooms[currentKP.parentId].foc
    };
    // rotate the point as the craft "rolled"
    // rolls around flight direction Y+
    float roll     = pushbrooms[currentKP.parentId].roll * (PI / 180.0f); // save roll in radians, like a real professional
    float radius   = pushbrooms[currentKP.parentId].axis_radius; // in km
    float altitude = pushbrooms[currentKP.parentId].altitude; // in km
    // find coordinate at point during scan, assumes no jitter
    // this is solvable as a quadratic, see Caleb's thesis for details
    float a = 1.0f + (tanf(roll - (PI/2.0f)) * tanf(roll - (PI/2.0f)));
    float b = -2.0f * radius * tanf(roll - (PI/2.0f));
    float c = radius*radius - ((altitude + radius) * (altitude + radius));
    // find the solution
    float solution1 = (-1.0f * b + sqrtf((b * b) - (4.0f * a * c))) / (2.0f * a);
    float solution2 = (-1.0f * b - sqrtf((b * b) - (4.0f * a * c))) / (2.0f * a);
    // find the position of the craft, the two solutions above produce opposite orbit locations
    // the correct solution will be positive in the z direction because the ground target is modeled
    // as the origin and the craft is modeled as observing from "above" the origin in the z+ direction
    float3 position;
    if (solution1 > 0) {
      position.x = solution1;
      position.y = 0.0f;
      position.z = tanf(roll - (PI/2.0f)) * solution1 * -1.0f; // multiply by -1.0f to make value positive
    } else {
      position.x = solution2;
      position.y = 0.0f;
      position.z = tanf(roll - (PI/2.0f)) * solution2 * -1.0f; // multiply by -1.0f to make value positive
    }
    // position curretly only exists in X-Z plane, translate it based on gsd & pixels moved to get an arc length
    float gsd = pushbrooms[currentKP.parentId].gsd; // ----> check if divide by 1000.0f was already done in image reading / 1000.0f; // convert from meters to km
    float arc_length = (gsd * (currentKP.loc.y - center.y)); // get "pixel distance" as real world scale in km
    // float arc_length = (pushbrooms[currentKP.parentId].dpix.y * (currentKP.loc.y - center.y)); // get "pixel distance" as real world scale in km
    float angle_out  = arc_length / radius;
    // rotate the keypoint to the correct orientation
    kp[k]    = rotatePoint(kp[k],{0.0f,roll,0.0f}); // do the roll, which is the off angle of the pushbroom scan
    // kp[k].z += radius;
    // kp[k]    = rotatePoint(kp[k],{angle_out, 0.0f, 0.0f}); // // rotate around the x+ axis to move forward in the "orbit"
    // kp[k].z -= radius;
    // rotate the position to the correct orientation
    // position.z += radius;
    position    = rotatePoint(position,{angle_out, 0.0f, 0.0f}); // rotate around the x+ axis to move forward in the "orbit"
    // position.z -= radius;
    // move the keypoint to the position
    // kp[k]   += position;
    kp[k].x = position.x - (kp[k].x);
    kp[k].y = position.y - (kp[k].y);
    kp[k].z = position.z - (kp[k].z);
    // get the vector of pointing
    // lines[i].vec = kp[k] - position;
    lines[i].vec = {
      position.x - kp[k].x,
      position.y - kp[k].y,
      position.z - kp[k].z
    };
    normalize(lines[i].vec);
    lines[i].pnt = position;
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

__global__ void ssrlcv::interpolateDepth(uint2 disparityMapSize, unsigned int influenceRadius, float* disparities, float* interpolated){
  unsigned long globalID = (blockIdx.y* gridDim.x+ blockIdx.x)*blockDim.x + threadIdx.x;
  if(globalID < (disparityMapSize.x-(2*influenceRadius))*(disparityMapSize.y-(2*influenceRadius))){
    float disparity = disparities[globalID];
    ulong2 loc = {globalID%disparityMapSize.x + influenceRadius,globalID/disparityMapSize.x + influenceRadius};
    for(unsigned long y = loc.y - influenceRadius; y >= loc.y + influenceRadius; ++y){
      for(unsigned long x = loc.x - influenceRadius; x >= loc.x + influenceRadius; ++x){
        disparity += disparities[y*disparityMapSize.x + x]*(1 - abs_diff(x,loc.x)/influenceRadius)*(1 - abs_diff(y,loc.y)/influenceRadius);
      }
    }
    interpolated[globalID] = disparity;
  }
}

// =============================================================================================================
//
// Filtering Kernels
//
// =============================================================================================================

__global__ void ssrlcv::filterTwoViewFromEstimatedPlane(unsigned long pointnum, Bundle::Line* lines, Bundle* bundles, float* cutoff, float3* point, float3* vector){
  // this is same method as the 2View for the first bit
  // see two view method for details
  unsigned long globalID = (blockIdx.y* gridDim.x+ blockIdx.x)*blockDim.x + threadIdx.x;
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
  float3 P = (s1 + s2)/2.0;

  // see: https://mathworld.wolfram.com/Point-PlaneDistance.html
  float3 A = point[0];  // a point on the plane
  float3 N = vector[0]; // the plane's normal

  float d     = N.x * A.x + N.y * A.y + N.z * A.z;
  float numer = abs( N.x * P.x + N.y * P.y + N.z * P.z - d);
  float denom = sqrtf((N.x * N.x) + (N.y * N.y) + (N.z * N.z));

  float dist = numer / denom;

  // if greater than cutoff remove it
  if (dist > *cutoff) {
    bundles[globalID].invalid = true;
  } else {
    bundles[globalID].invalid = false;
  }
}

__global__ void ssrlcv::filterNViewFromEstimatedPlane(unsigned long pointnum, Bundle::Line* lines, Bundle* bundles, float* cutoff, float3* point, float3* vector){

  // this is same method as the NView for the first bit
  // see multi view method for details
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
  float3 P; // the multiview estimated point
  if(inverse(S, Inverse)){
    multiply(Inverse, C, P);
  } else {
    // inversion failed
    bundles[globalID].invalid = true;
    return;
  }

  // see: https://mathworld.wolfram.com/Point-PlaneDistance.html
  float3 A = point[0];  // a point on the plane
  float3 N = vector[0]; // the plane's normal

  float d     = N.x * A.x + N.y * A.y + N.z * A.z;
  float numer = abs( N.x * P.x + N.y * P.y + N.z * P.z - d);
  float denom = sqrtf((N.x * N.x) + (N.y * N.y) + (N.z * N.z));

  float dist = numer / denom;

  // if greater than cutoff remove it
  if (dist > *cutoff) {
    bundles[globalID].invalid = true;
  } else {
    bundles[globalID].invalid = false;
  }

}

// =============================================================================================================
//
// 2 View Kernels
//
// =============================================================================================================

/**
* Does a trigulation with skew lines to find their closest intercetion.
*/
__global__ void ssrlcv::computeTwoViewTriangulate(unsigned long pointnum, Bundle::Line* lines, Bundle* bundles, float3* pointcloud){
  // this method is from wikipedia, last seen janurary 2020
  // https://en.wikipedia.org/wiki/Skew_lines#Nearest_Points
  unsigned long globalID = (blockIdx.y* gridDim.x+ blockIdx.x)*blockDim.x + threadIdx.x;
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
  bundles[globalID].invalid = false;
  // add the linaer errors locally within the block before
  float error = (s1.x - s2.x)*(s1.x - s2.x) + (s1.y - s2.y)*(s1.y - s2.y) + (s1.z - s2.z)*(s1.z - s2.z);
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
  float error = (s1.x - s2.x)*(s1.x - s2.x) + (s1.y - s2.y)*(s1.y - s2.y) + (s1.z - s2.z)*(s1.z - s2.z);
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

  float error = (s1.x - s2.x)*(s1.x - s2.x) + (s1.y - s2.y)*(s1.y - s2.y) + (s1.z - s2.z)*(s1.z - s2.z);

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
  float error = (s1.x - s2.x)*(s1.x - s2.x) + (s1.y - s2.y)*(s1.y - s2.y) + (s1.z - s2.z)*(s1.z - s2.z);
  if (isnan(error)){
    printf(" L1: %f %f %f \t L2: %f %f %f \n\t S1: %f %f %f \t S2: %f %f %f \n", L1.pnt.x, L1.pnt.y, L1.pnt.z, L2.pnt.x, L2.pnt.y, L2.pnt.z, s1.x, s1.y, s1.z, s2.x, s2.y, s2.z);
  }
  //float error = dotProduct(s1,s2)*dotProduct(s1,s2);
  //if(error != 0.0f) error = sqrtf(error);
  // only add errors that we like
  // float i_error;
  // filtering should only occur at the start of each adjustment step
  // TODO clean this up
  // i_error = error;
  bundles[globalID].invalid = false;
  atomicAdd(&localSum,error);
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
  float error = (s1.x - s2.x)*(s1.x - s2.x) + (s1.y - s2.y)*(s1.y - s2.y) + (s1.z - s2.z)*(s1.z - s2.z);
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
  } else {
    // inversion failed
    bundles[globalID].invalid = true;
    return;
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

  float a_error = 0;
  for(int i = bundles[globalID].index; i < bundles[globalID].index + bundles[globalID].numLines; i++){
    // see: https://mathworld.wolfram.com/Point-LineDistance3-Dimensional.html

    float3 linepoint1 = lines[i].pnt;
    float3 linepoint2 = lines[i].pnt + (lines[i].vec * 1000.0); // to avoid loss of significance

    float3 a = point - linepoint1;
    float3 b = point - linepoint2;
    float3 c = linepoint2 - linepoint1;

    float3 d = crossProduct(a,b);
    float numer = magnitude(d);
    float denom = magnitude(c);

    a_error = numer / denom;
    a_error *= a_error;
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
    // see: https://mathworld.wolfram.com/Point-LineDistance3-Dimensional.html

    float3 linepoint1 = lines[i].pnt;
    float3 linepoint2 = lines[i].pnt + (lines[i].vec * 1000.0); // to avoid loss of significance

    float3 a = point - linepoint1;
    float3 b = point - linepoint2;
    float3 c = linepoint2 - linepoint1;

    float3 d = crossProduct(a,b);
    float numer = magnitude(d);
    float denom = magnitude(c);

    a_error = numer / denom;
    a_error *= a_error;
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


  float a_error = 0;
  for(int i = bundles[globalID].index; i < bundles[globalID].index + bundles[globalID].numLines; i++){
    // see: https://mathworld.wolfram.com/Point-LineDistance3-Dimensional.html

    float3 linepoint1 = lines[i].pnt;
    float3 linepoint2 = lines[i].pnt + (lines[i].vec * 1000.0); // to avoid loss of significance

    float3 a = point - linepoint1;
    float3 b = point - linepoint2;
    float3 c = linepoint2 - linepoint1;

    float3 d = crossProduct(a,b);
    float numer = magnitude(d);
    float denom = magnitude(c);

    a_error = numer / denom;
    a_error *= a_error;
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

/**
* the CUDA kernel for Nview triangulation with angular error
*/
__global__ void ssrlcv::computeNViewTriangulate(float* angularError, float* lowCut, float* highCut, float* errors, unsigned long pointnum, Bundle::Line* lines, Bundle* bundles, float3* pointcloud){

    // get ready to do the stuff local memory space
    // this will later be added back to a global memory space
    __shared__ float localSum;
    if (threadIdx.x == 0) localSum = 0;
    __syncthreads();

    unsigned long globalID = (blockIdx.y* gridDim.x+ blockIdx.x)*blockDim.x + threadIdx.x;
    if (globalID > (pointnum-1)) return;

    printf("ERROR DEPRICATED FUNCTION computeNViewTriangulate() with upper and lower cut \t");

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


    float a_error = 0;
    for(int i = bundles[globalID].index; i < bundles[globalID].index + bundles[globalID].numLines; i++){
      float3 l = lines[i].vec;
      normalize(l);
      float3 v = point - lines[i].pnt;
      float  d = dotProduct(v,l);
      float3 close = (lines[i].pnt + l) * d;
      float dist = sqrtf((point.x - close.x)*(point.x - close.x) + (point.y - close.y)*(point.y - close.y) + (point.z - close.z)*(point.z - close.z));
      a_error += dist;
    }

    a_error /= (float) bundles[globalID].numLines;

    errors[globalID] = a_error;

    // filtering
    if (a_error > *highCut || a_error < *lowCut) {
      bundles[globalID].invalid = true;
    } else {
      bundles[globalID].invalid = false;
    }

    // after calculating local error add it all up
    atomicAdd(&localSum,a_error);
    __syncthreads();
    if (!threadIdx.x) atomicAdd(angularError,localSum);
}

// =============================================================================================================
//
// Bulk Point Cloud Alteration Kernels
//
// =============================================================================================================

/**
* the CUDA kernel for scalePointCloud
*/
__global__ void ssrlcv::computeScalePointCloud(float* scale, unsigned long pointnum, float3* points){
  unsigned long globalID = (blockIdx.y* gridDim.x+ blockIdx.x)*blockDim.x + threadIdx.x;
  if (globalID > (pointnum-1)) return;

  // scale the points
  points[globalID].x *= *scale;
  points[globalID].y *= *scale;
  points[globalID].z *= *scale;

}

/**
* the CUDA kernel for translatePointCloud
*/
__global__ void ssrlcv::computeTranslatePointCloud(float3* translate, unsigned long pointnum, float3* points){
  unsigned long globalID = (blockIdx.y* gridDim.x+ blockIdx.x)*blockDim.x + threadIdx.x;
  if (globalID > (pointnum-1)) return;

  // scale the points
  points[globalID].x += translate[0].x;
  points[globalID].y += translate[0].y;
  points[globalID].z += translate[0].z;
}

/**
* the CUDA kernel for rotatePointCloud
*/
__global__ void ssrlcv::computeRotatePointCloud(float3* rotation, unsigned long pointnum, float3* points){
  unsigned long globalID = (blockIdx.y* gridDim.x+ blockIdx.x)*blockDim.x + threadIdx.x;
  if (globalID > (pointnum-1)) return;

  // rotate the point
  points[globalID] = rotatePoint(points[globalID], rotation[0]);
}

/**
 * a CUDA kernel to compute the average point of a point cloud
 */
__global__ void ssrlcv::computeAveragePoint(float3* average, unsigned long pointnum, float3* points){
  __shared__ float3 localSum;
  if (threadIdx.x == 0) {
    localSum.x = 0.0f;
    localSum.y = 0.0f;
    localSum.z = 0.0f;
  }
  __syncthreads();

  unsigned long globalID = (blockIdx.y* gridDim.x+ blockIdx.x)*blockDim.x + threadIdx.x;
  if (globalID > (pointnum-1)) return;

  float3 delta;
  delta.x = points[globalID].x / pointnum;
  delta.y = points[globalID].y / pointnum;
  delta.z = points[globalID].z / pointnum;

  atomicAdd(&localSum.x,delta.x);
  atomicAdd(&localSum.y,delta.y);
  atomicAdd(&localSum.z,delta.z);
  __syncthreads();
  if (!threadIdx.x) {
    atomicAdd(&average[0].x,localSum.x);
    atomicAdd(&average[0].y,localSum.y);
    atomicAdd(&average[0].z,localSum.z);
  }
}

/**
 * @brief A helper function which allows us to calculate the absolute difference on an unsigned int.
 * 
 * @param a unsigned int
 * @param b unsigned int
 * @return __host__ 
 */
__host__ __device__ __forceinline__ unsigned int ssrlcv::abs_diff(unsigned int a, unsigned int b) {
  return (a > b) ? a - b : b - a;
} // abs_diff

















//
