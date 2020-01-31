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

  std::cout << "starting bundle generation ..." << std::endl;
  MemoryState origin[2] = {matchSet->matches->state,matchSet->keyPoints->state};
  if(origin[0] == cpu) matchSet->matches->transferMemoryTo(gpu);
  if(origin[1] == cpu) matchSet->keyPoints->transferMemoryTo(gpu);
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

  dim3 grid = {1,1,1};
  dim3 block = {1,1,1};
  getFlatGridBlock(bundles->numElements,grid,block,generateBundle);

  //in this kernel fill lines and bundles from keyPoints and matches
  std::cout << "Calling bundle generation kernel ..." << std::endl;
  generateBundle<<<grid, block>>>(bundles->numElements,bundles->device, lines->device, matchSet->matches->device, matchSet->keyPoints->device, d_cameras);
  std::cout << "Returned from bundle generation kernel ... \n" << std::endl;

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
*/
ssrlcv::Unity<float3>* ssrlcv::PointCloudFactory::twoViewTriangulate(BundleSet bundleSet){

  bundleSet.lines->transferMemoryTo(gpu);
  bundleSet.bundles->transferMemoryTo(gpu);

  Unity<float3>* pointcloud = new Unity<float3>(nullptr,bundleSet.bundles->numElements,gpu);

  dim3 grid = {1,1,1};
  dim3 block = {1,1,1};
  getFlatGridBlock(bundleSet.bundles->numElements,grid,block,generateBundle);

  std::cout << "Starting 2-view triangulation ..." << std::endl;
  computeTwoViewTriangulate<<<grid,block>>>(bundleSet.bundles->numElements,bundleSet.lines->device,bundleSet.bundles->device,pointcloud->device);
  std::cout << "2-view Triangulation done ... \n" << std::endl;

  cudaDeviceSynchronize();
  CudaCheckError();

  //transfer the poitns back to the CPU
  pointcloud->transferMemoryTo(cpu);
  pointcloud->clear(gpu);

  return pointcloud;
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

  MemoryState origin = matches->state;
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

  MemoryState origin = matches->state;
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

  MemoryState origin = matches->state;
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
  MemoryState origin = points->state;
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


// device methods


__global__ void ssrlcv::generateBundle(unsigned int numBundles, Bundle* bundles, Bundle::Line* lines, MultiMatch* matches, KeyPoint* keyPoints, Image::Camera* cameras){
  unsigned long globalID = (blockIdx.y* gridDim.x+ blockIdx.x)*blockDim.x + threadIdx.x;
  if (globalID > numBundles) return;
  MultiMatch match = matches[globalID];
  float3* kp = new float3[match.numKeyPoints]();
  int end =  (int) match.numKeyPoints + match.index;
  KeyPoint currentKP = {-1,{0.0f,0.0f}};
  bundles[globalID] = {match.numKeyPoints,match.index};
  for (int i = match.index, k= 0; i < end; i++,k++){
    // the current keypoint to transform
    currentKP = keyPoints[i];
    // set the dpix
    cameras[currentKP.parentId].dpix.x = (cameras[currentKP.parentId].foc * tanf(cameras[currentKP.parentId].fov / 2.0f)) / (cameras[currentKP.parentId].size.x / 2.0f );
    cameras[currentKP.parentId].dpix.y = cameras[currentKP.parentId].dpix.x; // assume square pixel for now
    // here we imagine the image plane is in the X Y plane AT a particular Z value, which is the focal length
    // begin movement to R3
    kp[k] = {
      cameras[currentKP.parentId].dpix.x * ((currentKP.loc.x) - (cameras[currentKP.parentId].size.x / 2.0f)),
      cameras[currentKP.parentId].dpix.y * ((currentKP.loc.y) - (cameras[currentKP.parentId].size.y / 2.0f)),
      cameras[currentKP.parentId].foc // this is the focal length
    }; // set the key point
    // rotate to correct orientation
    kp[k] = rotatePoint(kp[k], cameras[currentKP.parentId].cam_vec);
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
  float3 n1 = crossProduct(L2.vec,crossProduct(L1.vec,L2.vec));
  float3 n2 = crossProduct(L1.vec,crossProduct(L1.vec,L2.vec));

  // calculate some numerators
  float numer1 = dotProduct((L2.pnt - L1.pnt),n1);
  float numer2 = dotProduct((L1.pnt - L2.pnt),n2);

  // calculate some denominators
  float denom1 = dotProduct(L1.vec,n1);
  float denom2 = dotProduct(L2.vec,n2);

  // get the S points
  float3 s1 = L1.pnt + (numer1/denom1)*L1.vec; //dotProduct((numer1/denom1),L1.vec);
  float3 s2 = L2.pnt + (numer2/denom2)*L2.vec; //dotProduct((numer1/denom1),L2.vec);
  float3 point = (s1 + s2)/2.0;

  // fill in the value for the point cloud
  pointcloud[globalID] = s1;
}

// cannot be used with current stuff
// this is the old two view reprojection
__global__ void ssrlcv::two_view_reproject(int numMatches, float4* matches, float cam1C[3], float cam1V[3],float cam2C[3], float cam2V[3], float K_inv[9], float rotationTranspose1[9], float rotationTranspose2[9], float3* points){
   unsigned long globalID = (blockIdx.y* gridDim.x+ blockIdx.x)*blockDim.x + threadIdx.x;

  if(!(globalID<numMatches))return;
	//check out globalID cheat sheet jackson gave you for this
	int matchIndex = globalID; //need to define once I calculate grid/block size
	float4 match = matches[globalID];


	float pix1[3] =
	{
		match.x, match.y, 1
	};
	float pix2[3] =
	{
		match.z, match.w, 1
  };
  float K_inv_reg[3][3];
  for(int r = 0; r < 3; ++r){
    for(int c = 0; c < 3; ++c){
      K_inv_reg[r][c] = K_inv[r*3 + c];
    }
  }
  float rotationTranspose1_reg[3][3];
   for(int r = 0; r < 3; ++r){
    for(int c = 0; c < 3; ++c){
      rotationTranspose1_reg[r][c] = rotationTranspose1[r*3 + c];
    }
  }
  float rotationTranspose2_reg[3][3];
   for(int r = 0; r < 3; ++r){
    for(int c = 0; c < 3; ++c){
      rotationTranspose2_reg[r][c] = rotationTranspose2[r*3 + c];
    }
  }

	float inter1[3];
	float inter2[3];

	float temp[3];
	multiply(K_inv_reg, pix1, temp);
	multiply(rotationTranspose1_reg, temp, inter1);
	multiply(K_inv_reg, pix2, temp);
	multiply(rotationTranspose2_reg, temp, inter2);

	float worldP1[3] =
	{
		inter1[0]+cam1C[0], inter1[1]+cam1C[1], inter1[2]+cam1C[2]
	};

	float worldP2[3] =
	{
		inter2[0]+cam2C[0], inter2[1]+cam2C[1], inter2[2]+cam2C[2]
	};

	float v1[3] =
	{
		worldP1[0] - cam1C[0], worldP1[1] - cam1C[1], worldP1[2] - cam1C[2]
	};

	float v2[3] =
	{
		worldP2[0] - cam2C[0], worldP2[1] - cam2C[1], worldP2[2] - cam2C[2]
	};

	normalize(v1);
	normalize(v2);



	//match1 and match2?
	float M1[3][3] =
	{
		{ 1-(v1[0]*v1[0]), 0-(v1[0]*v1[1]), 0-(v1[0]*v1[2]) },
		{ 0-(v1[0]*v1[1]), 1-(v1[1]*v1[1]), 0-(v1[1]*v1[2]) },
		{ 0-(v1[0]*v1[2]), 0-(v1[1]*v1[2]), 1-(v1[2]*v1[2]) }
	};

	float M2[3][3] =
	{
		{ 1-(v2[0]*v2[0]), 0-(v2[0]*v2[1]), 0-(v2[0]*v2[2]) },
		{ 0-(v2[0]*v2[1]), 1-(v2[1]*v2[1]), 0-(v2[1]*v2[2]) },
		{ 0-(v2[0]*v2[2]), 0-(v2[1]*v2[2]), 1-(v2[2]*v2[2]) }
	};

	float q1[3];
	float q2[3];
	float Q[3];

	multiply( M1, worldP1, q1);
	multiply( M2, worldP2, q2);

	float M[3][3];
	float M_inv[3][3];

	for(int r = 0; r < 3; ++r)
	{
		for(int c = 0; c < 3; ++c)
		{
			M[r][c] = M1[r][c] + M2[r][c];
		}
		Q[r] = q1[r] + q2[r];
	}

	float solution[3];
	inverse(M, M_inv);
	multiply(M_inv, Q, solution);



  	points[matchIndex].x = solution[0];
  	points[matchIndex].y = solution[1];
  	points[matchIndex].z = solution[2];

}


























































// yee
