#include "PointCloudFactory.cuh"

ssrlcv::PointCloudFactory::PointCloudFactory(){

}

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
  getFlatGridBlock(bundles->numElements,grid,block);

  //in this kernel fill lines and bundles from keyPoints and matches
  std::cout << "calling kernel ..." << std::endl;
  generateBundle<<<grid, block>>>(bundles->numElements,bundles->device, lines->device, matchSet->matches->device, matchSet->keyPoints->device, d_cameras);
  std::cout << "returned from kernel ..." << std::endl;

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

// TODO fillout
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
  getFlatGridBlock(matches->numElements,grid,block);
  //
  // computeStereo<<<grid, block>>>(matches->numElements, matches->device, points_device, 8.0);
  computeStereo<<<grid, block>>>(matches->numElements, matches->device, points_device, scale);

  Unity<float3>* points = new Unity<float3>(points_device, matches->numElements,gpu);
  if(origin == cpu) matches->setMemoryState(cpu);

  return points;
}

// TODO fillout
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
  getFlatGridBlock(matches->numElements,grid,block);
  //
  computeStereo<<<grid, block>>>(matches->numElements, matches->device, points_device, scale);

  Unity<float3>* points = new Unity<float3>(points_device, matches->numElements,gpu);
  if(origin == cpu) matches->setMemoryState(cpu);

  return points;
}


// device methods


__global__ void ssrlcv::generateBundle(unsigned int numBundles, Bundle* bundles, Bundle::Line* lines, MultiMatch* matches, KeyPoint* keyPoints, Image::Camera* cameras){
  unsigned long globalID = (blockIdx.y* gridDim.x+ blockIdx.x)*blockDim.x + threadIdx.x;
  MultiMatch match = matches[globalID];
  float3* kp = new float3[match.numKeyPoints]();
  int end =  (int)match.numKeyPoints + match.index;
  KeyPoint currentKP = {-1,{0.0f,0.0f}};
  bundles[globalID] = {match.numKeyPoints,match.index};
  for (int i = match.index, k = 0; i < end; i++,k++){
    currentKP = keyPoints[i];
    printf("[%lu][%d] camera vec: <%f,%f,%f>\n", globalID,k, cameras[currentKP.parentId].cam_vec.x,cameras[currentKP.parentId].cam_vec.y,cameras[currentKP.parentId].cam_vec.z);
    normalize(cameras[currentKP.parentId].cam_vec);
    printf("[%lu][%d] norm camera vec: <%f,%f,%f>\n", globalID,k, cameras[currentKP.parentId].cam_vec.x,cameras[currentKP.parentId].cam_vec.y,cameras[currentKP.parentId].cam_vec.z);
    // set dpix values
    printf("[%lu][%d] dpix calc dump: (foc: %f) (fov: %f) (tanf: %f) (size: %d) \n", globalID,k, cameras[currentKP.parentId].foc, cameras[currentKP.parentId].fov, tanf(cameras[currentKP.parentId].fov / 2.0f), cameras[currentKP.parentId].size.x);
    cameras[currentKP.parentId].dpix.x = (cameras[currentKP.parentId].foc * tanf(cameras[currentKP.parentId].fov / 2.0f)) / (cameras[currentKP.parentId].size.x / 2.0f );
    cameras[currentKP.parentId].dpix.y = cameras[currentKP.parentId].dpix.x; // assume square pixel for now
    // temp
    printf("[%lu][%d] dpix calculated as: %f \n", globalID,k, cameras[currentKP.parentId].dpix.x);

    kp[k] = {
      cameras[currentKP.parentId].dpix.x * ((currentKP.loc.x) - (cameras[currentKP.parentId].size.x / 2.0f)),
      cameras[currentKP.parentId].dpix.y * ((-1.0f * currentKP.loc.y) - (cameras[currentKP.parentId].size.y / 2.0f)),
      0.0f
    }; // set the key point

    printf("[%lu][%d] kp, pre-rotation: (%f,%f,%f) \n", globalID,k, kp[k].x, kp[k].y, kp[k].z);
    kp[k] = rotatePoint(kp[k], getVectorAngles(cameras[currentKP.parentId].cam_vec));
    printf("[%lu][%d] kp, angles: (%f,%f,%f) \n", globalID,k, getVectorAngles(cameras[currentKP.parentId].cam_vec).x, getVectorAngles(cameras[currentKP.parentId].cam_vec).y, getVectorAngles(cameras[currentKP.parentId].cam_vec).z);
    printf("[%lu][%d] kp, post-rotation: (%f,%f,%f) \n", globalID,k, kp[k].x, kp[k].y, kp[k].z);
    // NOTE: will need to adjust foc with scale or x/y component here in the future
    kp[k].x = cameras[currentKP.parentId].cam_pos.x - (kp[k].x + (cameras[currentKP.parentId].cam_vec.x * cameras[currentKP.parentId].foc));
    kp[k].y = cameras[currentKP.parentId].cam_pos.y - (kp[k].y + (cameras[currentKP.parentId].cam_vec.y * cameras[currentKP.parentId].foc));
    kp[k].z = cameras[currentKP.parentId].cam_pos.z - (kp[k].z + (cameras[currentKP.parentId].cam_vec.z * cameras[currentKP.parentId].foc));
    printf("[%lu][%d] kp in R3: (%f,%f,%f)\n", globalID,k, kp[k].x, kp[k].y, kp[k].z);
    lines[i].vec = {
      cameras[currentKP.parentId].cam_pos.x - kp[k].x,
      cameras[currentKP.parentId].cam_pos.y - kp[k].y,
      cameras[currentKP.parentId].cam_pos.z - kp[k].z
    };
    normalize(lines[i].vec);
    printf("[%lu][%d] %f,%f,%f\n",globalID,k,lines[i].vec.x,lines[i].vec.y,lines[i].vec.z);
    lines[i].pnt = cameras[currentKP.parentId].cam_pos;
  }
}

__global__ void ssrlcv::computeStereo(unsigned int numMatches, Match* matches, float3* points, float scale){
  unsigned long globalID = (blockIdx.y* gridDim.x+ blockIdx.x)*blockDim.x + threadIdx.x;
  if (globalID < numMatches) {
    Match match = matches[globalID];
    float3 point = {match.keyPoints[0].loc.x,match.keyPoints[0].loc.y,0.0f};
    point.z = scale / sqrtf( dotProduct(match.keyPoints[0].loc-match.keyPoints[1].loc,match.keyPoints[0].loc-match.keyPoints[1].loc)) ;
    points[globalID] = point;
  }
}

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
