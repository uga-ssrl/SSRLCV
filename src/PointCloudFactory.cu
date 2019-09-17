#include "PointCloudFactory.cuh"

ssrlcv::PointCloudFactory::PointCloudFactory(){

}

/*
ssrlcv::Unity<float3>* ssrlcv::PointCloudFactory::reproject(Unity<Match>* matches, Image* target, Image* query){
  float3* pointCloud_device = nullptr;
  CudaSafeCall(cudaMalloc((void**)&pointCloud_device, matches->numElements*sizeof(float3)));
  Unity<float3>* pointCloud = new Unity<float3>(pointCloud_device,matches->numElements,gpu);

  // //initiliaze camera matrices
  // Camera cam1 = cData->cameras[0];
  // Camera cam2 = cData->cameras[1];
  // float cam1C[3] =
  // {
  //   cam1.val1, cam1.val2, cam1.val3
  // };
  // float cam1V[3] =
  // {
  //   -1*cam1.val4, -1*cam1.val5, -1*cam1.val6
  // };
  // float cam2C[3] =
  // {
  //   cam2.val1, cam2.val2, cam2.val3
  // };
  // float cam2V[3] =
  // {
  //   -1*cam2.val4, -1*cam2.val5, -1*cam2.val6
  // };
  //
  // //other matrix data needed by all threads
  // float K[3][3];  // intrinsic camera matrix
  // float K_inv[3][3];  // inverse of K
  //
  // K[0][0] = foc/dpix;
  // K[0][1] = 0;
  // K[0][2] = (float)(res/2.0);
  // K[1][0] = 0;
  // K[1][1] = foc/dpix;
  // K[1][2] = (float)(res/2.0);
  // K[2][0] = 0;
  // K[2][1] = 0;
  // K[2][2] = 1;
  // inverse3x3_cpu(K, K_inv);
  //
  // float x;
  // float y;
  // float z;
  // // Rotate cam1V about the x axis
  // x = cam1V[0];
  // y = cam1V[1];
  // z = cam1V[2];
  // float angle1;  // angle between cam1V and x axis
  // if (abs(z) < .00001)
  // {
  //   if (y > 0)
  //   {
  //     angle1 = PI/2;
  //   }
  //   else
  //   {
  //     angle1 = -1*PI/2;
  //   }
  // }
  // else
  // {
  //   angle1 = atan(y/z);
  //   if (z < 0 && y >= 0)
  //   {
  //     angle1 +=PI;
  //   }
  //   if (z < 0 && y < 0)
  //   {
  //     angle1 -= PI;
  //   }
  // }
  // float A1[3][3] =
  // {
  //   {1, 0, 0},
  //   {0, cos(angle1), -sin(angle1)},
  //   {0, sin(angle1), cos(angle1)}
  // };
  //
  // float temp[3];
  // //apply transform matrix we just got
  // multiply3x3x1_cpu(A1, cam1V, temp);
  //
  // //rotate around the y axis
  // x = temp[0];
  // y = temp[1];
  // z = temp[2];
  // float angle2;  // angle between temp and y axis
  // if (abs(z) < .00001)
  // {
  //   if (x <= 0)
  //   {
  //     angle2 = PI/2;
  //   }else
  //   {
  //     angle2 = -1*PI/2;
  //   }
  // }else
  // {
  //   angle2 = atan(-1*x / z);
  //   if(z < 0 && x < 0)
  //   {
  //     angle2 += PI;
  //   }
  //   if(z < 0 && x > 0)
  //   {
  //     angle2 -= PI;
  //   }
  // }
  //
  // float B1[3][3] =
  // {
  //   {cos(angle2), 0, sin(angle2)},
  //   {0, 1, 0},
  //   {-sin(angle2), 0, cos(angle2)}
  // };
  //
  // float rotCam1[3];
  // // apply transformation matrix B. store in rotcam1
  // multiply3x3x1_cpu(B1, temp, rotCam1);
  //
  // float rotationMatrix1[3][3];
  // float rotationTranspose1[3][3];
  //
  // //get rotation matrix as a single transform matrix
  // multiply3x3_cpu(B1, A1, rotationMatrix1);
  // transpose_cpu(rotationMatrix1, rotationTranspose1);
  // multiply3x3x1_cpu(rotationTranspose1, rotCam1, temp); // temp should be original cam1C now
  //
  // // Rotate cam2V about the x axis
  // x = cam2V[0];
  // y = cam2V[1];
  // z = cam2V[2];
  //
  // if(abs(z) < .00001)
  // {
  //   if(y > 0)
  //   {
  //     angle1 = PI/2;
  //   } else
  //   {
  //     angle1 = -1*PI/2;
  //   }
  // } else
  // {
  //   angle1 = atan(y / z);
  //   if(z<0 && y>=0)
  //   {
  //     angle1 += PI;
  //   }
  //   if(z<0 && y<0)
  //   {
  //     angle1 -= PI;
  //   }
  // }
  // float A2[3][3] =
  // {
  //   {1, 0, 0},
  //   {0, cos(angle1), -sin(angle1)},
  //   {0, sin(angle1), cos(angle1)}
  // };
  // // apply transformation matrix A
  // multiply3x3x1_cpu(A2, cam2V, temp);
  //
  // // Rotate about the y axis
  // x = temp[0];
  // y = temp[1];
  // z = temp[2];
  // if(abs(z) < .00001)
  // {
  //   if(x <= 0){
  //     angle2 = PI/2;
  //   }else
  //   {
  //     angle2 = -1*PI/2;
  //   }
  // } else
  // {
  //   angle2 = atan(-1*x / z);
  //   if(z<0 && x<0)
  //   {
  //     angle2 += PI;
  //   }
  //   if(z<0 && x>0)
  //   {
  //     angle2 -= PI;
  //   }
  // }
  // float B2[3][3] =
  // {
  //   {cos(angle2), 0, sin(angle2)},
  //   {0, 1, 0},
  //   {-sin(angle2), 0, cos(angle2)}
  // };
  // // apply transformation matrix B
  // float rotCam2[3];
  // multiply3x3x1_cpu(B2, temp, rotCam2);
  //
  // float rotationMatrix2[3][3];
  // float rotationTranspose2[3][3];
  //
  // // Get rotation matrix as a single transformation matrix
  // multiply3x3_cpu(B2, A2, rotationMatrix2);
  // transpose_cpu(rotationMatrix2, rotationTranspose2);
  // multiply3x3x1_cpu(rotationTranspose2, rotCam2, temp); // temp should be original cam2C now
  //
  // //linearize matrices
  // //position in linear matrix = 3*x +y, [x][y]
  // float K_inv_lin[9];
  // K_inv_lin[0] = K_inv[0][0];
  // K_inv_lin[1] = K_inv[0][1];
  // K_inv_lin[2] = K_inv[0][2];
  // K_inv_lin[3] = K_inv[1][0];
  // K_inv_lin[4] = K_inv[1][1];
  // K_inv_lin[5] = K_inv[1][2];
  // K_inv_lin[6] = K_inv[2][0];
  // K_inv_lin[7] = K_inv[2][1];
  // K_inv_lin[8] = K_inv[2][2];
  //
  // float rotTran1_lin[9];
  // rotTran1_lin[0] = rotationTranspose1[0][0];
  // rotTran1_lin[1] = rotationTranspose1[0][1];
  // rotTran1_lin[2] = rotationTranspose1[0][2];
  // rotTran1_lin[3] = rotationTranspose1[1][0];
  // rotTran1_lin[4] = rotationTranspose1[1][1];
  // rotTran1_lin[5] = rotationTranspose1[1][2];
  // rotTran1_lin[6] = rotationTranspose1[2][0];
  // rotTran1_lin[7] = rotationTranspose1[2][1];
  // rotTran1_lin[8] = rotationTranspose1[2][2];
  //
  // float rotTran2_lin[9];
  // rotTran2_lin[0] = rotationTranspose2[0][0];
  // rotTran2_lin[1] = rotationTranspose2[0][1];
  // rotTran2_lin[2] = rotationTranspose2[0][2];
  // rotTran2_lin[3] = rotationTranspose2[1][0];
  // rotTran2_lin[4] = rotationTranspose2[1][1];
  // rotTran2_lin[5] = rotationTranspose2[1][2];
  // rotTran2_lin[6] = rotationTranspose2[2][0];
  // rotTran2_lin[7] = rotationTranspose2[2][1];
  // rotTran2_lin[8] = rotationTranspose2[2][2];
  //
  // //initialize point cloud data to 0
  // //pointCloud->points = new  float3[POINT_CLOUD_SIZE];
  // float3* currentPoint;
  // for(int i = 0; i < POINT_CLOUD_SIZE; ++i)
  // {
  //   currentPoint = &(pointCloud->points[i]);
  //   currentPoint->x = 0.0f;
  //   currentPoint->y = 0.0f;
  //   currentPoint->z = 0.0f;
  // }
  //
  // //create pointers on the device. d_ indicates pointer to mem on device
  // float4* d_in_matches; 		 //where feature matches data is stored
  // float* d_in_cam1C; 		 //where camera data is stored
  // float* d_in_cam1V; 		 //where camera data is stored
  // float* d_in_cam2C; 		 //where camera data is stored
  // float* d_in_cam2V; 		 //where camera data is stored
  // float* d_in_k_inv;
  // float* d_in_rotTran1;
  //
  // float* d_in_rotTran2;
  // float3* d_out_pointCloud; //where point cloud output is stored
  //
  // //allocate the mem on the gpu
  // cudaMalloc((void**) &d_in_matches, FEATURE_DATA_BYTES);
  // cudaMalloc((void**) &d_in_cam1C, CAMERA_DATA_BYTES);
  // cudaMalloc((void**) &d_in_cam1V, CAMERA_DATA_BYTES);
  // cudaMalloc((void**) &d_in_cam2C, CAMERA_DATA_BYTES);
  // cudaMalloc((void**) &d_in_cam2V, CAMERA_DATA_BYTES);
  // cudaMalloc((void**) &d_in_k_inv, MATRIX_DAYA_BYTES);
  // cudaMalloc((void**) &d_in_rotTran1, MATRIX_DAYA_BYTES);
  // cudaMalloc((void**) &d_in_rotTran2, MATRIX_DAYA_BYTES);
  // cudaMalloc((void**) &d_out_pointCloud, POINT_CLOUD_BYTES);
  //
  // //transfer input data to mem on the gpu
  // cudaMemcpy(d_in_matches, fMatches->matches, FEATURE_DATA_BYTES, cudaMemcpyHostToDevice);
  // cudaMemcpy(d_in_cam1C, cam1C, CAMERA_DATA_BYTES, cudaMemcpyHostToDevice);
  // cudaMemcpy(d_in_cam1V, cam1V, CAMERA_DATA_BYTES, cudaMemcpyHostToDevice);
  // cudaMemcpy(d_in_cam2C, cam2C, CAMERA_DATA_BYTES, cudaMemcpyHostToDevice);
  // cudaMemcpy(d_in_cam2V, cam2V, CAMERA_DATA_BYTES, cudaMemcpyHostToDevice);
  // cudaMemcpy(d_in_k_inv, K_inv_lin, MATRIX_DAYA_BYTES, cudaMemcpyHostToDevice);
  // cudaMemcpy(d_in_rotTran1, rotTran1_lin, MATRIX_DAYA_BYTES, cudaMemcpyHostToDevice);
  // cudaMemcpy(d_in_rotTran2, rotTran2_lin, MATRIX_DAYA_BYTES, cudaMemcpyHostToDevice);
  // cudaMemcpy(d_out_pointCloud, pointCloud->points, POINT_CLOUD_BYTES, cudaMemcpyHostToDevice);
  //
  // //block and thread count
  // dim3 THREAD_COUNT = {512, 1, 1};
  // dim3 BLOCK_COUNT = {(unsigned int)ceil((POINT_CLOUD_SIZE+512)/512),1, 1}; //(unsigned int)ceil(POINT_CLOUD_SIZE/512)
  //
  // //call kernel
  // two_view_reproject<<<BLOCK_COUNT, THREAD_COUNT>>>(POINT_CLOUD_SIZE, d_in_matches, d_in_cam1C, d_in_cam1V, d_in_cam2C, d_in_cam2V, d_in_k_inv, d_in_rotTran1, d_in_rotTran2, d_out_pointCloud);
  //
  // //error check
  // CudaCheckError();
  //
  // //get result
  // cudaMemcpy(pointCloud->points, d_out_pointCloud, POINT_CLOUD_BYTES, cudaMemcpyDeviceToHost);
  //
  // pointCloud->numPoints = fMatches->numMatches;
  // //free mem on gpu
  // cudaFree(d_in_matches);
  // cudaFree(d_in_cam1C);
  // cudaFree(d_in_cam1V);
  // cudaFree(d_in_cam2C);
  // cudaFree(d_in_cam2V);
  // cudaFree(d_out_pointCloud);
  return pointCloud;
}
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
* Preforms a Stereo Disparity
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
  int blockSize = 1024;
  int gridSize = (int) ceil((float) matches->numElements / blockSize);

  //
  computeStereo<<<gridSize, blockSize>>>(matches->numElements, matches->device, points_device, scale);

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
    cameras[currentKP.parentId].cam_vec = normalizeVector(cameras[currentKP.parentId].cam_vec);
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
    lines[i].vec = normalizeVector(lines[i].vec);
    printf("[%lu][%d] %f,%f,%f\n",globalID,k,lines[i].vec.x,lines[i].vec.y,lines[i].vec.z);
    lines[i].pnt = cameras[currentKP.parentId].cam_pos;
  }
}

__global__ void ssrlcv::computeStereo(unsigned int numMatches, Match* matches, float3* points, float scale){
  unsigned long globalID = (blockIdx.y* gridDim.x+ blockIdx.x)*blockDim.x + threadIdx.x;
  if (globalID < numMatches) {
    Match match = matches[globalID];
    float3 point = {match.keyPoints[0].loc.x,match.keyPoints[0].loc.y,0.0f};
    point.z = sqrtf(scale*dotProduct({match.keyPoints[0].loc-match.keyPoints[1].loc},{match.keyPoints[0].loc-match.keyPoints[1].loc}));
    points[globalID] = point;
  }
}

__global__ void ssrlcv::two_view_reproject(int numMatches, float4* matches, float cam1C[3], float cam1V[3],float cam2C[3], float cam2V[3], float K_inv[9], float rotationTranspose1[9], float rotationTranspose2[9], float3* points){

  if(!(getGlobalIdx_1D_1D()<numMatches))return;
	//check out globalID cheat sheet jackson gave you for this
	int matchIndex = getGlobalIdx_1D_1D(); //need to define once I calculate grid/block size
	//printf("thread index %d", getGlobalIdx_1D_1D());
	float4 match = matches[getGlobalIdx_1D_1D()];


	float pix1[3] =
	{
		match.x, match.y, 1
	};
	float pix2[3] =
	{
		match.z, match.w, 1
	};


	float inter1[3];
	float inter2[3];

	float temp[3];
	multiply3x3x1_gpu(K_inv, pix1, temp);
	multiply3x3x1_gpu(rotationTranspose1, temp, inter1);
	multiply3x3x1_gpu(K_inv, pix2, temp);
	multiply3x3x1_gpu(rotationTranspose2, temp, inter2);

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

	normalize_gpu(v1);
	normalize_gpu(v2);



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

	multiply3x3x1_gpu( M1, worldP1, q1);
	multiply3x3x1_gpu( M2, worldP2, q2);

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
	inverse3x3_gpu(M, M_inv);
	multiply3x3x1_gpu(M_inv, Q, solution);



  	points[matchIndex].x = solution[0];
  	points[matchIndex].y = solution[1];
  	points[matchIndex].z = solution[2];

}


























































// yee
