#include "MatchFactory.cuh"

/*
fundamental matrix stuff
*/
float3 ssrlcv::multiply3x3x1(const float3 A[3], const float3 &B) {
  return {sum(A[0]*B),sum(A[1]*B),sum(A[2]*B)};
}
void ssrlcv::multiply3x3(const float3 A[3], const float3 B[3], float3 *C) {
  float3 bX = {B[0].x,B[1].x,B[2].x};
  float3 bY = {B[0].y,B[1].y,B[2].y};
  float3 bZ = {B[0].z,B[1].z,B[2].z};
  C[0] = {sum(A[0]*bX),sum(A[0]*bY),sum(A[0]*bZ)};
  C[1] = {sum(A[1]*bX),sum(A[1]*bY),sum(A[1]*bZ)};
  C[2] = {sum(A[2]*bX),sum(A[2]*bY),sum(A[2]*bZ)};
}
void ssrlcv::transpose3x3(const float3 M[3], float3 (&M_T)[3]) {
  M_T[0] = {M[0].x,M[1].x,M[2].x};
  M_T[1] = {M[0].y,M[1].y,M[2].y};
  M_T[2] = {M[0].z,M[1].z,M[2].z};
}
void ssrlcv::inverse3x3(float3 M[3], float3 (&Minv)[3]) {
  float d1 = M[1].y * M[2].z - M[2].y * M[1].z;
  float d2 = M[1].x * M[2].z - M[1].z * M[2].x;
  float d3 = M[1].x * M[2].y - M[1].y * M[2].x;
  float det = M[0].x*d1 - M[0].y*d2 + M[0].z*d3;
  if(det == 0) {
    // return pinv(M);
  }
  float invdet = 1/det;
  Minv[0].x = d1*invdet;
  Minv[0].y = (M[0].z*M[2].y - M[0].y*M[2].z) * invdet;
  Minv[0].z = (M[0].y*M[1].z - M[0].z*M[1].y) * invdet;
  Minv[1].x = -1 * d2 * invdet;
  Minv[1].y = (M[0].x*M[2].z - M[0].z*M[2].x) * invdet;
  Minv[1].z = (M[1].x*M[0].z - M[0].x*M[1].z) * invdet;
  Minv[2].x = d3 * invdet;
  Minv[2].y = (M[2].x*M[0].y - M[0].x*M[2].y) * invdet;
  Minv[2].z = (M[0].x*M[1].y - M[1].x*M[0].y) * invdet;
}
void ssrlcv::calcFundamentalMatrix_2View(Image_Descriptor query, Image_Descriptor target, float3 *F){
  if(query.fov != target.fov || query.foc != target.foc){
    std::cout<<"ERROR calculating fundamental matrix for 2view needs to bet taken with same camera (foc&fov are same)"<<std::endl;
    exit(-1);
  }
  float angle1;
  if(abs(query.cam_vec.z) < .00001) {
    if(query.cam_vec.y > 0)  angle1 = PI/2;
    else       angle1 = -1*PI/2;
  }
  else {
    angle1 = atan(query.cam_vec.y / query.cam_vec.z);
    if(query.cam_vec.z<0 && query.cam_vec.y>=0) {
      angle1 += PI;
    }
    if(query.cam_vec.z<0 && query.cam_vec.y<0) {
      angle1 -= PI;
    }
  }
  float3 A1[3] = {
    {1, 0, 0},
    {0, cos(angle1), -sin(angle1)},
    {0, sin(angle1), cos(angle1)}
  };

  float3 temp = multiply3x3x1(A1, query.cam_vec);

  float angle2 = 0.0f;
  if(abs(temp.z) < .00001) {
    if(temp.x <= 0)  angle1 = PI/2;
    else       angle1 = -1*PI/2;
  }
  else {
    angle2 = atan(-1*temp.x / temp.z);
    if(temp.z<0 && temp.x<0) {
      angle1 += PI;
    }
    if(temp.z<0 && temp.x>0) {
      angle2 -= PI;
    }
  }
  float3 B1[3] = {
    {cos(angle2), 0, sin(angle2)},
    {0, 1, 0},
    {-sin(angle2), 0, cos(angle2)}
  };

  float3 temp2 = multiply3x3x1(B1, temp);
  float3 rot1[3];
  multiply3x3(B1, A1, rot1);
  float3 rot1Transpose[3];
  transpose3x3(rot1,rot1Transpose);
  temp = multiply3x3x1(rot1Transpose, temp2);

  angle1 = 0.0f;
  if(abs(target.cam_vec.z) < .00001) {
    if(target.cam_vec.y > 0)  angle1 = PI/2;
    else       angle1 = -1*PI/2;
  }
  else {
    angle1 = atan(target.cam_vec.y / target.cam_vec.z);
    if(target.cam_vec.z<0 && target.cam_vec.y>=0) {
      angle1 += PI;
    }
    if(target.cam_vec.z<0 && target.cam_vec.y<0) {
      angle1 -= PI;
    }
  }
  float3 A2[3] = {
    {1, 0, 0},
    {0, cos(angle1), -sin(angle1)},
    {0, sin(angle1), cos(angle1)}
  };
  temp2 = multiply3x3x1(A2, target.cam_vec);

  angle2 = 0.0f;
  if(abs(temp2.z) < .00001) {
    if(temp2.x <= 0)  angle1 = PI/2;
    else       angle1 = -1*PI/2;
  }
  else {
    angle2 = atan(-1*temp2.x / temp2.z);
    if(temp2.z<0 && temp2.x<0) {
      angle1 += PI;
    }
    if(temp2.z<0 && temp2.x>0) {
      angle2 -= PI;
    }
  }
  float3 B2[3] = {
    {cos(angle2), 0, sin(angle2)},
    {0, 1, 0},
    {-sin(angle2), 0, cos(angle2)}
  };

  temp = multiply3x3x1(B2, temp2);

  float3 rot2[3];
  multiply3x3(B2, A2, rot2);
  float3 rot2Transpose[3];
  transpose3x3(rot2, rot2Transpose);

  temp2 = multiply3x3x1(rot2Transpose, temp);

  float2 dpix = {query.foc*tan(query.fov/2)/(query.size.x/2),
    query.foc*tan(query.fov/2)/(query.size.y/2)};

  float3 K[3] = {
    {query.foc/dpix.x, 0, ((float)query.size.x)/2.0f},
    {0, query.foc/dpix.y, ((float)query.size.y)/2.0f},
    {0, 0, 1}
  };
  float3 K_inv[3];
  inverse3x3(K,K_inv);
  float3 K_invTranspose[3];
  transpose3x3(K_inv,K_invTranspose);

  float3 R[3];
  multiply3x3(rot2Transpose, rot1, R);
  float3 S[3] = {
    {0, query.cam_pos.z - target.cam_pos.z, target.cam_pos.y - query.cam_pos.y},
    {query.cam_pos.z - target.cam_pos.z,0, query.cam_pos.x - target.cam_pos.x},
    {query.cam_pos.y - target.cam_pos.y, target.cam_pos.x - query.cam_pos.x, 0}
  };
  float3 E[3];;
  multiply3x3(R,S,E);
  float3 tempF[3];
  multiply3x3(K_invTranspose, E,tempF);
  multiply3x3(tempF, K_inv, F);
  std::cout << std::endl <<"between image "<<query.id<<" and "<<target.id
  <<" the final fundamental matrix result is: " << std::endl;
  for(int r = 0; r < 3; ++r) {
    std::cout << F[r].x << "  " << F[r].y << " "<<  F[r].z << std::endl;
  }
  std::cout<<std::endl;
}

ssrlcv::MatchFactory::MatchFactory(){

}


void ssrlcv::MatchFactory::refineMatches(ssrlcv::Unity<ssrlcv::Match>* matches, float cutoffRatio){
  if(cutoffRatio == 0.0f){
    std::cout<<"ERROR illegal value used for cutoff ratio: 0.0"<<std::endl;
    exit(-1);
  }
  MemoryState origin = matches->state;
  if(origin != both){
    matches->transferMemoryTo(both);
  }

  float max = 0.0f;
  float min = FLT_MAX;
  for(int i = 0; i < matches->numElements; ++i){
    if(matches->host[i].distance < min) min = matches->host[i].distance;
    if(matches->host[i].distance > max) max = matches->host[i].distance;
  }
  if(origin == gpu) matches->clear(cpu);

  printf("max dist = %f || min dist = %f\n",max,min);
  int* matchCounter_device = nullptr;
  CudaSafeCall(cudaMalloc((void**)&matchCounter_device, matches->numElements*sizeof(int)));

  dim3 grid = {1,1,1};
  dim3 block = {1,1,1};
  getFlatGridBlock(matches->numElements, grid, block);
  refineWCutoffRatio<<<grid,block>>>(matches->numElements, matches->device, matchCounter_device, {min, max}, cutoffRatio);
  cudaDeviceSynchronize();
  CudaCheckError();

  thrust::device_ptr<int> sum(matchCounter_device);
  thrust::inclusive_scan(sum, sum + matches->numElements, sum);
  unsigned long afterCompaction = 0;
  CudaSafeCall(cudaMemcpy(&(afterCompaction),matchCounter_device + (matches->numElements - 1), sizeof(int), cudaMemcpyDeviceToHost));

  Match* minimizedMatches_device = nullptr;
  CudaSafeCall(cudaMalloc((void**)&minimizedMatches_device, afterCompaction*sizeof(Match)));

  copyMatches<<<grid,block>>>(matches->numElements, matchCounter_device, minimizedMatches_device, matches->device);
  cudaDeviceSynchronize();
  CudaCheckError();
  unsigned long beforeCompaction = matches->numElements;

  matches->clear();
  matches->setData(minimizedMatches_device, afterCompaction, gpu);

  printf("numMatches after eliminating base on %f cutoffRatio = %d (was %d)\n",cutoffRatio,matches->numElements,beforeCompaction);

  matches->transferMemoryTo(origin);
  if(origin == cpu){
    matches->clear(gpu);
  }
}

ssrlcv::Unity<ssrlcv::Match>* ssrlcv::MatchFactory::generateMatchesBruteForce(ssrlcv::Image* query, ssrlcv::Unity<ssrlcv::Feature<ssrlcv::SIFT_Descriptor>>* queryFeatures,
ssrlcv::Image* target, ssrlcv::Unity<ssrlcv::Feature<ssrlcv::SIFT_Descriptor>>* targetFeatures){

  MemoryState origin[2] = {queryFeatures->state, targetFeatures->state};

  if(queryFeatures->fore == cpu) queryFeatures->transferMemoryTo(gpu);
  if(targetFeatures->fore == cpu) targetFeatures->transferMemoryTo(gpu);

  unsigned int numPossibleMatches = queryFeatures->numElements;

  Match* matches_device = nullptr;
  CudaSafeCall(cudaMalloc((void**)&matches_device, numPossibleMatches*sizeof(Match)));

  Unity<Match>* matches = new Unity<Match>(matches_device, numPossibleMatches, gpu);

  dim3 grid = {1,1,1};
  dim3 block = {1024,1,1};
  getGrid(matches->numElements,grid);

  clock_t timer = clock();

  matchFeaturesBruteForce<<<grid, block>>>(query->descriptor.id, queryFeatures->numElements, queryFeatures->device,
    target->descriptor.id, targetFeatures->numElements, targetFeatures->device, matches->device);

  cudaDeviceSynchronize();
  CudaCheckError();

  printf("done in %f seconds.\n\n",((float) clock() -  timer)/CLOCKS_PER_SEC);

  matches->transferMemoryTo(cpu);
  matches->clear(gpu);

  if(origin[0] != queryFeatures->state){
    queryFeatures->setMemoryState(origin[0]);
  }
  if(origin[1] != targetFeatures->state){
    targetFeatures->setMemoryState(origin[1]);
  }

  return matches;
}

ssrlcv::Unity<ssrlcv::Match>* ssrlcv::MatchFactory::generateMatchesConstrained(ssrlcv::Image* query, ssrlcv::Unity<ssrlcv::Feature<ssrlcv::SIFT_Descriptor>>* queryFeatures,
ssrlcv::Image* target, ssrlcv::Unity<ssrlcv::Feature<ssrlcv::SIFT_Descriptor>>* targetFeatures, float epsilon){

  MemoryState origin[2] = {queryFeatures->state, targetFeatures->state};

  if(queryFeatures->fore == cpu) queryFeatures->transferMemoryTo(gpu);
  if(targetFeatures->fore == cpu) targetFeatures->transferMemoryTo(gpu);

  unsigned int numPossibleMatches = queryFeatures->numElements;

  Match* matches_device = nullptr;
  CudaSafeCall(cudaMalloc((void**)&matches_device, numPossibleMatches*sizeof(Match)));

  Unity<Match>* matches = new Unity<Match>(matches_device, numPossibleMatches, gpu);

  dim3 grid = {1,1,1};
  dim3 block = {1024,1,1};
  getGrid(matches->numElements,grid);

  clock_t timer = clock();
  float3* fundamental = new float3[3];
  calcFundamentalMatrix_2View(query->descriptor, target->descriptor, fundamental);

  float3* fundamental_device;
  CudaSafeCall(cudaMalloc((void**)&fundamental_device, 3*sizeof(float3)));
  CudaSafeCall(cudaMemcpy(fundamental_device, fundamental, 3*sizeof(float3), cudaMemcpyHostToDevice));

  matchFeaturesConstrained<<<grid, block>>>(query->descriptor.id, queryFeatures->numElements, queryFeatures->device,
    target->descriptor.id, targetFeatures->numElements, targetFeatures->device, matches->device, epsilon, fundamental_device);
  cudaDeviceSynchronize();
  CudaCheckError();

  CudaSafeCall(cudaFree(fundamental_device));

  printf("done in %f seconds.\n\n",((float) clock() -  timer)/CLOCKS_PER_SEC);

  matches->transferMemoryTo(cpu);
  matches->clear(gpu);

  queryFeatures->transferMemoryTo(origin[0]);
  if(origin[0] == cpu){
    queryFeatures->clear(gpu);
  }
  targetFeatures->transferMemoryTo(origin[1]);
  if(origin[1] == cpu){
    targetFeatures->clear(gpu);
  }

  return matches;

}

ssrlcv::Unity<ssrlcv::Match>* ssrlcv::MatchFactory::generateSubPixelMatchesBruteForce(ssrlcv::Image* query, ssrlcv::Unity<ssrlcv::Feature<ssrlcv::SIFT_Descriptor>>* queryFeatures,
ssrlcv::Image* target, ssrlcv::Unity<ssrlcv::Feature<ssrlcv::SIFT_Descriptor>>* targetFeatures){

  MemoryState origin[2] = {queryFeatures->state, targetFeatures->state};

  if(queryFeatures->fore == cpu) queryFeatures->transferMemoryTo(gpu);
  if(targetFeatures->fore == cpu) targetFeatures->transferMemoryTo(gpu);

  Unity<Match>* matches = this->generateMatchesBruteForce(query, queryFeatures, target, targetFeatures);
  matches->transferMemoryTo(gpu);

  SubpixelM7x7* subDescriptors_device;
  CudaSafeCall(cudaMalloc((void**)&subDescriptors_device, matches->numElements*sizeof(SubpixelM7x7)));

  dim3 grid = {1,1,1};
  dim3 block = {9,9,1};
  getGrid(matches->numElements, grid);
  std::cout<<"initializing subPixelMatches..."<<std::endl;
  clock_t timer = clock();
  initializeSubPixels<<<grid, block>>>(matches->numElements, matches->device, subDescriptors_device,
    query->descriptor, queryFeatures->numElements, queryFeatures->device,
    target->descriptor, targetFeatures->numElements, targetFeatures->device);

  cudaDeviceSynchronize();
  CudaCheckError();
  printf("done in %f seconds.\n\n",((float) clock() -  timer)/CLOCKS_PER_SEC);

  Spline* splines_device;
  CudaSafeCall(cudaMalloc((void**)&splines_device, matches->numElements*2*sizeof(Spline)));

  grid = {1,1,1};
  block = {6,6,4};
  getGrid(matches->numElements*2, grid);

  std::cout<<"filling bicubic splines..."<<std::endl;
  timer = clock();
  fillSplines<<<grid,block>>>(matches->numElements, subDescriptors_device, splines_device);
  cudaDeviceSynchronize();
  CudaCheckError();
  printf("done in %f seconds.\n\n",((float) clock() -  timer)/CLOCKS_PER_SEC);
  CudaSafeCall(cudaFree(subDescriptors_device));

  std::cout<<"determining subpixel locations..."<<std::endl;
  timer = clock();
  determineSubPixelLocationsBruteForce<<<grid,block>>>(0.1, matches->numElements, matches->device, splines_device);
  cudaDeviceSynchronize();
  CudaCheckError();
  printf("done in %f seconds.\n\n",((float) clock() -  timer)/CLOCKS_PER_SEC);
  CudaSafeCall(cudaFree(splines_device));

  matches->transferMemoryTo(cpu);
  matches->clear(gpu);

  queryFeatures->transferMemoryTo(origin[0]);
  if(origin[0] == cpu){
    queryFeatures->clear(gpu);
  }
  targetFeatures->transferMemoryTo(origin[1]);
  if(origin[1] == cpu){
    targetFeatures->clear(gpu);
  }

  return matches;
}

ssrlcv::Unity<ssrlcv::Match>* ssrlcv::MatchFactory::generateSubPixelMatchesConstrained(ssrlcv::Image* query, ssrlcv::Unity<ssrlcv::Feature<ssrlcv::SIFT_Descriptor>>* queryFeatures,
ssrlcv::Image* target, ssrlcv::Unity<ssrlcv::Feature<ssrlcv::SIFT_Descriptor>>* targetFeatures, float epsilon){
  MemoryState origin[2] = {queryFeatures->state, targetFeatures->state};

  if(queryFeatures->fore == cpu) queryFeatures->transferMemoryTo(gpu);
  if(targetFeatures->fore == cpu) targetFeatures->transferMemoryTo(gpu);

  Unity<Match>* matches = this->generateMatchesConstrained(query, queryFeatures, target, targetFeatures, epsilon);
  matches->transferMemoryTo(gpu);

  SubpixelM7x7* subDescriptors_device;
  CudaSafeCall(cudaMalloc((void**)&subDescriptors_device, matches->numElements*sizeof(SubpixelM7x7)));

  dim3 grid = {1,1,1};
  dim3 block = {9,9,1};
  getGrid(matches->numElements, grid);
  std::cout<<"initializing subPixelMatches..."<<std::endl;
  clock_t timer = clock();
  initializeSubPixels<<<grid, block>>>(matches->numElements, matches->device, subDescriptors_device,
    query->descriptor, queryFeatures->numElements, queryFeatures->device,
    target->descriptor, targetFeatures->numElements, targetFeatures->device);

  cudaDeviceSynchronize();
  CudaCheckError();
  printf("done in %f seconds.\n\n",((float) clock() -  timer)/CLOCKS_PER_SEC);

  Spline* splines_device;
  CudaSafeCall(cudaMalloc((void**)&splines_device, matches->numElements*2*sizeof(Spline)));

  grid = {1,1,1};
  block = {6,6,4};
  getGrid(matches->numElements*2, grid);

  std::cout<<"filling bicubic splines..."<<std::endl;
  timer = clock();
  fillSplines<<<grid,block>>>(matches->numElements, subDescriptors_device, splines_device);
  cudaDeviceSynchronize();
  CudaCheckError();
  printf("done in %f seconds.\n\n",((float) clock() -  timer)/CLOCKS_PER_SEC);
  CudaSafeCall(cudaFree(subDescriptors_device));

  std::cout<<"determining subpixel locations..."<<std::endl;
  timer = clock();
  determineSubPixelLocationsBruteForce<<<grid,block>>>(0.1, matches->numElements, matches->device, splines_device);
  cudaDeviceSynchronize();
  CudaCheckError();
  printf("done in %f seconds.\n\n",((float) clock() -  timer)/CLOCKS_PER_SEC);
  CudaSafeCall(cudaFree(splines_device));

  matches->transferMemoryTo(cpu);
  matches->clear(gpu);

  queryFeatures->transferMemoryTo(origin[0]);
  if(origin[0] == cpu){
    queryFeatures->clear(gpu);
  }
  targetFeatures->transferMemoryTo(origin[1]);
  if(origin[1] == cpu){
    targetFeatures->clear(gpu);
  }

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

__device__ __host__ __forceinline__ float ssrlcv::sum(const float3 &a){
  return a.x + a.y + a.z;
}
__device__ __forceinline__ float ssrlcv::square(const float &a){
  return a*a;
}
__device__ __forceinline__ float ssrlcv::calcElucid(const int2 &a, const int2 &b){
  return sqrtf(dotProduct(a-b, a-b));
}
__device__ __forceinline__ float ssrlcv::calcElucid(const unsigned char a[128], const unsigned char b[128]){
  float dist = 0.0f;
  for(int i = 0; i < 128; ++i){
    dist += sqrtf(((float)(a[i] - b[i]))*((float)(a[i] - b[i])));
  }
  return dist;
}
__device__ __forceinline__ float ssrlcv::atomicMinFloat (float * addr, float value) {
  float old;
  old = (value >= 0) ? __int_as_float(atomicMin((int *)addr, __float_as_int(value))) :
    __uint_as_float(atomicMax((unsigned int *)addr, __float_as_uint(value)));
  return old;
}
__device__ __forceinline__ float ssrlcv::atomicMaxFloat (float * addr, float value) {
  float old;
  old = (value >= 0) ? __int_as_float(atomicMax((int *)addr, __float_as_int(value))) :
    __uint_as_float(atomicMin((unsigned int *)addr, __float_as_uint(value)));
  return old;
}
__device__ __forceinline__ float ssrlcv::findSubPixelContributer(const float2 &loc, const int &width){
  return ((loc.y - 12)*(width - 24)) + (loc.x - 12);
}

/*
matching
*/
__global__ void ssrlcv::matchFeaturesBruteForce(unsigned int queryImageID, unsigned long numFeaturesQuery,
ssrlcv::Feature<ssrlcv::SIFT_Descriptor>* featuresQuery, unsigned int targetImageID, unsigned long numFeaturesTarget,
ssrlcv::Feature<ssrlcv::SIFT_Descriptor>* featuresTarget, ssrlcv::Match* matches){
  unsigned long blockId = blockIdx.y * gridDim.x + blockIdx.x;
  if(blockId < numFeaturesQuery){
    Feature<SIFT_Descriptor> feature = featuresQuery[blockId];
    __shared__ int localMatch[1024];
    __shared__ float localDist[1024];
    localMatch[threadIdx.x] = -1;
    localDist[threadIdx.x] = FLT_MAX;
    __syncthreads();
    float currentDist = 0.0f;
    unsigned long numFeaturesTarget_register = numFeaturesQuery;
    for(int f = threadIdx.x; f < numFeaturesTarget_register; f += 1024){
      currentDist = 0.0f;
      for(int i = 0; i < 128; ++i){
        currentDist +=  square(((float)feature.descriptor.values[i])-((float)featuresTarget[f].descriptor.values[i]));
      }
      if(localDist[threadIdx.x] > currentDist){
        localDist[threadIdx.x] = currentDist;
        localMatch[threadIdx.x] = f;
      }
    }
    __syncthreads();
    if(threadIdx.x != 0) return;
    currentDist = FLT_MAX;
    int matchIndex = -1;
    for(int i = 0; i < 1024; ++i){
      if(currentDist > localDist[i]){
        currentDist = localDist[i];
        matchIndex = localMatch[i];
      }
    }
    Match match;
    match.features[0] = Feature<unsigned int>(feature.loc,queryImageID);
    match.features[1] = Feature<unsigned int>(featuresTarget[matchIndex].loc,targetImageID);
    match.distance = currentDist;
    matches[blockId] = match;
  }
}

__global__ void ssrlcv::matchFeaturesConstrained(unsigned int queryImageID, unsigned long numFeaturesQuery,
ssrlcv::Feature<ssrlcv::SIFT_Descriptor>* featuresQuery, unsigned int targetImageID, unsigned long numFeaturesTarget,
ssrlcv::Feature<ssrlcv::SIFT_Descriptor>* featuresTarget, ssrlcv::Match* matches, float epsilon, float3 fundamental[3]){
  unsigned long blockId = blockIdx.y * gridDim.x + blockIdx.x;
  if(blockId < numFeaturesQuery){
    Feature<SIFT_Descriptor> feature = featuresQuery[blockId];
    __shared__ int localMatch[1024];
    __shared__ float localDist[1024];
    localMatch[threadIdx.x] = -1;
    localDist[threadIdx.x] = FLT_MAX;
    __syncthreads();
    float currentDist = 0.0f;
    unsigned long numFeaturesTarget_register = numFeaturesQuery;

    float3 epipolar = {0.0f,0.0f,0.0f};
    epipolar.x = (fundamental[0].x*feature.loc.x) + (fundamental[0].y*feature.loc.y) + fundamental[0].z;
    epipolar.y = (fundamental[1].x*feature.loc.x) + (fundamental[1].y*feature.loc.y) + fundamental[1].z;
    epipolar.z = (fundamental[2].x*feature.loc.x) + (fundamental[2].y*feature.loc.y) + fundamental[2].z;

    float p = 0.0f;

    Feature<SIFT_Descriptor> currentFeature;
    float regEpsilon = epsilon;

    for(int f = threadIdx.x; f < numFeaturesTarget_register; f += 1024){

      currentFeature = featuresTarget[f];
      //ax + by + c = 0
      p = -1*((epipolar.x*currentFeature.loc.x) + epipolar.z)/epipolar.y;
      if(abs(currentFeature.loc.y - p) >= regEpsilon) continue;
      currentDist = 0.0f;
      for(int i = 0; i < 128; ++i){
        currentDist +=  square(((float)feature.descriptor.values[i])-((float)currentFeature.descriptor.values[i]));
      }
      if(localDist[threadIdx.x] > currentDist){
        localDist[threadIdx.x] = currentDist;
        localMatch[threadIdx.x] = f;
      }
    }
    __syncthreads();
    if(threadIdx.x != 0) return;
    currentDist = FLT_MAX;
    int matchIndex = -1;
    for(int i = 0; i < 1024; ++i){
      if(currentDist > localDist[i]){
        currentDist = localDist[i];
        matchIndex = localMatch[i];
      }
    }
    Match match;
    match.features[0] = Feature<unsigned int>(feature.loc,queryImageID);
    match.features[1] = Feature<unsigned int>(featuresTarget[matchIndex].loc,targetImageID);
    match.distance = currentDist;
    matches[blockId] = match;
  }
}


/*
subpixel stuff
*/
//TODO overload this kernel for different types of descriptors

//NOTE THIS MIGHT ONLY WORK FOR DENSE SIFT
__global__ void ssrlcv::initializeSubPixels(unsigned long numMatches, ssrlcv::Match* matches, ssrlcv::SubpixelM7x7* subPixelDescriptors,
ssrlcv::Image_Descriptor query, unsigned long numFeaturesQuery, ssrlcv::Feature<ssrlcv::SIFT_Descriptor>* featuresQuery,
ssrlcv::Image_Descriptor target, unsigned long numFeaturesTarget, ssrlcv::Feature<ssrlcv::SIFT_Descriptor>* featuresTarget){
  unsigned long blockId = blockIdx.y * gridDim.x + blockIdx.x;
  if(blockId < numMatches){
    __shared__ SubpixelM7x7 subDescriptor;
    Match match = matches[blockId];

    //this now needs to be actual indices to contributers
    int2 contrib = {((int)threadIdx.x) - 4, ((int)threadIdx.y) - 4};
    int contribQuery = findSubPixelContributer(match.features[0].loc + contrib, query.size.x);
    int contribTarget = findSubPixelContributer(match.features[1].loc + contrib, target.size.x);

    int pairedMatchIndex = findSubPixelContributer(match.features[1].loc, target.size.x);

    bool foundM1 = false;
    bool foundM2 = false;

    if(contribTarget >= 0 && contribTarget < numFeaturesTarget){
      subDescriptor.M1[threadIdx.x][threadIdx.y] = calcElucid(featuresQuery[blockId].descriptor.values, featuresTarget[contribTarget].descriptor.values);
      foundM1 = true;
    }
    if(contribQuery >= 0 && contribQuery < numFeaturesQuery){
      subDescriptor.M2[threadIdx.x][threadIdx.y] = calcElucid(featuresQuery[contribQuery].descriptor.values, featuresTarget[pairedMatchIndex].descriptor.values);
      foundM2 = true;
    }
    __syncthreads();
    //COME up with better way to do this
    if(!foundM1){
      float val = 0.0f;
      for(int x = 0; x < 9; ++x){
        for(int y = 0; y < 9; ++y){
          val += subDescriptor.M1[x][y];
        }
      }
      subDescriptor.M1[threadIdx.x][threadIdx.y] = val/81;
    }
    if(!foundM2){
      float val = 0.0f;
      for(int x = 0; x < 9; ++x){
        for(int y = 0; y < 9; ++y){
          val += subDescriptor.M2[x][y];
        }
      }
      subDescriptor.M2[threadIdx.x][threadIdx.y] = val/81;
    }
    __syncthreads();
    if(threadIdx.x == 0 && threadIdx.y == 0){
      subPixelDescriptors[blockId] = subDescriptor;
    }
  }
}

__global__ void ssrlcv::fillSplines(unsigned long numMatches, SubpixelM7x7* subPixelDescriptors, ssrlcv::Spline* splines){
  unsigned long blockId = blockIdx.y * gridDim.x + blockIdx.x;
  if(blockId < numMatches*2){
    float descriptor[9][9];
    for(int x = 0; x < 9; ++x){
      for(int y = 0; y < 9; ++y){
        descriptor[x][y] = (blockId%2 == 0) ? subPixelDescriptors[blockId/2].M1[x][y] : subPixelDescriptors[blockId/2].M2[x][y];
      }
    }

    __shared__ Spline spline;
    int2 corner = {
      ((int)threadIdx.z)%2,
      ((int)threadIdx.z)/2
    };
    int2 contributer = {
      ((int)threadIdx.x) + 2 + corner.x,
      ((int)threadIdx.y) + 2 + corner.y
    };
    float4 localCoeff;
    localCoeff.x = descriptor[contributer.x][contributer.y];
    localCoeff.y = descriptor[contributer.x + 1][contributer.y] - descriptor[contributer.x - 1][contributer.y];
    localCoeff.z = descriptor[contributer.x][contributer.y + 1] - descriptor[contributer.x][contributer.y - 1];
    localCoeff.w = descriptor[contributer.x + 1][contributer.y + 1] - descriptor[contributer.x - 1][contributer.y - 1];

    spline.coeff[threadIdx.x][threadIdx.y][corner.x][corner.y] = localCoeff.x;
    spline.coeff[threadIdx.x][threadIdx.y][corner.x][corner.y + 2] = localCoeff.y;
    spline.coeff[threadIdx.x][threadIdx.y][corner.x + 2][corner.y] = localCoeff.z;
    spline.coeff[threadIdx.x][threadIdx.y][corner.x + 2][corner.y + 2] = localCoeff.z;

    // Multiplying matrix a and b and storing in array mult.
    if(threadIdx.z != 0) return;
    float mult[4][4] = {0.0f};
    for(int i = 0; i < 4; ++i){
      for(int j = 0; j < 4; ++j){
        for(int c = 0; c < 4; ++c){
          mult[i][j] += splineHelper[i][c]*spline.coeff[threadIdx.x][threadIdx.y][c][j];
        }
      }
    }
    for(int i = 0; i < 4; ++i){
      for(int j = 0; j < 4; ++j){
        spline.coeff[threadIdx.x][threadIdx.y][i][j] = 0.0f;
      }
    }
    for(int i = 0; i < 4; ++i){
      for(int j = 0; j < 4; ++j){
        for(int c = 0; c < 4; ++c){
          spline.coeff[threadIdx.x][threadIdx.y][i][j] += mult[i][c]*splineHelperInv[c][j];
        }
      }
    }

    __syncthreads();
    splines[blockId] = spline;
  }
}

//NOTE THIS UPDATES FEATURE.LOC in MATCH
__global__ void ssrlcv::determineSubPixelLocationsBruteForce(float increment, unsigned long numMatches, ssrlcv::Match* matches, ssrlcv::Spline* splines){
  unsigned long blockId = blockIdx.y * gridDim.x + blockIdx.x;
  if(blockId < numMatches*2){
    __shared__ float minimum;
    minimum = FLT_MAX;
    __syncthreads();
    float localCoeff[4][4];
    for(int i = 0; i < 4; ++i){
      for(int j = 0; j < 4; ++j){
        localCoeff[i][j] = splines[blockId].coeff[threadIdx.x][threadIdx.y][i][j];
      }
    }
    float value = 0.0f;
    float localMin = FLT_MAX;
    float2 localSubLoc = {0.0f,0.0f};
    for(float x = -1.0f; x <= 1.0f; x+=increment){
      for(float y = -1.0f; y <= 1.0f; y+=increment){
        value = 0.0f;
        for(int i = 0; i < 4; ++i){
          for(int j = 0; j < 4; ++j){
            value += (localCoeff[i][j]*powf(x,i)*powf(y,j));
          }
        }
        if(value < localMin){
          localMin = value;
          localSubLoc = {x,y};
        }
      }
    }
    atomicMinFloat(&minimum, localMin);
    __syncthreads();
    if(localMin == minimum){
      if(blockId%2 == 0) matches[blockId/2].features[0].loc  = localSubLoc + matches[blockId/2].features[0].loc;
      else matches[blockId/2].features[1].loc = localSubLoc + matches[blockId/2].features[1].loc;
    }
    else return;
  }
}

/*
MATCH REFINEMENT
*/

//NOTE may be able to replace this with thrust stream compaction
__global__ void ssrlcv::refineWCutoffRatio(unsigned long numMatches, ssrlcv::Match* matches, int* matchCounter, float2 minMax, float cutoffRatio){
  unsigned long globalId = blockIdx.x*blockDim.x + threadIdx.x;
  if(globalId < numMatches){
    float2 regMinMax = minMax;
    if((matches[globalId].distance - regMinMax.x)/(regMinMax.y-regMinMax.x) < cutoffRatio){
      matchCounter[globalId] = 1;
    }
    else{
      matchCounter[globalId] = 0;
    }
  }
}
__global__ void ssrlcv::copyMatches(unsigned long numMatches, int* matchCounter, ssrlcv::Match* minimizedMatches, ssrlcv::Match* matches){
  unsigned long globalId = blockIdx.x*blockDim.x + threadIdx.x;
  if(globalId < numMatches){
    int counterVal = matchCounter[globalId];
    if(counterVal != 0 && counterVal > matchCounter[globalId - 1]){
      minimizedMatches[counterVal - 1] = matches[globalId];
    }
  }
}
