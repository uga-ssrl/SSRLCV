#include "FeatureFactory.cuh"

/*
HOST METHODS
*/
//Base feature factory


ssrlcv::FeatureFactory::FeatureFactory(){

}

ssrlcv::SIFT_FeatureFactory::SIFT_FeatureFactory(){

}

ssrlcv::Unity<ssrlcv::Feature<ssrlcv::SIFT_Descriptor>>* ssrlcv::SIFT_FeatureFactory::generateFeaturesDensly(ssrlcv::Image* image, unsigned int binDepth){
  Unity<Feature<SIFT_Descriptor>>* features = nullptr;

  if(image->quadtree == nullptr){
    std::cerr<<"ERROR: generateFeaturesDensly for Feature<SIFT_Descriptor> requires an image in a quadtree"<<std::endl;
    exit(-1);
  }
  if(image->quadtree->depth <= binDepth){
    std::cerr<<"ERROR: binDepth must be less than the quadtree depth"<<std::endl;
    exit(-1);
  }
  std::cout<<"generating features"<<std::endl;
  MemoryState origin[2] = {image->quadtree->nodes->state,image->quadtree->nodeDepthIndex->state};

  if(image->quadtree->nodes->fore == cpu){
    image->quadtree->nodes->transferMemoryTo(gpu);
  }
  if(image->quadtree->nodeDepthIndex->fore == gpu){
    image->quadtree->nodeDepthIndex->transferMemoryTo(cpu);
  }



  dim3 grid = {1,1,1};
  dim3 block = {1,1,1};
  unsigned int* featureNumbers_device = nullptr;
  unsigned int* featureAddresses_device = nullptr;

  unsigned int possibleFeatures = 0;
  unsigned int depthIndex = 0;
  if(binDepth == 0){
    possibleFeatures = image->quadtree->nodeDepthIndex->host[1];
  }
  else{
    depthIndex = image->quadtree->nodeDepthIndex->host[binDepth];
    possibleFeatures = image->quadtree->nodeDepthIndex->host[binDepth + 1] - depthIndex;
  }

  CudaSafeCall(cudaMalloc((void**)&featureNumbers_device,possibleFeatures*sizeof(unsigned int)));
  CudaSafeCall(cudaMalloc((void**)&featureAddresses_device,possibleFeatures*sizeof(unsigned int)));
  getFlatGridBlock(image->quadtree->nodeDepthIndex->host[1],grid,block);
  findValidFeatures<<<grid,block>>>(possibleFeatures,depthIndex,image->quadtree->nodes->device,featureNumbers_device,featureAddresses_device);
  cudaDeviceSynchronize();
  CudaCheckError();

  thrust::device_ptr<unsigned int> fN(featureNumbers_device);
  thrust::inclusive_scan(fN, fN + possibleFeatures, fN);

  unsigned int numSIFTFeatures = 0;
  CudaSafeCall(cudaMemcpy(&numSIFTFeatures,featureNumbers_device + (possibleFeatures - 1), sizeof(unsigned int), cudaMemcpyDeviceToHost));
  CudaSafeCall(cudaFree(featureNumbers_device));

  std::cout<<numSIFTFeatures<<" Feature<SIFT_Descriptor> were found during dense search"<<std::endl;

  thrust::device_ptr<unsigned int> fA(featureAddresses_device);
  thrust::remove(fA, fA + possibleFeatures, 0);

  grid = {1,1,1};
  block = {1,1,1};
  getFlatGridBlock(numSIFTFeatures, grid, block);
  Feature<SIFT_Descriptor>* features_device = nullptr;
  CudaSafeCall(cudaMalloc((void**)&features_device,numSIFTFeatures*sizeof(Feature<SIFT_Descriptor>)));
  features = new Unity<Feature<SIFT_Descriptor>>(features_device,numSIFTFeatures,gpu);

  fillValidFeatures<<<grid,block>>>(features->numElements,features->device,featureAddresses_device,image->quadtree->nodes->device);
  cudaDeviceSynchronize();
  CudaCheckError();

  CudaSafeCall(cudaFree(featureAddresses_device));

  this->fillDescriptors(image, features);


  if(origin[0] != image->quadtree->nodes->state){
    image->quadtree->nodes->setMemoryState(origin[0]);
  }
  if(origin[1] != image->quadtree->nodeDepthIndex->state){
    image->quadtree->nodeDepthIndex->setMemoryState(origin[1]);
  }

  return features;
}
void ssrlcv::SIFT_FeatureFactory::fillDescriptors(ssrlcv::Image* image, ssrlcv::Unity<ssrlcv::Feature<ssrlcv::SIFT_Descriptor>>* features){

  MemoryState origin = image->quadtree->data->state;

  dim3 grid = {1,1,1};
  dim3 block = {9,1,1};
  getGrid(features->numElements, grid);

  if(image->quadtree->data->fore == cpu){
    image->quadtree->data->transferMemoryTo(gpu);
  }
  std::cout<<"computing thetas for feature descriptors..."<<std::endl;
  clock_t timer = clock();
  computeThetas<<<grid, block>>>(features->numElements, features->device, image->quadtree->nodes->device, image->quadtree->data->device);
  cudaDeviceSynchronize();
  CudaCheckError();
  printf("done in %f seconds.\n\n",((float) clock() -  timer)/CLOCKS_PER_SEC);

  block = {16,16,1};
  std::cout<<"generating feature descriptors..."<<std::endl;
  timer = clock();
  fillDescriptorsDensly<<<grid,block>>>(features->numElements, features->device, image->quadtree->nodes->device, image->quadtree->data->device);
  cudaDeviceSynchronize();
  CudaCheckError();
  printf("done in %f seconds.\n\n",((float) clock() -  timer)/CLOCKS_PER_SEC);

  if(origin != image->quadtree->data->state){
    image->quadtree->data->setMemoryState(origin);
  }
}

/*
CUDA implementations
*/

__constant__ float ssrlcv::pi = 3.1415926535897932384626433832795028841971693993751058209749445923078164062;

/*
DEVICE METHODS
*/
__device__ __forceinline__ unsigned long ssrlcv::getGlobalIdx_2D_1D(){
  unsigned long blockId = blockIdx.y * gridDim.x + blockIdx.x;
  unsigned long threadId = blockId * blockDim.x + threadIdx.x;
  return threadId;
}
__device__ __forceinline__ float ssrlcv::getMagnitude(const int2 &vector){
  return sqrtf(dotProduct(vector, vector));
}
__device__ __forceinline__ float ssrlcv::getTheta(const int2 &vector){
  float theta = atan2f((float)vector.y, (float)vector.x) + pi;
  return fmod(theta,2.0f*pi);
}
__device__ __forceinline__ float ssrlcv::getTheta(const float2 &vector){
  float theta = atan2f(vector.y, vector.x) + pi;
  return fmod(theta,2.0f*pi);
}
__device__ __forceinline__ float ssrlcv::getTheta(const float2 &vector, const float &offset){
  float theta = (atan2f(vector.y, vector.x) + pi) - offset;
  return fmod(theta + 2.0f*pi,2.0f*pi);
}
__device__ void ssrlcv::trickleSwap(const float2 &compareWValue, float2* &arr, int index, const int &length){
  for(int i = index; i < length; ++i){
    if(compareWValue.x > arr[i].x){
      float2 temp = arr[i];
      arr[i] = compareWValue;
      if((temp.x == 0.0f && temp.y == 0.0f)|| index + 1 == length) return;
      return trickleSwap(temp, arr, index + 1, length);
    }
  }
}
__device__ __forceinline__ long4 ssrlcv::getOrientationContributers(const long2 &loc, const uint2 &imageSize){
  long4 orientationContributers;
  long pixelIndex = loc.y*imageSize.x + loc.x;
  orientationContributers.x = (loc.x == imageSize.x - 1) ? -1 : pixelIndex + 1;
  orientationContributers.y = (loc.x == 0) ? -1 : pixelIndex - 1;
  orientationContributers.z = (loc.y == imageSize.y - 1) ? -1 : (loc.y + 1)*imageSize.x + loc.x;
  orientationContributers.w = (loc.y == 0) ? -1 : (loc.y - 1)*imageSize.x + loc.x;
  return orientationContributers;
}
__device__ __forceinline__ int ssrlcv::floatToOrderedInt(float floatVal){
 int intVal = __float_as_int( floatVal );
 return (intVal >= 0 ) ? intVal : intVal ^ 0x7FFFFFFF;
}
__device__ __forceinline__ float ssrlcv::orderedIntToFloat(int intVal){
 return __int_as_float( (intVal >= 0) ? intVal : intVal ^ 0x7FFFFFFF);
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
__device__ __forceinline__ float ssrlcv::modulus(const float &x, const float &y){
    float z = x;
    int n;
    if(z < 0){
        n = (int)((-z)/y)+1;
        z += n*y;
    }
    n = (int)(z/y);
    z -= n*y;
    return z;
}
__device__ __forceinline__ float2 ssrlcv::rotateAboutPoint(const int2 &loc, const float &theta, const float2 &origin){
  float2 rotatedPoint = {(float) loc.x, (float) loc.y};
  rotatedPoint = rotatedPoint - origin;
  float2 temp = rotatedPoint;

  rotatedPoint.x = (temp.x*cosf(theta)) - (temp.y*sinf(theta)) + origin.x;
  rotatedPoint.y = (temp.x*sinf(theta)) + (temp.y*cosf(theta)) + origin.y;

  return rotatedPoint;
}
/*
KERNELS
*/


__global__ void ssrlcv::findValidFeatures(unsigned int numNodes, unsigned int nodeDepthIndex, Quadtree<unsigned char>::Node* nodes, unsigned int* featureNumbers, unsigned int* featureAddresses){
  unsigned int globalID = (blockIdx.y* gridDim.x+ blockIdx.x)*blockDim.x + threadIdx.x;
  if(globalID < numNodes){
    if(nodes[globalID + nodeDepthIndex].flag){
      featureNumbers[globalID] = 1;
      featureAddresses[globalID] = globalID + nodeDepthIndex;
    }
    else{
      featureNumbers[globalID] = 0;
      featureAddresses[globalID] = 0;
    }
  }
}
__global__ void ssrlcv::fillValidFeatures(unsigned int numFeatures, Feature<SIFT_Descriptor>* features, unsigned int* featureAddresses, Quadtree<unsigned char>::Node* nodes){
  unsigned long globalID = (blockIdx.y* gridDim.x+ blockIdx.x)*blockDim.x + threadIdx.x;
  if(globalID < numFeatures){
    Feature<SIFT_Descriptor> feature = Feature<SIFT_Descriptor>();
    feature.parent = featureAddresses[globalID];
    feature.loc = nodes[feature.parent].center;
    features[globalID] = feature;
  }
}

//NOTE currently not doing so
__global__ void ssrlcv::computeThetas(unsigned long numFeatures, Feature<SIFT_Descriptor>* features, Quadtree<unsigned char>::Node* nodes, unsigned char* pixels){
  unsigned long blockId = blockIdx.y * gridDim.x + blockIdx.x;
  if(blockId < numFeatures){
    __shared__ int2 orientationVectors[9];
    for(int i = 0; i < 9; ++i) orientationVectors[i] = {0,0};
    __syncthreads();
    Quadtree<unsigned char>::Node neighbor = nodes[nodes[features[blockId].parent].neighbors[threadIdx.x]];
    //contributers are neighbor of neighbors = 1,3,5,7
    //7 - 1, 5 - 3
    //vector = x-y,z-w
    int4 pixelValues;
    //need to assure this kernel that currentNeighbor will not be -1
    int currentNeighbor = neighbor.neighbors[7];
    for(int i = 0; i < nodes[currentNeighbor].numElements; ++i) pixelValues.x += pixels[nodes[currentNeighbor].dataIndex + i];
    pixelValues.x /= nodes[currentNeighbor].numElements;
    currentNeighbor = neighbor.neighbors[1];
    for(int i = 0; i < nodes[currentNeighbor].numElements; ++i) pixelValues.y += pixels[nodes[currentNeighbor].dataIndex + i];
    pixelValues.y /= nodes[currentNeighbor].numElements;
    currentNeighbor = neighbor.neighbors[5];
    for(int i = 0; i < nodes[currentNeighbor].numElements; ++i) pixelValues.z += pixels[nodes[currentNeighbor].dataIndex + i];
    pixelValues.z /= nodes[currentNeighbor].numElements;
    currentNeighbor = neighbor.neighbors[3];
    for(int i = 0; i < nodes[currentNeighbor].numElements; ++i) pixelValues.w += pixels[nodes[currentNeighbor].dataIndex + i];
    pixelValues.w /= nodes[currentNeighbor].numElements;

    orientationVectors[threadIdx.x].x = pixelValues.x - pixelValues.y;
    orientationVectors[threadIdx.x].y = pixelValues.z - pixelValues.w;

    __syncthreads();

    if(threadIdx.x != 0) return;

    float2 bestMagWTheta = {0.0f,0.0f};
    float2 tempMagWTheta = {0.0f,0.0f};
    int2 currentOrientationVector = {0,0};
    for(int i = 0; i < 9; ++i){
      currentOrientationVector = orientationVectors[i];
      tempMagWTheta = {getMagnitude(currentOrientationVector), getTheta(currentOrientationVector)};
      if(tempMagWTheta.x > bestMagWTheta.x) bestMagWTheta = tempMagWTheta;
    }
    features[blockId].descriptor.theta = bestMagWTheta.y;
  }
}

//TODO optimize memory usage in this kernel
__global__ void ssrlcv::fillDescriptorsDensly(unsigned long numFeatures, Feature<SIFT_Descriptor>* features, Quadtree<unsigned char>::Node* nodes, unsigned char* pixels){
  int neighborDirections[3][3] = {{0,1,2},{3,4,5},{6,7,8}};
  unsigned long blockId = blockIdx.y * gridDim.x + blockIdx.x;
  if(blockId < numFeatures){

    ssrlcv::Feature<SIFT_Descriptor> feature = features[blockId];
    __shared__ float2 descriptorGrid[16][16];
    __shared__ float localMax;
    __shared__ float localMin;
    descriptorGrid[threadIdx.x][threadIdx.y] = {0.0f,0.0f};
    localMax = 0.0f;
    localMin = FLT_MAX;
    /*
    FIRST DEFINE HOG GRID
    (x,y) = [(-8.5,-8.5),(8.5,8.5)]
      x' = xcos(theta) - ysin(theta) + feature.x
      y' = ycos(theta) + xsin(theta) + feature.y

    */
    float theta = feature.descriptor.theta;
    int2 distFromOrigin = {((int)threadIdx.x) - 8,((int)threadIdx.y) - 8};

    float2 descriptorGridPoint = {
      ((distFromOrigin.x*cosf(theta)) - (distFromOrigin.y*sinf(theta))) + feature.loc.x,
      ((distFromOrigin.x*sinf(theta)) + (distFromOrigin.y*cosf(theta))) + feature.loc.y
    };

    Quadtree<unsigned char>::Node node = nodes[feature.parent];
    int2 mover = {0,0};

    while(distFromOrigin.x != 0 || distFromOrigin.y != 0){
      if(distFromOrigin.x != 0){
        mover.x = (distFromOrigin.x > 0) ? 1 : -1;
        distFromOrigin.x -= mover.x;
      }
      if(distFromOrigin.y != 0){
        mover.y = (distFromOrigin.y > 0) ? 1 : -1;
        distFromOrigin.y -= mover.y;
      }
      node = nodes[node.neighbors[neighborDirections[mover.x+1][mover.y+1]]];
    }

    float4 pixelValues;
    int currentNeighbor = node.neighbors[7];
    for(int i = 0; i < nodes[currentNeighbor].numElements; ++i) pixelValues.x += pixels[nodes[currentNeighbor].dataIndex + i];
    pixelValues.x /= nodes[currentNeighbor].numElements;
    currentNeighbor = node.neighbors[1];
    for(int i = 0; i < nodes[currentNeighbor].numElements; ++i) pixelValues.y += pixels[nodes[currentNeighbor].dataIndex + i];
    pixelValues.y /= nodes[currentNeighbor].numElements;
    currentNeighbor = node.neighbors[5];
    for(int i = 0; i < nodes[currentNeighbor].numElements; ++i) pixelValues.z += pixels[nodes[currentNeighbor].dataIndex + i];
    pixelValues.z /= nodes[currentNeighbor].numElements;
    currentNeighbor = node.neighbors[3];
    for(int i = 0; i < nodes[currentNeighbor].numElements; ++i) pixelValues.w += pixels[nodes[currentNeighbor].dataIndex + i];
    pixelValues.w /= nodes[currentNeighbor].numElements;

    float2 vector = {pixelValues.x - pixelValues.y,  pixelValues.z - pixelValues.w};

    //expf stuff is the gaussian weighting function
    float expPow = -sqrtf(dotProduct(descriptorGridPoint - feature.loc,descriptorGridPoint - feature.loc))/16.0f;
    descriptorGrid[threadIdx.x][threadIdx.y] = {sqrtf(dotProduct(vector, vector))*expf(expPow),getTheta(vector,theta)};

    __syncthreads();
    /*
    NOW CREATE HOG AND GET DESCRIPTOR
    */

    if(threadIdx.x >= 4 || threadIdx.y >= 4) return;
    int2 gradDomain = {((int) threadIdx.x*4) + 1, ((int) threadIdx.x + 1)*4};
    int2 gradRange = {((int) threadIdx.y*4) + 1, ((int) threadIdx.y + 1)*4};

    float bin_descriptors[8] = {0.0f};
    float rad45 = 45.0f*(pi/180.0f);
    for(int x = gradDomain.x; x <= gradDomain.y; ++x){
      for(int y = gradRange.x; y <= gradRange.y; ++y){
        for(int o = 1; o < 9; ++o){
          if(o*rad45 > descriptorGrid[x][y].y){
            bin_descriptors[o - 1] += descriptorGrid[x][y].x;
            break;
          }
        }
      }
    }
    __syncthreads();
    /*
    NORMALIZE
    */
    for(int d = 0; d < 8; ++d){
      atomicMinFloat(&localMin, bin_descriptors[d]);
      atomicMaxFloat(&localMax, bin_descriptors[d]);
    }
    __syncthreads();
    for(int d = 0; d < 8; ++d){
      feature.descriptor.values[(threadIdx.y*4 + threadIdx.x)*8 + d] =
        __float2int_rn(255*(bin_descriptors[d]-localMin)/(localMax-localMin));
    }
    features[blockId].descriptor = feature.descriptor;
  }
}
