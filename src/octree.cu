#include "octree.cuh"

using namespace std;

// Define this to turn on error checking
#define CUDA_ERROR_CHECK

#define CudaSafeCall( err ) __cudaSafeCall( err, __FILE__, __LINE__ )
#define CudaCheckError()    __cudaCheckError( __FILE__, __LINE__ )


inline void __cudaSafeCall(cudaError err, const char *file, const int line) {
#ifdef CUDA_ERROR_CHECK
  if (cudaSuccess != err) {
      fprintf(stderr, "cudaSafeCall() failed at %s:%i : %s\n",
      file, line, cudaGetErrorString(err));
      exit(-1);
  }
#endif

  return;
}
inline void __cudaCheckError(const char *file, const int line) {
#ifdef CUDA_ERROR_CHECK
  cudaError err = cudaGetLastError();
  if (cudaSuccess != err) {
    fprintf(stderr, "cudaCheckError() failed at %s:%i : %s\n",
    file, line, cudaGetErrorString(err));
    exit(-1);
  }

    // More careful checking. However, this will affect performance.
    // Comment away if needed.
    //err = cudaDeviceSynchronize();
  if (cudaSuccess != err) {
    fprintf(stderr, "cudaCheckError() with sync failed at %s:%i : %s\n",
    file, line, cudaGetErrorString(err));
    exit(-1);
  }
#endif

  return;
}


//pretty much just a binary search in each dimension performed by threads
__global__ void getNodeKeys(float3* points, float3* nodeCenters, int* nodeKeys, float3 c, float W, int numPoints, int D){
  int tx = threadIdx.x;
  int bx = blockIdx.x;
  int globalID = bx * blockDim.x + tx;
  if(globalID < numPoints){
    float x = points[globalID].x;
    float y = points[globalID].y;
    float z = points[globalID].z;
    float leftx = c.x-W/2.0f, rightx = c.x + W/2.0f;
    float lefty = c.y-W/2.0f, righty = c.y + W/2.0f;
    float leftz = c.z-W/2.0f, rightz = c.z + W/2.0f;
    int key = 0;
    int depth = 1;
    while(depth <= D){
      if(x < c.x){
        key <<= 1;
        rightx = c.x;
        c.x = (leftx + rightx)/2.0f;
      }
      else{
        key = (key << 1) + 1;
        leftx = c.x;
        c.x = (leftx + rightx)/2.0f;
      }
      if(y < c.y){
        key <<= 1;
        righty = c.y;
        c.y = (lefty + righty)/2.0f;
      }
      else{
        key = (key << 1) + 1;
        lefty = c.y;
        c.y = (lefty + righty)/2.0f;
      }
      if(z < c.z){
        key <<= 1;
        rightz = c.z;
        c.z = (leftz + rightz)/2.0f;
      }
      else{
        key = (key << 1) + 1;
        leftz = c.z;
        c.z = (leftz + rightz)/2.0f;
      }
      depth++;
    }
    nodeKeys[globalID] = key;
    nodeCenters[globalID].x = c.x;
    nodeCenters[globalID].y = c.y;
    nodeCenters[globalID].z = c.z;
  }
}

__global__ void findAllNodes(int numNodes, int* nodeNumbers, int* nodeKeys){
  int tx = threadIdx.x;
  int bx = blockIdx.x;
  int globalID = bx * blockDim.x + tx;
  int tempCurrentKey = 0;
  int tempPrevKey = 0;
  if(globalID < numNodes){
    if(globalID == 0){
      nodeNumbers[globalID] = 0;
      return;
    }
    tempCurrentKey = nodeKeys[globalID] >> 3;
    tempPrevKey = nodeKeys[globalID - 1] >> 3;
    if(tempPrevKey == tempCurrentKey){
      nodeNumbers[globalID] = 0;
    }
    else{
      nodeNumbers[globalID] = 8;
    }
  }
}

Octree::Octree(){

}

Octree::~Octree(){

}

void Octree::parsePLY(string pathToFile){
  cout<<pathToFile + "'s data to be transfered to an empty octree."<<endl;
	ifstream plystream(pathToFile);
	string currentLine;
  vector<float3> points;
  vector<float3> normals;
  float minX = 0, minY = 0, minZ = 0, maxX = 0, maxY = 0, maxZ = 0;
	if (plystream.is_open()) {
		while (getline(plystream, currentLine)) {
      stringstream getMyFloats = stringstream(currentLine);
      float value = 0.0;
      int index = 0;
      float3 point;
      float3 normal;
      bool lineIsDone = false;
      while(getMyFloats >> value){
        switch(index){
          case 0:
            point.x = value;
            if(value > maxX) maxX = value;
            if(value < minX) minX = value;
            break;
          case 1:
            point.y = value;
            if(value > maxY) maxY = value;
            if(value < minY) minY = value;
            break;
          case 2:
            point.z = value;
            if(value > maxZ) maxZ = value;
            if(value < minZ) minZ = value;
            break;
          case 3:
            normal.x = value;
            break;
          case 4:
            normal.y = value;
            break;
          case 5:
            normal.z = value;
            break;
          default:
            lineIsDone = true;
            points.push_back(point);
            normals.push_back(normal);
            break;
        }
        if(lineIsDone) break;
        ++index;
      }
		}
    this->min = {minX,minY,minZ};
    this->max = {maxX,maxY,maxZ};

    this->center.x = (maxX + minX)/2;
    this->center.y = (maxY + minY)/2;
    this->center.z = (maxZ +  minZ)/2;

    this->width = maxX - minX;
    if(this->width < maxY - minY) this->width = maxY - minY;
    if(this->width < maxZ - minZ) this->width = maxZ - minZ;

    this->numPoints = (int) points.size();
    this->points = new float3[this->numPoints];
    this->normals = new float3[this->numPoints];
    this->nodeCenters = new float3[this->numPoints];
    this->nodePointIndexes = new int[this->numPoints];
    this->nodeKeys = new int[this->numPoints];

    for(int i = 0; i < points.size(); ++i){
      this->points[i] = points[i];
      this->normals[i] = normals[i];
      this->nodeCenters[i] = {0.0f,0.0f,0.0f};
      this->nodeKeys[i] = 0;
      //initializing here even though points are not sorted yet
      this->nodePointIndexes[i] = i;
    }
    printf("\nmin = %f,%f,%f\n",this->min.x,this->min.y,this->min.z);
    printf("max = %f,%f,%f\n",this->max.x,this->max.y,this->max.z);
    printf("center = %f,%f,%f\n",this->center.x,this->center.y,this->center.z);
    printf("number of points = %d\n\n", this->numPoints);
    cout<<pathToFile + "'s data has been transfered to an initialized octree.\n"<<endl;
	}
	else{
    cout << "Unable to open: " + pathToFile<< endl;
    exit(1);
  }
}

Octree::Octree(string pathToFile, int depth){
  this->parsePLY(pathToFile);
  this->depth = depth;
}

void Octree::allocateDeviceVariablesNODES(bool afterNodeDetermination){
  if(!afterNodeDetermination){
    CudaSafeCall(cudaMalloc((void**)&this->nodeCentersDevice, this->numPoints * sizeof(float3)));
    CudaSafeCall(cudaMalloc((void**)&this->nodeKeysDevice, this->numPoints * sizeof(int)));
    CudaSafeCall(cudaMalloc((void**)&this->nodePointIndexesDevice, this->numPoints * sizeof(int)));
  }
  else{
    CudaSafeCall(cudaMalloc((void**)&this->nodeNumbersDevice, this->numNodes * sizeof(int)));
    CudaSafeCall(cudaMalloc((void**)&this->nodeAddressesDevice, this->numNodes * sizeof(int)));
  }
}

void Octree::allocateDeviceVariablesGENERAL(){
  CudaSafeCall(cudaMalloc((void**)&this->pointsDevice, this->numPoints * sizeof(float3)));
  CudaSafeCall(cudaMalloc((void**)&this->normalsDevice, this->numPoints * sizeof(float3)));
}

void Octree::copyArraysHostToDeviceNODES(bool transferNodeCenters, bool transferNodeKeys, bool transferNodePointIndexes, bool transferNodeNumbers, bool transferNodeAddresses){
  if(transferNodeCenters) CudaSafeCall(cudaMemcpy(this->nodeCentersDevice, this->nodeCenters, this->numPoints * sizeof(float3), cudaMemcpyHostToDevice));
  if(transferNodeKeys) CudaSafeCall(cudaMemcpy(this->nodeKeysDevice, this->nodeKeys, this->numPoints * sizeof(int), cudaMemcpyHostToDevice));
  if(transferNodePointIndexes) CudaSafeCall(cudaMemcpy(this->nodePointIndexesDevice, this->nodePointIndexes, this->numPoints * sizeof(int), cudaMemcpyHostToDevice));
  if(transferNodeNumbers) CudaSafeCall(cudaMemcpy(this->nodeNumbersDevice, this->nodeNumbers, this->numNodes * sizeof(int), cudaMemcpyHostToDevice));
  if(transferNodeAddresses) CudaSafeCall(cudaMemcpy(this->nodeAddressesDevice, this->nodeAddresses, this->numNodes * sizeof(int), cudaMemcpyHostToDevice));
}

void Octree::copyArraysHostToDeviceGENERAL(bool transferPoints, bool transferNormals){
  if(points) CudaSafeCall(cudaMemcpy(this->pointsDevice, this->points, this->numPoints * sizeof(float3), cudaMemcpyHostToDevice));
  if(normals) CudaSafeCall(cudaMemcpy(this->normalsDevice, this->normals, this->numPoints * sizeof(float3), cudaMemcpyHostToDevice));
}

void Octree::copyArraysDeviceToHostNODES(bool transferNodeCenters, bool transferNodeKeys, bool transferNodePointIndexes, bool transferNodeNumbers, bool transferNodeAddresses){
  if(transferNodeCenters) CudaSafeCall(cudaMemcpy(this->nodeCenters, this->nodeCentersDevice, this->numPoints * sizeof(float3), cudaMemcpyDeviceToHost));
  if(transferNodeKeys) CudaSafeCall(cudaMemcpy(this->nodeKeys, this->nodeKeysDevice, this->numPoints * sizeof(int), cudaMemcpyDeviceToHost));
  if(transferNodePointIndexes) CudaSafeCall(cudaMemcpy(this->nodePointIndexes, this->nodePointIndexesDevice, this->numPoints * sizeof(int), cudaMemcpyDeviceToHost));
  if(transferNodeNumbers) CudaSafeCall(cudaMemcpy(this->nodeNumbers, this->nodeNumbersDevice, this->numNodes * sizeof(int), cudaMemcpyDeviceToHost));
  if(transferNodeAddresses) CudaSafeCall(cudaMemcpy(this->nodeAddresses, this->nodeAddressesDevice, this->numNodes * sizeof(int), cudaMemcpyDeviceToHost));

}

void Octree::copyArraysDeviceToHostGENERAL(bool transferPoints, bool transferNormals){
  if(points) CudaSafeCall(cudaMemcpy(this->points, this->pointsDevice, this->numPoints * sizeof(float3), cudaMemcpyDeviceToHost));
  if(normals) CudaSafeCall(cudaMemcpy(this->normals, this->normalsDevice, this->numPoints * sizeof(float3), cudaMemcpyDeviceToHost));
}

void Octree::executeKeyRetrieval(dim3 grid, dim3 block){

  getNodeKeys<<<grid,block>>>(this->pointsDevice, this->nodeCentersDevice, this->nodeKeysDevice, this->center, this->width, this->numPoints, this->depth);
  CudaCheckError();

}

void Octree::sortByKey(){
  int* keyTemp = new int[this->numPoints];
  for(int array = 0; array < 3; ++array){
    for(int i = 0; i < this->numPoints; ++i){
      keyTemp[i] = this->nodeKeys[i];
    }
    if(array == 0){
      thrust::sort_by_key(keyTemp, keyTemp + this->numPoints, this->points);
    }
    else if(array == 1){
      thrust::sort_by_key(keyTemp, keyTemp + this->numPoints, this->nodeCenters);
    }
    else if(array == 2){
      thrust::sort_by_key(keyTemp, keyTemp + this->numPoints, this->normals);
      thrust::sort_by_key(this->nodeKeys, this->nodeKeys + this->numPoints, this->normals);

    }
  }
}

//two new node arrays are instantiated here once numNodes is found out
void Octree::compactData(){
  thrust::pair<int*, float3*> nodeKeyCenters;//the last value of these node arrays
  thrust::pair<int*, int*> nodeKeyPointIndexes;//the last value of these node arrays

  int* keyTemp = new int[this->numPoints];
  for(int i = 0; i < this->numPoints; ++i){
    keyTemp[i] = this->nodeKeys[i];
  }
  nodeKeyCenters = thrust::unique_by_key(keyTemp, keyTemp + this->numPoints, this->nodeCenters);
  nodeKeyPointIndexes = thrust::unique_by_key(this->nodeKeys, this->nodeKeys + this->numPoints, this->nodePointIndexes);
  int numNodes = 0;
  for(int i = 0; this->nodeKeys[i] != *nodeKeyPointIndexes.first; ++i){
    ++numNodes;
  }

  /*
  //just used to check accuracy of unique_by_key
  printf("numNodes = %d (from keysTemp)\n", numNodes);
  numNodes = 0;
  for(int i = 0; this->nodeCenters[i].x != (*nodeKeyCenters.second).x ||
  this->nodeCenters[i].y != (*nodeKeyCenters.second).y ||
  this->nodeCenters[i].z != (*nodeKeyCenters.second).z; ++i){
    ++numNodes;
  }
  printf("numNodes = %d (from centers)\n", numNodes);
  numNodes = 0;
  for(int i = 0; keyTemp[i] != *nodeKeyCenters.first; ++i){
    ++numNodes;
  }
  printf("numNodes = %d (from keys)\n", numNodes);
  numNodes = 0;
  for(int i = 0; this->nodePointIndexes[i] != *nodeKeyPointIndexes.second; ++i){
    ++numNodes;
  }
  printf("numNodes = %d (from pointIndexes)\n", numNodes);
  */
  this->numNodes = numNodes;
  this->nodeNumbers = new int[this->numNodes];
  this->nodeAddresses = new int[this->numNodes];
  for(int i = 0; i < this->numNodes; ++i){
    this->nodeNumbers[i] = 0;
    this->nodeAddresses[i] = 0;
  }
}

void Octree::inclusiveScanForNodeAddresses(){
  thrust::inclusive_scan(this->nodeNumbers, this->nodeNumbers + this->numNodes, this->nodeAddresses);
}

void Octree::executeFindAllNodes(dim3 grid, dim3 block){

  findAllNodes<<<grid,block>>>(this->numNodes, this->nodeNumbersDevice, this->nodeKeysDevice);
  CudaCheckError();
}


void Octree::cudaFreeMemory(){
  CudaSafeCall(cudaFree(this->nodeKeysDevice));
  CudaSafeCall(cudaFree(this->nodeCentersDevice));
  CudaSafeCall(cudaFree(this->pointsDevice));
  CudaSafeCall(cudaFree(this->normalsDevice));
  CudaSafeCall(cudaFree(this->nodePointIndexesDevice));
  CudaSafeCall(cudaFree(this->nodeNumbersDevice));
  CudaSafeCall(cudaFree(this->nodeAddressesDevice ));
}
