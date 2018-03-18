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

__global__ void findAllNodes(int numUniqueNodes, int* nodeNumbers, Node* uniqueNodes){
  int tx = threadIdx.x;
  int bx = blockIdx.x;
  int globalID = bx * blockDim.x + tx;
  int tempCurrentKey = 0;
  int tempPrevKey = 0;
  if(globalID < numUniqueNodes){
    if(globalID == 0){
      nodeNumbers[globalID] = 0;
      return;
    }
    tempCurrentKey = uniqueNodes[globalID].key >> 3;
    tempPrevKey = uniqueNodes[globalID - 1].key >> 3;
    if(tempPrevKey == tempCurrentKey){
      nodeNumbers[globalID] = 0;
    }
    else{
      nodeNumbers[globalID] = 8;
    }
  }
}

__global__ void fill1NodeArray(Node* uniqueNodes, int* nodeAddresses, Node* finalNodeArray, int numUniqueNodes){
  int tx = threadIdx.x;
  int bx = blockIdx.x;
  int globalID = bx * blockDim.x + tx;
  Node currentNode;
  int address = 0;
  int currentDKey = 0;
  if(globalID < numUniqueNodes){
    currentNode = uniqueNodes[globalID];
    currentDKey = currentNode.key&((1<<3)-1);
    address = nodeAddresses[globalID] + currentDKey;//actually last three printBits
    finalNodeArray[address] = currentNode;
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
    this->totalNodes = 0;
    this->numUniqueNodes = 0;

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

void Octree::copyPointsToDevice(){
  CudaSafeCall(cudaMemcpy(this->pointsDevice, this->points, this->numPoints * sizeof(float3), cudaMemcpyHostToDevice));
}
void Octree::copyPointsToHost(){
  CudaSafeCall(cudaMemcpy(this->points, this->pointsDevice, this->numPoints * sizeof(float3), cudaMemcpyDeviceToHost));

}
void Octree::copyNormalsToDevice(){
  CudaSafeCall(cudaMemcpy(this->normalsDevice, this->normals, this->numPoints * sizeof(float3), cudaMemcpyHostToDevice));
}
void Octree::copyNormalsToHost(){
  CudaSafeCall(cudaMemcpy(this->normals, this->normalsDevice, this->numPoints * sizeof(float3), cudaMemcpyDeviceToHost));

}
void Octree::copyNodeCentersToDevice(){
  CudaSafeCall(cudaMemcpy(this->nodeCentersDevice, this->nodeCenters, this->numPoints * sizeof(float3), cudaMemcpyHostToDevice));
}
void Octree::copyNodeCentersToHost(){
  CudaSafeCall(cudaMemcpy(this->nodeCenters, this->nodeCentersDevice, this->numPoints * sizeof(float3), cudaMemcpyDeviceToHost));

}
void Octree::copyNodeKeysToDevice(){
  CudaSafeCall(cudaMemcpy(this->nodeKeysDevice, this->nodeKeys, this->numPoints * sizeof(int), cudaMemcpyHostToDevice));
}
void Octree::copyNodeKeysToHost(){
  CudaSafeCall(cudaMemcpy(this->nodeKeys, this->nodeKeysDevice, this->numPoints * sizeof(int), cudaMemcpyDeviceToHost));

}
void Octree::copyNodePointIndexesToDevice(){
  CudaSafeCall(cudaMemcpy(this->nodePointIndexesDevice, this->nodePointIndexes, this->numPoints * sizeof(int), cudaMemcpyHostToDevice));
}
void Octree::copyNodePointIndexesToHost(){
  CudaSafeCall(cudaMemcpy(this->nodePointIndexes, this->nodePointIndexesDevice, this->numPoints * sizeof(int), cudaMemcpyDeviceToHost));
}
void Octree::copyFinalNodeArrayToDevice(){
  CudaSafeCall(cudaMemcpy(this->finalNodeArrayDevice, this->finalNodeArray, this->totalNodes * sizeof(Node), cudaMemcpyHostToDevice));

}
void Octree::copyFinalNodeArrayToHost(){
  CudaSafeCall(cudaMemcpy(this->finalNodeArray, this->finalNodeArrayDevice, this->totalNodes * sizeof(Node), cudaMemcpyDeviceToHost));

}

void Octree::executeKeyRetrieval(dim3 grid, dim3 block){

  getNodeKeys<<<grid,block>>>(this->pointsDevice, this->nodeCentersDevice, this->nodeKeysDevice, this->center, this->width, this->numPoints, this->depth);
  CudaCheckError();

}

void Octree::sortByKey(){
  int* keyTemp = new int[this->numPoints];
  int* keyTempDevice;
  CudaSafeCall(cudaMalloc((void**)&keyTempDevice, this->numPoints*sizeof(int)));

  for(int array = 0; array < 2; ++array){
    for(int i = 0; i < this->numPoints; ++i){
      keyTemp[i] = this->nodeKeys[i];
    }
    thrust::device_ptr<float3> P(this->pointsDevice);
    thrust::device_ptr<float3> C(this->nodeCentersDevice);
    thrust::device_ptr<float3> N(this->normalsDevice);
    thrust::device_ptr<int> K(this->nodeKeysDevice);

    CudaSafeCall(cudaMemcpy(keyTempDevice, keyTemp, this->numPoints * sizeof(int), cudaMemcpyHostToDevice));
    thrust::device_ptr<int> KT(keyTempDevice);
    if(array == 0){
      thrust::sort_by_key(KT, KT + this->numPoints, P);
    }
    else if(array == 1){
      thrust::sort_by_key(KT, KT + this->numPoints, C);
      thrust::sort_by_key(K, K + this->numPoints, N);
    }
  }
}

//three new node arrays are instantiated here once numUniqueNodes is found out
void Octree::compactData(){
  thrust::pair<int*, float3*> nodeKeyCenters;//the last value of these node arrays
  thrust::pair<int*, int*> nodeKeyPointIndexes;//the last value of these node arrays

  int* keyTemp = new int[this->numPoints];
  for(int i = 0; i < this->numPoints; ++i){
    keyTemp[i] = this->nodeKeys[i];
  }
  nodeKeyCenters = thrust::unique_by_key(keyTemp, keyTemp + this->numPoints, this->nodeCenters);
  nodeKeyPointIndexes = thrust::unique_by_key(this->nodeKeys, this->nodeKeys + this->numPoints, this->nodePointIndexes);
  int numUniqueNodes = 0;
  for(int i = 0; this->nodeKeys[i] != *nodeKeyPointIndexes.first; ++i){
    ++numUniqueNodes;
  }
  this->numUniqueNodes = numUniqueNodes;

}

void Octree::fillInUniqueNodes(){
  this->uniqueNodeArray = new Node[this->numUniqueNodes];
  for(int i = 0; i < this->numUniqueNodes; ++i){
    this->uniqueNodeArray[i].center = this->nodeCenters[i];
    this->uniqueNodeArray[i].pointIndex = this->nodePointIndexes[i];
    if(i + 1 == this->numUniqueNodes){
      this->uniqueNodeArray[i].numPoints = this->nodePointIndexes[i + 1] - this->nodePointIndexes[i];
    }
    else{
      this->uniqueNodeArray[i].numPoints = this->numPoints - this->nodePointIndexes[i];

    }
    this->uniqueNodeArray[i].key = this->nodeKeys[i];
  }
}

void Octree::executeFindAllNodes(dim3 grid, dim3 block, int numUniqueNodes, int* nodeNumbers, Node* uniqueNodes){

  findAllNodes<<<grid,block>>>(numUniqueNodes, nodeNumbers, uniqueNodes);
  CudaCheckError();
}
