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


/*
HELPER METHODS AND CUDA KERNELS
*/
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

__global__ void findAllNodes(int numUniqueNodes, int* nodeNumbers, int* uniqueNodeKeys){
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
    tempCurrentKey = uniqueNodeKeys[globalID] >> 3;
    tempPrevKey = uniqueNodeKeys[globalID - 1] >> 3;
    if(tempPrevKey == tempCurrentKey){
      nodeNumbers[globalID] = 0;
    }
    else{
      nodeNumbers[globalID] = 8;
    }
  }
}

//cannot do this!!!!!!! no return of device pointer
int* calculateNodeAddresses(int numUniqueNodes, int* uniqueNodeKeys){
  dim3 grid = {1,1,1};
  dim3 block = {1,1,1};
  if(numUniqueNodes < 65535) grid.x = (unsigned int) numUniqueNodes;
  else{
    grid.x = 65535;
    while(grid.x*block.x < numUniqueNodes){
      ++block.x;
    }
    while(grid.x*block.x > numUniqueNodes){
      --grid.x;
    }
  }
  int* nodeNumbers = new int[numUniqueNodes];
  int* nodeAddresses = new int[numUniqueNodes];
  for(int i = 0; i < numUniqueNodes; ++i){
    nodeNumbers[i] = 0;
    nodeAddresses[i] = 0;
  }
  int* uniqueNodeKeysDevice;
  int* nodeNumbersDevice;
  int* nodeAddressesDevice;
  CudaSafeCall(cudaMalloc((void**)&nodeNumbersDevice, numUniqueNodes * sizeof(int)));
  CudaSafeCall(cudaMalloc((void**)&nodeAddressesDevice, numUniqueNodes * sizeof(int)));
  CudaSafeCall(cudaMalloc((void**)&uniqueNodeKeysDevice, numUniqueNodes * sizeof(int)));

  CudaSafeCall(cudaMemcpy(nodeNumbersDevice, nodeNumbers, numUniqueNodes * sizeof(int), cudaMemcpyHostToDevice));
  CudaSafeCall(cudaMemcpy(nodeAddressesDevice, nodeAddresses, numUniqueNodes * sizeof(int), cudaMemcpyHostToDevice));
  CudaSafeCall(cudaMemcpy(uniqueNodeKeysDevice, uniqueNodeKeys, numUniqueNodes * sizeof(int), cudaMemcpyHostToDevice));

  findAllNodes<<<grid,block>>>(numUniqueNodes, nodeNumbersDevice, uniqueNodeKeysDevice);
  CudaCheckError();
  cudaDeviceSynchronize();
  thrust::device_ptr<int> nN(nodeNumbersDevice);
  thrust::device_ptr<int> nA(nodeAddressesDevice);
  thrust::inclusive_scan(nN, nN + numUniqueNodes, nA);
  CudaSafeCall(cudaMemcpy(nodeAddresses, nodeAddressesDevice, numUniqueNodes * sizeof(int), cudaMemcpyDeviceToHost));

  CudaSafeCall(cudaFree(nodeAddressesDevice));
  CudaSafeCall(cudaFree(nodeNumbersDevice));
  CudaSafeCall(cudaFree(uniqueNodeKeysDevice));
  return nodeAddresses;

}

__global__ void fillBlankNodeArray(Node* uniqueNodes, int* nodeAddresses, Node* outputNodeArray, int numUniqueNodes){
  int tx = threadIdx.x;
  int bx = blockIdx.x;
  int globalID = bx * blockDim.x + tx;
  int address = 0;
  if(globalID < numUniqueNodes && globalID != 0 && nodeAddresses[globalID - 1] != nodeAddresses[globalID]){
    for(int i = 0; i < 8; ++i){
      Node currentNode;
      currentNode.numPoints = 0;
      currentNode.key = (uniqueNodes[globalID].key<<3) + i;
      address = nodeAddresses[globalID] + i;
      outputNodeArray[address] = currentNode;
    }
  }
}

__global__ void fillNodeArrayWithUniques(Node* uniqueNodes, int* nodeAddresses, Node* outputNodeArray, int numUniqueNodes ){
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
    outputNodeArray[address] = currentNode;
  }
}


/*
OCTREE CLASS FUNCTIONS
*/
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
    this->finestNodeCenters = new float3[this->numPoints];
    this->finestNodePointIndexes = new int[this->numPoints];
    this->finestNodeKeys = new int[this->numPoints];
    this->totalNodes = 0;
    this->numFinestUniqueNodes = 0;

    for(int i = 0; i < points.size(); ++i){
      this->points[i] = points[i];
      this->normals[i] = normals[i];
      this->finestNodeCenters[i] = {0.0f,0.0f,0.0f};
      this->finestNodeKeys[i] = 0;
      //initializing here even though points are not sorted yet
      this->finestNodePointIndexes[i] = i;
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

void Octree::copyFinestNodeCentersToDevice(){
  CudaSafeCall(cudaMemcpy(this->finestNodeCentersDevice, this->finestNodeCenters, this->numPoints * sizeof(float3), cudaMemcpyHostToDevice));
}
void Octree::copyFinestNodeCentersToHost(){
  CudaSafeCall(cudaMemcpy(this->finestNodeCenters, this->finestNodeCentersDevice, this->numPoints * sizeof(float3), cudaMemcpyDeviceToHost));

}
void Octree::copyFinestNodeKeysToDevice(){
  CudaSafeCall(cudaMemcpy(this->finestNodeKeysDevice, this->finestNodeKeys, this->numPoints * sizeof(int), cudaMemcpyHostToDevice));
}
void Octree::copyFinestNodeKeysToHost(){
  CudaSafeCall(cudaMemcpy(this->finestNodeKeys, this->finestNodeKeysDevice, this->numPoints * sizeof(int), cudaMemcpyDeviceToHost));

}
void Octree::copyFinestNodePointIndexesToDevice(){
  CudaSafeCall(cudaMemcpy(this->finestNodePointIndexesDevice, this->finestNodePointIndexes, this->numPoints * sizeof(int), cudaMemcpyHostToDevice));
}
void Octree::copyFinestNodePointIndexesToHost(){
  CudaSafeCall(cudaMemcpy(this->finestNodePointIndexes, this->finestNodePointIndexesDevice, this->numPoints * sizeof(int), cudaMemcpyDeviceToHost));
}

void Octree::executeKeyRetrieval(dim3 grid, dim3 block){

  getNodeKeys<<<grid,block>>>(this->pointsDevice, this->finestNodeCentersDevice, this->finestNodeKeysDevice, this->center, this->width, this->numPoints, this->depth);
  CudaCheckError();

}

void Octree::sortByKey(){
  int* keyTemp = new int[this->numPoints];
  int* keyTempDevice;
  CudaSafeCall(cudaMalloc((void**)&keyTempDevice, this->numPoints*sizeof(int)));

  for(int array = 0; array < 2; ++array){
    for(int i = 0; i < this->numPoints; ++i){
      keyTemp[i] = this->finestNodeKeys[i];
    }
    thrust::device_ptr<float3> P(this->pointsDevice);
    thrust::device_ptr<float3> C(this->finestNodeCentersDevice);
    thrust::device_ptr<float3> N(this->normalsDevice);
    thrust::device_ptr<int> K(this->finestNodeKeysDevice);

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

void Octree::compactData(){
  thrust::pair<int*, float3*> nodeKeyCenters;//the last value of these node arrays
  thrust::pair<int*, int*> nodeKeyPointIndexes;//the last value of these node arrays

  int* keyTemp = new int[this->numPoints];
  for(int i = 0; i < this->numPoints; ++i){
    keyTemp[i] = this->finestNodeKeys[i];
  }
  nodeKeyCenters = thrust::unique_by_key(keyTemp, keyTemp + this->numPoints, this->finestNodeCenters);
  nodeKeyPointIndexes = thrust::unique_by_key(this->finestNodeKeys, this->finestNodeKeys + this->numPoints, this->finestNodePointIndexes);
  int numUniqueNodes = 0;
  for(int i = 0; this->finestNodeKeys[i] != *nodeKeyPointIndexes.first; ++i){
    ++numUniqueNodes;
  }
  this->numFinestUniqueNodes = numUniqueNodes;

}

void Octree::fillUniqueNodesAtFinestLevel(){
  //we have keys, centers, numpoints, point indexes
  this->uniqueNodesAtFinestLevel = new Node[this->numFinestUniqueNodes];
  for(int i = 0; i < this->numFinestUniqueNodes; ++i){
    Node currentNode;
    currentNode.key = this->finestNodeKeys[i];
    currentNode.center = this->finestNodeCenters[i];
    currentNode.pointIndex = this->finestNodePointIndexes[i];
    if(i + 1 != this->numFinestUniqueNodes){
      currentNode.numPoints = this->finestNodePointIndexes[i + 1] - this->finestNodePointIndexes[i];
    }
    else{
      currentNode.numPoints = this->numPoints - this->finestNodePointIndexes[i] - 1;
    }




    this->uniqueNodesAtFinestLevel[i] = currentNode;
  }
}

void Octree::createFinalNodeArray(){

  Node* uniqueNodesDevice;
  CudaSafeCall(cudaMalloc((void**)&uniqueNodesDevice, this->numFinestUniqueNodes*sizeof(Node)));
  CudaSafeCall(cudaMemcpy(uniqueNodesDevice, this->uniqueNodesAtFinestLevel, this->numFinestUniqueNodes*sizeof(Node), cudaMemcpyHostToDevice));

  Node** nodeArray2DDevice;
  CudaSafeCall(cudaMalloc((void**)&nodeArray2DDevice, (this->depth + 1)*sizeof(Node*)));
  Node** nodeArray2D = new Node*[this->depth + 1];
  CudaSafeCall(cudaMemcpy(nodeArray2D, nodeArray2DDevice, (this->depth + 1)*sizeof(Node*), cudaMemcpyDeviceToHost));

  Node* currentNodeDevice;//not being used????

  int* currentNodeAddressDevice;

  int* depthIndices = new int[this->depth + 1];
  int numUniqueNodes = this->numFinestUniqueNodes;
  int* currentKeys = new int[numUniqueNodes];
  //loop through depths

  for(int d = this->depth; d >= 0; --d){
    dim3 grid = {1,1,1};
    dim3 block = {1,1,1};
    if(numUniqueNodes < 65535) grid.x = (unsigned int) numUniqueNodes;
    else{
      grid.x = 65535;
      while(grid.x*block.x < numUniqueNodes){
        ++block.x;
      }
      while(grid.x*block.x > numUniqueNodes){
        --grid.x;
      }
    }
    int* currentNodeAddress = new int[numUniqueNodes];
    for(int i = 0; i < numUniqueNodes; ++i) currentNodeAddress = 0;
    CudaSafeCall(cudaMalloc((void**)&currentNodeAddressDevice, numUniqueNodes*sizeof(int)));
    if(d == this->depth){
      currentNodeAddress = calculateNodeAddresses(numUniqueNodes, this->finestNodeKeys);
    }
    else{
      currentNodeAddress = calculateNodeAddresses(numUniqueNodes, currentKeys);
    }

    //broken here
    CudaSafeCall(cudaMemcpy(currentNodeAddressDevice, currentNodeAddress, numUniqueNodes*sizeof(int), cudaMemcpyHostToDevice));
    int numNodesAtDepth = d > 0 ? currentNodeAddress[numUniqueNodes - 1] + 8 : 1;

    CudaSafeCall(cudaMalloc((void**)&nodeArray2D[this->depth - d], numNodesAtDepth* sizeof(Node)));
    //populate node array from base nodeAddress >>>
    //params = uniqueNodesDevice, nodeArray2D[this->depth - d], currentNodeAddressDevice,
    //          numUniqueNodes, numNodesAtDepth, ???currentNodeDepth???
    //to populate node array you look at the nodeAddresses
    //if the address of i != i - 1 then that is a parent and you need to add 8 nodes with the starting address being the parents

    fillBlankNodeArray<<<grid,block>>>(uniqueNodesDevice, currentNodeAddressDevice, nodeArray2D[this->depth - d], numUniqueNodes);
    CudaCheckError();
    cudaDeviceSynchronize();
    fillNodeArrayWithUniques<<<grid,block>>>(uniqueNodesDevice, currentNodeAddressDevice, nodeArray2D[this->depth - d], numUniqueNodes);

    //DOUBLE FREE OR CORRUPTION ERROR AFTER THIS
    /*
    CudaSafeCall(cudaFree(uniqueNodesDevice));
    CudaSafeCall(cudaFree(currentNodeAddressDevice));
    delete[] currentNodeAddress;
    delete[] currentKeys;
    numUniqueNodes = numNodesAtDepth / 8;

    int* currentKeys = new int[numUniqueNodes];

    if(d > 0){
      CudaSafeCall(cudaMalloc((void**)&uniqueNodesDevice, numUniqueNodes*sizeof(Node)));
      //get keys for that depth!!!!!!
      //params  = nodeArray2D[D-d], currentKeys(my addition), numNodesAtDepth




    }

    depthIndices[this->depth - d] = this->totalNodes;
    this->totalNodes += numNodesAtDepth;
    //???currentNodeDepth*=2.0 //this is most likely distance from the center of the octree
    */
  }
  /*
  this->finalNodeArray = new Node[this->totalNodes];
  CudaSafeCall(cudaMalloc((void**)&this->finalNodeArrayDevice, this->totalNodes*sizeof(Node)));
  for(int i = 0; i <= this->depth; ++i){
    if(i < this->depth){
      CudaSafeCall(cudaMemcpy(this->finalNodeArrayDevice + depthIndices[i], nodeArray2D[i], (depthIndices[i+1]-depthIndices[i])*sizeof(Node), cudaMemcpyDeviceToDevice));
    }
    else{
      CudaSafeCall(cudaMemcpy(this->finalNodeArrayDevice + depthIndices[i], nodeArray2D[i], sizeof(Node), cudaMemcpyDeviceToDevice));
    }
    CudaSafeCall(cudaFree(nodeArray2D[i]));
  }
  CudaSafeCall(cudaFree(nodeArray2DDevice));
  delete[] nodeArray2D;
  */
}
