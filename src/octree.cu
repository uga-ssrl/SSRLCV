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
OCTREE NODE array
*/
__device__ __host__ Node::Node(){
  this->pointIndex = -1;
  this->center = {0.0f,0.0f,0.0f};
  this->key = 0;
  this->numPoints = 0;
  this->parent = -1;
  for(int i = 0; i < 27; ++i){
    if(i < 6){
      this->faces[i] = -1;
    }
    if(i < 8){
      this->children[i] = -1;
      this->vertices[i] = -1;
    }
    if(i < 12){
      this->edges[i] = -1;
    }
    this->neighbors[i] = -1;
  }
}

/*
HELPER METHODS AND CUDA KERNELS
*/
__device__ __host__ void printBits(size_t const size, void const * const ptr){
    unsigned char *b = (unsigned char*) ptr;
    unsigned char byte;
    int i, j;
    printf("bits - ");
    for (i=size-1;i>=0;i--)
    {
        for (j=7;j>=0;j--)
        {
            byte = (b[i] >> j) & 1;
            printf("%u", byte);
        }
    }
    printf("\n");
}
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

    tempCurrentKey = uniqueNodes[globalID].key>>3;
    tempPrevKey = uniqueNodes[globalID - 1].key>>3;
    if(tempPrevKey == tempCurrentKey){
      nodeNumbers[globalID] = 0;
    }
    else{
      nodeNumbers[globalID] = 8;
    }
  }
}

void calculateNodeAddresses(int numUniqueNodes, Node* uniqueNodesDevice, int* nodeAddressesDevice, int* nodeNumbersDevice){
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

  findAllNodes<<<grid,block>>>(numUniqueNodes, nodeNumbersDevice, uniqueNodesDevice);
  CudaCheckError();
  cudaDeviceSynchronize();
  thrust::device_ptr<int> nN(nodeNumbersDevice);
  thrust::device_ptr<int> nA(nodeAddressesDevice);
  thrust::inclusive_scan(nN, nN + numUniqueNodes, nA);


}

__global__ void fillBlankNodeArray(Node* uniqueNodes, int* nodeNumbers, int* nodeAddresses, Node* outputNodeArray, int numUniqueNodes){
  int tx = threadIdx.x;
  int bx = blockIdx.x;
  int globalID = bx * blockDim.x + tx;
  int address = 0;
  if(globalID < numUniqueNodes && (globalID == 0 || nodeNumbers[globalID] == 8)){
    int siblingKey = uniqueNodes[globalID].key;
    siblingKey &= ~(1u);
    siblingKey &= ~(1u << 1);
    siblingKey &= ~(1u << 2);
    for(int i = 0; i < 8; ++i){
      Node currentNode = Node();
      currentNode.key = siblingKey + i;
      address = nodeAddresses[globalID] + i;
      outputNodeArray[address] = currentNode;
      //centers?
    }
  }
}

__global__ void fillFinestNodeArrayWithUniques(Node* uniqueNodes, int* nodeAddresses, Node* outputNodeArray, int numUniqueNodes, int* pointNodeIndex){
  int tx = threadIdx.x;
  int bx = blockIdx.x;
  int globalID = bx * blockDim.x + tx;
  int address = 0;
  int currentDKey = 0;
  Node currentNode = uniqueNodes[globalID];
  if(globalID < numUniqueNodes){
    currentDKey = currentNode.key&((1<<3)-1);
    address = nodeAddresses[globalID] + currentDKey;
    for(int i = currentNode.pointIndex; i < currentNode.numPoints + currentNode.pointIndex; ++i){
      pointNodeIndex[i] = address;
    }
    outputNodeArray[address] = currentNode;
  }
}

__global__ void fillNodeArrayWithUniques(Node* uniqueNodes, int* nodeAddresses, Node* outputNodeArray, int numUniqueNodes){
  int tx = threadIdx.x;
  int bx = blockIdx.x;
  int globalID = bx * blockDim.x + tx;
  int address = 0;
  int currentDKey = 0;
  Node currentNode = uniqueNodes[globalID];
  if(globalID < numUniqueNodes){
    currentDKey = currentNode.key&((1<<3)-1);
    address = nodeAddresses[globalID] + currentDKey;
    outputNodeArray[address] = currentNode;
  }
}

__global__ void generateParentalUniqueNodes(Node* uniqueNodes, Node* nodeArrayD, int numNodesAtDepth){
  int numUniqueNodesAtDepth = numNodesAtDepth / 8;
  int tx = threadIdx.x;
  int bx = blockIdx.x;
  int globalID = bx * blockDim.x + tx;
  int nodeArrayIndex = globalID*8;
  Node parentNode = Node();
  int parentKey;
  if(globalID < numUniqueNodesAtDepth){
    parentKey = nodeArrayD[nodeArrayIndex].key;
    parentKey >>= 3;
    parentNode.key = parentKey;
    parentNode.pointIndex = nodeArrayD[nodeArrayIndex].pointIndex;
    for(int i = 0; i < 8; ++i){
      parentNode.numPoints += nodeArrayD[nodeArrayIndex + i].numPoints;
      nodeArrayD[nodeArrayIndex + i].parent = globalID;
      parentNode.children[i] = nodeArrayIndex + i;//index of NodeArry(d+1)
      //center?
      uniqueNodes[globalID] = parentNode;
    }
  }
}

//each node should have atleast 8 neighbors
__global__ void computeNeighboringNodes(Node* nodeArray, int numNodes, int depthIndex, int childDepthIndex, int depthDistance,int* parentLUT, int* childLUT, int* numNeighbors){
  int blockID = blockIdx.y * gridDim.x + blockIdx.x;
  if(blockID < numNodes){
    blockID += depthIndex;
    nodeArray[blockID].neighbors[13] = blockID;
    int parentIndex = nodeArray[blockID].parent;
    if(childDepthIndex != -1){
      for(int i = 0; i < 8; ++i){
        if(nodeArray[blockID].children[i] != -1) nodeArray[blockID].children[i] += childDepthIndex;
      }
    }
    if(parentIndex != -1){
       nodeArray[blockID].parent += (depthIndex + numNodes);
       parentIndex += (depthIndex + numNodes);
    }
    else{
      //then it is highest node
      return;
    }
    __syncthreads();
    int depthKey = nodeArray[blockID].key&((1<<3)-1);
    int parentLUTIndex = parentLUT[depthKey*8 + threadIdx.x];
    int childLUTIndex = childLUT[depthKey*8 + threadIdx.x];
    if(depthKey > 7 || depthKey < 0){
      printf("ERROR in depthKey generation...depthKey = %d\n",depthKey);
    }
    int neighborParentIndex = 0;
    neighborParentIndex = nodeArray[parentIndex].neighbors[parentLUTIndex];
    if(neighborParentIndex != -1){
      //printf("depth = %d-%d, current = %d, parent = %d, %d, %d\n",depthIndex, depthIndex + numNodes, blockID, parentIndex,neighborParentIndex,nodeArray[neighborParentIndex].children[childLUTIndex]);
      if(nodeArray[neighborParentIndex].children[childLUTIndex] != -1 ){
        nodeArray[blockID].neighbors[threadIdx.x] = nodeArray[neighborParentIndex].children[childLUTIndex];
        atomicAdd(numNeighbors, 1);

      }

    }
    else{
      nodeArray[blockID].neighbors[threadIdx.x] = -1;
    }
    //if(blockID == depthIndex && threadIdx.x == 26){
    //  printBits(sizeof(int), &nodeArray[blockID].key);
    //  printf("%d,%d,%d,%d\n", parentIndex,blockID,parentLUTIndex,neighborParentIndex);
    //}
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
      int numPoints= 0;
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
            numPoints++;
            lineIsDone = true;
            points.push_back(point);
            normals.push_back(normal);
            break;
        }
        if(lineIsDone) break;
        ++index;
      }
		}

    cout<<numPoints<<endl;
    this->min = {minX,minY,minZ};
    this->max = {maxX,maxY,maxZ};

    this->center.x = (maxX + minX)/2;
    this->center.y = (maxY + minY)/2;
    this->center.z = (maxZ + minZ)/2;

    this->width = maxX - minX;
    if(this->width < maxY - minY) this->width = maxY - minY;
    if(this->width < maxZ - minZ) this->width = maxZ - minZ;

    this->numPoints = (int) points.size();
    this->points = new float3[this->numPoints];
    this->normals = new float3[this->numPoints];
    this->finestNodeCenters = new float3[this->numPoints];
    this->finestNodePointIndexes = new int[this->numPoints];
    this->finestNodeKeys = new int[this->numPoints];
    this->pointNodeIndex = new int[this->numPoints];
    this->totalNodes = 0;
    this->numFinestUniqueNodes = 0;

    for(int i = 0; i < points.size(); ++i){
      this->points[i] = points[i];
      this->normals[i] = normals[i];
      this->finestNodeCenters[i] = {0.0f,0.0f,0.0f};
      this->finestNodeKeys[i] = 0;
      this->pointNodeIndex[i] = -1;
      //initializing here even though points are not sorted yet
      this->finestNodePointIndexes[i] = i;
    }
    printf("\nmin = %f,%f,%f\n",this->min.x,this->min.y,this->min.z);
    printf("max = %f,%f,%f\n",this->max.x,this->max.y,this->max.z);
    printf("bounding box width = %f\n", this->width);
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
  for(int i = 0; this->finestNodeKeys[i] != *nodeKeyPointIndexes.first; ++i,++numUniqueNodes);
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
  cout<<this->numFinestUniqueNodes<<"\n"<<endl;

  Node* uniqueNodesDevice;
  CudaSafeCall(cudaMalloc((void**)&uniqueNodesDevice, this->numFinestUniqueNodes*sizeof(Node)));
  CudaSafeCall(cudaMemcpy(uniqueNodesDevice, this->uniqueNodesAtFinestLevel, this->numFinestUniqueNodes*sizeof(Node), cudaMemcpyHostToDevice));

  Node** nodeArray2DDevice;
  CudaSafeCall(cudaMalloc((void**)&nodeArray2DDevice, (this->depth + 1)*sizeof(Node*)));
  Node** nodeArray2D = new Node*[this->depth + 1];
  CudaSafeCall(cudaMemcpy(nodeArray2D, nodeArray2DDevice, (this->depth + 1)*sizeof(Node*), cudaMemcpyDeviceToHost));

  int* nodeAddressesDevice;
  int* nodeNumbersDevice;

  this->depthIndex = new int[this->depth + 1];
  CudaSafeCall(cudaMalloc((void**)&this->depthIndexDevice, (this->depth + 1)*sizeof(int)));

  int numUniqueNodes = this->numFinestUniqueNodes;

  for(int d = this->depth; d >= 0; --d){
    cout<<"depth "<<d<<endl;
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
        if(grid.x*block.x < numUniqueNodes){
          ++grid.x;//to ensure that numThreads > numUniqueNodes
          break;
        }
      }
    }
    int* nodeAddressesHost = new int[numUniqueNodes];

    for(int i = 0; i < numUniqueNodes; ++i){
      nodeAddressesHost[i] = 0;
    }

    CudaSafeCall(cudaMalloc((void**)&nodeNumbersDevice, numUniqueNodes * sizeof(int)));
    CudaSafeCall(cudaMalloc((void**)&nodeAddressesDevice, numUniqueNodes*sizeof(int)));
    //this is just to fill the arrays with 0s
    CudaSafeCall(cudaMemcpy(nodeNumbersDevice, nodeAddressesHost, numUniqueNodes * sizeof(int), cudaMemcpyHostToDevice));
    CudaSafeCall(cudaMemcpy(nodeAddressesDevice, nodeAddressesHost, numUniqueNodes * sizeof(int), cudaMemcpyHostToDevice));

    calculateNodeAddresses(numUniqueNodes, uniqueNodesDevice, nodeAddressesDevice, nodeNumbersDevice);
    CudaSafeCall(cudaMemcpy(nodeAddressesHost, nodeAddressesDevice, numUniqueNodes* sizeof(int), cudaMemcpyDeviceToHost));

    int numNodesAtDepth = d > 0 ? nodeAddressesHost[numUniqueNodes - 1] + 8: 1;
    cout<<"NUM NODES AT DEPTH = "<<numNodesAtDepth<<"\nNUM UNIQUE NODES AT DEPTH = "<<numUniqueNodes<<endl;
    delete[] nodeAddressesHost;

    CudaSafeCall(cudaMalloc((void**)&nodeArray2D[this->depth - d], numNodesAtDepth* sizeof(Node)));

    fillBlankNodeArray<<<grid,block>>>(uniqueNodesDevice, nodeNumbersDevice,  nodeAddressesDevice, nodeArray2D[this->depth - d], numUniqueNodes);
    CudaCheckError();
    cudaDeviceSynchronize();
    if(this->depth == d){
      int* pointNodeIndexDevice;
      CudaSafeCall(cudaMalloc((void**)&pointNodeIndexDevice, this->numPoints*sizeof(int)));
      CudaSafeCall(cudaMemcpy(pointNodeIndexDevice, this->pointNodeIndex, numUniqueNodes* sizeof(int), cudaMemcpyHostToDevice));
      fillFinestNodeArrayWithUniques<<<grid,block>>>(uniqueNodesDevice, nodeAddressesDevice, nodeArray2D[this->depth - d], numUniqueNodes, pointNodeIndexDevice);
      CudaCheckError();
      CudaSafeCall(cudaMemcpy(this->pointNodeIndex, pointNodeIndexDevice, this->numPoints*sizeof(int), cudaMemcpyDeviceToHost));
      CudaSafeCall(cudaFree(pointNodeIndexDevice));
    }
    else{
      fillNodeArrayWithUniques<<<grid,block>>>(uniqueNodesDevice, nodeAddressesDevice, nodeArray2D[this->depth - d], numUniqueNodes);
      CudaCheckError();
    }
    CudaSafeCall(cudaFree(uniqueNodesDevice));
    CudaSafeCall(cudaFree(nodeAddressesDevice));
    CudaSafeCall(cudaFree(nodeNumbersDevice));

    numUniqueNodes = numNodesAtDepth / 8;

    //get unique nodes at next depth
    if(d > 0){
      CudaSafeCall(cudaMalloc((void**)&uniqueNodesDevice, numUniqueNodes*sizeof(Node)));
      if(numUniqueNodes < 65535) grid.x = (unsigned int) numUniqueNodes;
      else{
        grid.x = 65535;
        while(grid.x*block.x < numUniqueNodes){
          ++block.x;
        }
        while(grid.x*block.x > numUniqueNodes){
          --grid.x;
          if(grid.x*block.x < numUniqueNodes){
            ++grid.x;//to ensure that numThreads > numUniqueNodes
            break;
          }
        }
      }
      generateParentalUniqueNodes<<<grid,block>>>(uniqueNodesDevice, nodeArray2D[this->depth - d], numNodesAtDepth);
      CudaCheckError();
    }
    this->depthIndex[this->depth - d] = this->totalNodes;
    cout<<"DEPTH INDEX = "<<this->totalNodes<<endl;
    this->totalNodes += numNodesAtDepth;
  }
  cout<<"2D NODE ARRAY COMPLETED\n"<<endl;
  this->finalNodeArray = new Node[this->totalNodes];
  CudaSafeCall(cudaMalloc((void**)&this->finalNodeArrayDevice, this->totalNodes*sizeof(Node)));
  for(int i = 0; i <= this->depth; ++i){
    if(i < this->depth){
      CudaSafeCall(cudaMemcpy(this->finalNodeArrayDevice + this->depthIndex[i], nodeArray2D[i], (this->depthIndex[i+1]-this->depthIndex[i])*sizeof(Node), cudaMemcpyDeviceToDevice));
    }
    else{
      CudaSafeCall(cudaMemcpy(this->finalNodeArrayDevice + this->depthIndex[i], nodeArray2D[i], sizeof(Node), cudaMemcpyDeviceToDevice));
    }
    CudaSafeCall(cudaFree(nodeArray2D[i]));
  }
  CudaSafeCall(cudaMemcpy(this->finalNodeArray, this->finalNodeArrayDevice, this->totalNodes*sizeof(Node), cudaMemcpyDeviceToHost));
  CudaSafeCall(cudaMemcpy(this->depthIndexDevice, this->depthIndex, (this->depth + 1)*sizeof(int), cudaMemcpyHostToDevice));
  //for(int i =this->depthIndex[0]; i < this->depthIndex[1] - this->depthIndex[0]; ++i){
  //for(int i = 0 ; i < 64; ++i){
  //  printBits(sizeof(int), &this->finalNodeArray[i].key);
  //}

  delete[] nodeArray2D;
  cout<<"NODE ARRAY FLATTENED AND COMPLETED"<<endl;
}

void Octree::fillLUTs(){
  int c[6][6][6];
  int p[6][6][6];

  int numbParent = 0;
  for (int k = 5; k >= 0; k -= 2){
    for (int i = 0; i < 6; i += 2){
    	for (int j = 5; j >= 0; j -= 2){
    		int numb = 0;
    		for (int l = 0; l < 2; l++){
    		  for (int m = 0; m < 2; m++){
    				for (int n = 0; n < 2; n++){
    					c[i+m][j-n][k-l] = numb++;
    					p[i+m][j-n][k-l] = numbParent;
    				}
    			}
        }
        numbParent++;
      }
    }
  }

  int numbLUT = 0;
  for (int k = 3; k > 1; k--){
    for (int i = 2; i < 4; i++){
    	for (int j = 3; j > 1; j--){
    		int numb = 0;
    		for (int n = 1; n >= -1; n--){
    			for (int l = -1; l <= 1; l++){
    				for (int m = 1; m >= -1; m--){
    					this->parentLUT[numbLUT][numb] = p[i+l][j+m][k+n];
    					this->childLUT[numbLUT][numb++] = c[i+l][j+m][k+n];
    				}
    			}
        }
        numbLUT++;
      }
    }
  }
  int flatParentLUT[8*27];
  int flatChildLUT[8*27];
  int flatCounter = 0;
  for(int row = 0; row < 8; ++row){
    for(int col = 0; col < 27; ++col){
      flatParentLUT[flatCounter] = this->parentLUT[row][col];
      flatChildLUT[flatCounter] = this->childLUT[row][col];
      flatCounter++;
    }
  }
  CudaSafeCall(cudaMalloc((void**)&this->parentLUTDevice, 8*27*sizeof(int)));
  CudaSafeCall(cudaMalloc((void**)&this->childLUTDevice, 8*27*sizeof(int)));
  CudaSafeCall(cudaMemcpy(this->parentLUTDevice, flatParentLUT, 8*27*sizeof(int), cudaMemcpyHostToDevice));
  CudaSafeCall(cudaMemcpy(this->childLUTDevice, flatChildLUT, 8*27*sizeof(int), cudaMemcpyHostToDevice));

}

void Octree::printLUTs(){
  cout<<"\nPARENT LUT"<<endl;
  for(int row = 0; row <  8; ++row){
    for(int col = 0; col < 27; ++col){
      cout<<this->parentLUT[row][col]<<" ";
    }
    cout<<endl;
  }
  cout<<"\nCHILD LUT"<<endl;
  for(int row = 0; row <  8; ++row){
    for(int col = 0; col < 27; ++col){
      cout<<this->childLUT[row][col]<<" ";
    }
    cout<<endl;
  }

}

void Octree::fillNeighborhoods(){

  //need to use highest number of nodes in a depth instead of totalNodes
  dim3 grid = {1,1,1};
  dim3 block = {27,1,1};
  int numNodesAtDepth;
  int depthStartingIndex;
  int childDepthStartingIndex;
  int atomicCounter = 0;
  int* atomicCounterDevice;
  CudaSafeCall(cudaMalloc((void**)&atomicCounterDevice, sizeof(int)));
  for(int i = this->depth; i >= 0 ; --i){
    numNodesAtDepth = 1;
    childDepthStartingIndex = -1;
    depthStartingIndex = this->depthIndex[i];
    if(i != depth){
      numNodesAtDepth = this->depthIndex[i + 1] - depthStartingIndex;
    }
    if(i != 0){
      childDepthStartingIndex = this->depthIndex[i-1];
    }
    if(numNodesAtDepth < 65535) grid.x = (unsigned int) numNodesAtDepth;
    else{
      grid.x = 65535;
      while(grid.x*grid.y < numNodesAtDepth){
        ++grid.y;
      }
      while(grid.x*grid.y > numNodesAtDepth){
        --grid.x;
        if(grid.x*grid.y < numNodesAtDepth){
          ++grid.x;//to ensure that numThreads > totalNodes
          break;
        }
      }
    }
    atomicCounter = 0;
    CudaSafeCall(cudaMemcpy(atomicCounterDevice, &atomicCounter, sizeof(int), cudaMemcpyHostToDevice));
    cout<<"COMPUTE NEIGHBORHOOD ON DEPTH "<<this->depth - i<<" HAS INITIATED AND INCLUDES "<<numNodesAtDepth<<" NODES STARTING AT "<<depthStartingIndex<<endl;
    computeNeighboringNodes<<<grid, block>>>(this->finalNodeArrayDevice, numNodesAtDepth, depthStartingIndex, childDepthStartingIndex,this->depth - i, this->parentLUTDevice, this->childLUTDevice, atomicCounterDevice);
    CudaCheckError();
    CudaSafeCall(cudaMemcpy(&atomicCounter, atomicCounterDevice, sizeof(int), cudaMemcpyDeviceToHost));
    cout<<"NUM NEIGHBORS = "<<atomicCounter<<endl;
  }
  CudaSafeCall(cudaMemcpy(this->finalNodeArray, this->finalNodeArrayDevice, this->totalNodes * sizeof(Node), cudaMemcpyDeviceToHost));

}
void Octree::computeVertexArray(){

}
void Octree::computeEdgeArray(){

}
void Octree::computeFaceArray(){

}
