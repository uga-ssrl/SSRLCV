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

void calculateNodeAddresses(int numUniqueNodes, Node* uniqueNodesDevice, int* nodeAddressesDevice){
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
  for(int i = 0; i < numUniqueNodes; ++i){
    nodeNumbers[i] = 0;
  }
  int* nodeNumbersDevice;
  CudaSafeCall(cudaMalloc((void**)&nodeNumbersDevice, numUniqueNodes * sizeof(int)));

  CudaSafeCall(cudaMemcpy(nodeNumbersDevice, nodeNumbers, numUniqueNodes * sizeof(int), cudaMemcpyHostToDevice));
  //this is just copying 0s not actually making nodeNumbers and nodeAddresses the same
  CudaSafeCall(cudaMemcpy(nodeAddressesDevice, nodeNumbers, numUniqueNodes * sizeof(int), cudaMemcpyHostToDevice));

  findAllNodes<<<grid,block>>>(numUniqueNodes, nodeNumbersDevice, uniqueNodesDevice);
  CudaCheckError();
  cudaDeviceSynchronize();
  thrust::device_ptr<int> nN(nodeNumbersDevice);
  thrust::device_ptr<int> nA(nodeAddressesDevice);
  thrust::inclusive_scan(nN, nN + numUniqueNodes, nA);

  CudaSafeCall(cudaFree(nodeNumbersDevice));

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
      currentNode.key = ((uniqueNodes[globalID].key>>3)<<3) + i;//needs to be the parent of that nodes key + i
      currentNode.parent = -1;
      for(int c = 0; c < 8; ++c) currentNode.children[i] = -1;
      address = nodeAddresses[globalID] + i;
      outputNodeArray[address] = currentNode;
      //will have centers
    }
  }
}

__global__ void fillFinestNodeArrayWithUniques(Node* uniqueNodes, int* nodeAddresses, Node* outputNodeArray, int numUniqueNodes, int* pointNodeIndex){
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

__global__ void generateParentalUniqueNodes(Node* uniqueNodes, Node* nodeArrayD, int numNodesAtDepth){
  int numUniqueNodesAtDepth = numNodesAtDepth / 8;
  int tx = threadIdx.x;
  int bx = blockIdx.x;
  int globalID = bx * blockDim.x + tx;
  globalID *= 8;//this is to make sure initiation strides are 8
  Node parentNode;
  int parentKey;
  if(globalID < numUniqueNodesAtDepth){
    parentKey = nodeArrayD[globalID].key>>3;
    parentKey <<= 3;
    parentNode.key = parentKey;
    parentNode.pointIndex = nodeArrayD[globalID].pointIndex;
    for(int i = 0; i < 8; ++i){
      parentNode.numPoints += nodeArrayD[globalID + i].numPoints;
      nodeArrayD[globalID + i].parent = globalID;
      parentNode.children[i] = globalID + i;//index of NodeArry(d+1)
      //center?
      uniqueNodes[globalID] = parentNode;
    }
  }
}

__global__ void computeNeighboringNodes(Node* nodeArray, int numNodes, int* parentLUT, int* childLUT, int* depthIndex, int depth){
  int tx = threadIdx.x;
  int bx = blockIdx.x;
  int globalID = bx * blockDim.x + tx;
  int numNodesAtDepth;
  for(int i = 0; i < depth; ++i){

    if(globalID < numNodes){

    }
    __syncthreads();
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

  int* currentNodeAddressDevice;

  this->depthIndex = new int[this->depth + 1];
  int numUniqueNodes = this->numFinestUniqueNodes;

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
        if(grid.x*block.x < numUniqueNodes){
          ++grid.x;//to ensure that numThreads > numUniqueNodes
          break;
        }
      }
    }

    CudaSafeCall(cudaMalloc((void**)&currentNodeAddressDevice, numUniqueNodes*sizeof(int)));
    calculateNodeAddresses(numUniqueNodes, uniqueNodesDevice, currentNodeAddressDevice);

    int* nodeAddressesHost = new int[numUniqueNodes];
    CudaSafeCall(cudaMemcpy(nodeAddressesHost, currentNodeAddressDevice, numUniqueNodes* sizeof(int), cudaMemcpyDeviceToHost));

    int numNodesAtDepth = d > 0 ? nodeAddressesHost[numUniqueNodes - 1] + 8 : 1;
    delete[] nodeAddressesHost;

    CudaSafeCall(cudaMalloc((void**)&nodeArray2D[this->depth - d], numNodesAtDepth* sizeof(Node)));

    fillBlankNodeArray<<<grid,block>>>(uniqueNodesDevice, currentNodeAddressDevice, nodeArray2D[this->depth - d], numUniqueNodes);
    CudaCheckError();
    cudaDeviceSynchronize();
    if(this->depth == d){
      int* pointNodeIndexDevice;
      CudaSafeCall(cudaMalloc((void**)&pointNodeIndexDevice, this->numPoints*sizeof(int)));
      CudaSafeCall(cudaMemcpy(pointNodeIndexDevice, this->pointNodeIndex, numUniqueNodes* sizeof(int), cudaMemcpyHostToDevice));
      fillFinestNodeArrayWithUniques<<<grid,block>>>(uniqueNodesDevice, currentNodeAddressDevice, nodeArray2D[this->depth - d], numUniqueNodes, pointNodeIndexDevice);
      CudaCheckError();
      CudaSafeCall(cudaMemcpy(this->pointNodeIndex, pointNodeIndexDevice, this->numPoints*sizeof(int), cudaMemcpyDeviceToHost));
      CudaSafeCall(cudaFree(pointNodeIndexDevice));
    }
    else{
      fillNodeArrayWithUniques<<<grid,block>>>(uniqueNodesDevice, currentNodeAddressDevice, nodeArray2D[this->depth - d], numUniqueNodes);
      CudaCheckError();
      cudaDeviceSynchronize();
    }

    //need to find centers????????? is that necessary
    //missing parameters when comparing to hoppes algorithm on: numNodesD and currentNodeDepth
    //https://devtalk.nvidia.com/default/topic/609551/teaching-and-curriculum-support/
    //my-cuda-programming-lecture-and-teaching-of-poisson-parallel-surface-reconstruction-in-a-summer-scho/

    CudaSafeCall(cudaFree(uniqueNodesDevice));
    CudaSafeCall(cudaFree(currentNodeAddressDevice));
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
      cudaDeviceSynchronize();
    }
    //now we have unique nodes at next depth

    this->depthIndex[this->depth - d] = this->totalNodes;
    this->totalNodes += numNodesAtDepth;
    cout<<this->totalNodes<<" "<<this->depthIndex[this->depth -d]<<" "<<numNodesAtDepth



    <<endl;
    //necessary????
    //???currentNodeDepth*=2.0 //this is most likely distance from the center of the octree
  }
  cout<<"2D NODE ARRAY COMPLETED"<<endl;
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
  CudaSafeCall(cudaMemcpy(this->depthIndexDevice, this->depthIndex, (this->depth + 1)*sizeof(Node), cudaMemcpyHostToDevice));

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

  CudaSafeCall(cudaMalloc((void**)&this->parentLUTDevice, 8*27*sizeof(int)));
  CudaSafeCall(cudaMalloc((void**)&this->childLUTDevice, 8*27*sizeof(int)));
  CudaSafeCall(cudaMemcpy(this->parentLUTDevice, this->parentLUT, 8*27*sizeof(int), cudaMemcpyHostToDevice));
  CudaSafeCall(cudaMemcpy(this->childLUTDevice, this->childLUT, 8*27*sizeof(int), cudaMemcpyHostToDevice));

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
  dim3 block = {1,1,1};
  if(this->totalNodes < 65535) grid.x = (unsigned int) this->totalNodes;
  else{
    grid.x = 65535;
    while(grid.x*block.x < this->totalNodes){
      ++block.x;
    }
    while(grid.x*block.x > this->totalNodes){
      --grid.x;
      if(grid.x*block.x < this->totalNodes){
        ++grid.x;//to ensure that numThreads > totalNodes
        break;
      }
    }
  }
  computeNeighboringNodes<<<grid, block>>>(this->finalNodeArrayDevice, this->totalNodes, this->parentLUTDevice, this->childLUTDevice, this->depthIndexDevice, this->depth);
  CudaCheckError();
  CudaSafeCall(cudaMemcpy(this->finalNodeArray, this->finalNodeArrayDevice, this->totalNodes * sizeof(Node), cudaMemcpyDeviceToHost));

}
void Octree::computeVertexArray(){

}
void Octree::computeEdgeArray(){

}
void Octree::computeFaceArray(){

}
