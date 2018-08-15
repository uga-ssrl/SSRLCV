#include "octree.cuh"

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
  // err = cudaDeviceSynchronize();
  if (cudaSuccess != err) {
    fprintf(stderr, "cudaCheckError() with sync failed at %s:%i : %s\n",
    file, line, cudaGetErrorString(err));
    exit(-1);
  }
#endif

  return;
}


__constant__ int3 coordPlacementIdentity[8] = {
  {-1,-1,-1},
  {-1,-1,1},
  {-1,1,-1},
  {-1,1,1},
  {1,-1,-1},
  {1,-1,1},
  {1,1,-1},
  {1,1,1}
};

__constant__ int2 vertexEdgeIdentity[12] = {
  {0,1},
  {0,2},
  {1,3},
  {2,3},
  {0,4},
  {1,5},
  {2,6},
  {3,7},
  {4,5},
  {4,6},
  {5,7},
  {6,7}
};

__constant__ int4 vertexFaceIdentity[6] = {
  {0,1,2,3},
  {0,1,4,5},
  {0,2,4,6},
  {1,3,5,7},
  {2,3,6,7},
  {4,5,6,7}
};

__constant__ int4 edgeFaceIdentity[6] = {
  {0,1,2,3},
  {0,4,5,8},
  {1,4,6,9},
  {2,5,7,10},
  {3,6,7,11},
  {8,9,10,11}
};

__device__ __host__ Vertex::Vertex(){
  for(int i = 0; i < 8; ++i){
    this->nodes[i] = -1;
  }
  this->depth = -1;
  this->coord = {0.0f,0.0f,0.0f};
  this->color = {0,0,0};
}

__device__ __host__ Edge::Edge(){
  for(int i = 0; i < 4; ++i){
    this->nodes[i] = -1;
  }
  this->depth = -1;
  this->v1 = -1;
  this->v2 = -1;
  this->color = {0,0,0};

}

__device__ __host__ Face::Face(){
  this->nodes[0] = -1;
  this->nodes[1] = -1;
  this->depth = -1;
  this->e1 = -1;
  this->e2 = -1;
  this->e3 = -1;
  this->e4 = -1;
  this->color = {0,0,0};

}

__device__ __host__ Node::Node(){
  this->pointIndex = -1;
  this->center = {0.0f,0.0f,0.0f};
  this->color = {0,0,0};
  this->key = 0;
  this->width = 0.0f;
  this->numPoints = 0;
  this->parent = -1;
  this->depth = -1;
  this->numFinestChildren = 0;
  this->finestChildIndex = -1;
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

struct is_not_neg{
  __host__ __device__
  bool operator()(const int x)
  {
    return (x >= 0);
  }
};

__device__ __forceinline__ int floatToOrderedInt(float floatVal){
 int intVal = __float_as_int( floatVal );
 return (intVal >= 0 ) ? intVal : intVal ^ 0x7FFFFFFF;
}
__device__ __forceinline__ float orderedIntToFloat(int intVal){
 return __int_as_float( (intVal >= 0) ? intVal : intVal ^ 0x7FFFFFFF);
}
__device__ __host__ void printBits(size_t const size, void const * const ptr){
  unsigned char *b = (unsigned char*) ptr;
  unsigned char byte;
  int i, j;
  printf("bits - ");
  for (i=size-1;i>=0;i--){
    for (j=7;j>=0;j--){
      byte = (b[i] >> j) & 1;
      printf("%u", byte);
    }
  }
  printf("\n");
}
__global__ void getNodeKeys(float3* points, float3* nodeCenters, int* nodeKeys, float3 c, float W, int numPoints, int D){
  int globalID = blockIdx.x * blockDim.x + threadIdx.x;
  if(globalID < numPoints){
    float x = points[globalID].x;
    float y = points[globalID].y;
    float z = points[globalID].z;
    int key = 0;
    int depth = 1;
    W /= 2.0f;
    while(depth <= D){
      W /= 2.0f;
      if(x < c.x){
        key <<= 1;
        c.x -= W;
      }
      else{
        key = (key << 1) + 1;
        c.x += W;
      }
      if(y < c.y){
        key <<= 1;
        c.y -= W;
      }
      else{
        key = (key << 1) + 1;
        c.y += W;
      }
      if(z < c.z){
        key <<= 1;
        c.z -= W;
      }
      else{
        key = (key << 1) + 1;
        c.z += W;
      }
      depth++;
    }
    nodeKeys[globalID] = key;
    nodeCenters[globalID] = c;
  }
}

//createFinalNodeArray kernels
__global__ void findAllNodes(int numUniqueNodes, int* nodeNumbers, Node* uniqueNodes){
  int globalID = blockIdx.x * blockDim.x + threadIdx.x;
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
void calculateNodeAddresses(dim3 grid, dim3 block, int numUniqueNodes, Node* uniqueNodesDevice, int* nodeAddressesDevice, int* nodeNumbersDevice){
  findAllNodes<<<grid,block>>>(numUniqueNodes, nodeNumbersDevice, uniqueNodesDevice);
  cudaDeviceSynchronize();
  CudaCheckError();
  thrust::device_ptr<int> nN(nodeNumbersDevice);
  thrust::device_ptr<int> nA(nodeAddressesDevice);
  thrust::inclusive_scan(nN, nN + numUniqueNodes, nA);

}
__global__ void fillBlankNodeArray(Node* uniqueNodes, int* nodeNumbers, int* nodeAddresses, Node* outputNodeArray, int numUniqueNodes, int currentDepth, float totalWidth){
  int globalID = blockIdx.x * blockDim.x + threadIdx.x;
  int address = 0;
  if(globalID < numUniqueNodes && (globalID == 0 || nodeNumbers[globalID] == 8)){
    int siblingKey = uniqueNodes[globalID].key;
    uchar3 color = uniqueNodes[globalID].color;
    siblingKey &= 0xfffffff8;//will clear last 3 bits
    for(int i = 0; i < 8; ++i){
      address = nodeAddresses[globalID] + i;
      outputNodeArray[address] = Node();
      outputNodeArray[address].color = color;
      outputNodeArray[address].depth = currentDepth;
      outputNodeArray[address].key = siblingKey + i;
    }
  }
}
__global__ void fillFinestNodeArrayWithUniques(Node* uniqueNodes, int* nodeAddresses, Node* outputNodeArray, int numUniqueNodes, int* pointNodeIndex){
  int globalID = blockIdx.x * blockDim.x + threadIdx.x;
  int address = 0;
  int currentDKey = 0;
  if(globalID < numUniqueNodes){
    currentDKey = (uniqueNodes[globalID].key&(0x00000007));//will clear all but last 3 bits
    address = nodeAddresses[globalID] + currentDKey;
    for(int i = uniqueNodes[globalID].pointIndex; i < uniqueNodes[globalID].numPoints + uniqueNodes[globalID].pointIndex; ++i){
      pointNodeIndex[i] = address;
    }
    outputNodeArray[address].key = uniqueNodes[globalID].key;
    outputNodeArray[address].depth = uniqueNodes[globalID].depth;
    outputNodeArray[address].center = uniqueNodes[globalID].center;
    outputNodeArray[address].color = uniqueNodes[globalID].color;
    outputNodeArray[address].pointIndex = uniqueNodes[globalID].pointIndex;
    outputNodeArray[address].numPoints = uniqueNodes[globalID].numPoints;
    outputNodeArray[address].finestChildIndex = address;//itself
    outputNodeArray[address].numFinestChildren = 1;//itself
  }
}
__global__ void fillNodeArrayWithUniques(Node* uniqueNodes, int* nodeAddresses, Node* outputNodeArray, Node* childNodeArray,int numUniqueNodes){
  int globalID = blockIdx.x * blockDim.x + threadIdx.x;
  int address = 0;
  int currentDKey = 0;
  if(globalID < numUniqueNodes){
    currentDKey = (uniqueNodes[globalID].key&(0x00000007));//will clear all but last 3 bits
    address = nodeAddresses[globalID] + currentDKey;
    for(int i = 0; i < 8; ++i){
      outputNodeArray[address].children[i] = uniqueNodes[globalID].children[i];
      childNodeArray[uniqueNodes[globalID].children[i]].parent = address;
    }
    outputNodeArray[address].key = uniqueNodes[globalID].key;
    outputNodeArray[address].depth = uniqueNodes[globalID].depth;
    outputNodeArray[address].center = uniqueNodes[globalID].center;
    outputNodeArray[address].color = uniqueNodes[globalID].color;
    outputNodeArray[address].pointIndex = uniqueNodes[globalID].pointIndex;
    outputNodeArray[address].numPoints = uniqueNodes[globalID].numPoints;
    outputNodeArray[address].finestChildIndex = uniqueNodes[globalID].finestChildIndex;
    outputNodeArray[address].numFinestChildren = uniqueNodes[globalID].numFinestChildren;
  }
}
//TODO try and optimize
__global__ void generateParentalUniqueNodes(Node* uniqueNodes, Node* nodeArrayD, int numNodesAtDepth, float totalWidth){
  int numUniqueNodesAtParentDepth = numNodesAtDepth / 8;
  int globalID = blockIdx.x * blockDim.x + threadIdx.x;
  int nodeArrayIndex = globalID*8;
  if(globalID < numUniqueNodesAtParentDepth){
    uniqueNodes[globalID] = Node();//may not be necessary
    int firstUniqueChild = -1;
    bool childIsUnique[8] = {false};
    for(int i = 0; i < 8; ++i){
      if(nodeArrayD[nodeArrayIndex + i].pointIndex != -1){
        if(firstUniqueChild == -1){
          firstUniqueChild = i;
        }
        childIsUnique[i] = true;
      }
    }
    uniqueNodes[globalID].key = (nodeArrayD[nodeArrayIndex + firstUniqueChild].key>>3);
    uniqueNodes[globalID].pointIndex = nodeArrayD[nodeArrayIndex + firstUniqueChild].pointIndex;
    int depth =  nodeArrayD[nodeArrayIndex + firstUniqueChild].depth;
    uniqueNodes[globalID].depth = depth - 1;
    //should be the lowest index on the lowest child
    uniqueNodes[globalID].finestChildIndex = nodeArrayD[nodeArrayIndex + firstUniqueChild].finestChildIndex;

    float3 center = {0.0f,0.0f,0.0f};
    float widthOfNode = totalWidth/powf(2,depth);
    center.x = nodeArrayD[nodeArrayIndex + firstUniqueChild].center.x - (widthOfNode*0.5*coordPlacementIdentity[firstUniqueChild].x);
    center.y = nodeArrayD[nodeArrayIndex + firstUniqueChild].center.y - (widthOfNode*0.5*coordPlacementIdentity[firstUniqueChild].y);
    center.z = nodeArrayD[nodeArrayIndex + firstUniqueChild].center.z - (widthOfNode*0.5*coordPlacementIdentity[firstUniqueChild].z);
    uniqueNodes[globalID].center = center;

    for(int i = 0; i < 8; ++i){
      if(childIsUnique[i]){
        uniqueNodes[globalID].numPoints += nodeArrayD[nodeArrayIndex + i].numPoints;
        uniqueNodes[globalID].numFinestChildren += nodeArrayD[nodeArrayIndex + i].numFinestChildren;
      }
      else{
        nodeArrayD[nodeArrayIndex + i].center.x = center.x + (widthOfNode*0.5*coordPlacementIdentity[i].x);
        nodeArrayD[nodeArrayIndex + i].center.y = center.y + (widthOfNode*0.5*coordPlacementIdentity[i].y);
        nodeArrayD[nodeArrayIndex + i].center.z = center.z + (widthOfNode*0.5*coordPlacementIdentity[i].z);
      }
      uniqueNodes[globalID].children[i] = nodeArrayIndex + i;
      nodeArrayD[nodeArrayIndex + i].width = widthOfNode;
    }
  }
}
__global__ void computeNeighboringNodes(Node* nodeArray, int numNodes, int depthIndex,int* parentLUT, int* childLUT, int* numNeighbors, int childDepthIndex){
  int blockID = blockIdx.y * gridDim.x + blockIdx.x;
  if(blockID < numNodes){
    int neighborParentIndex = 0;
    nodeArray[blockID + depthIndex].neighbors[13] = blockID + depthIndex;
    __syncthreads();//threads wait until all other threads have finished above operations
    if(nodeArray[blockID + depthIndex].parent != -1){
      int parentIndex = nodeArray[blockID + depthIndex].parent + depthIndex + numNodes;
      int depthKey = nodeArray[blockID + depthIndex].key&(0x00000007);//will clear all but last 3 bits
      int lutIndexHelper = (depthKey*27) + threadIdx.x;
      int parentLUTIndex = parentLUT[lutIndexHelper];
      int childLUTIndex = childLUT[lutIndexHelper];
      neighborParentIndex = nodeArray[parentIndex].neighbors[parentLUTIndex];
      if(neighborParentIndex != -1){
        atomicAdd(numNeighbors, 1);
        nodeArray[blockID + depthIndex].neighbors[threadIdx.x] = nodeArray[neighborParentIndex].children[childLUTIndex];
      }
    }
    __syncthreads();//index updates
    //doing this mostly to prevent memcpy overhead
    if(childDepthIndex != -1 && threadIdx.x < 8 &&
      nodeArray[blockID + depthIndex].children[threadIdx.x] != -1){
      nodeArray[blockID + depthIndex].children[threadIdx.x] += childDepthIndex;
    }
    if(nodeArray[blockID + depthIndex].parent != -1 && threadIdx.x == 0){
      nodeArray[blockID + depthIndex].parent += depthIndex + numNodes;
    }
    else if(threadIdx.x == 0){//this means you are at root
      nodeArray[blockID + depthIndex].width = 2*nodeArray[nodeArray[blockID + depthIndex].children[0]].width;

    }
  }
}

__global__ void findNormalNeighborsAndComputeCMatrix(int numNodesAtDepth, int depthIndex, int maxNeighbors, Node* nodeArray, float3* points, float* cMatrix, int* neighborIndices, int* numNeighbors){
  int blockID = blockIdx.y * gridDim.x + blockIdx.x;
  if(blockID < numNodesAtDepth){
    float3 centroid = {0.0f,0.0f,0.0f};
    int n = 0;
    int regDepthIndex = depthIndex;
    int numPointsInNode = nodeArray[blockID + regDepthIndex].numPoints;
    int neighbor = -1;
    int regMaxNeighbors = maxNeighbors;
    int regPointIndex = nodeArray[blockID + regDepthIndex].pointIndex;
    float3 coord = {0.0f,0.0f,0.0f};
    float3 neighborCoord = {0.0f,0.0f,0.0f};
    float currentDistanceSq = 0.0f;
    float largestDistanceSq = 0.0f;
    int indexOfFurthestNeighbor = -1;
    int regNNPointIndex = 0;
    int numPointsInNeighbor = 0;
    float* distanceSq = new float[regMaxNeighbors];
    for(int threadID = threadIdx.x; threadID < numPointsInNode; threadID += blockDim.x){
      n = 0;
      coord = points[regPointIndex + threadID];
      currentDistanceSq = 0.0f;
      largestDistanceSq = 0.0f;
      indexOfFurthestNeighbor = -1;
      regNNPointIndex = 0;
      numPointsInNeighbor = 0;
      for(int i = 0; i < regMaxNeighbors; ++i) distanceSq[i] = 0.0f;
      for(int neigh = 0; neigh < 27; ++neigh){
        neighbor = nodeArray[blockID + regDepthIndex].neighbors[neigh];
        if(neighbor != -1){
          numPointsInNeighbor = nodeArray[neighbor].numPoints;
          regNNPointIndex = nodeArray[neighbor].pointIndex;
          for(int p = 0; p < numPointsInNeighbor; ++p){
            neighborCoord = points[regNNPointIndex + p];
            currentDistanceSq = ((coord.x - neighborCoord.x)*(coord.x - neighborCoord.x)) +
              ((coord.y - neighborCoord.y)*(coord.y - neighborCoord.y)) +
              ((coord.z - neighborCoord.z)*(coord.z - neighborCoord.z));
            if(n < regMaxNeighbors){
              if(currentDistanceSq > largestDistanceSq){
                largestDistanceSq = currentDistanceSq;
                indexOfFurthestNeighbor = n;
              }
              distanceSq[n] = currentDistanceSq;
              neighborIndices[(regPointIndex + threadID)*regMaxNeighbors + n] = regNNPointIndex + p;
              cMatrix[(regPointIndex + threadID)*regMaxNeighbors*3 + (n*3)] = neighborCoord.x;
              cMatrix[(regPointIndex + threadID)*regMaxNeighbors*3 + (n*3 + 1)] = neighborCoord.y;
              cMatrix[(regPointIndex + threadID)*regMaxNeighbors*3 + (n*3 + 2)] = neighborCoord.z;
              ++n;
            }
            else if(n == regMaxNeighbors && currentDistanceSq >= largestDistanceSq) continue;
            else{
              neighborIndices[(regPointIndex + threadID)*regMaxNeighbors + indexOfFurthestNeighbor] = regNNPointIndex + p;
              cMatrix[(regPointIndex + threadID)*regMaxNeighbors*3 + (indexOfFurthestNeighbor*3)] = neighborCoord.x;
              cMatrix[(regPointIndex + threadID)*regMaxNeighbors*3 + (indexOfFurthestNeighbor*3 + 1)] = neighborCoord.y;
              cMatrix[(regPointIndex + threadID)*regMaxNeighbors*3 + (indexOfFurthestNeighbor*3 + 2)] = neighborCoord.z;
              distanceSq[indexOfFurthestNeighbor] = currentDistanceSq;
              largestDistanceSq = 0.0f;
              for(int i = 0; i < regMaxNeighbors; ++i){
                if(distanceSq[i] > largestDistanceSq){
                  largestDistanceSq = distanceSq[i];
                  indexOfFurthestNeighbor = i;
                }
              }
            }
          }
        }
      }
      numNeighbors[regPointIndex + threadID] = n;
      for(int np = 0; np < n; ++np){
        centroid.x += cMatrix[(regPointIndex + threadID)*regMaxNeighbors*3 + (np*3)];
        centroid.y += cMatrix[(regPointIndex + threadID)*regMaxNeighbors*3 + (np*3 + 1)];
        centroid.z += cMatrix[(regPointIndex + threadID)*regMaxNeighbors*3 + (np*3 + 2)];
      }
      centroid = {centroid.x/n, centroid.y/n, centroid.z/n};
      for(int np = 0; np < n; ++np){
        cMatrix[(regPointIndex + threadID)*regMaxNeighbors*3 + (np*3)] -= centroid.x;
        cMatrix[(regPointIndex + threadID)*regMaxNeighbors*3 + (np*3 + 1)] -= centroid.y;
        cMatrix[(regPointIndex + threadID)*regMaxNeighbors*3 + (np*3 + 2)] -= centroid.z;
      }
    }
    delete[] distanceSq;
  }
}
__global__ void transposeFloatMatrix(int m, int n, float* matrix){
  int globalID = blockIdx.x * blockDim.x + threadIdx.x;
  if(globalID < m*n){
    int2 regLocation = {globalID/n,globalID%n};
    float regPastValue = matrix[globalID];
    __syncthreads();
    matrix[regLocation.y*m + regLocation.x] = regPastValue;
  }
}
__global__ void setNormal(int currentPoint, float* vt, float3* normals){
  normals[currentPoint] = {vt[2],vt[5],vt[8]};
}
__global__ void checkForAbiguity(int numPoints, int numCameras, float3* normals, float3* points, float3* cameraPositions, bool* ambiguous){
  int blockID = blockIdx.y * gridDim.x + blockIdx.x;
  if(blockID < numPoints && threadIdx.x < numCameras){
    float3 regCameraPosition = cameraPositions[threadIdx.x];
    float3 coord = points[blockID];
    float3 norm = normals[blockID];
    __shared__ int directionCheck;
    directionCheck = 0;
    __syncthreads();
    coord = {regCameraPosition.x - coord.x,regCameraPosition.y - coord.y,regCameraPosition.z - coord.z};
    float dot = (coord.x*norm.x) + (coord.y*norm.y) + (coord.z*norm.z);
    if(dot < 0) atomicSub(&directionCheck,1);
    else atomicAdd(&directionCheck,1);
    __syncthreads();
    if(abs(directionCheck) == numCameras){
      if(directionCheck < 0){
        normals[blockID] = {-1.0f*norm.x,-1.0f*norm.y,-1.0f*norm.z};
      }
      ambiguous[blockID] = false;
    }
    else{
      ambiguous[blockID] = true;
    }
  }
}
__global__ void reorient(int numNodesAtDepth, int depthIndex, Node* nodeArray, int* numNeighbors, int maxNeighbors, float3* normals, int* neighborIndices, bool* ambiguous){
  int blockID = blockIdx.y * gridDim.x + blockIdx.x;
  if(blockID < numNodesAtDepth){
    __shared__ bool ambiguityExists;
    ambiguityExists = true;
    __syncthreads();
    int regDepthIndex = depthIndex;
    int numPointsInNode = nodeArray[blockID + regDepthIndex].numPoints;
    int regPointIndex = nodeArray[blockID + regDepthIndex].pointIndex;
    int2 directionCounter = {0,0};
    float3 norm = {0.0f,0.0f,0.0f};
    float3 neighNorm = {0.0f,0.0f,0.0f};
    int regNumNeighbors = 0;
    int regNeighborIndex = 0;
    bool amb = true;
    while(ambiguityExists){
      ambiguityExists = false;
      for(int threadID = threadIdx.x; threadID < numPointsInNode; threadID += blockDim.x){
        if(!ambiguous[regPointIndex + threadID]) continue;
        amb = true;
        directionCounter = {0,0};
        norm = normals[regPointIndex + threadID];
        regNumNeighbors = numNeighbors[regPointIndex + threadID];
        for(int np = 0; np < regNumNeighbors; ++np){
          regNeighborIndex = neighborIndices[(regPointIndex + threadID)*maxNeighbors + np];
          if(ambiguous[regNeighborIndex]) continue;
          amb = false;
          neighNorm = normals[regNeighborIndex];
          if((norm.x*neighNorm.x)+(norm.y*neighNorm.y)+(norm.z*neighNorm.z) < 0){
            ++directionCounter.x;
          }
          else{
            ++directionCounter.y;
          }
        }
        if(!amb){
          ambiguous[blockID] = false;
          if(directionCounter.x < directionCounter.y){
            normals[blockID] = {-1.0f*norm.x,-1.0f*norm.y,-1.0f*norm.z};
          }
        }
        else{
          ambiguityExists = true;
        }
      }
      if(ambiguityExists) __syncthreads();
    }
  }
}

//vertex edge and face array kernels
__global__ void findVertexOwners(Node* nodeArray, int numNodes, int depthIndex, int* vertexLUT, int* numVertices, int* ownerInidices, int* vertexPlacement){
  int blockID = blockIdx.y * gridDim.x + blockIdx.x;
  if(blockID < numNodes){
    int vertexID = (blockID*8) + threadIdx.x;
    int sharesVertex = -1;
    for(int i = 0; i < 7; ++i){//iterate through neighbors that share vertex
      sharesVertex = vertexLUT[(threadIdx.x*7) + i];
      if(nodeArray[blockID + depthIndex].neighbors[sharesVertex] != -1 && sharesVertex < 13){//less than itself
        return;
      }
    }
    //if thread reaches this point, that means that this vertex is owned by the current node
    //also means owner == current node
    ownerInidices[vertexID] = blockID + depthIndex;
    vertexPlacement[vertexID] = threadIdx.x;
    atomicAdd(numVertices, 1);
  }
}
__global__ void fillUniqueVertexArray(Node* nodeArray, Vertex* vertexArray, int numVertices, int vertexIndex,int depthIndex, int depth, float width, int* vertexLUT, int* ownerInidices, int* vertexPlacement){
  int globalID = blockIdx.x * blockDim.x + threadIdx.x;
  if(globalID < numVertices){

    int ownerNodeIndex = ownerInidices[globalID];
    int ownedIndex = vertexPlacement[globalID];

    nodeArray[ownerNodeIndex].vertices[ownedIndex] = globalID + vertexIndex;

    float depthHalfWidth = width/powf(2, depth + 1);
    Vertex vertex = Vertex();
    vertex.coord.x = nodeArray[ownerNodeIndex].center.x + (depthHalfWidth*coordPlacementIdentity[ownedIndex].x);
    vertex.coord.y = nodeArray[ownerNodeIndex].center.y + (depthHalfWidth*coordPlacementIdentity[ownedIndex].y);
    vertex.coord.z = nodeArray[ownerNodeIndex].center.z + (depthHalfWidth*coordPlacementIdentity[ownedIndex].z);
    vertex.color = nodeArray[ownerNodeIndex].color;
    vertex.depth = depth;
    vertex.nodes[0] = ownerNodeIndex;
    int neighborSharingVertex = -1;
    for(int i = 0; i < 7; ++i){
      neighborSharingVertex = nodeArray[ownerNodeIndex].neighbors[vertexLUT[(ownedIndex*7) + i]];
      vertex.nodes[i + 1] =  neighborSharingVertex;
      if(neighborSharingVertex == -1) continue;
      nodeArray[neighborSharingVertex].vertices[6 - i] = globalID + vertexIndex;
    }
    vertexArray[globalID] = vertex;
  }
}
__global__ void findEdgeOwners(Node* nodeArray, int numNodes, int depthIndex, int* edgeLUT, int* numEdges, int* ownerInidices, int* edgePlacement){
  int blockID = blockIdx.y * gridDim.x + blockIdx.x;
  if(blockID < numNodes){
    int edgeID = (blockID*12) + threadIdx.x;
    int sharesEdge = -1;
    for(int i = 0; i < 3; ++i){//iterate through neighbors that share edge
      sharesEdge = edgeLUT[(threadIdx.x*3) + i];
      if(nodeArray[blockID + depthIndex].neighbors[sharesEdge] != -1 && sharesEdge < 13){//less than itself
        return;
      }
    }
    //if thread reaches this point, that means that this edge is owned by the current node
    //also means owner == current node
    ownerInidices[edgeID] = blockID + depthIndex;
    edgePlacement[edgeID] = threadIdx.x;
    atomicAdd(numEdges, 1);
  }
}
__global__ void fillUniqueEdgeArray(Node* nodeArray, Edge* edgeArray, int numEdges, int edgeIndex, int depthIndex, int depth, float width, int* edgeLUT, int* ownerInidices, int* edgePlacement){
  int globalID = blockIdx.x * blockDim.x + threadIdx.x;
  if(globalID < numEdges){
    int ownerNodeIndex = ownerInidices[globalID];
    int ownedIndex = edgePlacement[globalID];
    nodeArray[ownerNodeIndex].edges[ownedIndex] = globalID + edgeIndex;

    float depthHalfWidth = width/powf(2, depth + 1);
    Edge edge = Edge();
    edge.v1 = nodeArray[ownerNodeIndex].vertices[vertexEdgeIdentity[ownedIndex].x];
    edge.v2 = nodeArray[ownerNodeIndex].vertices[vertexEdgeIdentity[ownedIndex].y];
    edge.color = nodeArray[ownerNodeIndex].color;
    edge.depth = depth;
    edge.nodes[0] = ownerNodeIndex;
    int neighborSharingEdge = -1;
    int placement = 0;
    int neighborPlacement = 0;
    for(int i = 0; i < 3; ++i){
      neighborPlacement = edgeLUT[(ownedIndex*3) + i];
      neighborSharingEdge = nodeArray[ownerNodeIndex].neighbors[neighborPlacement];
      edge.nodes[i + 1] =  neighborSharingEdge;
      if(neighborSharingEdge == -1) continue;
      placement = ownedIndex + 13 - neighborPlacement;
      if(neighborPlacement <= 8 || ((ownedIndex == 4 || ownedIndex == 5) && neighborPlacement < 12)){
        --placement;
      }
      else if(neighborPlacement >= 18 || ((ownedIndex == 6 || ownedIndex == 7) && neighborPlacement > 14)){
        ++placement;
      }
      nodeArray[neighborSharingEdge].edges[placement] = globalID + edgeIndex;
    }
    edgeArray[globalID] = edge;
  }
}
__global__ void findFaceOwners(Node* nodeArray, int numNodes, int depthIndex, int* faceLUT, int* numFaces, int* ownerInidices, int* facePlacement){
  int blockID = blockIdx.y * gridDim.x + blockIdx.x;
  if(blockID < numNodes){
    int faceID = (blockID*6) + threadIdx.x;
    int sharesFace = -1;
    sharesFace = faceLUT[threadIdx.x];
    if(nodeArray[blockID + depthIndex].neighbors[sharesFace] != -1 && sharesFace < 13){//less than itself
      return;
    }
    //if thread reaches this point, that means that this face is owned by the current node
    //also means owner == current node
    ownerInidices[faceID] = blockID + depthIndex;
    facePlacement[faceID] = threadIdx.x;
    atomicAdd(numFaces, 1);
  }

}
__global__ void fillUniqueFaceArray(Node* nodeArray, Face* faceArray, int numFaces, int faceIndex, int depthIndex, int depth, float width, int* faceLUT, int* ownerInidices, int* facePlacement){
  int globalID = blockIdx.x * blockDim.x + threadIdx.x;
  if(globalID < numFaces){

    int ownerNodeIndex = ownerInidices[globalID];
    int ownedIndex = facePlacement[globalID];

    nodeArray[ownerNodeIndex].faces[ownedIndex] = globalID + faceIndex;

    float depthHalfWidth = width/powf(2, depth + 1);
    Face face = Face();

    face.e1 = nodeArray[ownerNodeIndex].edges[edgeFaceIdentity[ownedIndex].x];
    face.e2 = nodeArray[ownerNodeIndex].edges[edgeFaceIdentity[ownedIndex].y];
    face.e3 = nodeArray[ownerNodeIndex].edges[edgeFaceIdentity[ownedIndex].z];
    face.e4 = nodeArray[ownerNodeIndex].edges[edgeFaceIdentity[ownedIndex].w];
    face.color = nodeArray[ownerNodeIndex].color;
    face.depth = depth;
    face.nodes[0] = ownerNodeIndex;
    int neighborSharingFace = -1;
    neighborSharingFace = nodeArray[ownerNodeIndex].neighbors[faceLUT[ownedIndex]];
    face.nodes[1] =  neighborSharingFace;
    if(neighborSharingFace != -1)nodeArray[neighborSharingFace].faces[5 - ownedIndex] = globalID + faceIndex;
    faceArray[globalID] = face;

  }
}

Octree::Octree(){
  this->simpleOctree = true;
  this->depth = 1;
  this->pointNodeDeviceReady = false;
  this->vertexArrayDeviceReady = false;
  this->edgeArrayDeviceReady = false;
  this->faceArrayDeviceReady = false;
  this->pointsDeviceReady = false;
  this->normalsDeviceReady = false;
  this->colorsDeviceReady = false;
}

Octree::~Octree(){
  delete[] this->vertexIndex;
  delete[] this->edgeIndex;
  delete[] this->faceIndex;
  delete[] this->finalNodeArray;
  delete[] this->vertexArray;
  delete[] this->edgeArray;
  delete[] this->faceArray;
  delete[] this->points;
  delete[] this->normals;
  delete[] this->colors;
  delete[] this->depthIndex;
  delete[] this->pointNodeIndex;
  CudaSafeCall(cudaFree(this->finalNodeArrayDevice));
  if(this->pointNodeDeviceReady) CudaSafeCall(cudaFree(this->pointNodeIndexDevice));
  if(this->vertexArrayDeviceReady) CudaSafeCall(cudaFree(this->vertexArrayDevice));
  if(this->edgeArrayDeviceReady) CudaSafeCall(cudaFree(this->edgeArrayDevice));
  if(this->faceArrayDeviceReady) CudaSafeCall(cudaFree(this->faceArrayDevice));
  if(this->pointsDeviceReady) CudaSafeCall(cudaFree(this->pointsDevice));
  if(this->normalsDeviceReady) CudaSafeCall(cudaFree(this->normalsDevice));
  if(this->colorsDeviceReady) CudaSafeCall(cudaFree(this->colorsDevice));

}

//TODO clean the main switch statement up
void Octree::parsePLY(){
  std::cout<<this->pathToFile + " is being used as the ply"<<std::endl;
	std::ifstream plystream(this->pathToFile);
	std::string currentLine;
  float minX = std::numeric_limits<float>::max(),
  minY = std::numeric_limits<float>::max(),
  minZ = std::numeric_limits<float>::max(),
  maxX = std::numeric_limits<float>::min(),
  maxY = std::numeric_limits<float>::min(),
  maxZ = std::numeric_limits<float>::min();

  std::string temp = "";
  bool headerIsDone = false;
  int currentPoint = 0;
	if (plystream.is_open()) {
		while (getline(plystream, currentLine)) {
      std::istringstream stringBuffer = std::istringstream(currentLine);
      if(!headerIsDone){
        if(currentLine.find("element vertex") != std::string::npos){
          stringBuffer >> temp;
          stringBuffer >> temp;
          stringBuffer >> this->numPoints;
          this->points = new float3[this->numPoints];
          this->normals = new float3[this->numPoints];
          this->colors = new uchar3[this->numPoints];
          this->finestNodeCenters = new float3[this->numPoints];
          this->finestNodePointIndexes = new int[this->numPoints];
          this->finestNodeKeys = new int[this->numPoints];
          this->pointNodeIndex = new int[this->numPoints];
        }
        else if(currentLine.find("nx") != std::string::npos){
          this->normalsComputed = true;
          std::cout<<"normals are precomputed"<<std::endl;
        }
        else if(currentLine.find("blue") != std::string::npos){
          this->hasColor = true;
        }
        else if(currentLine.find("end_header") != std::string::npos){
          headerIsDone = true;
        }
        continue;
      }
      else if(currentPoint >= this->numPoints) break;

      float value = 0.0;
      int index = 0;
      float3 point = {0.0f, 0.0f, 0.0f};
      float3 normal = {0.0f, 0.0f, 0.0f};
      uchar3 color = {255, 255, 255};
      bool lineIsDone = false;

      while(stringBuffer >> value){
        switch(index){
          case 0:
            point.x = value;
            if(value > maxX) maxX = value;
            if(value < minX) minX = value;
            ++index;
            break;
          case 1:
            point.y = value;
            if(value > maxY) maxY = value;
            if(value < minY) minY = value;
            ++index;
            break;
          case 2:
            point.z = value;
            if(value > maxZ) maxZ = value;
            if(value < minZ) minZ = value;
            if(!this->hasColor && !this->normalsComputed){
              lineIsDone = true;
            }
            ++index;
            break;
          case 3:
            if(this->normalsComputed) normal.x = value;
            else if(this->hasColor) color.x = value;
            else lineIsDone = true;
            ++index;
            break;
          case 4:
            if(this->normalsComputed) normal.y = value;
            else if(this->hasColor) color.y = value;
            else lineIsDone = true;
            ++index;
            break;
          case 5:
            if(this->normalsComputed) normal.z = value;
            else if(this->hasColor){
              color.z = value;
              lineIsDone = true;
            }
            else lineIsDone = true;
            ++index;
            break;
          case 6:
            if(this->normalsComputed && this->hasColor){
              color.x = value;
            }
            else lineIsDone = true;
            ++index;
            break;
          case 7:
            if(this->normalsComputed && this->hasColor){
               color.y = value;
            }
            else lineIsDone = true;
            ++index;
            break;
          case 8:
            if(this->normalsComputed && this->hasColor){
              color.z = value;
            }
            lineIsDone = true;
            ++index;
            break;
          default:
            lineIsDone = true;
            break;
        }
        if(lineIsDone){
          this->points[currentPoint] = point;
          this->normals[currentPoint] = normal;
          this->colors[currentPoint] = color;
          this->finestNodePointIndexes[currentPoint] = currentPoint;
          this->finestNodeCenters[currentPoint] = {0.0f,0.0f,0.0f};
          this->finestNodeKeys[currentPoint] = 0;
          this->pointNodeIndex[currentPoint] = -1;
          break;
        }
      }
      ++currentPoint;
		}
    this->center.x = (maxX + minX)/2.0f;
    this->center.y = (maxY + minY)/2.0f;
    this->center.z = (maxZ + minZ)/2.0f;

    this->width = maxX - minX;
    if(this->width < maxY - minY) this->width = maxY - minY;
    if(this->width < maxZ - minZ) this->width = maxZ - minZ;

    minX = this->center.x - (this->width/2.0f);
    minY = this->center.y - (this->width/2.0f);
    minZ = this->center.z - (this->width/2.0f);
    maxX = this->center.x + (this->width/2.0f);
    maxY = this->center.y + (this->width/2.0f);
    maxZ = this->center.z + (this->width/2.0f);

    this->min = {minX,minY,minZ};
    this->max = {maxX,maxY,maxZ};
    this->totalNodes = 0;
    this->numFinestUniqueNodes = 0;

    printf("\nmin = %f,%f,%f\n",this->min.x,this->min.y,this->min.z);
    printf("max = %f,%f,%f\n",this->max.x,this->max.y,this->max.z);
    printf("bounding box width = %f\n", this->width);
    printf("center = %f,%f,%f\n",this->center.x,this->center.y,this->center.z);
    printf("number of points = %d\n\n", this->numPoints);
	}
	else{
    std::cout << "Unable to open: " + this->pathToFile<< std::endl;
    exit(1);
  }
}

Octree::Octree(std::string pathToFile, int depth){
  this->pathToFile = pathToFile;
  this->hasColor = false;
  this->normalsComputed = false;
  this->parsePLY();
  this->depth = depth;
  this->simpleOctree = false;
  this->pointNodeDeviceReady = false;
  this->vertexArrayDeviceReady = false;
  this->edgeArrayDeviceReady = false;
  this->faceArrayDeviceReady = false;
  this->pointsDeviceReady = false;
  this->normalsDeviceReady = false;
  this->colorsDeviceReady = false;
}

void Octree::init_octree_gpu(){
  clock_t cudatimer;
  cudatimer = clock();

  CudaSafeCall(cudaMalloc((void**)&this->finestNodeCentersDevice, this->numPoints * sizeof(float3)));
  CudaSafeCall(cudaMalloc((void**)&this->finestNodeKeysDevice, this->numPoints * sizeof(int)));
  CudaSafeCall(cudaMalloc((void**)&this->finestNodePointIndexesDevice, this->numPoints * sizeof(int)));

  this->copyPointsToDevice();
  this->copyFinestNodeCentersToDevice();
  this->copyFinestNodeKeysToDevice();
  if(this->normalsComputed)  this->copyNormalsToDevice();
  if(this->hasColor)  this->copyColorsToDevice();


  printf("octree initial allocation & base variable copy took %f seconds.\n",((float) clock() - cudatimer)/CLOCKS_PER_SEC);
}

void Octree::copyPointsToDevice(){
  this->pointsDeviceReady = true;
  CudaSafeCall(cudaMalloc((void**)&this->pointsDevice, this->numPoints * sizeof(float3)));
  CudaSafeCall(cudaMemcpy(this->pointsDevice, this->points, this->numPoints * sizeof(float3), cudaMemcpyHostToDevice));
}
void Octree::copyPointsToHost(){
  if(this->pointsDeviceReady){
    this->pointsDeviceReady = false;
    CudaSafeCall(cudaMemcpy(this->points, this->pointsDevice, this->numPoints * sizeof(float3), cudaMemcpyDeviceToHost));
    CudaSafeCall(cudaFree(this->pointsDevice));
  }
  else{
    std::cout<<"WARNING - points already on host"<<std::endl;
  }
}
void Octree::copyNormalsToDevice(){
  this->normalsDeviceReady = true;
  CudaSafeCall(cudaMalloc((void**)&this->normalsDevice, this->numPoints * sizeof(float3)));
  CudaSafeCall(cudaMemcpy(this->normalsDevice, this->normals, this->numPoints * sizeof(float3), cudaMemcpyHostToDevice));
}
void Octree::copyNormalsToHost(){
  if(this->normalsDeviceReady){
    this->normalsDeviceReady = false;
    CudaSafeCall(cudaMemcpy(this->normals, this->normalsDevice, this->numPoints * sizeof(float3), cudaMemcpyDeviceToHost));
    CudaSafeCall(cudaFree(this->normalsDevice));
  }
  else{
    std::cout<<"WARNING - normals already on host"<<std::endl;
  }
}
void Octree::copyColorsToDevice(){
  this->colorsDeviceReady = true;
  CudaSafeCall(cudaMalloc((void**)&this->colorsDevice, this->numPoints * sizeof(uchar3)));
  CudaSafeCall(cudaMemcpy(this->colorsDevice, this->colors, this->numPoints * sizeof(uchar3), cudaMemcpyHostToDevice));
}
void Octree::copyColorsToHost(){
  if(this->colorsDeviceReady){
    this->colorsDeviceReady = false;
    CudaSafeCall(cudaMemcpy(this->colors, this->colorsDevice, this->numPoints * sizeof(uchar3), cudaMemcpyDeviceToHost));
    CudaSafeCall(cudaFree(this->colorsDevice));
  }
  else{
    std::cout<<"WARNING - colors already on host"<<std::endl;
  }
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
void Octree::freePrereqArrays(){
  clock_t cudatimer;
  cudatimer = clock();

  delete[] this->finestNodeCenters;
  delete[] this->finestNodePointIndexes;
  delete[] this->finestNodeKeys;
  delete[] this->uniqueNodesAtFinestLevel;
  CudaSafeCall(cudaFree(this->finestNodeCentersDevice));
  CudaSafeCall(cudaFree(this->finestNodePointIndexesDevice));
  CudaSafeCall(cudaFree(this->finestNodeKeysDevice));

  printf("octree freePrereqArrays took %f seconds.\n",((float) clock() - cudatimer)/CLOCKS_PER_SEC);
}

void Octree::copyNodesToDevice(){
  CudaSafeCall(cudaMemcpy(this->finalNodeArrayDevice, this->finalNodeArray, this->totalNodes * sizeof(Node), cudaMemcpyHostToDevice));
}
void Octree::copyNodesToHost(){
  CudaSafeCall(cudaMemcpy(this->finalNodeArray, this->finalNodeArrayDevice, this->totalNodes * sizeof(Node), cudaMemcpyDeviceToHost));
}

void Octree::copyPointNodeIndexesToDevice(){
  this->pointNodeDeviceReady = true;
  CudaSafeCall(cudaMalloc((void**)&pointNodeIndexDevice, this->numPoints * sizeof(int)));
  CudaSafeCall(cudaMemcpy(this->pointNodeIndexDevice, this->pointNodeIndex, this->numPoints * sizeof(int), cudaMemcpyHostToDevice));
}
void Octree::copyPointNodeIndexesToHost(){
  if(this->pointNodeDeviceReady){
    this->pointNodeDeviceReady = false;
    CudaSafeCall(cudaMemcpy(this->pointNodeIndex, this->pointNodeIndexDevice, this->numPoints * sizeof(int), cudaMemcpyDeviceToHost));
    CudaSafeCall(cudaFree(this->pointNodeIndexDevice));
  }
  else{
    std::cout<<"WARNING - pointNodeIndices already on host"<<std::endl;
  }
}
void Octree::copyVerticesToDevice(){
  this->vertexArrayDeviceReady = true;
  CudaSafeCall(cudaMalloc((void**)&this->vertexArrayDevice, this->totalVertices*sizeof(Vertex)));
  CudaSafeCall(cudaMemcpy(this->vertexArrayDevice, this->vertexArray, this->totalVertices * sizeof(Vertex), cudaMemcpyHostToDevice));
}
void Octree::copyVerticesToHost(){
  if(this->vertexArrayDeviceReady){
    this->vertexArrayDeviceReady = false;
    CudaSafeCall(cudaMemcpy(this->vertexArray, this->vertexArrayDevice, this->totalVertices * sizeof(Vertex), cudaMemcpyDeviceToHost));
    CudaSafeCall(cudaFree(this->vertexArrayDevice));
  }
  else{
    std::cout<<"WARNING - vertexArray already on host"<<std::endl;
  }
}
void Octree::copyEdgesToDevice(){
  this->edgeArrayDeviceReady = true;
  CudaSafeCall(cudaMalloc((void**)&this->edgeArrayDevice, this->totalEdges*sizeof(Edge)));
  CudaSafeCall(cudaMemcpy(this->edgeArrayDevice, this->edgeArray, this->totalEdges * sizeof(Edge), cudaMemcpyHostToDevice));
}
void Octree::copyEdgesToHost(){
  if(this->edgeArrayDeviceReady){
    this->edgeArrayDeviceReady = false;
    CudaSafeCall(cudaMemcpy(this->edgeArray, this->edgeArrayDevice, this->totalEdges * sizeof(Edge), cudaMemcpyDeviceToHost));
    CudaSafeCall(cudaFree(this->edgeArrayDevice));
  }
  else{
    std::cout<<"WARNING - edgeArray already on host"<<std::endl;
  }
}
void Octree::copyFacesToDevice(){
  this->faceArrayDeviceReady = true;
  CudaSafeCall(cudaMalloc((void**)&this->faceArrayDevice, this->totalFaces*sizeof(Face)));
  CudaSafeCall(cudaMemcpy(this->faceArrayDevice, this->faceArray, this->totalFaces * sizeof(Face), cudaMemcpyHostToDevice));
}
void Octree::copyFacesToHost(){
  if(this->faceArrayDeviceReady){
    this->faceArrayDeviceReady = false;
    CudaSafeCall(cudaMemcpy(this->faceArray, this->faceArrayDevice, this->totalFaces * sizeof(Face), cudaMemcpyDeviceToHost));
    CudaSafeCall(cudaFree(this->faceArrayDevice));
  }
  else{
    std::cout<<"WARNING - faceArray already on host"<<std::endl;
  }
}

void Octree::generateKeys(){
  clock_t cudatimer;
  cudatimer = clock();

  dim3 grid = {1,1,1};
  dim3 block = {1,1,1};
  if(this->numPoints < 65535) grid.x = (unsigned int) this->numPoints;
  else{
    grid.x = 65535;
    while(grid.x*block.x < this->numPoints){
      ++block.x;
    }
    while(grid.x*block.x > this->numPoints){
      --grid.x;
      if(grid.x*block.x < this->numPoints){
        ++grid.x;//to ensure that numThreads > this->numPoints
        break;
      }
    }
  }

  getNodeKeys<<<grid,block>>>(this->pointsDevice, this->finestNodeCentersDevice, this->finestNodeKeysDevice, this->center, this->width, this->numPoints, this->depth);
  CudaCheckError();

  printf("octree generateNodeKeys took %f seconds.\n",((float) clock() - cudatimer)/CLOCKS_PER_SEC);
}
void Octree::prepareFinestUniquNodes(){
  clock_t cudatimer;
  cudatimer = clock();

  //SORT DATA

  thrust::device_ptr<int> kys = thrust::device_pointer_cast(this->finestNodeKeysDevice);
  thrust::device_ptr<float3> pnts = thrust::device_pointer_cast(this->pointsDevice);
  thrust::device_ptr<float3> cnts = thrust::device_pointer_cast(this->finestNodeCentersDevice);

  thrust::device_vector<float3> sortedPnts(this->numPoints);
  thrust::device_vector<float3> sortedCnts(this->numPoints);

  thrust::counting_iterator<int> iter(0);
  thrust::device_vector<int> indices(this->numPoints);
  thrust::copy(iter, iter + this->numPoints, indices.begin());

  thrust::sort_by_key(kys, kys + this->numPoints, indices.begin());
  this->copyFinestNodeKeysToHost();

  //sort based on indices
  thrust::gather(indices.begin(), indices.end(), pnts, sortedPnts.begin());
  thrust::gather(indices.begin(), indices.end(), cnts, sortedCnts.begin());
  CudaSafeCall(cudaMemcpy(this->points, thrust::raw_pointer_cast(sortedPnts.data()), this->numPoints*sizeof(float3),cudaMemcpyDeviceToHost));
  CudaSafeCall(cudaMemcpy(this->finestNodeCenters, thrust::raw_pointer_cast(sortedCnts.data()), this->numPoints*sizeof(float3),cudaMemcpyDeviceToHost));
  this->copyFinestNodeCentersToDevice();

  if(this->hasColor){
    thrust::device_ptr<uchar3> clrs(this->colorsDevice);
    thrust::device_vector<uchar3> sortedClrs(this->numPoints);
    thrust::gather(indices.begin(), indices.end(), clrs, sortedClrs.begin());
    CudaSafeCall(cudaMemcpy(this->colors, thrust::raw_pointer_cast(sortedClrs.data()), this->numPoints*sizeof(uchar3),cudaMemcpyDeviceToHost));
  }
  if(this->normalsComputed){
    thrust::device_ptr<float3> nmls(this->normalsDevice);
    thrust::device_vector<float3> sortedNmls(this->numPoints);
    thrust::gather(indices.begin(), indices.end(), nmls, sortedNmls.begin());
    CudaSafeCall(cudaMemcpy(this->normals, thrust::raw_pointer_cast(sortedNmls.data()), this->numPoints*sizeof(float3),cudaMemcpyDeviceToHost));
  }

  if(this->pointsDeviceReady){
    CudaSafeCall(cudaFree(this->pointsDevice));
    this->pointsDeviceReady = false;
  }
  if(this->normalsDeviceReady){
    CudaSafeCall(cudaFree(this->normalsDevice));
    this->normalsDeviceReady = false;
  }
  if(this->colorsDeviceReady){
    CudaSafeCall(cudaFree(this->colorsDevice));
    this->colorsDeviceReady = false;
  }

  //TODO OPTIMIZE THIS PORTION
  //COMPACT DATA

  thrust::pair<int*, int*> new_end;//the last value of these node arrays

  new_end = thrust::unique_by_key(this->finestNodeKeys, this->finestNodeKeys + this->numPoints, this->finestNodePointIndexes);

  bool foundFirst = false;
  int numUniqueNodes = 0;
  // for(int i = 0; i < this->numPoints; ++i){
  //   printf("%d, {%f,%f,%f} ",i,this->points[i].x,this->points[i].y,this->points[i].z);
  //   printBits(sizeof(int), &this->finestNodeKeys[i]);
  // }
  while(numUniqueNodes != this->numPoints || numUniqueNodes == 0){
    if(this->finestNodeKeys[numUniqueNodes] == *new_end.first){
      if(foundFirst) break;
      else foundFirst = true;
    }
    numUniqueNodes++;
  }
  this->numFinestUniqueNodes = numUniqueNodes;
  //FILL FINEST NODES

  this->uniqueNodesAtFinestLevel = new Node[this->numFinestUniqueNodes];
  for(int i = 0; i < this->numFinestUniqueNodes; ++i){
    Node currentNode;
    currentNode.key = this->finestNodeKeys[i];
    //this should correct
    currentNode.center = this->finestNodeCenters[this->finestNodePointIndexes[i]];

    currentNode.pointIndex = this->finestNodePointIndexes[i];
    currentNode.depth = this->depth;
    if(i + 1 != this->numFinestUniqueNodes){
      currentNode.numPoints = this->finestNodePointIndexes[i + 1] - this->finestNodePointIndexes[i];
    }
    else{
      currentNode.numPoints = this->numPoints - this->finestNodePointIndexes[i];

    }
    int3 colorHolder = {0,0,0};
    for(int p = currentNode.pointIndex; p < currentNode.pointIndex + currentNode.numPoints; ++p){
      colorHolder.x += this->colors[p].x;
      colorHolder.y += this->colors[p].y;
      colorHolder.z += this->colors[p].z;
    }
    uchar3 color;
    colorHolder.x /= currentNode.numPoints;
    colorHolder.y /= currentNode.numPoints;
    colorHolder.z /= currentNode.numPoints;
    color = {(unsigned char) colorHolder.x,(unsigned char) colorHolder.y,(unsigned char) colorHolder.z};
    currentNode.color = {color.x,color.y,color.z};
    this->uniqueNodesAtFinestLevel[i] = currentNode;
  }

  printf("octree prepareFinestUniquNodes took %f seconds.\n", ((float) clock() - cudatimer)/CLOCKS_PER_SEC);
}

void Octree::createFinalNodeArray(){
  clock_t cudatimer;
  cudatimer = clock();

  Node* uniqueNodesDevice;
  CudaSafeCall(cudaMalloc((void**)&uniqueNodesDevice, this->numFinestUniqueNodes*sizeof(Node)));
  CudaSafeCall(cudaMemcpy(uniqueNodesDevice, this->uniqueNodesAtFinestLevel, this->numFinestUniqueNodes*sizeof(Node), cudaMemcpyHostToDevice));
  Node** nodeArray2D = new Node*[this->depth + 1];

  int* nodeAddressesDevice;
  int* nodeNumbersDevice;

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

    CudaSafeCall(cudaMalloc((void**)&nodeNumbersDevice, numUniqueNodes * sizeof(int)));
    CudaSafeCall(cudaMalloc((void**)&nodeAddressesDevice, numUniqueNodes * sizeof(int)));
    //this is just to fill the arrays with 0s
    calculateNodeAddresses(grid, block, numUniqueNodes, uniqueNodesDevice, nodeAddressesDevice, nodeNumbersDevice);

    int numNodesAtDepth = 0;
    CudaSafeCall(cudaMemcpy(&numNodesAtDepth, nodeAddressesDevice + (numUniqueNodes - 1), sizeof(int), cudaMemcpyDeviceToHost));

    numNodesAtDepth = (d > 0) ? numNodesAtDepth + 8: 1;


    CudaSafeCall(cudaMalloc((void**)&nodeArray2D[this->depth - d], numNodesAtDepth* sizeof(Node)));

    fillBlankNodeArray<<<grid,block>>>(uniqueNodesDevice, nodeNumbersDevice,  nodeAddressesDevice, nodeArray2D[this->depth - d], numUniqueNodes, d, this->width);
    CudaCheckError();
    cudaDeviceSynchronize();
    if(this->depth == d){
      this->copyPointNodeIndexesToDevice();
      fillFinestNodeArrayWithUniques<<<grid,block>>>(uniqueNodesDevice, nodeAddressesDevice,nodeArray2D[this->depth - d], numUniqueNodes, this->pointNodeIndexDevice);
      this->copyPointNodeIndexesToHost();
      CudaCheckError();

    }
    else{
      fillNodeArrayWithUniques<<<grid,block>>>(uniqueNodesDevice, nodeAddressesDevice, nodeArray2D[this->depth - d], nodeArray2D[this->depth - d - 1], numUniqueNodes);
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
      generateParentalUniqueNodes<<<grid,block>>>(uniqueNodesDevice, nodeArray2D[this->depth - d], numNodesAtDepth, this->width);
      CudaCheckError();
    }
    this->depthIndex[this->depth - d] = this->totalNodes;
    this->totalNodes += numNodesAtDepth;
  }
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
  delete[] nodeArray2D;

  printf("octree buildFinalNodeArray took %f seconds.\n\n", ((float) clock() - cudatimer)/CLOCKS_PER_SEC);
  printf("TOTAL NODES = %d\n\n",this->totalNodes);

}

void Octree::printLUTs(){
  std::cout<<"\nPARENT LUT"<<std::endl;
  for(int row = 0; row <  8; ++row){
    for(int col = 0; col < 27; ++col){
      std::cout<<this->parentLUT[row][col]<<" ";
    }
    std::cout<<std::endl;
  }
  std::cout<<"\nCHILD LUT"<<std::endl;
  for(int row = 0; row <  8; ++row){
    for(int col = 0; col < 27; ++col){
      std::cout<<this->childLUT[row][col]<<" ";
    }
    std::cout<<std::endl;
  }
  std::cout<<"\nVERTEX LUT"<<std::endl;
  for(int row = 0; row <  8; ++row){
    for(int col = 0; col < 7; ++col){
      std::cout<<this->vertexLUT[row][col]<<" ";
    }
    std::cout<<std::endl;
  }
  std::cout<<"\nEDGE LUT"<<std::endl;
  for(int row = 0; row <  12; ++row){
    for(int col = 0; col < 3; ++col){
      std::cout<<this->edgeLUT[row][col]<<" ";
    }
    std::cout<<std::endl;
  }
  std::cout<<"\nFACE LUT"<<std::endl;
  for(int row = 0; row <  6; ++row){
    std::cout<<this->faceLUT[row]<<" ";
  }
  std::cout<<std::endl<<std::endl;
}
void Octree::fillLUTs(){
  clock_t cudatimer;
  cudatimer = clock();

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

  int flatParentLUT[216];
  int flatChildLUT[216];
  int flatEdgeLUT[36];
  int flatVertexLUT[56];
  int flatCounter = 0;
  int flatVertexCounter = 0;
  int edgeCounter = 0;
  for(int row = 0; row < 12; ++row){
    for(int col = 0; col < 27; ++col){
      if(row < 8){
        flatParentLUT[flatCounter] = this->parentLUT[row][col];
        if(col < 7){
          flatVertexLUT[flatVertexCounter] = this->vertexLUT[row][col];
          flatVertexCounter++;
        }
        flatChildLUT[flatCounter] = this->childLUT[row][col];
        flatCounter++;
      }
      if(col < 3){
        flatEdgeLUT[edgeCounter] = this->edgeLUT[row][col];
        ++edgeCounter;
      }
    }
  }
  CudaSafeCall(cudaMalloc((void**)&this->edgeLUTDevice, 36*sizeof(int)));
  CudaSafeCall(cudaMalloc((void**)&this->faceLUTDevice, 6*sizeof(int)));
  CudaSafeCall(cudaMalloc((void**)&this->parentLUTDevice, 216*sizeof(int)));
  CudaSafeCall(cudaMalloc((void**)&this->vertexLUTDevice, 56*sizeof(int)));
  CudaSafeCall(cudaMalloc((void**)&this->childLUTDevice, 216*sizeof(int)));
  CudaSafeCall(cudaMemcpy(this->edgeLUTDevice, flatEdgeLUT, 36*sizeof(int), cudaMemcpyHostToDevice));
  CudaSafeCall(cudaMemcpy(this->faceLUTDevice, this->faceLUT, 6*sizeof(int), cudaMemcpyHostToDevice));
  CudaSafeCall(cudaMemcpy(this->parentLUTDevice, flatParentLUT, 216*sizeof(int), cudaMemcpyHostToDevice));
  CudaSafeCall(cudaMemcpy(this->vertexLUTDevice, flatVertexLUT, 56*sizeof(int), cudaMemcpyHostToDevice));
  CudaSafeCall(cudaMemcpy(this->childLUTDevice, flatChildLUT, 216*sizeof(int), cudaMemcpyHostToDevice));

  printf("octree fillLUTs took %f seconds.\n", ((float) clock() - cudatimer)/CLOCKS_PER_SEC);
}

void Octree::fillNeighborhoods(){
  clock_t cudatimer;
  cudatimer = clock();

  //need to use highest number of nodes in a depth instead of totalNodes
  dim3 grid = {1,1,1};
  dim3 block = {27,1,1};
  int numNodesAtDepth;
  int depthStartingIndex;
  int atomicCounter = 0;
  int* atomicCounterDevice;
  int childDepthIndex;
  CudaSafeCall(cudaMalloc((void**)&atomicCounterDevice, sizeof(int)));
  for(int i = this->depth; i >= 0 ; --i){
    numNodesAtDepth = 1;
    depthStartingIndex = this->depthIndex[i];
    childDepthIndex = -1;
    if(i != this->depth){
      numNodesAtDepth = this->depthIndex[i + 1] - depthStartingIndex;
    }
    if(i != 0){
      childDepthIndex = this->depthIndex[i - 1];
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
    CudaSafeCall(cudaMemcpy(atomicCounterDevice, &atomicCounter, sizeof(int), cudaMemcpyHostToDevice));
    computeNeighboringNodes<<<grid, block>>>(this->finalNodeArrayDevice, numNodesAtDepth, depthStartingIndex, this->parentLUTDevice, this->childLUTDevice, atomicCounterDevice, childDepthIndex);
    CudaCheckError();
    CudaSafeCall(cudaMemcpy(&atomicCounter, atomicCounterDevice, sizeof(int), cudaMemcpyDeviceToHost));
    atomicCounter = 0;
  }

  CudaSafeCall(cudaFree(atomicCounterDevice));
  CudaSafeCall(cudaFree(this->childLUTDevice));
  CudaSafeCall(cudaFree(this->parentLUTDevice));

  printf("octree findAllNeighbors took %f seconds.\n", ((float) clock() - cudatimer)/CLOCKS_PER_SEC);
}
void Octree::computeVertexArray(){
  clock_t cudatimer;
  cudatimer = clock();

  int numNodesAtDepth = 0;
  dim3 grid = {1,1,1};
  dim3 block = {1,1,1};
  int* atomicCounter;
  int numVertices = 0;
  CudaSafeCall(cudaMalloc((void**)&atomicCounter, sizeof(int)));
  CudaSafeCall(cudaMemcpy(atomicCounter, &numVertices, sizeof(int), cudaMemcpyHostToDevice));
  Vertex** vertexArray2DDevice;
  CudaSafeCall(cudaMalloc((void**)&vertexArray2DDevice, (this->depth + 1)*sizeof(Vertex*)));
  Vertex** vertexArray2D = new Vertex*[this->depth + 1];

  this->vertexIndex = new int[this->depth + 1];

  this->totalVertices = 0;
  int prevCount = 0;
  int* ownerInidicesDevice;
  int* vertexPlacementDevice;
  int* compactedOwnerArrayDevice;
  int* compactedVertexPlacementDevice;
  for(int i = 0; i <= this->depth; ++i){
    //reset previously allocated resources
    grid.y = 1;
    block.x = 8;
    if(i == this->depth){
      numNodesAtDepth = 1;
    }
    else{
      numNodesAtDepth = this->depthIndex[i + 1] - this->depthIndex[i];
    }
    if(numNodesAtDepth < 65535) grid.x = (unsigned int) numNodesAtDepth;
    else{
      grid.x = 65535;
      while(grid.x*grid.y < numNodesAtDepth){
        ++grid.y;
      }
      while(grid.x*grid.y > numNodesAtDepth){
        --grid.x;
      }
      if(grid.x*grid.y < numNodesAtDepth){
        ++grid.x;
      }
    }
    int* ownerInidices = new int[numNodesAtDepth*8];
    for(int v = 0;v < numNodesAtDepth*8; ++v){
      ownerInidices[v] = -1;
    }
    CudaSafeCall(cudaMalloc((void**)&ownerInidicesDevice,numNodesAtDepth*8*sizeof(int)));
    CudaSafeCall(cudaMalloc((void**)&vertexPlacementDevice,numNodesAtDepth*8*sizeof(int)));
    CudaSafeCall(cudaMemcpy(ownerInidicesDevice, ownerInidices, numNodesAtDepth*8*sizeof(int), cudaMemcpyHostToDevice));
    CudaSafeCall(cudaMemcpy(vertexPlacementDevice, ownerInidices, numNodesAtDepth*8*sizeof(int), cudaMemcpyHostToDevice));
    delete[] ownerInidices;

    prevCount = numVertices;
    this->vertexIndex[i] = numVertices;

    findVertexOwners<<<grid, block>>>(this->finalNodeArrayDevice, numNodesAtDepth,
      this->depthIndex[i], this->vertexLUTDevice, atomicCounter, ownerInidicesDevice, vertexPlacementDevice);
    CudaCheckError();
    CudaSafeCall(cudaMemcpy(&numVertices, atomicCounter, sizeof(int), cudaMemcpyDeviceToHost));
    if(i == this->depth  && numVertices - prevCount != 8){
      std::cout<<"ERROR GENERATING VERTICES, vertices at depth 0 != 8 -> "<<numVertices - prevCount<<std::endl;
      exit(-1);
    }

    CudaSafeCall(cudaMalloc((void**)&vertexArray2D[i], (numVertices - prevCount)*sizeof(Vertex)));
    CudaSafeCall(cudaMalloc((void**)&compactedOwnerArrayDevice,(numVertices - prevCount)*sizeof(int)));
    CudaSafeCall(cudaMalloc((void**)&compactedVertexPlacementDevice,(numVertices - prevCount)*sizeof(int)));

    thrust::device_ptr<int> arrayToCompact(ownerInidicesDevice);
    thrust::device_ptr<int> arrayOut(compactedOwnerArrayDevice);
    thrust::device_ptr<int> placementToCompact(vertexPlacementDevice);
    thrust::device_ptr<int> placementOut(compactedVertexPlacementDevice);

    thrust::copy_if(arrayToCompact, arrayToCompact + (numNodesAtDepth*8), arrayOut, is_not_neg());
    CudaCheckError();
    thrust::copy_if(placementToCompact, placementToCompact + (numNodesAtDepth*8), placementOut, is_not_neg());
    CudaCheckError();

    CudaSafeCall(cudaFree(ownerInidicesDevice));
    CudaSafeCall(cudaFree(vertexPlacementDevice));

    //reset and allocated resources
    grid.y = 1;
    block.x = 1;
    if(numVertices - prevCount < 65535) grid.x = (unsigned int) numVertices - prevCount;
    else{
      grid.x = 65535;
      while(grid.x*block.x < numVertices - prevCount){
        ++block.x;
      }
      while(grid.x*block.x > numVertices - prevCount){
        --grid.x;
        if(grid.x*block.x < numVertices - prevCount){
          ++grid.x;
          break;
        }
      }
    }

    fillUniqueVertexArray<<<grid, block>>>(this->finalNodeArrayDevice, vertexArray2D[i],
      numVertices - prevCount, this->vertexIndex[i], this->depthIndex[i], this->depth - i,
      this->width, this->vertexLUTDevice, compactedOwnerArrayDevice, compactedVertexPlacementDevice);
    CudaCheckError();
    CudaSafeCall(cudaFree(compactedOwnerArrayDevice));
    CudaSafeCall(cudaFree(compactedVertexPlacementDevice));

  }
  this->totalVertices = numVertices;
  this->vertexArray = new Vertex[numVertices];
  for(int i = 0; i < numVertices; ++i) this->vertexArray[i] = Vertex();//might not be necessary
  this->copyVerticesToDevice();
  for(int i = 0; i <= this->depth; ++i){
    if(i < this->depth){
      CudaSafeCall(cudaMemcpy(this->vertexArrayDevice + this->vertexIndex[i], vertexArray2D[i], (this->vertexIndex[i+1] - this->vertexIndex[i])*sizeof(Vertex), cudaMemcpyDeviceToDevice));
    }
    else{
      CudaSafeCall(cudaMemcpy(this->vertexArrayDevice + this->vertexIndex[i], vertexArray2D[i], 8*sizeof(Vertex), cudaMemcpyDeviceToDevice));
    }
    CudaSafeCall(cudaFree(vertexArray2D[i]));
  }
  this->copyVerticesToHost();
  CudaSafeCall(cudaFree(this->vertexLUTDevice));
  CudaSafeCall(cudaFree(vertexArray2DDevice));

  printf("octree createVertexArray took %f seconds.\n", ((float) clock() - cudatimer)/CLOCKS_PER_SEC);
}
void Octree::computeEdgeArray(){
  clock_t cudatimer;
  cudatimer = clock();

  int numNodesAtDepth = 0;
  dim3 grid = {1,1,1};
  dim3 block = {1,1,1};
  int* atomicCounter;
  int numEdges = 0;
  CudaSafeCall(cudaMalloc((void**)&atomicCounter, sizeof(int)));
  CudaSafeCall(cudaMemcpy(atomicCounter, &numEdges, sizeof(int), cudaMemcpyHostToDevice));
  Edge** edgeArray2DDevice;
  CudaSafeCall(cudaMalloc((void**)&edgeArray2DDevice, (this->depth + 1)*sizeof(Edge*)));
  Edge** edgeArray2D = new Edge*[this->depth + 1];

  this->edgeIndex = new int[this->depth + 1];

  this->totalEdges = 0;
  int prevCount = 0;
  int* ownerInidicesDevice;
  int* edgePlacementDevice;
  int* compactedOwnerArrayDevice;
  int* compactedEdgePlacementDevice;
  for(int i = 0; i <= this->depth; ++i){
    //reset previously allocated resources
    grid.y = 1;
    block.x = 12;
    if(i == this->depth){
      numNodesAtDepth = 1;
    }
    else{
      numNodesAtDepth = this->depthIndex[i + 1] - this->depthIndex[i];
    }
    if(numNodesAtDepth < 65535) grid.x = (unsigned int) numNodesAtDepth;
    else{
      grid.x = 65535;
      while(grid.x*grid.y < numNodesAtDepth){
        ++grid.y;
      }
      while(grid.x*grid.y > numNodesAtDepth){
        --grid.x;

      }
      if(grid.x*grid.y < numNodesAtDepth){
        ++grid.x;
      }
    }
    int* ownerInidices = new int[numNodesAtDepth*12];
    for(int v = 0;v < numNodesAtDepth*12; ++v){
      ownerInidices[v] = -1;
    }
    CudaSafeCall(cudaMalloc((void**)&ownerInidicesDevice,numNodesAtDepth*12*sizeof(int)));
    CudaSafeCall(cudaMalloc((void**)&edgePlacementDevice,numNodesAtDepth*12*sizeof(int)));
    CudaSafeCall(cudaMemcpy(ownerInidicesDevice, ownerInidices, numNodesAtDepth*12*sizeof(int), cudaMemcpyHostToDevice));
    CudaSafeCall(cudaMemcpy(edgePlacementDevice, ownerInidices, numNodesAtDepth*12*sizeof(int), cudaMemcpyHostToDevice));
    delete[] ownerInidices;

    prevCount = numEdges;
    this->edgeIndex[i] = numEdges;
    findEdgeOwners<<<grid, block>>>(this->finalNodeArrayDevice, numNodesAtDepth,
      this->depthIndex[i], this->edgeLUTDevice, atomicCounter, ownerInidicesDevice, edgePlacementDevice);
    CudaCheckError();
    CudaSafeCall(cudaMemcpy(&numEdges, atomicCounter, sizeof(int), cudaMemcpyDeviceToHost));
    if(i == this->depth  && numEdges - prevCount != 12){
      std::cout<<"ERROR GENERATING EDGES, edges at depth 0 != 12 -> "<<numEdges - prevCount<<std::endl;
      exit(-1);
    }

    CudaSafeCall(cudaMalloc((void**)&edgeArray2D[i], (numEdges - prevCount)*sizeof(Edge)));
    CudaSafeCall(cudaMalloc((void**)&compactedOwnerArrayDevice,(numEdges - prevCount)*sizeof(int)));
    CudaSafeCall(cudaMalloc((void**)&compactedEdgePlacementDevice,(numEdges - prevCount)*sizeof(int)));

    thrust::device_ptr<int> arrayToCompact(ownerInidicesDevice);
    thrust::device_ptr<int> arrayOut(compactedOwnerArrayDevice);
    thrust::device_ptr<int> placementToCompact(edgePlacementDevice);
    thrust::device_ptr<int> placementOut(compactedEdgePlacementDevice);

    thrust::copy_if(arrayToCompact, arrayToCompact + (numNodesAtDepth*12), arrayOut, is_not_neg());
    CudaCheckError();
    thrust::copy_if(placementToCompact, placementToCompact + (numNodesAtDepth*12), placementOut, is_not_neg());
    CudaCheckError();

    CudaSafeCall(cudaFree(ownerInidicesDevice));
    CudaSafeCall(cudaFree(edgePlacementDevice));

    //reset and allocated resources
    grid.y = 1;
    block.x = 1;
    if(numEdges - prevCount < 65535) grid.x = (unsigned int) numEdges - prevCount;
    else{
      grid.x = 65535;
      while(grid.x*block.x < numEdges - prevCount){
        ++block.x;
      }
      while(grid.x*block.x > numEdges - prevCount){
        --grid.x;
        if(grid.x*block.x < numEdges - prevCount){
          ++grid.x;
          break;
        }
      }
    }

    fillUniqueEdgeArray<<<grid, block>>>(this->finalNodeArrayDevice, edgeArray2D[i],
      numEdges - prevCount, this->edgeIndex[i],this->depthIndex[i], this->depth - i,
      this->width, this->edgeLUTDevice, compactedOwnerArrayDevice, compactedEdgePlacementDevice);
    CudaCheckError();
    CudaSafeCall(cudaFree(compactedOwnerArrayDevice));
    CudaSafeCall(cudaFree(compactedEdgePlacementDevice));

  }
  this->totalEdges = numEdges;
  this->edgeArray = new Edge[numEdges];
  for(int i = 0; i < numEdges; ++i) this->edgeArray[i] = Edge();//might not be necessary
  this->copyEdgesToDevice();
  for(int i = 0; i <= this->depth; ++i){
    if(i < this->depth){
      CudaSafeCall(cudaMemcpy(this->edgeArrayDevice + this->edgeIndex[i], edgeArray2D[i], (this->edgeIndex[i+1] - this->edgeIndex[i])*sizeof(Edge), cudaMemcpyDeviceToDevice));
    }
    else{
      CudaSafeCall(cudaMemcpy(this->edgeArrayDevice + this->edgeIndex[i], edgeArray2D[i], 12*sizeof(Edge), cudaMemcpyDeviceToDevice));
    }
    CudaSafeCall(cudaFree(edgeArray2D[i]));
  }
  this->copyEdgesToHost();
  CudaSafeCall(cudaFree(this->edgeLUTDevice));
  CudaSafeCall(cudaFree(edgeArray2DDevice));
  printf("octree createEdgeArray took %f seconds.\n", ((float) clock() - cudatimer)/CLOCKS_PER_SEC);
}
void Octree::computeFaceArray(){
  clock_t cudatimer;
  cudatimer = clock();

  int numNodesAtDepth = 0;
  dim3 grid = {1,1,1};
  dim3 block = {1,1,1};
  int* atomicCounter;
  int numFaces = 0;
  CudaSafeCall(cudaMalloc((void**)&atomicCounter, sizeof(int)));
  CudaSafeCall(cudaMemcpy(atomicCounter, &numFaces, sizeof(int), cudaMemcpyHostToDevice));
  Face** faceArray2DDevice;
  CudaSafeCall(cudaMalloc((void**)&faceArray2DDevice, (this->depth + 1)*sizeof(Face*)));
  Face** faceArray2D = new Face*[this->depth + 1];

  this->faceIndex = new int[this->depth + 1];

  this->totalFaces = 0;
  int prevCount = 0;
  int* ownerInidicesDevice;
  int* facePlacementDevice;
  int* compactedOwnerArrayDevice;
  int* compactedFacePlacementDevice;
  for(int i = 0; i <= this->depth; ++i){
    //reset previously allocated resources
    grid.y = 1;
    block.x = 6;
    if(i == this->depth){
      numNodesAtDepth = 1;
    }
    else{
      numNodesAtDepth = this->depthIndex[i + 1] - this->depthIndex[i];
    }
    if(numNodesAtDepth < 65535) grid.x = (unsigned int) numNodesAtDepth;
    else{
      grid.x = 65535;
      while(grid.x*grid.y < numNodesAtDepth){
        ++grid.y;
      }
      while(grid.x*grid.y > numNodesAtDepth){
        --grid.x;

      }
      if(grid.x*grid.y < numNodesAtDepth){
        ++grid.x;
      }
    }
    int* ownerInidices = new int[numNodesAtDepth*6];
    for(int v = 0;v < numNodesAtDepth*6; ++v){
      ownerInidices[v] = -1;
    }
    CudaSafeCall(cudaMalloc((void**)&ownerInidicesDevice,numNodesAtDepth*6*sizeof(int)));
    CudaSafeCall(cudaMalloc((void**)&facePlacementDevice,numNodesAtDepth*6*sizeof(int)));
    CudaSafeCall(cudaMemcpy(ownerInidicesDevice, ownerInidices, numNodesAtDepth*6*sizeof(int), cudaMemcpyHostToDevice));
    CudaSafeCall(cudaMemcpy(facePlacementDevice, ownerInidices, numNodesAtDepth*6*sizeof(int), cudaMemcpyHostToDevice));
    delete[] ownerInidices;

    prevCount = numFaces;
    this->faceIndex[i] = numFaces;
    findFaceOwners<<<grid, block>>>(this->finalNodeArrayDevice, numNodesAtDepth,
      this->depthIndex[i], this->faceLUTDevice, atomicCounter, ownerInidicesDevice, facePlacementDevice);
    CudaCheckError();
    CudaSafeCall(cudaMemcpy(&numFaces, atomicCounter, sizeof(int), cudaMemcpyDeviceToHost));
    if(i == this->depth  && numFaces - prevCount != 6){
      std::cout<<"ERROR GENERATING FACES, faces at depth 0 != 6 -> "<<numFaces - prevCount<<std::endl;
      exit(-1);
    }

    CudaSafeCall(cudaMalloc((void**)&faceArray2D[i], (numFaces - prevCount)*sizeof(Face)));
    CudaSafeCall(cudaMalloc((void**)&compactedOwnerArrayDevice,(numFaces - prevCount)*sizeof(int)));
    CudaSafeCall(cudaMalloc((void**)&compactedFacePlacementDevice,(numFaces - prevCount)*sizeof(int)));

    thrust::device_ptr<int> arrayToCompact(ownerInidicesDevice);
    thrust::device_ptr<int> arrayOut(compactedOwnerArrayDevice);
    thrust::device_ptr<int> placementToCompact(facePlacementDevice);
    thrust::device_ptr<int> placementOut(compactedFacePlacementDevice);

    thrust::copy_if(arrayToCompact, arrayToCompact + (numNodesAtDepth*6), arrayOut, is_not_neg());
    CudaCheckError();
    thrust::copy_if(placementToCompact, placementToCompact + (numNodesAtDepth*6), placementOut, is_not_neg());
    CudaCheckError();

    CudaSafeCall(cudaFree(ownerInidicesDevice));
    CudaSafeCall(cudaFree(facePlacementDevice));

    //reset and allocated resources
    grid.y = 1;
    block.x = 1;
    if(numFaces - prevCount < 65535) grid.x = (unsigned int) numFaces - prevCount;
    else{
      grid.x = 65535;
      while(grid.x*block.x < numFaces - prevCount){
        ++block.x;
      }
      while(grid.x*block.x > numFaces - prevCount){
        --grid.x;
        if(grid.x*block.x < numFaces - prevCount){
          ++grid.x;
          break;
        }
      }
    }

    fillUniqueFaceArray<<<grid, block>>>(this->finalNodeArrayDevice, faceArray2D[i],
      numFaces - prevCount, numFaces,this->depthIndex[i], this->depth - i,
      this->width, this->faceLUTDevice, compactedOwnerArrayDevice, compactedFacePlacementDevice);
    CudaCheckError();
    CudaSafeCall(cudaFree(compactedOwnerArrayDevice));
    CudaSafeCall(cudaFree(compactedFacePlacementDevice));

  }
  this->totalFaces = numFaces;
  this->faceArray = new Face[numFaces];
  for(int i = 0; i < numFaces; ++i) this->faceArray[i] = Face();//might not be necessary
  this->copyFacesToDevice();
  for(int i = 0; i <= this->depth; ++i){
    if(i < this->depth){
      CudaSafeCall(cudaMemcpy(this->faceArrayDevice + this->faceIndex[i], faceArray2D[i], (this->faceIndex[i+1] - this->faceIndex[i])*sizeof(Face), cudaMemcpyDeviceToDevice));
    }
    else{
      CudaSafeCall(cudaMemcpy(this->faceArrayDevice + this->faceIndex[i], faceArray2D[i], 6*sizeof(Face), cudaMemcpyDeviceToDevice));
    }
    CudaSafeCall(cudaFree(faceArray2D[i]));
  }
  this->copyFacesToHost();
  CudaSafeCall(cudaFree(this->faceLUTDevice));
  CudaSafeCall(cudaFree(faceArray2DDevice));
  printf("octree createFaceArray took %f seconds.\n", ((float) clock() - cudatimer)/CLOCKS_PER_SEC);
}

void Octree::checkForGeneralNodeErrors(){
  clock_t cudatimer;
  cudatimer = clock();
  this->copyNodesToHost();
  float regionOfError = this->width/pow(2,depth + 1);
  bool error = false;
  int numFuckedNodes = 0;
  int orphanNodes = 0;
  int nodesWithOutChildren = 0;
  int nodesThatCantFindChildren = 0;
  int noPoints = 0;
  int numSiblingParents = 0;
  int numChildNeighbors = 0;
  bool parentNeighbor = false;
  bool childNeighbor = false;
  int numParentNeighbors = 0;
  int numVerticesMissing = 0;
  int numEgesMissing = 0;
  int numFacesMissing = 0;
  int numCentersOUTSIDE = 0;
  for(int i = 0; i < this->totalNodes; ++i){
    if(this->finalNodeArray[i].depth < 0){
      numFuckedNodes++;
    }
    if(this->finalNodeArray[i].parent != -1
      && this->finalNodeArray[i].depth == this->finalNodeArray[this->finalNodeArray[i].parent].depth){
      ++numSiblingParents;
    }
    if(this->finalNodeArray[i].parent == -1 && this->finalNodeArray[i].depth != 0){
      orphanNodes++;
    }
    int checkForChildren = 0;
    for(int c = 0; c < 8 && this->finalNodeArray[i].depth < 10; ++c){
      if(this->finalNodeArray[i].children[c] == -1){
        checkForChildren++;
      }
      if(this->finalNodeArray[i].children[c] == 0 &&
        this->finalNodeArray[i].depth != this->depth - 1){
        std::cout<<"NODE THAT IS NOT AT 2nd TO FINEST DEPTH HAS A CHILD WITH INDEX 0 IN FINEST DEPTH"<<std::endl;
      }
    }
    if(this->finalNodeArray[i].numPoints == 0){
      noPoints++;
    }
    if(this->finalNodeArray[i].depth != 0 &&
      this->finalNodeArray[this->finalNodeArray[i].parent].children[this->finalNodeArray[i].key&((1<<3)-1)] == -1){

      nodesThatCantFindChildren++;
    }
    if(checkForChildren == 8){
      nodesWithOutChildren++;
    }
    if(this->finalNodeArray[i].depth == 0){
      if(this->finalNodeArray[i].numFinestChildren < this->numFinestUniqueNodes){
        std::cout<<"DEPTH 0 DOES NOT INCLUDE ALL FINEST UNIQUE NODES "<<this->finalNodeArray[i].numFinestChildren<<",";
        std::cout<<this->numFinestUniqueNodes<<", NUM FULL FINEST NODES SHOULD BE "<<this->depthIndex[1]<<std::endl;
        exit(-1);
      }
      if(this->finalNodeArray[i].numPoints != this->numPoints){
        std::cout<<"DEPTH 0 DOES NOT CONTAIN ALL POINTS "<<this->finalNodeArray[i].numPoints<<","<<this->numPoints<<std::endl;
        exit(-1);
      }
    }
    childNeighbor = false;
    parentNeighbor = false;
    for(int n = 0; n < 27; ++n){
      if(this->finalNodeArray[i].neighbors[n] != -1){
        if(this->finalNodeArray[i].depth < this->finalNodeArray[this->finalNodeArray[i].neighbors[n]].depth){
          childNeighbor = true;
        }
        else if(this->finalNodeArray[i].depth > this->finalNodeArray[this->finalNodeArray[i].neighbors[n]].depth){
          parentNeighbor = true;
        }
      }
    }
    for(int v = 0; v < 8; ++v){
      if(this->finalNodeArray[i].vertices[v] == -1){
        ++numVerticesMissing;
      }
    }
    for(int e = 0; e < 12; ++e){
      if(this->finalNodeArray[i].edges[e] == -1){
        ++numEgesMissing;
      }
    }
    for(int f = 0; f < 6; ++f){
      if(this->finalNodeArray[i].faces[f] == -1){
        ++numFacesMissing;
      }
    }
    if(parentNeighbor){
      ++numParentNeighbors;
    }
    if(childNeighbor){
      ++numChildNeighbors;
    }
    if((this->finalNodeArray[i].center.x < this->min.x ||
    this->finalNodeArray[i].center.y < this->min.y ||
    this->finalNodeArray[i].center.z < this->min.z ||
    this->finalNodeArray[i].center.x > this->max.x ||
    this->finalNodeArray[i].center.y > this->max.y ||
    this->finalNodeArray[i].center.z > this->max.z )){
      ++numCentersOUTSIDE;
    }
  }
  if(numCentersOUTSIDE > 0){
    printf("ERROR %d centers outside of bounding box\n",numCentersOUTSIDE);
    error = true;
  }
  if(numSiblingParents > 0){
    std::cout<<"ERROR "<<numSiblingParents<<" NODES THINK THEIR PARENT IS IN THE SAME DEPTH AS THEMSELVES"<<std::endl;
    error = true;
  }
  if(numChildNeighbors > 0){
    std::cout<<"ERROR "<<numChildNeighbors<<" NODES WITH SIBLINGS AT HIGHER DEPTH"<<std::endl;
    error = true;
  }
  if(numParentNeighbors > 0){
    std::cout<<"ERROR "<<numParentNeighbors<<" NODES WITH SIBLINGS AT LOWER DEPTH"<<std::endl;
    error = true;
  }
  if(numFuckedNodes > 0){
    std::cout<<numFuckedNodes<<" ERROR IN NODE CONCATENATION OR GENERATION"<<std::endl;
    error = true;
  }
  if(orphanNodes > 0){
    std::cout<<orphanNodes<<" ERROR THERE ARE ORPHAN NODES"<<std::endl;
    error = true;
  }
  if(nodesThatCantFindChildren > 0){
    std::cout<<"ERROR "<<nodesThatCantFindChildren<<" PARENTS WITHOUT CHILDREN"<<std::endl;
    error = true;
  }
  if(numVerticesMissing > 0){
    std::cout<<"ERROR "<<numVerticesMissing<<" VERTICES MISSING"<<std::endl;
    error = true;
  }
  if(numEgesMissing > 0){
    std::cout<<"ERROR "<<numEgesMissing<<" EDGES MISSING"<<std::endl;
    error = true;
  }
  if(numFacesMissing > 0){
    std::cout<<"ERROR "<<numFacesMissing<<" FACES MISSING"<<std::endl;
    error = true;
  }
  if(error) exit(-1);
  else std::cout<<"NO ERRORS DETECTED IN OCTREE"<<std::endl;
  std::cout<<"NODES WITHOUT POINTS = "<<noPoints<<std::endl;
  std::cout<<"NODES WITH POINTS = "<<this->totalNodes - noPoints<<std::endl<<std::endl;

  printf("octree checkForErrors took %f seconds.\n\n", ((float) clock() - cudatimer)/CLOCKS_PER_SEC);
}

//TODO make camera parameter acquisition dynamic and not hard coded
//TODO possibly use maxPointsInOneNode > 1024 && minPointsInOneNode <= 1 as an outlier check
//TODO will need to remove neighborDistance as a parameter
void Octree::computeNormals(int minNeighForNorms, int maxNeighbors){
  std::cout<<"\n";
  clock_t cudatimer;
  cudatimer = clock();
  if(this->colorsDeviceReady) this->copyColorsToHost();
  if(!this->pointsDeviceReady) this->copyPointsToDevice();
  this->copyNodesToHost();

  int numNodesAtDepth = 0;
  int currentNumNeighbors = 0;
  int currentNeighborIndex = -1;
  int maxPointsInOneNode = 0;
  int minPossibleNeighbors = std::numeric_limits<int>::max();
  int depthIndex = 0;
  int currentDepth = 0;

  for(int i = 0; i < this->totalNodes; ++i){
    currentNumNeighbors = 0;
    if(minPossibleNeighbors < minNeighForNorms){
      ++currentDepth;
      i = this->depthIndex[currentDepth];
      minPossibleNeighbors = std::numeric_limits<int>::max();
      maxPointsInOneNode = 0;
    }
    if(this->depth - this->finalNodeArray[i].depth != currentDepth){
      if(minPossibleNeighbors >= minNeighForNorms) break;
      ++currentDepth;
    }
    if(maxPointsInOneNode < this->finalNodeArray[i].numPoints){
      maxPointsInOneNode = this->finalNodeArray[i].numPoints;
    }
    for(int n = 0; n < 27; ++n){
      currentNeighborIndex = this->finalNodeArray[i].neighbors[n];
      if(currentNeighborIndex != -1) currentNumNeighbors += this->finalNodeArray[currentNeighborIndex].numPoints;
    }
    if(minPossibleNeighbors > currentNumNeighbors){
      minPossibleNeighbors = currentNumNeighbors;
    }
  }
  depthIndex = this->depthIndex[currentDepth];
  numNodesAtDepth = this->depthIndex[currentDepth + 1] - depthIndex;
  std::cout<<"Continuing with depth "<<this->depth - currentDepth<<" nodes starting at "<<depthIndex<<" with "<<numNodesAtDepth<<" nodes"<<std::endl;
  std::cout<<"Continuing with "<<minPossibleNeighbors<<" minPossibleNeighbors"<<std::endl;
  std::cout<<"Continuing with "<<maxNeighbors<<" maxNeighborsAllowed"<<std::endl;
  std::cout<<"Continuing with "<<maxPointsInOneNode<<" maxPointsInOneNode"<<std::endl;

  /*north korean mountain range camera parameters*/
  float3 position1 = {10.0f,12.0f,999.0f};
  float3 position2 = {-12.0f,-15.0f,967.0f};
  float3 cameraPositions[2] = {position1, position2};
  unsigned int numCameras = 2;
  if(numCameras > 1024){
    std::cout<<"ERROR numCameras > 1024"<<std::endl;
    exit(-1);
  }

  uint size = this->numPoints*maxNeighbors*3;
  float* cMatrixDevice;
  int* neighborIndicesDevice;
  int* numRealNeighborsDevice;
  int* numRealNeighbors = new int[this->numPoints];

  for(int i = 0; i < this->numPoints; ++i){
    numRealNeighbors[i] = 0;
  }
  int* temp = new int[size/3];
  for(int i = 0; i < size/3; ++i){
    temp[i] = -1;
  }

  CudaSafeCall(cudaMalloc((void**)&numRealNeighborsDevice, this->numPoints*sizeof(int)));
  CudaSafeCall(cudaMalloc((void**)&cMatrixDevice, size*sizeof(float)));
  CudaSafeCall(cudaMalloc((void**)&neighborIndicesDevice, (size/3)*sizeof(int)));
  CudaSafeCall(cudaMemcpy(numRealNeighborsDevice, numRealNeighbors, this->numPoints*sizeof(int), cudaMemcpyHostToDevice));
  CudaSafeCall(cudaMemcpy(neighborIndicesDevice, temp, (size/3)*sizeof(int), cudaMemcpyHostToDevice));
  delete[] temp;

  dim3 grid = {1,1,1};
  dim3 block = {1,1,1};

  block.x = (maxPointsInOneNode > 1024) ? 1024 : maxPointsInOneNode;
  if(numNodesAtDepth < 65535) grid.x = (unsigned int) numNodesAtDepth;
  else{
    grid.x = 65535;
    while(grid.x*grid.y < numNodesAtDepth){
      ++grid.y;
    }
    while(grid.x*grid.y > numNodesAtDepth){
      --grid.x;
    }
    if(grid.x*grid.y < numNodesAtDepth){
      ++grid.x;
    }
  }
  findNormalNeighborsAndComputeCMatrix<<<grid, block>>>(numNodesAtDepth, depthIndex, maxNeighbors,
    this->finalNodeArrayDevice, this->pointsDevice, cMatrixDevice, neighborIndicesDevice, numRealNeighborsDevice);
  CudaCheckError();
  CudaSafeCall(cudaMemcpy(numRealNeighbors, numRealNeighborsDevice, this->numPoints*sizeof(int), cudaMemcpyDeviceToHost));
  this->copyNormalsToDevice();

  cusolverDnHandle_t cusolverH = NULL;
  cublasHandle_t cublasH = NULL;
  cublasStatus_t cublas_status = CUBLAS_STATUS_SUCCESS;
  cusolverStatus_t cusolver_status = CUSOLVER_STATUS_SUCCESS;

  float *d_A, *d_S, *d_U, *d_VT, *d_work, *d_rwork;
  int* devInfo;

  cusolver_status = cusolverDnCreate(&cusolverH);
  assert(CUSOLVER_STATUS_SUCCESS == cublas_status);

  cublas_status = cublasCreate(&cublasH);
  assert(CUBLAS_STATUS_SUCCESS == cublas_status);

  int n = 3;
  int m = 0;
  int lwork = 0;

  //TODO changed this to gesvdjBatched (this will enable doing multiple svds at once)
  for(int p = 0; p < this->numPoints; ++p){
    m = numRealNeighbors[p];
    lwork = 0;
    if(m < minNeighForNorms){
      std::cout<<"ERROR...point does not have enough neighbors...increase min neighbors"<<std::endl;
      exit(-1);
    }
    CudaSafeCall(cudaMalloc((void**)&d_A, m*n*sizeof(float)));
    CudaSafeCall(cudaMalloc((void**)&d_S, n*sizeof(float)));
    CudaSafeCall(cudaMalloc((void**)&d_U, m*m*sizeof(float)));
    CudaSafeCall(cudaMalloc((void**)&d_VT, n*n*sizeof(float)));
    CudaSafeCall(cudaMalloc((void**)&devInfo, sizeof(int)));
    CudaSafeCall(cudaMemcpy(d_A, cMatrixDevice + (p*maxNeighbors*n), m*n*sizeof(float), cudaMemcpyDeviceToDevice));
    transposeFloatMatrix<<<m*n,1>>>(m,n,d_A);
    cudaDeviceSynchronize();
    CudaCheckError();

    //query working space of SVD
    cusolver_status = cusolverDnSgesvd_bufferSize(cusolverH, m, n, &lwork);

    assert(cusolver_status == CUSOLVER_STATUS_SUCCESS);

    CudaSafeCall(cudaMalloc((void**)&d_work, lwork*sizeof(float)));
    //SVD

    cusolver_status = cusolverDnSgesvd(cusolverH, 'A', 'A', m, n,
      d_A, m, d_S, d_U, m, d_VT, n, d_work, lwork, d_rwork, devInfo);
    cudaDeviceSynchronize();
    assert(cusolver_status == CUSOLVER_STATUS_SUCCESS);

    //FIND 2 ROWS OF S WITH HEIGHEST VALUES
    //TAKE THOSE ROWS IN VT AND GET CROSS PRODUCT = NORMALS ESTIMATE
    //TODO maybe find better way to cache this and not use only one block
    setNormal<<<1, 1>>>(p, d_VT, this->normalsDevice);
    CudaCheckError();

    CudaSafeCall(cudaFree(d_A));
    CudaSafeCall(cudaFree(d_S));
    CudaSafeCall(cudaFree(d_U));
    CudaSafeCall(cudaFree(d_VT));
    CudaSafeCall(cudaFree(d_work));
    CudaSafeCall(cudaFree(devInfo));
  }
  std::cout<<"normals have been estimated by use of svd"<<std::endl;
  if (cublasH) cublasDestroy(cublasH);
  if (cusolverH) cusolverDnDestroy(cusolverH);

  delete[] numRealNeighbors;
  CudaSafeCall(cudaFree(cMatrixDevice));

  float3* cameraPositionsDevice;
  bool* ambiguityDevice;
  CudaSafeCall(cudaMalloc((void**)&cameraPositionsDevice, numCameras*sizeof(float3)));
  CudaSafeCall(cudaMalloc((void**)&ambiguityDevice, this->numPoints*sizeof(bool)));
  CudaSafeCall(cudaMemcpy(cameraPositionsDevice, cameraPositions, numCameras*sizeof(float3), cudaMemcpyHostToDevice));

  dim3 grid2 = {1,1,1};
  if(this->numPoints < 65535) grid2.x = (unsigned int) this->numPoints;
  else{
    grid2.x = 65535;
    while(grid2.x*grid2.y < this->numPoints){
      ++grid2.y;
    }
    while(grid2.x*grid2.y > this->numPoints){
      --grid2.x;
    }
    if(grid2.x*grid2.y < this->numPoints){
      ++grid2.x;
    }
  }
  dim3 block2 = {numCameras,1,1};

  checkForAbiguity<<<grid2, block2>>>(this->numPoints, numCameras, this->normalsDevice,
    this->pointsDevice, cameraPositionsDevice, ambiguityDevice);
  CudaCheckError();
  CudaSafeCall(cudaFree(cameraPositionsDevice));

  // bool* ambiguity = new bool[this->numPoints];
  // CudaSafeCall(cudaMemcpy(ambiguity, ambiguityDevice, this->numPoints*sizeof(bool), cudaMemcpyDeviceToHost));
  // int numAmbiguous = 0;
  // for(int i = 0; i < this->numPoints; ++i){
  //   if(ambiguity[i]) ++numAmbiguous;
  // }
  // std::cout<<"numAmbiguous = "<<numAmbiguous<<"/"<<this->numPoints<<std::endl;
  // delete[] ambiguity;

  reorient<<<grid, block>>>(numNodesAtDepth, depthIndex, this->finalNodeArrayDevice, numRealNeighborsDevice, maxNeighbors, this->normalsDevice,
    neighborIndicesDevice, ambiguityDevice);
  CudaCheckError();
  this->copyNormalsToHost();
  this->copyPointsToHost();//could be brought back to host before this method
  this->normalsComputed = true;

  CudaSafeCall(cudaFree(numRealNeighborsDevice));
  CudaSafeCall(cudaFree(neighborIndicesDevice));
  CudaSafeCall(cudaFree(ambiguityDevice));

  printf("octree computeNormals took %f seconds.\n\n", ((float) clock() - cudatimer)/CLOCKS_PER_SEC);
}

void Octree::writeVertexPLY(){
  std::string newFile = "out" + this->pathToFile.substr(4, this->pathToFile.length() - 4) + "_vertices_" + std::to_string(this->depth)+ ".ply";
  std::ofstream plystream(newFile);
  if (plystream.is_open()) {
    std::ostringstream stringBuffer = std::ostringstream("");
    stringBuffer << "ply\nformat ascii 1.0\ncomment object: SSRL test\n";
    stringBuffer << "element vertex ";
    stringBuffer <<  this->totalVertices;
    stringBuffer << "\nproperty float x\nproperty float y\nproperty float z\n";
    stringBuffer << "property uchar red\nproperty uchar green\nproperty uchar blue\n";
    stringBuffer << "end_header\n";
    plystream << stringBuffer.str();
    for(int i = 0; i < this->totalVertices; ++i){
      stringBuffer = std::ostringstream("");
      stringBuffer << this->vertexArray[i].coord.x;
      stringBuffer << " ";
      stringBuffer << this->vertexArray[i].coord.y;
      stringBuffer << " ";
      stringBuffer << this->vertexArray[i].coord.z;
      stringBuffer << " ";
      stringBuffer << (int) this->vertexArray[i].color.x;
      stringBuffer << " ";
      stringBuffer << (int) this->vertexArray[i].color.y;
      stringBuffer << " ";
      stringBuffer << (int) this->vertexArray[i].color.z;
      stringBuffer << "\n";
      plystream << stringBuffer.str();
    }
    std::cout<<newFile + " has been created.\n"<<std::endl;
  }
  else{
    std::cout << "Unable to open: " + newFile<< std::endl;
    exit(1);
  }
}
void Octree::writeEdgePLY(){
  std::string newFile = "out" + this->pathToFile.substr(4, this->pathToFile.length() - 4) + "_edges_" + std::to_string(this->depth)+ ".ply";
  std::ofstream plystream(newFile);
  if (plystream.is_open()) {
    std::ostringstream stringBuffer = std::ostringstream("");
    stringBuffer << "ply\nformat ascii 1.0\ncomment object: SSRL test\n";
    stringBuffer << "element vertex ";
    stringBuffer <<  this->totalVertices;
    stringBuffer << "\nproperty float x\nproperty float y\nproperty float z\n";
    stringBuffer << "property uchar red\nproperty uchar green\nproperty uchar blue\n";
    stringBuffer << "element edge ";
    stringBuffer <<  this->totalEdges;
    stringBuffer << "\nproperty int vertex1\nproperty int vertex2\n";
    stringBuffer << "property uchar red\nproperty uchar green\nproperty uchar blue\n";
    stringBuffer << "end_header\n";
    plystream << stringBuffer.str();
    for(int i = 0; i < this->totalVertices; ++i){
      stringBuffer = std::ostringstream("");
      stringBuffer << this->vertexArray[i].coord.x;
      stringBuffer << " ";
      stringBuffer << this->vertexArray[i].coord.y;
      stringBuffer << " ";
      stringBuffer << this->vertexArray[i].coord.z;
      stringBuffer << " ";
      stringBuffer << (int) this->vertexArray[i].color.x;
      stringBuffer << " ";
      stringBuffer << (int) this->vertexArray[i].color.y;
      stringBuffer << " ";
      stringBuffer << (int) this->vertexArray[i].color.z;
      stringBuffer << "\n";
      plystream << stringBuffer.str();
    }
    for(int i = 0; i < this->totalEdges; ++i){
      stringBuffer = std::ostringstream("");
      stringBuffer << this->edgeArray[i].v1;
      stringBuffer << " ";
      stringBuffer << this->edgeArray[i].v2;
      stringBuffer << " ";
      stringBuffer << (int) this->edgeArray[i].color.x;
      stringBuffer << " ";
      stringBuffer << (int) this->edgeArray[i].color.y;
      stringBuffer << " ";
      stringBuffer << (int) this->edgeArray[i].color.z;
      stringBuffer << "\n";
      plystream << stringBuffer.str();
    }
    std::cout<<newFile + " has been created.\n"<<std::endl;
  }
  else{
    std::cout << "Unable to open: " + newFile<< std::endl;
    exit(1);
  }
}
void Octree::writeCenterPLY(){
  std::string newFile = "out" + this->pathToFile.substr(4, this->pathToFile.length() - 4) + "_centers_" + std::to_string(this->depth)+ ".ply";
  std::ofstream plystream(newFile);
  if (plystream.is_open()) {
    std::ostringstream stringBuffer = std::ostringstream("");
    stringBuffer << "ply\nformat ascii 1.0\ncomment object: SSRL test\n";
    stringBuffer << "element vertex ";
    stringBuffer <<  this->totalNodes;
    stringBuffer << "\nproperty float x\nproperty float y\nproperty float z\n";
    stringBuffer << "property uchar red\nproperty uchar green\nproperty uchar blue\n";
    stringBuffer << "end_header\n";
    plystream << stringBuffer.str();
    for(int i = 0; i < this->totalNodes; ++i){
      stringBuffer = std::ostringstream("");
      stringBuffer << this->finalNodeArray[i].center.x;
      stringBuffer << " ";
      stringBuffer << this->finalNodeArray[i].center.y;
      stringBuffer << " ";
      stringBuffer << this->finalNodeArray[i].center.z;
      stringBuffer << " ";
      stringBuffer << (int) this->finalNodeArray[i].color.x;
      stringBuffer << " ";
      stringBuffer << (int) this->finalNodeArray[i].color.y;
      stringBuffer << " ";
      stringBuffer << (int) this->finalNodeArray[i].color.z;
      stringBuffer << "\n";
      plystream << stringBuffer.str();
    }
    std::cout<<newFile + " has been created.\n"<<std::endl;
  }
  else{
    std::cout << "Unable to open: " + newFile<< std::endl;
    exit(1);
  }
}
void Octree::writeNormalPLY(){
  std::string newFile = "out" + this->pathToFile.substr(4, this->pathToFile.length() - 4) + "_normals_" + std::to_string(this->depth)+ ".ply";
	std::ofstream plystream(newFile);
	if (plystream.is_open()) {
    std::ostringstream stringBuffer = std::ostringstream("");
    stringBuffer << "ply\nformat ascii 1.0\ncomment object: SSRL test\n";
    stringBuffer << "element vertex ";
    stringBuffer << this->numPoints;
    stringBuffer << "\nproperty float x\nproperty float y\nproperty float z\n";
    stringBuffer << "property float nx\nproperty float ny\nproperty float nz\n";
    stringBuffer << "end_header\n";
    plystream << stringBuffer.str();
    for(int i = 0; i < this->numPoints; ++i){
      stringBuffer = std::ostringstream("");
      stringBuffer << this->points[i].x;
      stringBuffer << " ";
      stringBuffer << this->points[i].y;
      stringBuffer << " ";
      stringBuffer << this->points[i].z;
      stringBuffer << " ";
      stringBuffer << this->normals[i].x;
      stringBuffer << " ";
      stringBuffer << this->normals[i].y;
      stringBuffer << " ";
      stringBuffer << this->normals[i].z;
      stringBuffer << "\n";
      plystream << stringBuffer.str();
    }
    std::cout<<newFile + " has been created.\n"<<std::endl;
	}
	else{
    std::cout << "Unable to open: " + newFile<< std::endl;
    exit(1);
  }
}
void Octree::writeDepthPLY(int d){
  if(d < 0 || d > this->depth){
    std::cout<<"ERROR DEPTH FOR WRITEDEPTHPLY IS OUT OF BOUNDS"<<std::endl;
    exit(-1);
  }
  std::string newFile = "out" + this->pathToFile.substr(4, this->pathToFile.length() - 4) +
  "_finestNodes_" + std::to_string(d) + "_"+ std::to_string(this->depth)+ ".ply";
  std::ofstream plystream(newFile);
  if (plystream.is_open()) {
    int verticesToWrite = (depth != 0) ? this->vertexIndex[this->depth - d + 1] : this->totalVertices;
    int facesToWrite = (depth != 0) ? this->faceIndex[this->depth - d + 1] - this->faceIndex[this->depth - d] : 6;
    int faceStartingIndex = this->faceIndex[this->depth - d];
    std::ostringstream stringBuffer = std::ostringstream("");
    stringBuffer << "ply\nformat ascii 1.0\ncomment object: SSRL test\n";
    stringBuffer << "element vertex ";
    stringBuffer << verticesToWrite;
    stringBuffer << "\nproperty float x\nproperty float y\nproperty float z\n";
    stringBuffer << "property uchar red\nproperty uchar green\nproperty uchar blue\n";
    stringBuffer << "element face ";
    stringBuffer << facesToWrite;
    stringBuffer << "\nproperty list uchar int vertex_index\n";
    stringBuffer << "end_header\n";
    plystream << stringBuffer.str();
    for(int i = 0; i < verticesToWrite; ++i){
      stringBuffer = std::ostringstream("");
      stringBuffer << this->vertexArray[i].coord.x;
      stringBuffer << " ";
      stringBuffer << this->vertexArray[i].coord.y;
      stringBuffer << " ";
      stringBuffer << this->vertexArray[i].coord.z;
      stringBuffer << " ";
      stringBuffer << 50;
      stringBuffer << " ";
      stringBuffer << 50;
      stringBuffer << " ";
      stringBuffer << 50;
      stringBuffer << "\n";
      plystream << stringBuffer.str();
    }
    for(int i = faceStartingIndex; i < facesToWrite + faceStartingIndex; ++i){
      stringBuffer = std::ostringstream("");
      stringBuffer << "4 ";
      stringBuffer << this->edgeArray[this->faceArray[i].e1].v1;
      stringBuffer << " ";
      stringBuffer << this->edgeArray[this->faceArray[i].e1].v2;
      stringBuffer << " ";
      stringBuffer << this->edgeArray[this->faceArray[i].e4].v2;
      stringBuffer << " ";
      stringBuffer << this->edgeArray[this->faceArray[i].e4].v1;
      stringBuffer << "\n";
      plystream << stringBuffer.str();
    }
    std::cout<<newFile + " has been created.\n"<<std::endl;
  }
  else{
    std::cout << "Unable to open: " + newFile<< std::endl;
    exit(1);
  }
}
