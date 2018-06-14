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

//TODO use these, recently added


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


/*
HELPER METHODS AND CUDA KERNELS
*/
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
  int tx = threadIdx.x;
  int bx = blockIdx.x;
  int globalID = bx * blockDim.x + tx;
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
void calculateNodeAddresses(dim3 grid, dim3 block, int numUniqueNodes, Node* uniqueNodesDevice, int* nodeAddressesDevice, int* nodeNumbersDevice){
  findAllNodes<<<grid,block>>>(numUniqueNodes, nodeNumbersDevice, uniqueNodesDevice);
  CudaCheckError();
  cudaDeviceSynchronize();
  thrust::device_ptr<int> nN(nodeNumbersDevice);
  thrust::device_ptr<int> nA(nodeAddressesDevice);
  thrust::inclusive_scan(nN, nN + numUniqueNodes, nA);

}
//TODO maybe optimize this center determination in fill Blank node array
__global__ void fillBlankNodeArray(Node* uniqueNodes, int* nodeNumbers, int* nodeAddresses, Node* outputNodeArray, int numUniqueNodes, int currentDepth, float totalWidth){
  int tx = threadIdx.x;
  int bx = blockIdx.x;
  int globalID = bx * blockDim.x + tx;
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
  int tx = threadIdx.x;
  int bx = blockIdx.x;
  int globalID = bx * blockDim.x + tx;
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
  int tx = threadIdx.x;
  int bx = blockIdx.x;
  int globalID = bx * blockDim.x + tx;
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
  int tx = threadIdx.x;
  int bx = blockIdx.x;
  int globalID = bx * blockDim.x + tx;
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
    // center.x = nodeArrayD[nodeArrayIndex + firstUniqueChild].center.x - (widthOfNode*0.5*coordPlacementIdentity[firstUniqueChild].x);
    // center.y = nodeArrayD[nodeArrayIndex + firstUniqueChild].center.y - (widthOfNode*0.5*coordPlacementIdentity[firstUniqueChild].y);
    // center.z = nodeArrayD[nodeArrayIndex + firstUniqueChild].center.z - (widthOfNode*0.5*coordPlacementIdentity[firstUniqueChild].z);
    uniqueNodes[globalID].center = center;

    for(int i = 0; i < 8; ++i){
      if(childIsUnique[i]){
        uniqueNodes[globalID].numPoints += nodeArrayD[nodeArrayIndex + i].numPoints;
        uniqueNodes[globalID].numFinestChildren += nodeArrayD[nodeArrayIndex + i].numFinestChildren;
      }
      else{
        // nodeArrayD[nodeArrayIndex + i].center.x = center.x + (widthOfNode*0.5*coordPlacementIdentity[i].x);
        // nodeArrayD[nodeArrayIndex + i].center.y = center.y + (widthOfNode*0.5*coordPlacementIdentity[i].y);
        // nodeArrayD[nodeArrayIndex + i].center.z = center.z + (widthOfNode*0.5*coordPlacementIdentity[i].z);
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

//TODO ensure that ordering of vertices is correct
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
  int tx = threadIdx.x;
  int bx = blockIdx.x;
  int globalID = bx * blockDim.x + tx;
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

//TODO ensure that ordering of edges is correct
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
  int tx = threadIdx.x;
  int bx = blockIdx.x;
  int globalID = bx * blockDim.x + tx;
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
      if(neighborPlacement < 9){
        --placement;
      }
      else if(neighborPlacement > 17){
        ++placement;
      }
      nodeArray[neighborSharingEdge].edges[placement] = globalID + edgeIndex;
    }
    edgeArray[globalID] = edge;
  }
}

//TODO ensure that ordering of faces is correct
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
  int tx = threadIdx.x;
  int bx = blockIdx.x;
  int globalID = bx * blockDim.x + tx;
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


/*
OCTREE CLASS FUNCTIONS
*/
Octree::Octree(){
  this->simpleOctree = true;
}

Octree::~Octree(){
  //TODO add deletes and cudaFrees here
}

/*
TODO remove normal reading from this and replace it with colors
normals will be determined after the octree has be built

TODO also optimize memory usage with colors added, illegal memory access occurs when not freeing enough
*/
void Octree::parsePLY(){
  cout<<this->pathToFile + " is being used as the ply"<<endl;
	ifstream plystream(this->pathToFile);
	string currentLine;
  float minX = FLT_MAX, minY = FLT_MAX, minZ = FLT_MAX, maxX = -FLT_MAX, maxY = -FLT_MAX, maxZ = -FLT_MAX;
  string temp = "";
  bool headerIsDone = false;
  int currentPoint = 0;
  this->hasColor = false;
  this->normalsComputed = false;
	if (plystream.is_open()) {
		while (getline(plystream, currentLine)) {
      istringstream stringBuffer = istringstream(currentLine);
      if(!headerIsDone){
        if(currentLine.find("element vertex") != string::npos){
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
        else if(currentLine.find("nx") != string::npos){
          this->normalsComputed = true;
        }
        else if(currentLine.find("blue") != string::npos){
          this->hasColor = true;
        }
        else if(currentLine.find("end_header") != string::npos){
          headerIsDone = true;
        }
        continue;
      }
      else if(currentPoint >= this->numPoints) break;

      float value = 0.0;
      int index = 0;
      float3 point;
      float3 normal;
      uchar3 color;
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
            ++index;
            break;
          case 3:
            if(this->normalsComputed) normal.x = value;
            else if(this->hasColor) color.x = value;
            ++index;
            break;
          case 4:
            if(this->normalsComputed) normal.y = value;
            else if(this->hasColor) color.y = value;
            ++index;
            break;
          case 5:
            if(this->normalsComputed) normal.z = value;
            else if(this->hasColor) color.z = value;
            ++index;
            break;
          case 6:
            if(this->normalsComputed && this->hasColor){
              color.x = value;
            }
            else{
              lineIsDone = true;
              this->points[currentPoint] = point;
              this->normals[currentPoint] = normal;
              this->colors[currentPoint] = color;
              this->finestNodePointIndexes[currentPoint] = currentPoint;
              this->finestNodeCenters[currentPoint] = {0.0f,0.0f,0.0f};
              this->finestNodeKeys[currentPoint] = 0;
              this->pointNodeIndex[currentPoint] = -1;
            }
            ++index;
            break;
          case 7:
            if(this->normalsComputed && this->hasColor){
               color.y = value;
               ++index;
            }
            else{
              lineIsDone = true;
              this->points[currentPoint] = point;
              this->normals[currentPoint] = normal;
              this->colors[currentPoint] = color;
              this->finestNodePointIndexes[currentPoint] = currentPoint;
              this->finestNodeCenters[currentPoint] = {0.0f,0.0f,0.0f};
              this->finestNodeKeys[currentPoint] = 0;
              this->pointNodeIndex[currentPoint] = -1;
            }
            break;
          case 8:
            if(this->normalsComputed && this->hasColor){
              color.z = value;
              ++index;
            }
            else{
              lineIsDone = true;
              this->points[currentPoint] = point;
              this->normals[currentPoint] = normal;
              this->colors[currentPoint] = color;
              this->finestNodePointIndexes[currentPoint] = currentPoint;
              this->finestNodeCenters[currentPoint] = {0.0f,0.0f,0.0f};
              this->finestNodeKeys[currentPoint] = 0;
              this->pointNodeIndex[currentPoint] = -1;
            }
            break;
          default:
            lineIsDone = true;
            this->points[currentPoint] = point;
            this->normals[currentPoint] = normal;
            this->colors[currentPoint] = color;
            this->finestNodePointIndexes[currentPoint] = currentPoint;
            this->finestNodeCenters[currentPoint] = {0.0f,0.0f,0.0f};
            this->finestNodeKeys[currentPoint] = 0;
            this->pointNodeIndex[currentPoint] = -1;
            break;
        }
        if(lineIsDone) break;
      }
      ++currentPoint;
		}

    this->min = {minX,minY,minZ};
    this->max = {maxX,maxY,maxZ};

    this->center.x = (maxX + minX)/2.0f;
    this->center.y = (maxY + minY)/2.0f;
    this->center.z = (maxZ + minZ)/2.0f;

    this->width = maxX - minX;
    if(this->width < maxY - minY) this->width = maxY - minY;
    if(this->width < maxZ - minZ) this->width = maxZ - minZ;

    this->totalNodes = 0;
    this->numFinestUniqueNodes = 0;

    printf("\nmin = %f,%f,%f\n",this->min.x,this->min.y,this->min.z);
    printf("max = %f,%f,%f\n",this->max.x,this->max.y,this->max.z);
    printf("bounding box width = %f\n", this->width);
    printf("center = %f,%f,%f\n",this->center.x,this->center.y,this->center.z);
    printf("number of points = %d\n\n", this->numPoints);
    cout<<this->pathToFile + "'s data has been transfered to an initialized octree.\n"<<endl;
	}
	else{
    cout << "Unable to open: " + this->pathToFile<< endl;
    exit(1);
  }
}

Octree::Octree(string pathToFile, int depth){
  this->pathToFile = pathToFile;
  this->parsePLY();
  this->depth = depth;
  this->simpleOctree = false;
}

void Octree::writeFinestPLY(){
  this->copyFinestNodeCentersToHost();
  string newFile = this->pathToFile.substr(0, this->pathToFile.length() - 4) + "_finest.ply";
  ofstream plystream(newFile);
  if (plystream.is_open()) {
    ostringstream stringBuffer = ostringstream("");
    stringBuffer << "ply\nformat ascii 1.0\ncomment object: SSRL test\n";
    stringBuffer << "element vertex ";
    stringBuffer <<  this->numPoints;
    stringBuffer << "\nproperty float x\nproperty float y\nproperty float z\n";
    stringBuffer << "end_header\n";
    plystream << stringBuffer.str();
    for(int i = 0; i < this->numPoints; ++i){
      stringBuffer = ostringstream("");
      stringBuffer << this->finestNodeCenters[i].x;
      stringBuffer << " ";
      stringBuffer << this->finestNodeCenters[i].y;
      stringBuffer << " ";
      stringBuffer << this->finestNodeCenters[i].z;
      stringBuffer << "\n";
      plystream << stringBuffer.str();
    }
    cout<<newFile + " has been created.\n"<<endl;
  }
  else{
    cout << "Unable to open: " + newFile<< endl;
    exit(1);
  }
}
void Octree::writeVertexPLY(){
  string newFile = this->pathToFile.substr(0, this->pathToFile.length() - 4) + "_vertices.ply";
  ofstream plystream(newFile);
  if (plystream.is_open()) {
    ostringstream stringBuffer = ostringstream("");
    stringBuffer << "ply\nformat ascii 1.0\ncomment object: SSRL test\n";
    stringBuffer << "element vertex ";
    stringBuffer <<  this->totalVertices;
    stringBuffer << "\nproperty float x\nproperty float y\nproperty float z\n";
    stringBuffer << "property uchar red\nproperty uchar green\nproperty uchar blue\n";
    stringBuffer << "end_header\n";
    plystream << stringBuffer.str();
    for(int i = 0; i < this->totalVertices; ++i){
      stringBuffer = ostringstream("");
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
    cout<<newFile + " has been created.\n"<<endl;
  }
  else{
    cout << "Unable to open: " + newFile<< endl;
    exit(1);
  }
}
void Octree::writeEdgePLY(){
  string newFile = this->pathToFile.substr(0, this->pathToFile.length() - 4) + "_edges.ply";
  ofstream plystream(newFile);
  if (plystream.is_open()) {
    ostringstream stringBuffer = ostringstream("");
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
      stringBuffer = ostringstream("");
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
      stringBuffer = ostringstream("");
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
    cout<<newFile + " has been created.\n"<<endl;
  }
  else{
    cout << "Unable to open: " + newFile<< endl;
    exit(1);
  }
}
void Octree::writeCenterPLY(){
  string newFile = this->pathToFile.substr(0, this->pathToFile.length() - 4) + "_centers.ply";
  ofstream plystream(newFile);
  if (plystream.is_open()) {
    ostringstream stringBuffer = ostringstream("");
    stringBuffer << "ply\nformat ascii 1.0\ncomment object: SSRL test\n";
    stringBuffer << "element vertex ";
    stringBuffer <<  this->totalNodes;
    stringBuffer << "\nproperty float x\nproperty float y\nproperty float z\n";
    stringBuffer << "property uchar red\nproperty uchar green\nproperty uchar blue\n";
    stringBuffer << "end_header\n";
    plystream << stringBuffer.str();
    for(int i = 0; i < this->totalNodes; ++i){
      stringBuffer = ostringstream("");
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
    cout<<newFile + " has been created.\n"<<endl;
  }
  else{
    cout << "Unable to open: " + newFile<< endl;
    exit(1);
  }
}
void Octree::writeNormalPLY(){
  string newFile = this->pathToFile.substr(0, this->pathToFile.length() - 4) + "_normals.ply";
	ofstream plystream(newFile);
	if (plystream.is_open()) {
    ostringstream stringBuffer = ostringstream("");
    stringBuffer << "ply\nformat ascii 1.0\ncomment object: SSRL test\n";
    stringBuffer << "element vertex ";
    stringBuffer << this->numPoints;
    stringBuffer << "\nproperty float x\nproperty float y\nproperty float z\n";
    stringBuffer << "property float nx\nproperty float ny\nproperty float nz\n";
    stringBuffer << "end_header\n";
    plystream << stringBuffer.str();
    for(int i = 0; i < this->numPoints; ++i){
      stringBuffer = ostringstream("");
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
    cout<<newFile + " has been created.\n"<<endl;
	}
	else{
    cout << "Unable to open: " + newFile<< endl;
    exit(1);
  }
}

void Octree::init_octree_gpu(){
  clock_t cudatimer;
  cudatimer = clock();
  this->pointNodeDeviceReady = false;
  this->vertexArrayDeviceReady = false;
  this->edgeArrayDeviceReady = false;
  this->faceArrayDeviceReady = false;
  this->pointsDeviceReady = false;
  this->normalsDeviceReady = false;
  this->colorsDeviceReady = false;

  CudaSafeCall(cudaMalloc((void**)&this->finestNodeCentersDevice, this->numPoints * sizeof(float3)));
  CudaSafeCall(cudaMalloc((void**)&this->finestNodeKeysDevice, this->numPoints * sizeof(int)));
  CudaSafeCall(cudaMalloc((void**)&this->finestNodePointIndexesDevice, this->numPoints * sizeof(int)));

  this->copyPointsToDevice();
  this->copyFinestNodeCentersToDevice();
  this->copyFinestNodeKeysToDevice();
  this->copyNormalsToDevice();
  this->copyColorsToDevice();

  cudatimer = clock() - cudatimer;
  printf("initial allocation & base variable copy took %f seconds.\n\n",((float) cudatimer)/CLOCKS_PER_SEC);
}

/*TODO
determine if you want to make memory methods include frees and cudaMalloc
if you do not then you need to decide which variables are most important to keep
on the device
*/

void Octree::copyPointsToDevice(){
  this->pointsDeviceReady = true;
  CudaSafeCall(cudaMalloc((void**)&this->pointsDevice, this->numPoints * sizeof(float3)));
  CudaSafeCall(cudaMemcpy(this->pointsDevice, this->points, this->numPoints * sizeof(float3), cudaMemcpyHostToDevice));
}
void Octree::copyPointsToHost(){
  this->pointsDeviceReady = false;
  CudaSafeCall(cudaMemcpy(this->points, this->pointsDevice, this->numPoints * sizeof(float3), cudaMemcpyDeviceToHost));
  CudaSafeCall(cudaFree(this->pointsDevice));
}
void Octree::copyNormalsToDevice(){
  this->normalsDeviceReady = true;
  CudaSafeCall(cudaMalloc((void**)&this->normalsDevice, this->numPoints * sizeof(float3)));
  CudaSafeCall(cudaMemcpy(this->normalsDevice, this->normals, this->numPoints * sizeof(float3), cudaMemcpyHostToDevice));
}
void Octree::copyNormalsToHost(){
  this->normalsDeviceReady = false;
  CudaSafeCall(cudaMemcpy(this->normals, this->normalsDevice, this->numPoints * sizeof(float3), cudaMemcpyDeviceToHost));
  CudaSafeCall(cudaFree(this->normalsDevice));
}
void Octree::copyColorsToDevice(){
  this->colorsDeviceReady = true;
  CudaSafeCall(cudaMalloc((void**)&this->colorsDevice, this->numPoints * sizeof(uchar3)));
  CudaSafeCall(cudaMemcpy(this->colorsDevice, this->colors, this->numPoints * sizeof(uchar3), cudaMemcpyHostToDevice));
}
void Octree::copyColorsToHost(){
  this->colorsDeviceReady = false;
  CudaSafeCall(cudaMemcpy(this->colors, this->colorsDevice, this->numPoints * sizeof(uchar3), cudaMemcpyDeviceToHost));
  CudaSafeCall(cudaFree(this->colorsDevice));
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
  delete[] this->finestNodeCenters;
  delete[] this->finestNodePointIndexes;
  delete[] this->finestNodeKeys;
  delete[] this->uniqueNodesAtFinestLevel;
  CudaSafeCall(cudaFree(this->finestNodeCentersDevice));
  CudaSafeCall(cudaFree(this->finestNodePointIndexesDevice));
  CudaSafeCall(cudaFree(this->finestNodeKeysDevice));
  cout<<"PREREQUISITE/FINEST DEPTH ARRAYS HAVE BEEN DELETED"<<endl;
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
  this->pointNodeDeviceReady = false;

  CudaSafeCall(cudaMemcpy(this->pointNodeIndex, this->pointNodeIndexDevice, this->numPoints * sizeof(int), cudaMemcpyDeviceToHost));
  CudaSafeCall(cudaFree(this->pointNodeIndexDevice));
}
void Octree::copyVerticesToDevice(){
  this->vertexArrayDeviceReady = true;

  CudaSafeCall(cudaMalloc((void**)&this->vertexArrayDevice, this->totalVertices*sizeof(Vertex)));
  CudaSafeCall(cudaMemcpy(this->vertexArrayDevice, this->vertexArray, this->totalVertices * sizeof(Vertex), cudaMemcpyHostToDevice));
}
void Octree::copyVerticesToHost(){
  this->vertexArrayDeviceReady = false;

  CudaSafeCall(cudaMemcpy(this->vertexArray, this->vertexArrayDevice, this->totalVertices * sizeof(Vertex), cudaMemcpyDeviceToHost));
  CudaSafeCall(cudaFree(this->vertexArrayDevice));
}
void Octree::copyEdgesToDevice(){
  this->edgeArrayDeviceReady = true;

  CudaSafeCall(cudaMalloc((void**)&this->edgeArrayDevice, this->totalEdges*sizeof(Edge)));
  CudaSafeCall(cudaMemcpy(this->edgeArrayDevice, this->edgeArray, this->totalEdges * sizeof(Edge), cudaMemcpyHostToDevice));
}
void Octree::copyEdgesToHost(){
  this->edgeArrayDeviceReady = false;

  CudaSafeCall(cudaMemcpy(this->edgeArray, this->edgeArrayDevice, this->totalEdges * sizeof(Edge), cudaMemcpyDeviceToHost));
  CudaSafeCall(cudaFree(this->edgeArrayDevice));
}
void Octree::copyFacesToDevice(){
  this->faceArrayDeviceReady = true;

  CudaSafeCall(cudaMalloc((void**)&this->faceArrayDevice, this->totalFaces*sizeof(Face)));
  CudaSafeCall(cudaMemcpy(this->faceArrayDevice, this->faceArray, this->totalFaces * sizeof(Face), cudaMemcpyHostToDevice));
}
void Octree::copyFacesToHost(){
  this->faceArrayDeviceReady = false;

  CudaSafeCall(cudaMemcpy(this->faceArray, this->faceArrayDevice, this->totalFaces * sizeof(Face), cudaMemcpyDeviceToHost));
  CudaSafeCall(cudaFree(this->faceArrayDevice));
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
  cudatimer = clock() - cudatimer;
  printf("getnodeKeys kernel took %f seconds.\n\n",((float) cudatimer)/CLOCKS_PER_SEC);
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

  CudaSafeCall(cudaFree(this->pointsDevice));
  this->pointsDeviceReady = false;
  CudaSafeCall(cudaFree(this->normalsDevice));
  this->normalsDeviceReady = false;
  CudaSafeCall(cudaFree(this->colorsDevice));
  this->colorsDeviceReady = false;


  //TODO OPTIMIZE THIS PORTION
  //COMPACT DATA

  thrust::pair<int*, int*> new_end;//the last value of these node arrays

  new_end = thrust::unique_by_key(thrust::host, this->finestNodeKeys, this->finestNodeKeys + this->numPoints, this->finestNodePointIndexes);

  bool foundFirst = false;
  int numUniqueNodes = 0;
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
  cudatimer = clock() - cudatimer;
  printf("octree prepareFinestUniquNodes took %f seconds.\n\n", ((float) cudatimer)/CLOCKS_PER_SEC);
}

void Octree::createFinalNodeArray(){

  Node* uniqueNodesDevice;
  CudaSafeCall(cudaMalloc((void**)&uniqueNodesDevice, this->numFinestUniqueNodes*sizeof(Node)));
  CudaSafeCall(cudaMemcpy(uniqueNodesDevice, this->uniqueNodesAtFinestLevel, this->numFinestUniqueNodes*sizeof(Node), cudaMemcpyHostToDevice));
  Node** nodeArray2DDevice;
  CudaSafeCall(cudaMalloc((void**)&nodeArray2DDevice, (this->depth + 1)*sizeof(Node*)));
  Node** nodeArray2D = new Node*[this->depth + 1];
  //is this necessary? does not seem to be doing anything
  CudaSafeCall(cudaMemcpy(nodeArray2D, nodeArray2DDevice, (this->depth + 1)*sizeof(Node*), cudaMemcpyDeviceToHost));

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
    int* nodeAddressesHost = new int[numUniqueNodes];

    for(int i = 0; i < numUniqueNodes; ++i){
      nodeAddressesHost[i] = 0;
    }

    CudaSafeCall(cudaMalloc((void**)&nodeNumbersDevice, numUniqueNodes * sizeof(int)));
    CudaSafeCall(cudaMalloc((void**)&nodeAddressesDevice, numUniqueNodes * sizeof(int)));
    //this is just to fill the arrays with 0s
    CudaSafeCall(cudaMemcpy(nodeNumbersDevice, nodeAddressesHost, numUniqueNodes * sizeof(int), cudaMemcpyHostToDevice));
    CudaSafeCall(cudaMemcpy(nodeAddressesDevice, nodeAddressesHost, numUniqueNodes * sizeof(int), cudaMemcpyHostToDevice));
    calculateNodeAddresses(grid, block, numUniqueNodes, uniqueNodesDevice, nodeAddressesDevice, nodeNumbersDevice);
    CudaSafeCall(cudaMemcpy(nodeAddressesHost, nodeAddressesDevice, numUniqueNodes* sizeof(int), cudaMemcpyDeviceToHost));

    int numNodesAtDepth = (d > 0) ? nodeAddressesHost[numUniqueNodes - 1] + 8: 1;
    cout<<"NUM NODES|UNIQUE NODES AT DEPTH "<<d<<" = "<<numNodesAtDepth<<"|"<<numUniqueNodes;
    delete[] nodeAddressesHost;

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
    cout<<" - DEPTH INDEX = "<<this->totalNodes<<endl;
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
  delete[] nodeArray2D;
  CudaSafeCall(cudaFree(nodeArray2DDevice));

  cout<<"NODE ARRAY FLATTENED AND COMPLETED"<<endl;
  printf("TOTAL NODES = %d\n\n",this->totalNodes);

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
  cout<<"\nVERTEX LUT"<<endl;
  for(int row = 0; row <  8; ++row){
    for(int col = 0; col < 7; ++col){
      cout<<this->vertexLUT[row][col]<<" ";
    }
    cout<<endl;
  }
  cout<<"\nEDGE LUT"<<endl;
  for(int row = 0; row <  12; ++row){
    for(int col = 0; col < 3; ++col){
      cout<<this->edgeLUT[row][col]<<" ";
    }
    cout<<endl;
  }
  cout<<"\nFACE LUT"<<endl;
  for(int row = 0; row <  6; ++row){
    cout<<this->faceLUT[row]<<" ";
  }
  cout<<endl<<endl;
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
}
void Octree::fillNeighborhoods(){

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
    cout<<"COMPUTED NEIGHBORHOOD ON DEPTH "<<this->depth - i<<" NUM NEIGHBORS = "<<atomicCounter<<endl;
    atomicCounter = 0;
  }
  cout<<endl;
  this->copyNodesToHost();
  CudaSafeCall(cudaFree(this->childLUTDevice));
  CudaSafeCall(cudaFree(this->parentLUTDevice));

}

void Octree::computeVertexArray(){
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
  //no idea why this is a thing - first instance of this operation is in createFinalNodeArray
  CudaSafeCall(cudaMemcpy(vertexArray2D, vertexArray2DDevice, (this->depth + 1)*sizeof(Vertex*), cudaMemcpyDeviceToHost));

  int* vertexIndex = new int[this->depth + 1];

  this->totalVertices = 0;
  int prevCount = 0;
  int* ownerInidicesDevice;
  int* vertexPlacementDevice;
  int* compactedOwnerArrayDevice;
  int* compactedVertexPlacementDevice;
  for(int i = 0; i <= this->depth; ++i){
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
    vertexIndex[i] = numVertices;
    //reset previously allocated resources
    block.x = 8;
    findVertexOwners<<<grid, block>>>(this->finalNodeArrayDevice, numNodesAtDepth,
      this->depthIndex[i], this->vertexLUTDevice, atomicCounter, ownerInidicesDevice, vertexPlacementDevice);
    CudaCheckError();
    CudaSafeCall(cudaMemcpy(&numVertices, atomicCounter, sizeof(int), cudaMemcpyDeviceToHost));
    if(i == this->depth  && numVertices - prevCount != 8){
      cout<<"ERROR GENERATING VERTICES, vertices at depth 0 != 8 -> "<<numVertices - prevCount<<endl;
      exit(-1);
    }
    CudaSafeCall(cudaMalloc((void**)&vertexArray2D[i], (numVertices - prevCount)*sizeof(Vertex)));
    cout<<numVertices - prevCount<<" VERTICES AT DEPTH "<<this->depth - i<<" - ";
    cout<<"TOTAL VERTICES = "<<numVertices<<endl;


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

    //uncomment this if you want to check vertex array
    /*
    int* compactedOwnerArray = new int[numVertices - prevCount];
    CudaSafeCall(cudaMemcpy(compactedOwnerArray, compactedOwnerArrayDevice, (numVertices - prevCount)*sizeof(int), cudaMemcpyDeviceToHost));
    if(i == 0){
      for(int a = 0; a < numVertices - prevCount; ++a){
        if(compactedOwnerArray[a] == -1){
          cout<<"ERROR IN COMPACTING VERTEX IDENTIFIER ARRAY"<<endl;
          exit(-1);
        }
      }
    }
    delete[] compactedOwnerArray;
    */

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
      numVertices - prevCount, vertexIndex[i], this->depthIndex[i], this->depth - i,
      this->width, this->vertexLUTDevice, compactedOwnerArrayDevice, compactedVertexPlacementDevice);
    CudaCheckError();
    CudaSafeCall(cudaFree(compactedOwnerArrayDevice));
    CudaSafeCall(cudaFree(compactedVertexPlacementDevice));

  }
  this->totalVertices = numVertices;
  cout<<"CREATING FULL VERTEX ARRAY"<<endl;
  this->vertexArray = new Vertex[numVertices];
  for(int i = 0; i < numVertices; ++i) this->vertexArray[i] = Vertex();//might not be necessary
  this->copyVerticesToDevice();
  for(int i = 0; i <= this->depth; ++i){
    if(i < this->depth){
      CudaSafeCall(cudaMemcpy(this->vertexArrayDevice + vertexIndex[i], vertexArray2D[i], (vertexIndex[i+1] - vertexIndex[i])*sizeof(Vertex), cudaMemcpyDeviceToDevice));
    }
    else{
      CudaSafeCall(cudaMemcpy(this->vertexArrayDevice + vertexIndex[i], vertexArray2D[i], 8*sizeof(Vertex), cudaMemcpyDeviceToDevice));
    }
    CudaSafeCall(cudaFree(vertexArray2D[i]));
  }
  cout<<"VERTEX ARRAY COMPLETED"<<endl<<endl;
  this->copyVerticesToHost();
  CudaSafeCall(cudaFree(this->vertexLUTDevice));
  CudaSafeCall(cudaFree(vertexArray2DDevice));
  delete[] vertexIndex;

}
void Octree::computeEdgeArray(){
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
  //no idea why this is a thing - first instance of this operation is in createFinalNodeArray
  CudaSafeCall(cudaMemcpy(edgeArray2D, edgeArray2DDevice, (this->depth + 1)*sizeof(Edge*), cudaMemcpyDeviceToHost));

  int* edgeIndex = new int[this->depth + 1];

  this->totalEdges = 0;
  int prevCount = 0;
  int* ownerInidicesDevice;
  int* edgePlacementDevice;
  int* compactedOwnerArrayDevice;
  int* compactedEdgePlacementDevice;
  for(int i = 0; i <= this->depth; ++i){
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
    edgeIndex[i] = numEdges;
    block.x = 12;//reset previously allocated resources
    findEdgeOwners<<<grid, block>>>(this->finalNodeArrayDevice, numNodesAtDepth,
      this->depthIndex[i], this->edgeLUTDevice, atomicCounter, ownerInidicesDevice, edgePlacementDevice);
    CudaCheckError();
    CudaSafeCall(cudaMemcpy(&numEdges, atomicCounter, sizeof(int), cudaMemcpyDeviceToHost));
    if(i == this->depth  && numEdges - prevCount != 12){
      cout<<"ERROR GENERATING EDGES, edges at depth 0 != 12 -> "<<numEdges - prevCount<<endl;
      exit(-1);
    }
    CudaSafeCall(cudaMalloc((void**)&edgeArray2D[i], (numEdges - prevCount)*sizeof(Edge)));
    cout<<numEdges - prevCount<<" EDGES AT DEPTH "<<this->depth - i<<" - ";
    cout<<"TOTAL EDGES = "<<numEdges<<endl;


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

    //uncomment this if you want to check edge array
    /*
    int* compactedOwnerArray = new int[numEdges - prevCount];
    CudaSafeCall(cudaMemcpy(compactedOwnerArray, compactedOwnerArrayDevice, (numEdges - prevCount)*sizeof(int), cudaMemcpyDeviceToHost));
    if(i == 0){
      for(int a = 0; a < numEdges - prevCount; ++a){
        if(compactedOwnerArray[a] == -1){
          cout<<a<<" ERROR IN COMPACTING EDGE IDENTIFIER ARRAY"<<endl;
          exit(-1);
        }
      }
    }
    delete[] compactedOwnerArray;
    */

    grid.y = 1;//reset and allocated resources
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
      numEdges - prevCount, edgeIndex[i],this->depthIndex[i], this->depth - i,
      this->width, this->edgeLUTDevice, compactedOwnerArrayDevice, compactedEdgePlacementDevice);
    CudaCheckError();
    CudaSafeCall(cudaFree(compactedOwnerArrayDevice));
    CudaSafeCall(cudaFree(compactedEdgePlacementDevice));

  }
  this->totalEdges = numEdges;
  cout<<"CREATING FULL EDGE ARRAY"<<endl;
  this->edgeArray = new Edge[numEdges];
  for(int i = 0; i < numEdges; ++i) this->edgeArray[i] = Edge();//might not be necessary
  this->copyEdgesToDevice();
  for(int i = 0; i <= this->depth; ++i){
    if(i < this->depth){
      CudaSafeCall(cudaMemcpy(this->edgeArrayDevice + edgeIndex[i], edgeArray2D[i], (edgeIndex[i+1] - edgeIndex[i])*sizeof(Edge), cudaMemcpyDeviceToDevice));
    }
    else{
      CudaSafeCall(cudaMemcpy(this->edgeArrayDevice + edgeIndex[i], edgeArray2D[i], 12*sizeof(Edge), cudaMemcpyDeviceToDevice));
    }
    CudaSafeCall(cudaFree(edgeArray2D[i]));
  }
  cout<<"EDGE ARRAY COMPLETED"<<endl<<endl;
  this->copyEdgesToHost();
  CudaSafeCall(cudaFree(this->edgeLUTDevice));
  CudaSafeCall(cudaFree(edgeArray2DDevice));
}
void Octree::computeFaceArray(){
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
  //no idea why this is a thing - first instance of this operation is in createFinalNodeArray
  CudaSafeCall(cudaMemcpy(faceArray2D, faceArray2DDevice, (this->depth + 1)*sizeof(Face*), cudaMemcpyDeviceToHost));

  int* faceIndex = new int[this->depth + 1];

  this->totalFaces = 0;
  int prevCount = 0;
  int* ownerInidicesDevice;
  int* facePlacementDevice;
  int* compactedOwnerArrayDevice;
  int* compactedFacePlacementDevice;
  for(int i = 0; i <= this->depth; ++i){
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
    faceIndex[i] = numFaces;
    block.x = 6;//reset previously allocated resources
    findFaceOwners<<<grid, block>>>(this->finalNodeArrayDevice, numNodesAtDepth,
      this->depthIndex[i], this->faceLUTDevice, atomicCounter, ownerInidicesDevice, facePlacementDevice);
    CudaCheckError();
    CudaSafeCall(cudaMemcpy(&numFaces, atomicCounter, sizeof(int), cudaMemcpyDeviceToHost));
    if(i == this->depth  && numFaces - prevCount != 6){
      cout<<"ERROR GENERATING FACES, faces at depth 0 != 6 -> "<<numFaces - prevCount<<endl;
      exit(-1);
    }
    CudaSafeCall(cudaMalloc((void**)&faceArray2D[i], (numFaces - prevCount)*sizeof(Face)));
    cout<<numFaces - prevCount<<" FACES AT DEPTH "<<this->depth - i<<" - ";
    cout<<"TOTAL FACES = "<<numFaces<<endl;


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

    //if you want to check face array uncomment
    /*
    int* compactedOwnerArray = new int[numFaces - prevCount];
    CudaSafeCall(cudaMemcpy(compactedOwnerArray, compactedOwnerArrayDevice, (numFaces - prevCount)*sizeof(int), cudaMemcpyDeviceToHost));
    if(i == 0){
      for(int a = 0; a < numFaces - prevCount; ++a){
        if(compactedOwnerArray[a] == -1){
          cout<<"ERROR IN COMPACTING FACE IDENTIFIER ARRAY"<<endl;
          exit(-1);
        }
      }
    }
    delete[] compactedOwnerArray;
    */

    grid.y = 1;//reset and allocated resources
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
  cout<<"CREATING FULL FACE ARRAY"<<endl;
  this->faceArray = new Face[numFaces];
  for(int i = 0; i < numFaces; ++i) this->faceArray[i] = Face();//might not be necessary
  this->copyFacesToDevice();
  for(int i = 0; i <= this->depth; ++i){
    if(i < this->depth){
      CudaSafeCall(cudaMemcpy(this->faceArrayDevice + faceIndex[i], faceArray2D[i], (faceIndex[i+1] - faceIndex[i])*sizeof(Face), cudaMemcpyDeviceToDevice));
    }
    else{
      CudaSafeCall(cudaMemcpy(this->faceArrayDevice + faceIndex[i], faceArray2D[i], 6*sizeof(Face), cudaMemcpyDeviceToDevice));
    }
    CudaSafeCall(cudaFree(faceArray2D[i]));
  }
  cout<<"FACE ARRAY COMPLETED"<<endl<<endl;
  this->copyFacesToHost();
  CudaSafeCall(cudaFree(this->faceLUTDevice));
  CudaSafeCall(cudaFree(faceArray2DDevice));
}

void Octree::checkForGeneralNodeErrors(){
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
        cout<<"NODE THAT IS NOT AT 2nd TO FINEST DEPTH HAS A CHILD WITH INDEX 0 IN FINEST DEPTH"<<endl;
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
        cout<<"DEPTH 0 DOES NOT INCLUDE ALL FINEST UNIQUE NODES "<<this->finalNodeArray[i].numFinestChildren<<",";
        cout<<this->numFinestUniqueNodes<<", NUM FULL FINEST NODES SHOULD BE "<<this->depthIndex[1]<<endl;
        exit(-1);
      }
      if(this->finalNodeArray[i].numPoints != this->numPoints){
        cout<<"DEPTH 0 DOES NOT CONTAIN ALL POINTS "<<this->finalNodeArray[i].numPoints<<","<<this->numPoints<<endl;
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
    if((this->finalNodeArray[i].center.x < this->min.x - regionOfError ||
    this->finalNodeArray[i].center.y < this->min.y - regionOfError ||
    this->finalNodeArray[i].center.z < this->min.z - regionOfError||
    this->finalNodeArray[i].center.x > this->max.x + regionOfError||
    this->finalNodeArray[i].center.y > this->max.y + regionOfError||
    this->finalNodeArray[i].center.z > this->max.z + regionOfError)){
      ++numCentersOUTSIDE;
    }
  }
  if(numCentersOUTSIDE > 0){
    printf("ERROR %d centers outside of bounding box\n",numCentersOUTSIDE);
    error = true;
  }
  if(numSiblingParents > 0){
    cout<<"ERROR "<<numSiblingParents<<" NODES THINK THEIR PARENT IS IN THE SAME DEPTH AS THEMSELVES"<<endl;
    error = true;
  }
  if(numChildNeighbors > 0){
    cout<<"ERROR "<<numChildNeighbors<<" NODES WITH SIBLINGS AT HIGHER DEPTH"<<endl;
    error = true;
  }
  if(numParentNeighbors > 0){
    cout<<"ERROR "<<numParentNeighbors<<" NODES WITH SIBLINGS AT LOWER DEPTH"<<endl;
    error = true;
  }
  if(numFuckedNodes > 0){
    cout<<numFuckedNodes<<" ERROR IN NODE CONCATENATION OR GENERATION"<<endl;
    error = true;
  }
  if(orphanNodes > 0){
    cout<<orphanNodes<<" ERROR THERE ARE ORPHAN NODES"<<endl;
    error = true;
  }
  if(nodesThatCantFindChildren > 0){
    cout<<"ERROR "<<nodesThatCantFindChildren<<" PARENTS WITHOUT CHILDREN"<<endl;
    error = true;
  }
  if(numVerticesMissing > 0){
    cout<<"ERROR "<<numVerticesMissing<<" VERTICES MISSING"<<endl;
    error = true;
  }
  if(numEgesMissing > 0){
    cout<<"ERROR "<<numEgesMissing<<" EDGES MISSING"<<endl;
    error = true;
  }
  if(numFacesMissing > 0){
    cout<<"ERROR "<<numFacesMissing<<" FACES MISSING"<<endl;
    error = true;
  }
  //if(error) exit(-1);
  else cout<<"NO ERRORS DETECTED IN OCTREE"<<endl;
  cout<<"NODES WITHOUT POINTS = "<<noPoints<<endl;
  cout<<"NODES WITH POINTS = "<<this->totalNodes - noPoints<<endl<<endl;
}



//TODO implement!
void Octree::computeNormals(){

}
