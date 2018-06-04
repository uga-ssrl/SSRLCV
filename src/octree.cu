#include "octree.cuh"

using namespace std;

#define MAX_POSSIBLE_DEPTH = 10//TODO need to find a way around using this
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

__constant__ int d_MAX_POSSIBLE_DEPTH = 10;//TODO need to find a way around using this

__device__ __host__ Vertex::Vertex(){
  for(int i = 0; i < 8; ++i){
    this->nodes[i] = -1;
  }
  this->depth = -1;
  this->coord = {0.0f,0.0f,0.0f};
}

__device__ __host__ Edge::Edge(){
  for(int i = 0; i < 4; ++i){
    this->nodes[i] = -1;
  }
  this->depth = -1;
  this->p1 = {0.0f,0.0f,0.0f};
  this->p2 = {0.0f,0.0f,0.0f};

}

__device__ __host__ Face::Face(){
  this->nodes[0] = -1;
  this->nodes[1] = -1;
  this->depth = -1;
  this->p1 = {0.0f,0.0f,0.0f};
  this->p2 = {0.0f,0.0f,0.0f};
  this->p3 = {0.0f,0.0f,0.0f};
  this->p4 = {0.0f,0.0f,0.0f};

}

__device__ __host__ Node::Node(){
  this->pointIndex = -1;
  this->center = {0.0f,0.0f,0.0f};
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
__global__ void fillBlankNodeArray(Node* uniqueNodes, int* nodeNumbers, int* nodeAddresses, Node* outputNodeArray, int numUniqueNodes, int currentDepth){
  int tx = threadIdx.x;
  int bx = blockIdx.x;
  int globalID = bx * blockDim.x + tx;
  int address = 0;
  if(globalID < numUniqueNodes && (globalID == 0 || nodeNumbers[globalID] == 8)){
    int siblingKey = uniqueNodes[globalID].key;
    siblingKey &= 0xfffffff8;//will clear last 3 bits
    for(int i = 0; i < 8; ++i){
      address = nodeAddresses[globalID] + i;
      outputNodeArray[address] = Node();
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
    currentDKey = uniqueNodes[globalID].key&(0x00000007);//will clear all but last 3 bits
    address = nodeAddresses[globalID] + currentDKey;
    for(int i = uniqueNodes[globalID].pointIndex; i < uniqueNodes[globalID].numPoints + uniqueNodes[globalID].pointIndex; ++i){
      pointNodeIndex[i] = address;
    }
    outputNodeArray[address].key = uniqueNodes[globalID].key;
    outputNodeArray[address].depth = uniqueNodes[globalID].depth;
    outputNodeArray[address].center = uniqueNodes[globalID].center;
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
    currentDKey = uniqueNodes[globalID].key&(0x00000007);//will clear all but last 3 bits
    address = nodeAddresses[globalID] + currentDKey;
    for(int i = 0; i < 8; ++i){
      outputNodeArray[address].children[i] = uniqueNodes[globalID].children[i];
      childNodeArray[uniqueNodes[globalID].children[i]].parent = address;
    }
    outputNodeArray[address].key = uniqueNodes[globalID].key;
    outputNodeArray[address].depth = uniqueNodes[globalID].depth;
    outputNodeArray[address].center = uniqueNodes[globalID].center;
    outputNodeArray[address].pointIndex = uniqueNodes[globalID].pointIndex;
    outputNodeArray[address].numPoints = uniqueNodes[globalID].numPoints;
    outputNodeArray[address].finestChildIndex = uniqueNodes[globalID].finestChildIndex;
    outputNodeArray[address].numFinestChildren = uniqueNodes[globalID].numFinestChildren;
  }
}
//TODO make this kernel increment numFinestChildren and set finestChildIndex
__global__ void generateParentalUniqueNodes(Node* uniqueNodes, Node* nodeArrayD, int numNodesAtDepth, float totalWidth){
  int numUniqueNodesAtParentDepth = numNodesAtDepth / 8;
  int tx = threadIdx.x;
  int bx = blockIdx.x;
  int globalID = bx * blockDim.x + tx;
  int nodeArrayIndex = globalID*8;
  if(globalID < numUniqueNodesAtParentDepth){
    uniqueNodes[globalID] = Node();//may not be necessary
    uniqueNodes[globalID].key = nodeArrayD[nodeArrayIndex].key>>3;
    uniqueNodes[globalID].pointIndex = nodeArrayD[nodeArrayIndex].pointIndex;
    int depth =  nodeArrayD[nodeArrayIndex].depth;
    uniqueNodes[globalID].depth = depth - 1;
    float3 center = nodeArrayD[nodeArrayIndex].center;
    float widthOfNode = totalWidth/powf(2,depth);
    center.x += (widthOfNode/2);
    center.y += (widthOfNode/2);
    center.z += (widthOfNode/2);
    uniqueNodes[globalID].center = center;
    int currentFinestChildIndex = 0x77777777;
    for(int i = 0; i < 8; ++i){
      uniqueNodes[globalID].numPoints += nodeArrayD[nodeArrayIndex + i].numPoints;
      uniqueNodes[globalID].children[i] = nodeArrayIndex + i;
      uniqueNodes[globalID].numFinestChildren += nodeArrayD[nodeArrayIndex + i].numFinestChildren;
      nodeArrayD[globalID + i].width = widthOfNode;
      if(currentFinestChildIndex > nodeArrayD[nodeArrayIndex + i].finestChildIndex && -1 != nodeArrayD[nodeArrayIndex + i].finestChildIndex){
        uniqueNodes[globalID].finestChildIndex = nodeArrayD[nodeArrayIndex + i].finestChildIndex;//no need to update due to them being the finest
      }
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
      if(depthIndex == 820008){
        //printf("THIS IS YOUR PARENTAL INDEX %d, \n",nodeArray[blockID].parent);
      }
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
      //else{//already instantiated as -1
      //  nodeArray[blockID].neighbors[threadIdx.x] = -1;
      //}
    }
    __syncthreads();//index updates
    //doing this mostly to prevent memcpy overhead
    if(childDepthIndex != -1 && threadIdx.x < 8){
      nodeArray[blockID + depthIndex].children[threadIdx.x] += childDepthIndex;
    }
    if(nodeArray[blockID + depthIndex].parent != -1 && threadIdx.x == 0){
      nodeArray[blockID + depthIndex].parent += (depthIndex + numNodes);
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
    blockID += depthIndex;
    int sharesVertex = -1;
    for(int i = 0; i < 7; ++i){//iterate through neighbors that share vertex
      sharesVertex = vertexLUT[(threadIdx.x*7) + i];
      if(nodeArray[blockID].neighbors[sharesVertex] != -1 && sharesVertex < 13){//less than itself
        return;
      }
    }
    //if thread reaches this point, that means that this vertex is owned by the current node
    //also means owner == current node
    ownerInidices[vertexID] = blockID;
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
    int3 vertexCoordIdentity[8];
    vertexCoordIdentity[0] = {-1,-1,-1};
    vertexCoordIdentity[1] = {-1,1,-1};
    vertexCoordIdentity[2] = {1,-1,-1};
    vertexCoordIdentity[3] = {1,1,-1};
    vertexCoordIdentity[4] = {-1,-1,1};
    vertexCoordIdentity[5] = {-1,1,1};
    vertexCoordIdentity[6] = {1,-1,1};
    vertexCoordIdentity[7] = {1,1,1};

    float depthHalfWidth = width/powf(2, depth + 1);
    Vertex vertex = Vertex();
    vertex.coord.x = nodeArray[ownerNodeIndex].center.x + depthHalfWidth*vertexCoordIdentity[ownedIndex].x;
    vertex.coord.y = nodeArray[ownerNodeIndex].center.y + depthHalfWidth*vertexCoordIdentity[ownedIndex].y;
    vertex.coord.z = nodeArray[ownerNodeIndex].center.z + depthHalfWidth*vertexCoordIdentity[ownedIndex].z;
    vertex.depth = depth;
    int neighborSharingVertex = -1;
    for(int i = ownedIndex*7; i < ownedIndex + 7; ++i){
      neighborSharingVertex = nodeArray[ownerNodeIndex].neighbors[vertexLUT[i]];
      vertex.nodes[i - (ownedIndex*7)] =  neighborSharingVertex;
      if(neighborSharingVertex != -1) nodeArray[neighborSharingVertex].vertices[7 - (i - (ownedIndex*7))] = globalID + vertexIndex;
    }
    vertexArray[globalID] = vertex;
  }
}

//TODO ensure that ordering of edges is correct
__global__ void findEdgeOwners(Node* nodeArray, int numNodes, int depthIndex, int* edgeLUT, int* numEdges, int* ownerInidices, int* edgePlacement){
  int blockID = blockIdx.y * gridDim.x + blockIdx.x;
  if(blockID < numNodes){
    int edgeID = (blockID*12) + threadIdx.x;
    blockID += depthIndex;
    int sharesEdge = -1;
    for(int i = 0; i < 3; ++i){//iterate through neighbors that share edge
      sharesEdge = edgeLUT[(threadIdx.x*3) + i];
      if(nodeArray[blockID].neighbors[sharesEdge] != -1 && sharesEdge < 13){//less than itself
        return;
      }
    }
    //if thread reaches this point, that means that this edge is owned by the current node
    //also means owner == current node
    ownerInidices[edgeID] = blockID;
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
    int2 vertexEdgeIdentity[12];
    vertexEdgeIdentity[0] = {0,1};
    vertexEdgeIdentity[1] = {0,2};
    vertexEdgeIdentity[2] = {1,3};
    vertexEdgeIdentity[3] = {2,3};
    vertexEdgeIdentity[4] = {0,4};
    vertexEdgeIdentity[5] = {1,5};
    vertexEdgeIdentity[6] = {2,6};
    vertexEdgeIdentity[7] = {3,7};
    vertexEdgeIdentity[8] = {4,5};
    vertexEdgeIdentity[9] = {4,6};
    vertexEdgeIdentity[10] = {5,7};
    vertexEdgeIdentity[11] = {6,7};

    int3 edgeCoordIdentity[8];
    edgeCoordIdentity[0] = {-1,-1,-1};
    edgeCoordIdentity[1] = {-1,1,-1};
    edgeCoordIdentity[2] = {1,-1,-1};
    edgeCoordIdentity[3] = {1,1,-1};
    edgeCoordIdentity[4] = {-1,-1,1};
    edgeCoordIdentity[5] = {-1,1,1};
    edgeCoordIdentity[6] = {1,-1,1};
    edgeCoordIdentity[7] = {1,1,1};

    float depthHalfWidth = width/powf(2, depth + 1);
    Edge edge = Edge();

    edge.p1.x = nodeArray[ownerNodeIndex].center.x + depthHalfWidth*edgeCoordIdentity[vertexEdgeIdentity[ownedIndex].x].x;
    edge.p1.y = nodeArray[ownerNodeIndex].center.y + depthHalfWidth*edgeCoordIdentity[vertexEdgeIdentity[ownedIndex].x].y;
    edge.p1.z = nodeArray[ownerNodeIndex].center.z + depthHalfWidth*edgeCoordIdentity[vertexEdgeIdentity[ownedIndex].x].z;
    edge.p2.x = nodeArray[ownerNodeIndex].center.x + depthHalfWidth*edgeCoordIdentity[vertexEdgeIdentity[ownedIndex].y].x;
    edge.p2.y = nodeArray[ownerNodeIndex].center.y + depthHalfWidth*edgeCoordIdentity[vertexEdgeIdentity[ownedIndex].y].y;
    edge.p2.z = nodeArray[ownerNodeIndex].center.z + depthHalfWidth*edgeCoordIdentity[vertexEdgeIdentity[ownedIndex].y].z;
    edge.depth = depth;

    int neighborSharingEdge = -1;
    for(int i = ownedIndex*3; i < ownedIndex + 3; ++i){
      neighborSharingEdge = nodeArray[ownerNodeIndex].neighbors[edgeLUT[i]];
      edge.nodes[i - (ownedIndex*3)] =  neighborSharingEdge;
      if(neighborSharingEdge != -1) nodeArray[neighborSharingEdge].edges[11 - (i - (ownedIndex*3))] = globalID + edgeIndex;
    }
    edgeArray[globalID] = edge;
  }
}

//TODO ensure that ordering of faces is correct
__global__ void findFaceOwners(Node* nodeArray, int numNodes, int depthIndex, int* faceLUT, int* numFaces, int* ownerInidices, int* facePlacement){
  int blockID = blockIdx.y * gridDim.x + blockIdx.x;
  if(blockID < numNodes){
    int faceID = (blockID*6) + threadIdx.x;
    blockID += depthIndex;
    int sharesFace = -1;
    sharesFace = faceLUT[threadIdx.x];
    if(nodeArray[blockID].neighbors[sharesFace] != -1 && sharesFace < 13){//less than itself
      return;
    }
    //if thread reaches this point, that means that this face is owned by the current node
    //also means owner == current node
    ownerInidices[faceID] = blockID;
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
    int4 vertexFaceIdentity[6];
    vertexFaceIdentity[0] = {0,1,2,3};
    vertexFaceIdentity[1] = {0,1,4,5};
    vertexFaceIdentity[2] = {0,2,4,6};
    vertexFaceIdentity[3] = {1,3,5,7};
    vertexFaceIdentity[4] = {2,3,6,7};
    vertexFaceIdentity[5] = {4,5,6,7};

    int3 faceCoordIdentity[8];
    faceCoordIdentity[0] = {-1,-1,-1};
    faceCoordIdentity[1] = {-1,1,-1};
    faceCoordIdentity[2] = {1,-1,-1};
    faceCoordIdentity[3] = {1,1,-1};
    faceCoordIdentity[4] = {-1,-1,1};
    faceCoordIdentity[5] = {-1,1,1};
    faceCoordIdentity[6] = {1,-1,1};
    faceCoordIdentity[7] = {1,1,1};

    float depthHalfWidth = width/powf(2, depth + 1);
    Face face = Face();

    face.p1.x = nodeArray[ownerNodeIndex].center.x + depthHalfWidth*faceCoordIdentity[vertexFaceIdentity[ownedIndex].x].x;
    face.p1.y = nodeArray[ownerNodeIndex].center.y + depthHalfWidth*faceCoordIdentity[vertexFaceIdentity[ownedIndex].x].y;
    face.p1.z = nodeArray[ownerNodeIndex].center.z + depthHalfWidth*faceCoordIdentity[vertexFaceIdentity[ownedIndex].x].z;
    face.p2.x = nodeArray[ownerNodeIndex].center.x + depthHalfWidth*faceCoordIdentity[vertexFaceIdentity[ownedIndex].y].x;
    face.p2.y = nodeArray[ownerNodeIndex].center.y + depthHalfWidth*faceCoordIdentity[vertexFaceIdentity[ownedIndex].y].y;
    face.p2.z = nodeArray[ownerNodeIndex].center.z + depthHalfWidth*faceCoordIdentity[vertexFaceIdentity[ownedIndex].y].z;
    face.p3.x = nodeArray[ownerNodeIndex].center.x + depthHalfWidth*faceCoordIdentity[vertexFaceIdentity[ownedIndex].z].x;
    face.p3.y = nodeArray[ownerNodeIndex].center.y + depthHalfWidth*faceCoordIdentity[vertexFaceIdentity[ownedIndex].z].y;
    face.p3.z = nodeArray[ownerNodeIndex].center.z + depthHalfWidth*faceCoordIdentity[vertexFaceIdentity[ownedIndex].z].z;
    face.p4.x = nodeArray[ownerNodeIndex].center.x + depthHalfWidth*faceCoordIdentity[vertexFaceIdentity[ownedIndex].w].x;
    face.p4.y = nodeArray[ownerNodeIndex].center.y + depthHalfWidth*faceCoordIdentity[vertexFaceIdentity[ownedIndex].w].y;
    face.p4.z = nodeArray[ownerNodeIndex].center.z + depthHalfWidth*faceCoordIdentity[vertexFaceIdentity[ownedIndex].w].z;
    face.depth = depth;
    int neighborSharingFace = -1;



    neighborSharingFace = nodeArray[ownerNodeIndex].neighbors[faceLUT[ownedIndex]];

    face.nodes[5 - ownedIndex] =  neighborSharingFace;

    if(neighborSharingFace != -1) nodeArray[neighborSharingFace].faces[5 - ownedIndex] = globalID + faceIndex;


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
void Octree::parsePLY(string pathToFile){
  cout<<pathToFile + "'s data to be transfered to an empty octree."<<endl;
	ifstream plystream(pathToFile);
	string currentLine;
  vector<float3> points;
  vector<float3> normals;//this will not be used in here once normal determination is done
  vector<uchar3> colors;
  float minX = 0, minY = 0, minZ = 0, maxX = 0, maxY = 0, maxZ = 0;
	if (plystream.is_open()) {
		while (getline(plystream, currentLine)) {
      stringstream getMyFloats = stringstream(currentLine);
      float value = 0.0;
      int index = 0;
      float3 point;
      float3 normal;
      uchar3 color;
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
            normal.x = (value*-1);
            color.x = 255;//will have value once normal is removed after testing
            break;
          case 4:
            normal.y = (value*-1);
            color.y = 255;//will have value once normal is removed after testing
            break;
          case 5:
            normal.z = (value*-1);
            color.z = 255;//will have value once normal is removed after testing
            break;
          default:
            lineIsDone = true;
            points.push_back(point);
            normals.push_back(normal);//only for testing
            colors.push_back(color);
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
    this->center.z = (maxZ + minZ)/2;

    this->width = maxX - minX;
    if(this->width < maxY - minY) this->width = maxY - minY;
    if(this->width < maxZ - minZ) this->width = maxZ - minZ;

    this->numPoints = (int) points.size();
    this->points = new float3[this->numPoints];
    this->normals = new float3[this->numPoints];
    this->colors = new uchar3[this->numPoints];
    this->finestNodeCenters = new float3[this->numPoints];
    this->finestNodePointIndexes = new int[this->numPoints];
    this->finestNodeKeys = new int[this->numPoints];
    this->pointNodeIndex = new int[this->numPoints];
    this->totalNodes = 0;
    this->numFinestUniqueNodes = 0;

    for(int i = 0; i < points.size(); ++i){
      this->points[i] = points[i];
      this->normals[i] = normals[i];
      this->colors[i] = colors[i];
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
  this->simpleOctree = false;
}

void Octree::init_octree_gpu(){
  clock_t cudatimer;
  cudatimer = clock();

  CudaSafeCall(cudaMalloc((void**)&this->pointsDevice, this->numPoints * sizeof(float3)));
  CudaSafeCall(cudaMalloc((void**)&this->normalsDevice, this->numPoints * sizeof(float3)));
  CudaSafeCall(cudaMalloc((void**)&this->colorsDevice, this->numPoints * sizeof(uchar3)));
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
void Octree::copyColorsToDevice(){
  CudaSafeCall(cudaMemcpy(this->colorsDevice, this->colors, this->numPoints * sizeof(uchar3), cudaMemcpyHostToDevice));
}
void Octree::copyColorsToHost(){
  CudaSafeCall(cudaMemcpy(this->colors, this->colorsDevice, this->numPoints * sizeof(uchar3), cudaMemcpyDeviceToHost));
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
  delete[] finestNodeCenters;
  delete[] finestNodePointIndexes;
  delete[] finestNodeKeys;
  delete[] uniqueNodesAtFinestLevel;
  CudaSafeCall(cudaFree(finestNodeCentersDevice));
  CudaSafeCall(cudaFree(finestNodePointIndexesDevice));
  CudaSafeCall(cudaFree(finestNodeKeysDevice));
  cout<<"PREREQUISITE/FINEST DEPTH ARRAYS HAVE BEEN DELETED"<<endl;
}

void Octree::copyNodesToDevice(){
  CudaSafeCall(cudaMemcpy(this->finalNodeArrayDevice, this->finalNodeArray, this->totalNodes * sizeof(Node), cudaMemcpyHostToDevice));
}
void Octree::copyNodesToHost(){
  CudaSafeCall(cudaMemcpy(this->finalNodeArray, this->finalNodeArrayDevice, this->totalNodes * sizeof(Node), cudaMemcpyDeviceToHost));
}
void Octree::copyVerticesToDevice(){
  CudaSafeCall(cudaMemcpy(this->vertexArrayDevice, this->vertexArray, this->totalVertices * sizeof(Vertex), cudaMemcpyHostToDevice));
}
void Octree::copyVerticesToHost(){
  CudaSafeCall(cudaMemcpy(this->vertexArray, this->vertexArrayDevice, this->totalVertices * sizeof(Vertex), cudaMemcpyDeviceToHost));
}
void Octree::copyEdgesToDevice(){
  CudaSafeCall(cudaMemcpy(this->edgeArrayDevice, this->edgeArray, this->totalEdges * sizeof(Edge), cudaMemcpyHostToDevice));
}
void Octree::copyEdgesToHost(){
  CudaSafeCall(cudaMemcpy(this->edgeArray, this->edgeArrayDevice, this->totalEdges * sizeof(Edge), cudaMemcpyDeviceToHost));
}
void Octree::copyFacesToDevice(){
  CudaSafeCall(cudaMemcpy(this->faceArrayDevice, this->faceArray, this->totalFaces * sizeof(Face), cudaMemcpyHostToDevice));
}
void Octree::copyFacesToHost(){
  CudaSafeCall(cudaMemcpy(this->faceArray, this->faceArrayDevice, this->totalFaces * sizeof(Face), cudaMemcpyDeviceToHost));
}

//TODO optimize if possible and remove normals from this section
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
void Octree::sortByKey(){
  clock_t cudatimer;
  cudatimer = clock();

  int* keyTemp = new int[this->numPoints];
  int* keyTempDevice;
  CudaSafeCall(cudaMalloc((void**)&keyTempDevice, this->numPoints*sizeof(int)));

  for(int array = 0; array < 3; ++array){
    for(int i = 0; i < this->numPoints; ++i){
      keyTemp[i] = this->finestNodeKeys[i];
    }
    thrust::device_ptr<float3> pnts(this->pointsDevice);
    thrust::device_ptr<float3> cnts(this->finestNodeCentersDevice);
    thrust::device_ptr<float3> nmls(this->normalsDevice);
    thrust::device_ptr<uchar3> clrs(this->colorsDevice);
    thrust::device_ptr<int> kys(this->finestNodeKeysDevice);

    CudaSafeCall(cudaMemcpy(keyTempDevice, keyTemp, this->numPoints * sizeof(int), cudaMemcpyHostToDevice));
    thrust::device_ptr<int> temp(keyTempDevice);
    if(array == 0){
      thrust::sort_by_key(temp, temp + this->numPoints, pnts);
    }
    else if(array == 1){
      thrust::sort_by_key(temp, temp + this->numPoints, cnts);
      thrust::sort_by_key(kys, kys + this->numPoints, clrs);//this also sorts the keys
    }
    else{
      thrust::sort_by_key(temp, temp + this->numPoints, nmls);
    }
  }

  CudaSafeCall(cudaFree(keyTempDevice));

  cudatimer = clock() - cudatimer;
  printf("octree sort_by_key took %f seconds.\n\n",((float) cudatimer)/CLOCKS_PER_SEC);
}
//TODO use device pointers
void Octree::compactData(){
  clock_t cudatimer;
  cudatimer = clock();
  this->copyPointsToHost();
  this->copyNormalsToHost();
  this->copyColorsToHost();
  this->copyFinestNodeCentersToHost();
  this->copyFinestNodeKeysToHost();

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
    numUniqueNodes++;
  }
  this->numFinestUniqueNodes = numUniqueNodes;
  cudatimer = clock() - cudatimer;
  printf("octree unique_by_key took %f seconds.\n\n",((float) cudatimer)/CLOCKS_PER_SEC);

}
void Octree::fillUniqueNodesAtFinestLevel(){//only fills with base information
  this->uniqueNodesAtFinestLevel = new Node[this->numFinestUniqueNodes];
  for(int i = 0; i < this->numFinestUniqueNodes; ++i){
    Node currentNode;
    currentNode.key = this->finestNodeKeys[i];
    currentNode.center = this->finestNodeCenters[i];
    currentNode.pointIndex = this->finestNodePointIndexes[i];
    currentNode.depth = this->depth;
    if(i + 1 != this->numFinestUniqueNodes){
      currentNode.numPoints = this->finestNodePointIndexes[i + 1] - this->finestNodePointIndexes[i];
    }
    else{
      currentNode.numPoints = this->numPoints - this->finestNodePointIndexes[i];
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

    int numNodesAtDepth = d > 0 ? nodeAddressesHost[numUniqueNodes - 1] + 8: 1;
    cout<<"NUM NODES|UNIQUE NODES AT DEPTH "<<d<<" = "<<numNodesAtDepth<<"|"<<numUniqueNodes;
    delete[] nodeAddressesHost;

    CudaSafeCall(cudaMalloc((void**)&nodeArray2D[this->depth - d], numNodesAtDepth* sizeof(Node)));

    fillBlankNodeArray<<<grid,block>>>(uniqueNodesDevice, nodeNumbersDevice,  nodeAddressesDevice, nodeArray2D[this->depth - d], numUniqueNodes, d);
    CudaCheckError();
    cudaDeviceSynchronize();
    if(this->depth == d){
      int* pointNodeIndexDevice;
      CudaSafeCall(cudaMalloc((void**)&pointNodeIndexDevice, this->numPoints * sizeof(int)));
      CudaSafeCall(cudaMemcpy(pointNodeIndexDevice, this->pointNodeIndex, this->numPoints * sizeof(int), cudaMemcpyHostToDevice));
      fillFinestNodeArrayWithUniques<<<grid,block>>>(uniqueNodesDevice, nodeAddressesDevice,nodeArray2D[this->depth - d], numUniqueNodes, pointNodeIndexDevice);
      CudaCheckError();
      CudaSafeCall(cudaMemcpy(this->pointNodeIndex, pointNodeIndexDevice, this->numPoints * sizeof(int), cudaMemcpyDeviceToHost));
      CudaSafeCall(cudaFree(pointNodeIndexDevice));
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
  for(int i = 0; i < this->totalNodes; ++i) this->finalNodeArray[i] = Node();//might not be necessary
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
  int vertexCounter = 0;
  int edgeCounter = 0;
  for(int row = 0; row < 12; ++row){
    vertexCounter = 0;
    for(int col = 0; col < 27; ++col){
      if(row < 8){
        flatParentLUT[flatCounter] = this->parentLUT[row][col];
        if(col == 0){
          this->vertexLUT[row][vertexCounter] = this->parentLUT[row][col];
          flatVertexLUT[flatVertexCounter] = this->parentLUT[row][col];
          vertexCounter++;
          flatVertexCounter++;
        }
        else if(this->parentLUT[row][col] > this->vertexLUT[row][vertexCounter - 1] && flatParentLUT[flatCounter] != 13){
          this->vertexLUT[row][vertexCounter] = flatParentLUT[flatCounter];
          flatVertexLUT[flatVertexCounter] = this->parentLUT[row][col];
          vertexCounter++;
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
  this->copyNodesToHost();
  CudaSafeCall(cudaFree(this->childLUTDevice));
  CudaSafeCall(cudaFree(this->parentLUTDevice));

}

void Octree::checkForGeneralNodeErrors(){
  int numFuckedNodes = 0;
  int orphanNodes = 0;
  int nodesWithOutChildren = 0;
  int nodesThatCantFindChildren = 0;
  int noPoints = 0;
  for(int i = 0; i < this->totalNodes; ++i){
    if(this->finalNodeArray[i].depth < 0){
      numFuckedNodes++;

    }
    if(this->finalNodeArray[i].parent == -1 && this->finalNodeArray[i].depth != 0){
      orphanNodes++;
    }
    int checkForChildren = 0;
    for(int c = 0; c < 8 && this->finalNodeArray[i].depth < 10; ++c){
      if(this->finalNodeArray[i].children[c] == -1){
        checkForChildren++;
      }
    }
    if(this->finalNodeArray[i].numPoints == 0){
      noPoints++;
    }
    if(this->finalNodeArray[i].depth != 0 && this->finalNodeArray[this->finalNodeArray[i].parent].children[this->finalNodeArray[i].key&((1<<3)-1)] == -1){

      //cout<<"PARENT ("<<this->finalNodeArray[i].parent<<
      //") DOES NOT KNOW CHILD ("<<(this->finalNodeArray[i].key&((1<<3)-1))<<
      //") EXISTS STARTING AT DEPTH "<<this->finalNodeArray[i].depth<<endl;
      //exit(-1);
      nodesThatCantFindChildren++;
    }
    if(this->finalNodeArray[i].depth == 9 && this->finalNodeArray[i].numFinestChildren == 0){
      //cout<<"FINEST NODES ARE NOT BEING COUNTED IN PARENTS"<<endl;
      //exit(-1);
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
  }
  if(numFuckedNodes > 0){
    cout<<numFuckedNodes<<" ERROR IN NODE CONCATENATION OR GENERATION"<<endl;
    exit(-1);
  }
  if(orphanNodes > 0){
    cout<<orphanNodes<<" ERROR THERE ARE ORPHAN NODES"<<endl;
    exit(-1);
  }
  if(nodesThatCantFindChildren > 0){
    cout<<nodesThatCantFindChildren<<" ERROR CHILDREN WITHOUT PARENTS"<<endl;
    exit(-1);
  }

  cout<<"NODES WITHOUT POINTS = "<<noPoints<<endl;
  cout<<"NODES WITH POINTS = "<<this->totalNodes - noPoints<<endl<<endl;
}

void Octree::computeVertexArray(){
  int numNodesAtDepth = 0;
  dim3 grid = {1,1,1};
  dim3 block = {1,1,1};
  int* atomicCounter;
  int numVertices = 0;
  CudaSafeCall(cudaMalloc((void**)&atomicCounter, sizeof(int)));
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
    block.x = 8;//reset previously allocated resources
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

    grid.y = 1;//reset and allocated resources
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
      numVertices - prevCount, numVertices,this->depthIndex[i], this->depth - i,
      this->width, this->vertexLUTDevice, compactedOwnerArrayDevice, compactedVertexPlacementDevice);
    CudaCheckError();
    CudaSafeCall(cudaFree(compactedOwnerArrayDevice));
    CudaSafeCall(cudaFree(compactedVertexPlacementDevice));

  }
  this->totalVertices = numVertices;
  cout<<"CREATING FULL VERTEX ARRAY"<<endl;
  this->vertexArray = new Vertex[numVertices];
  for(int i = 0; i < numVertices; ++i) this->vertexArray[i] = Vertex();//might not be necessary
  CudaSafeCall(cudaMalloc((void**)&this->vertexArrayDevice, numVertices*sizeof(Vertex)));
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
}
void Octree::computeEdgeArray(){
  int numNodesAtDepth = 0;
  dim3 grid = {1,1,1};
  dim3 block = {1,1,1};
  int* atomicCounter;
  int numEdges = 0;
  CudaSafeCall(cudaMalloc((void**)&atomicCounter, sizeof(int)));
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
      numEdges - prevCount, numEdges,this->depthIndex[i], this->depth - i,
      this->width, this->edgeLUTDevice, compactedOwnerArrayDevice, compactedEdgePlacementDevice);
    CudaCheckError();
    CudaSafeCall(cudaFree(compactedOwnerArrayDevice));
    CudaSafeCall(cudaFree(compactedEdgePlacementDevice));

  }
  this->totalEdges = numEdges;
  cout<<"CREATING FULL EDGE ARRAY"<<endl;
  this->edgeArray = new Edge[numEdges];
  for(int i = 0; i < numEdges; ++i) this->edgeArray[i] = Edge();//might not be necessary
  CudaSafeCall(cudaMalloc((void**)&this->edgeArrayDevice, numEdges*sizeof(Edge)));
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
  CudaSafeCall(cudaMemcpy(this->edgeArray, this->edgeArrayDevice, this->totalEdges*sizeof(Edge), cudaMemcpyDeviceToHost));
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
  CudaSafeCall(cudaMalloc((void**)&this->faceArrayDevice, numFaces*sizeof(Face)));
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
  CudaSafeCall(cudaMemcpy(this->faceArray, this->faceArrayDevice, this->totalFaces*sizeof(Face), cudaMemcpyDeviceToHost));
  CudaSafeCall(cudaFree(this->faceLUTDevice));
  CudaSafeCall(cudaFree(faceArray2DDevice));
}

//TODO implement!
void Octree::computeNormals(){

}
