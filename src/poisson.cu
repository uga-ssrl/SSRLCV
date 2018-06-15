#include "poisson.cuh"

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

__device__ __host__ float3 operator+(const float3 &a, const float3 &b) {
  return {a.x+b.x, a.y+b.y, a.z+b.z};
}
__device__ __host__ float3 operator-(const float3 &a, const float3 &b) {
  return {a.x-b.x, a.y-b.y, a.z-b.z};
}
__device__ __host__ float3 operator/(const float3 &a, const float3 &b) {
  return {a.x/b.x, a.y/b.y, a.z/b.z};
}
__device__ __host__ float3 operator*(const float3 &a, const float3 &b) {
  return {a.x*b.x, a.y*b.y, a.z*b.z};
}
__device__ __host__ float dotProduct(const float3 &a, const float3 &b){
  return (a.x*b.x) + (a.y*b.y) + (a.z*b.z);
}
__device__ __host__ float3 operator+(const float3 &a, const float &b){
  return {a.x+b, a.y+b, a.z+b};
}
__device__ __host__ float3 operator-(const float3 &a, const float &b){
  return {a.x-b, a.y-b, a.z-b};
}
__device__ __host__ float3 operator/(const float3 &a, const float &b){
  return {a.x/b, a.y/b, a.z/b};
}
__device__ __host__ float3 operator*(const float3 &a, const float &b){
  return {a.x*b, a.y*b, a.z*b};
}
__device__ __host__ float3 operator+(const float &a, const float3 &b) {
  return {a+b.x, a+b.y, a+b.z};
}
__device__ __host__ float3 operator-(const float &a, const float3 &b) {
  return {a-b.x, a-b.y, a-b.z};
}
__device__ __host__ float3 operator/(const float &a, const float3 &b) {
  return {a/b.x, a/b.y, a/b.z};
}
__device__ __host__ float3 operator*(const float &a, const float3 &b) {
  return {a*b.x, a*b.y, a*b.z};
}
__device__ __host__ bool operator==(const float3 &a, const float3 &b){
  return (a.x==b.x)&&(a.y==b.y)&&(a.z==b.z);
}

//TODO maybe get the third convolution to get closer to gausian filter
__device__ __host__ float3 blender(const float3 &a, const float3 &b, const float &bw){
  float t[3] = {(a.x-b.x)/bw,(a.y-b.y)/bw,(a.z-b.z)/bw};
  float result[3] = {0.0f};
  for(int i = 0; i < 3; ++i){
    if(t[i] > 0.5 && t[i] <= 1.5){
      result[i] = (t[i]-1.5)*(t[i]-1.5)/(bw*bw*bw);
    }
    else if(t[i] < -0.5 && t[i] >= -1.5){
      result[i] = (t[i]+1.5)*(t[i]+1.5)/(bw*bw*bw);
    }
    else if(t[i] <= 0.5 && t[i] >= -0.5){
      result[i] = (1.5-(t[i]*t[i]))/(2.0f*bw*bw*bw);
    }
    else return {0.0f,0.0f,0.0f};
  }
  return {result[0],result[1],result[2]};
}
__device__ __host__ float3 blenderPrime(const float3 &a, const float3 &b, const float &bw){
  float t[3] = {(a.x-b.x)/bw,(a.y-b.y)/bw,(a.z-b.z)/bw};
  float result[3] = {0.0f};
  for(int i = 0; i < 3; ++i){
    if(t[i] > 0.5 && t[i] <= 1.5){
      result[i] = (2.0f*t[i] + 3.0f)/(bw*bw*bw);
    }
    else if(t[i] < -0.5 && t[i] >= -1.5){
      result[i] = (2.0f*t[i] - 3.0f)/(bw*bw*bw);
    }
    else if(t[i] <= 0.5 && t[i] >= -0.5){
      result[i] = (-2.0f*t[i])/(2.0f*bw*bw*bw);
    }
    else return {0.0f,0.0f,0.0f};
  }
  return {result[0],result[1],result[2]};
}
__device__ __host__ float3 blenderPrimePrime(const float3 &a, const float3 &b, const float &bw){
  float t[3] = {(a.x-b.x)/bw,(a.y-b.y)/bw,(a.z-b.z)/bw};
  float result[3] = {0.0f};
  for(int i = 0; i < 3; ++i){
    if(t[i] > 0.5 && t[i] <= 1.5){
      result[i] = 2.0f/(bw*bw*bw);
    }
    else if(t[i] < -0.5 && t[i] >= -1.5){
      result[i] = 2.0f/(bw*bw*bw);
    }
    else if(t[i] <= 0.5 && t[i] >= -0.5){
      result[i] = -1.0f/(bw*bw*bw);
    }
    else return {0.0f,0.0f,0.0f};
  }
  return {result[0],result[1],result[2]};
}

__device__ __host__ int3 splitCrunchBits3(const unsigned int &size, const int &key){
  int3 xyz = {0,0,0};
  for(int i = size - 1;i >= 0;){
    xyz.x = (xyz.x << 1) + ((key >> i) & 1);
    --i;
    xyz.y = (xyz.y << 1) + ((key >> i) & 1);
    --i;
    xyz.z = (xyz.z << 1) + ((key >> i) & 1);
    --i;
  }
  return xyz;
}

__global__ void computeVectorFeild(Node* nodeArray, int numFinestNodes, float3* vectorField, float3* normals, float3* points){
  int blockID = blockIdx.y * gridDim.x + blockIdx.x;
  if(blockID < numFinestNodes){
    __shared__ float3 vec;
    vec = {0.0f, 0.0f, 0.0f};
    __syncthreads();
    int neighborIndex = nodeArray[blockID].neighbors[threadIdx.x];
    if(neighborIndex != -1){
      int currentPoint = nodeArray[neighborIndex].pointIndex;
      int stopIndex = nodeArray[neighborIndex].numPoints + currentPoint;
      float3 blend = {0.0f,0.0f,0.0f};
      float width = nodeArray[blockID].width;
      float3 center = nodeArray[blockID].center;
      for(int i = currentPoint; i < stopIndex; ++i){
        //n = 2 Fo(q) make bounds {0.0f, 1.0f}
          //blend = 1.0f - blend;
        //n = 2 Fo(q) make bounds {-1.0f, 0.0f}
          //blend = blend + 1.0f;
        //n currently = 3
        blend = blender(points[i],center,width)*normals[i];
        if(blend.x == 0.0f && blend.y == 0.0f && blend.z == 0.0f) continue;
        atomicAdd(&vec.x, blend.x);
        atomicAdd(&vec.y, blend.y);
        atomicAdd(&vec.z, blend.z);
      }
    }
    __syncthreads();
    if(threadIdx.x != 0) return;
    else vectorField[blockID] = vec;
  }
}
__global__ void computeDivergenceFine(int depthOfOctree, Node* nodeArray, int numNodes, int depthIndex, float3* vectorField, float* divCoeff, float* fPrimeLUT){
  int blockID = blockIdx.y * gridDim.x + blockIdx.x;
  if(blockID < numNodes){
    __shared__ float coeff;
    int neighborIndex = nodeArray[blockID + depthIndex].neighbors[threadIdx.x];
    if(neighborIndex != -1){
      int numFinestChildren = nodeArray[neighborIndex].numFinestChildren;
      int finestChildIndex = nodeArray[neighborIndex].finestChildIndex;
      int3 xyz1;
      int3 xyz2;
      xyz1 = splitCrunchBits3(depthOfOctree*3, nodeArray[blockID + depthIndex].key);
      int mult = pow(2,depthOfOctree + 1) - 1;
      for(int i = finestChildIndex; i < finestChildIndex + numFinestChildren; ++i){
        xyz2 = splitCrunchBits3(depthOfOctree*3, nodeArray[i].key);
        atomicAdd(&coeff, dotProduct(vectorField[i], {fPrimeLUT[xyz1.x*mult + xyz2.x],fPrimeLUT[xyz1.y*mult + xyz2.y],fPrimeLUT[xyz1.z*mult + xyz2.z]}));
      }
      __syncthreads();
      //may want only one thread doing this should not matter though
      divCoeff[blockID + depthIndex] = coeff;
    }
  }
}
__global__ void findRelatedChildren(Node* nodeArray, int numNodes, int depthIndex, int2* relativityIndicators){
  int blockID = blockIdx.y * gridDim.x + blockIdx.x;
  if(blockID < numNodes){
    __shared__ int numRelativeChildren;
    __shared__ int firstRelativeChild;
    numRelativeChildren = 0;
    firstRelativeChild = 2147483647;//max int
    int neighborIndex = nodeArray[blockID + depthIndex].neighbors[threadIdx.x];
    if(neighborIndex != -1){
      //may not be helping anything by doing this but it prevents 2 accesses to global memory
      int registerChildChecker = nodeArray[neighborIndex].numFinestChildren;
      int registerChildIndex = nodeArray[neighborIndex].finestChildIndex;
      if(registerChildIndex != -1 && registerChildChecker != 0){
        atomicAdd(&numRelativeChildren, nodeArray[neighborIndex].numFinestChildren);
        atomicMin(&firstRelativeChild, nodeArray[neighborIndex].finestChildIndex);
      }
    }
    __syncthreads();
    //may want only one thread doing this should not matter though
    relativityIndicators[blockID].x = firstRelativeChild;
    relativityIndicators[blockID].y = numRelativeChildren;
  }
}
__global__ void computeDivergenceCoarse(int depthOfOctree, Node* nodeArray, int2* relativityIndicators, int currentNode, int depthIndex, float3* vectorField, float* divCoeff, float* fPrimeLUT){
  int globalID = blockIdx.x *blockDim.x + threadIdx.x;
  if(globalID < relativityIndicators[currentNode].y){
    globalID += relativityIndicators[currentNode].x;
    int3 xyz1;
    int3 xyz2;
    xyz1 = splitCrunchBits3(depthOfOctree*3, nodeArray[currentNode + depthIndex].key);
    xyz2 = splitCrunchBits3(depthOfOctree*3, nodeArray[globalID].key);
    int mult = pow(2,depthOfOctree + 1) - 1;
    //TODO try and find a way to optimize this so that it is not using atomics and global memory
    float fx,fy,fz;
    fx = fPrimeLUT[xyz1.x*mult + xyz2.x];
    fy = fPrimeLUT[xyz1.y*mult + xyz2.y];
    fz = fPrimeLUT[xyz1.z*mult + xyz2.z];
    float divergenceContributer = dotProduct(vectorField[globalID], {fx,fy,fz});
    atomicAdd(&divCoeff[currentNode + depthIndex], divergenceContributer);
  }
}

//TODO find out why most laplacians are 0
__global__ void computeLd(int depthOfOctree, Node* nodeArray, int numNodes, int depthIndex, float* offDiagonal, int* offDiagonalIndex, float* diagonals, int* numNonZeroEntries,float* fLUT, float* fPrimePrimeLUT){
  int blockID = blockIdx.y * gridDim.x + blockIdx.x;
  if(blockID < numNodes){
    __shared__ int numNonZero;
    numNonZero = 0;
    __syncthreads();
    int neighborIndex = nodeArray[blockID + depthIndex].neighbors[threadIdx.x];
    if(neighborIndex != -1){
      int3 xyz1;
      int3 xyz2;
      xyz1 = splitCrunchBits3(depthOfOctree*3, nodeArray[blockID + depthIndex].key);
      xyz2 = splitCrunchBits3(depthOfOctree*3, nodeArray[neighborIndex].key);
      int mult = pow(2,depthOfOctree + 1) - 1;
      float laplacianValue = (fPrimePrimeLUT[xyz1.x*mult + xyz2.x]*fLUT[xyz1.y*mult + xyz2.y]*fLUT[xyz1.z*mult + xyz2.z])+
      (fLUT[xyz1.x*mult + xyz2.x]*fPrimePrimeLUT[xyz1.y*mult + xyz2.y]*fLUT[xyz1.z*mult + xyz2.z])+
      (fLUT[xyz1.x*mult + xyz2.x]*fLUT[xyz1.y*mult + xyz2.y]*fPrimePrimeLUT[xyz1.z*mult + xyz2.z]);
      if(laplacianValue != 0.0f){
        if(threadIdx.x == 13){
          diagonals[blockID] = laplacianValue;
        }
        else{
          atomicAdd(&numNonZero, 1);
          offDiagonal[blockID*26 + (threadIdx.x > 13 ? threadIdx.x - 1 : threadIdx.x)] = laplacianValue;
          offDiagonalIndex[blockID*26 + (threadIdx.x > 13 ? threadIdx.x - 1 : threadIdx.x)] = neighborIndex;
        }
        //printf("%d,%d -> laplacian = %.9f\n",blockID + depthIndex,neighborIndex, laplacianValue);
      }
      else{
        offDiagonalIndex[threadIdx.x] = -1;
      }
    }
    else{
      offDiagonalIndex[threadIdx.x] = -1;
    }
    __syncthreads();
    numNonZeroEntries[blockID] = numNonZero;
  }
}

__global__ void updateDivergence(int depthOfOctree, Node* nodeArray, int numNodes, int depthIndex, float* divCoeff, float* fLUT, float* fPrimePrimeLUT, float* nodeImplicit){
  int blockID = blockIdx.y * gridDim.x + blockIdx.x;
  if(blockID < numNodes){
    int parent = nodeArray[blockID + depthIndex].parent;
    int parentNeighbor = nodeArray[parent].neighbors[threadIdx.x];
    if(parentNeighbor != -1){
      float nodeImplicitValue = nodeImplicit[parentNeighbor];
      int3 xyz1;
      int3 xyz2;
      xyz1 = splitCrunchBits3(depthOfOctree*3, nodeArray[blockID + depthIndex].key);
      xyz2 = splitCrunchBits3(depthOfOctree*3, nodeArray[parentNeighbor].key);
      int mult = pow(2,depthOfOctree + 1) - 1;
      float laplacianValue = (fPrimePrimeLUT[xyz1.x*mult + xyz2.x]*fLUT[xyz1.y*mult + xyz2.y]*fLUT[xyz1.z*mult + xyz2.z])+(fLUT[xyz1.x*mult + xyz2.x]*fPrimePrimeLUT[xyz1.y*mult + xyz2.y]*fLUT[xyz1.z*mult + xyz2.z])+(fLUT[xyz1.x*mult + xyz2.x]*fLUT[xyz1.y*mult + xyz2.y]*fPrimePrimeLUT[xyz1.z*mult + xyz2.z]);
      atomicAdd(&divCoeff[blockID + depthIndex], -1.0f*laplacianValue*nodeImplicitValue);
    }
  }
}

Poisson::Poisson(Octree* octree){
  this->octree = octree;
  float* divergenceVector = new float[this->octree->totalNodes];
  for(int i = 0; i < this->octree->totalNodes; ++i){
    divergenceVector[i] = 0.0f;
  }
  CudaSafeCall(cudaMalloc((void**)&this->divergenceVectorDevice, this->octree->totalNodes*sizeof(float)));
  CudaSafeCall(cudaMemcpy(this->divergenceVectorDevice, divergenceVector, this->octree->totalNodes*sizeof(float), cudaMemcpyHostToDevice));
  this->octree->copyPointsToDevice();
  this->octree->copyNormalsToDevice();
}

Poisson::~Poisson(){

}

//TODO OPTMIZE THIS YOU FUCK TARD
void Poisson::computeLUTs(){
  clock_t timer;
  timer = clock();

  float currentWidth = this->octree->width;
  float3 currentCenter = this->octree->center;
  float3 tempCenter = {0.0f,0.0f,0.0f};
  int pow2 = 1;
  vector<float3> centers;
  queue<float3> centersTemp;
  centersTemp.push(currentCenter);
  for(int d = 0; d <= this->octree->depth; ++d){
    for(int i = 0; i < pow2; ++i){
      tempCenter = centersTemp.front();
      centersTemp.pop();
      centers.push_back(tempCenter);
      centersTemp.push(tempCenter - (currentWidth/4));
      centersTemp.push(tempCenter + (currentWidth/4));
    }
    currentWidth /= 2;
    pow2 *= 2;
  }
  int numCenters = centers.size();
  printf("number of absolute unique centers = %d\n\n",numCenters);

  unsigned int size = (pow(2, this->octree->depth + 1) - 1);
  float** f = new float*[size];
  float** ff = new float*[size];
  float** fff = new float*[size];
  for(int i = 0; i < size; ++i){
    f[i] = new float[size];
    ff[i] = new float[size];
    fff[i] = new float[size];
  }

  float totalWidth = this->octree->width;
  int pow2i = 1;
  int offseti = 0;
  int pow2j = 1;
  int offsetj = 0;
  for(int i = 0; i <= this->octree->depth; ++i){
    offseti = pow2i - 1;
    pow2j = 1;
    for(int j = 0; j <= this->octree->depth; ++j){
      offsetj = pow2j - 1;
      for(int k = offseti; k < offseti + pow2i; ++k){
        for(int l = offsetj; l < offsetj + pow2j; ++l){
          f[k][l] = dotProduct(blender(centers[l],centers[k],totalWidth/pow2i),blender(centers[k],centers[l],totalWidth/pow2j));
          ff[k][l] = dotProduct(blender(centers[l],centers[k],totalWidth/pow2i),blenderPrime(centers[k],centers[l],totalWidth/pow2j));
          fff[k][l] = dotProduct(blender(centers[l],centers[k],totalWidth/pow2i),blenderPrimePrime(centers[k],centers[l],totalWidth/pow2j));
        }
      }
      pow2j *= 2;
    }
    pow2i *= 2;
  }
  this->fLUT = new float[size*size];
  this->fPrimeLUT = new float[size*size];
  this->fPrimePrimeLUT = new float[size*size];
  for(int i = 0; i < size; ++i){
    for(int j = 0; j < size; ++j){
      this->fLUT[i*size + j] = f[i][j];
      this->fPrimeLUT[i*size + j] = ff[i][j];
      this->fPrimePrimeLUT[i*size + j] = fff[i][j];
    }
  }
  timer = clock() - timer;
  printf("blending LUT generation took %f seconds fully on the CPU.\n\n",((float) timer)/CLOCKS_PER_SEC);
}

//TODO should optimize computeDivergenceCoarse
//TODO THERE ARE MEMORY ACCESS PROBLEMS ORIGINATING PROBABLY FROM LUT STUFF!!!!!!!!!!!!!! FIXXXXXXXXx
void Poisson::computeDivergenceVector(){
  clock_t cudatimer;
  cudatimer = clock();
  /*
  FIRST COMPUTE VECTOR FIELD
  */

  int numNodesAtDepth = 0;
  dim3 grid = {1,1,1};
  dim3 block = {1,1,1};
  numNodesAtDepth = this->octree->depthIndex[1];
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
  block.x = 27;
  float3* vectorField = new float3[numNodesAtDepth];
  for(int i = 0; i < numNodesAtDepth; ++i){
    vectorField[i] = {0.0f,0.0f,0.0f};
  }
  float3* vectorFieldDevice;
  CudaSafeCall(cudaMalloc((void**)&vectorFieldDevice, numNodesAtDepth*sizeof(float3)));
  CudaSafeCall(cudaMemcpy(vectorFieldDevice, vectorField, numNodesAtDepth*sizeof(float3), cudaMemcpyHostToDevice));
  computeVectorFeild<<<grid,block>>>(this->octree->finalNodeArrayDevice, numNodesAtDepth, vectorFieldDevice, this->octree->normalsDevice, this->octree->pointsDevice);
  cudaDeviceSynchronize();//force this to finish as it is necessary for next kernels
  CudaCheckError();
  /*
  CudaSafeCall(cudaMemcpy(vectorField, vectorFieldDevice, numNodesAtDepth*sizeof(float3), cudaMemcpyDeviceToHost));
  for(int i = 0; i < numNodesAtDepth; ++i){
    if(vectorField[i].x != 0.0f && vectorField[i].y != 0.0f && vectorField[i].z != 0.0f){
      cout<<vectorField[i].x<<","<<vectorField[i].y<<","<<vectorField[i].z<<endl;
    }
  }
  */
  delete[] vectorField;
  cudatimer = clock() - cudatimer;
  printf("Vector field generation kernel took %f seconds.\n\n",((float) cudatimer)/CLOCKS_PER_SEC);
  cudatimer = clock();
  /*
  NOW COMPUTE DIVERGENCE VECTOR AFTER FINDING VECTOR FIELD
  */

  unsigned int size = (pow(2, this->octree->depth + 1) - 1);
  CudaSafeCall(cudaMalloc((void**)&this->fPrimeLUTDevice, size*size*sizeof(float)));
  CudaSafeCall(cudaMemcpy(this->fPrimeLUTDevice, this->fPrimeLUT, size*size*sizeof(float), cudaMemcpyHostToDevice));

  int2* relativityIndicators;
  int2* relativityIndicatorsDevice;
  for(int d = 0; d <= this->octree->depth; ++d){
    block = {27,1,1};
    grid = {1,1,1};
    if(d != this->octree->depth){
      numNodesAtDepth = this->octree->depthIndex[d + 1] - this->octree->depthIndex[d];
    }
    else numNodesAtDepth = 1;

    if(numNodesAtDepth < 65535) grid.x = (unsigned int) numNodesAtDepth;
    else{
      grid.x = 65535;
      while(grid.x*grid.y < numNodesAtDepth){
        ++grid.y;
      }
      while(grid.x*grid.y > numNodesAtDepth){
        --grid.x;
        if(grid.x*grid.y < numNodesAtDepth){
          ++grid.x;//to ensure that numThreads > nodes
          break;
        }
      }
    }
    if(d <= 5){//evaluate divergence coefficients at finer depths
      computeDivergenceFine<<<grid, block>>>(this->octree->depth, this->octree->finalNodeArrayDevice, numNodesAtDepth, this->octree->depthIndex[d], vectorFieldDevice, this->divergenceVectorDevice, this->fPrimeLUTDevice);
      CudaCheckError();
    }
    else{//evaluate divergence coefficients at coarser depths
      relativityIndicators = new int2[numNodesAtDepth];
      for(int i = 0; i < numNodesAtDepth; ++i){
        relativityIndicators[i] = {0,0};
      }
      CudaSafeCall(cudaMalloc((void**)&relativityIndicatorsDevice, numNodesAtDepth*sizeof(int2)));
      CudaSafeCall(cudaMemcpy(relativityIndicatorsDevice, relativityIndicators, numNodesAtDepth*sizeof(int2), cudaMemcpyHostToDevice));
      findRelatedChildren<<<grid, block>>>(this->octree->finalNodeArrayDevice, numNodesAtDepth, this->octree->depthIndex[d], relativityIndicatorsDevice);
      cudaDeviceSynchronize();
      CudaCheckError();
      CudaSafeCall(cudaMemcpy(relativityIndicators, relativityIndicatorsDevice, numNodesAtDepth*sizeof(int2), cudaMemcpyDeviceToHost));
      for(int currentNode = 0; currentNode < numNodesAtDepth; ++currentNode){
        block.x = 1;
        grid.y = 1;
        if(relativityIndicators[currentNode].y == 0) continue;//TODO ensure this assumption is valid
        else if(relativityIndicators[currentNode].y < 65535) grid.x = (unsigned int) relativityIndicators[currentNode].y;
        else{
          grid.x = 65535;
          while(grid.x*block.x < relativityIndicators[currentNode].y){
            ++block.x;
          }
          while(grid.x*block.x > relativityIndicators[currentNode].y){
            --grid.x;
            if(grid.x*block.x < relativityIndicators[currentNode].y){
              ++grid.x;//to ensure that numThreads > nodes
              break;
            }
          }
        }
        computeDivergenceCoarse<<<grid, block>>>(this->octree->depth, this->octree->finalNodeArrayDevice, relativityIndicatorsDevice, currentNode, this->octree->depthIndex[d], vectorFieldDevice, this->divergenceVectorDevice, this->fPrimeLUTDevice);
        CudaCheckError();
      }
      CudaSafeCall(cudaFree(relativityIndicatorsDevice));
      delete[] relativityIndicators;
    }
  }
  CudaSafeCall(cudaFree(vectorFieldDevice));
  CudaSafeCall(cudaFree(this->fPrimeLUTDevice));
  delete[] this->fPrimeLUT;

  cudatimer = clock() - cudatimer;
  printf("Divergence vector generation kernel took %f seconds.\n\n",((float) cudatimer)/CLOCKS_PER_SEC);
}

//TODO separate diagonals and non diagonals
void Poisson::computeImplicitFunction(){
  clock_t cudatimer;
  cudatimer = clock();

  unsigned int size = (pow(2, this->octree->depth + 1) - 1);
  float* nodeImplicit = new float[this->octree->totalNodes];
  for(int i = 0; i < this->octree->totalNodes; ++i){
    nodeImplicit[i] = 0.0f;
  }
  CudaSafeCall(cudaMalloc((void**)&this->fLUTDevice, size*size*sizeof(float)));
  CudaSafeCall(cudaMalloc((void**)&this->fPrimePrimeLUTDevice, size*size*sizeof(float)));
  CudaSafeCall(cudaMalloc((void**)&this->nodeImplicitDevice, this->octree->totalNodes*sizeof(float)));
  CudaSafeCall(cudaMemcpy(this->fLUTDevice, this->fLUT, size*size*sizeof(float), cudaMemcpyHostToDevice));
  CudaSafeCall(cudaMemcpy(this->fPrimePrimeLUTDevice, this->fPrimePrimeLUT, size*size*sizeof(float), cudaMemcpyHostToDevice));
  CudaSafeCall(cudaMemcpy(this->nodeImplicitDevice, nodeImplicit, this->octree->totalNodes*sizeof(float), cudaMemcpyHostToDevice));

  int numNodesAtDepth = 0;
  float* diagonalsDevice;
  float* offDiagonalsDevice;
  int* offDiagonalIndicesDevice;
  int* numOffDiagonalsDevice;
  float* temp;
  int* numNonZeroEntries;
  float* diag;

  dim3 grid;
  dim3 block;

  for(int d = this->octree->depth; d >= 0; --d){
    //update divergence coefficients based on solutions at coarser depths
    grid = {1,1,1};
    block = {27,1,1};
    if(d != this->octree->depth){
      numNodesAtDepth = this->octree->depthIndex[d + 1] - this->octree->depthIndex[d];
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
      for(int dcoarse = d + 1; dcoarse <= this->octree->depth; ++dcoarse){
        updateDivergence<<<grid, block>>>(this->octree->depth, this->octree->finalNodeArrayDevice, numNodesAtDepth,
          this->octree->depthIndex[d], this->divergenceVectorDevice,
          this->fLUTDevice, this->fPrimePrimeLUTDevice, this->nodeImplicitDevice);
        CudaCheckError();
      }
    }

    /*
    MULTIGRID SOLVER
    */

    //setup

    temp = new float[numNodesAtDepth*26];
    diag = new float[numNodesAtDepth];
    numNonZeroEntries = new int[numNodesAtDepth];
    for(int i = 0; i < numNodesAtDepth*26; ++i){
      temp[i] = 0;
      if(i % 26 == 0){
        numNonZeroEntries[i/26] = -1;
        diag[i/26] = 0.0f;
      }
      temp[i] = 0.0f;
    }

    CudaSafeCall(cudaMalloc((void**)&diagonalsDevice, numNodesAtDepth*sizeof(float)));
    CudaSafeCall(cudaMalloc((void**)&offDiagonalsDevice, numNodesAtDepth*26*sizeof(float)));
    CudaSafeCall(cudaMalloc((void**)&offDiagonalIndicesDevice, numNodesAtDepth*26*sizeof(int)));
    CudaSafeCall(cudaMalloc((void**)&numOffDiagonalsDevice, numNodesAtDepth*sizeof(int)));
    CudaSafeCall(cudaMemcpy(diagonalsDevice, diag, numNodesAtDepth*sizeof(float), cudaMemcpyHostToDevice));
    CudaSafeCall(cudaMemcpy(offDiagonalsDevice, temp, numNodesAtDepth*26*sizeof(float), cudaMemcpyHostToDevice));
    CudaSafeCall(cudaMemcpy(offDiagonalIndicesDevice, temp, numNodesAtDepth*26*sizeof(int), cudaMemcpyHostToDevice));
    CudaSafeCall(cudaMemcpy(numOffDiagonalsDevice, numNonZeroEntries, numNodesAtDepth*sizeof(int), cudaMemcpyHostToDevice));
    computeLd<<<grid, block>>>(this->octree->depth, this->octree->finalNodeArrayDevice, numNodesAtDepth, this->octree->depthIndex[d],
      offDiagonalsDevice, offDiagonalIndicesDevice, diagonalsDevice, numOffDiagonalsDevice,
      this->fLUTDevice, this->fPrimePrimeLUTDevice);
    CudaCheckError();

    CudaSafeCall(cudaMemcpy(diag, diagonalsDevice, numNodesAtDepth*sizeof(float), cudaMemcpyDeviceToHost));
    // for(int i = 0; i < numNodesAtDepth; ++i){
    //   cout<<i<<"-"<<diag[i]<<endl;
    // }

    delete[] diag;
    delete[] numNonZeroEntries;
    delete[] temp;


    //multiplication???


    CudaSafeCall(cudaFree(diagonalsDevice));
    CudaSafeCall(cudaFree(offDiagonalsDevice));
    CudaSafeCall(cudaFree(offDiagonalIndicesDevice));
    CudaSafeCall(cudaFree(numOffDiagonalsDevice));

  }

  CudaSafeCall(cudaFree(this->fLUTDevice));
  CudaSafeCall(cudaFree(this->fPrimePrimeLUTDevice));
  CudaSafeCall(cudaFree(this->divergenceVectorDevice));
  delete[] this->fLUT;
  delete[] this->fPrimePrimeLUT;
  delete[] nodeImplicit;
  cudatimer = clock() - cudatimer;
  printf("Node Implicit f(n) generation kernel took %f seconds.\n\n",((float) cudatimer)/CLOCKS_PER_SEC);
}

void Poisson::marchingCubes(){
  this->octree->copyPointsToDevice();
  this->octree->copyNormalsToDevice();

  CudaSafeCall(cudaFree(this->nodeImplicitDevice));
}
