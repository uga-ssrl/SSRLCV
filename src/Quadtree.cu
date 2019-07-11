#include "Quadtree.cuh"

/*
Accepted types for data
-listed to avoid link issues (add to list if new type is desired)
*/

template class ssrlcv::Quadtree<unsigned char>;
template class ssrlcv::Quadtree<float2>;
template class ssrlcv::Quadtree<unsigned int>;
template class ssrlcv::Quadtree<ssrlcv::LocalizedData<unsigned char>>;

/*
CLASS AND STRUCT METHODS
*/

template<typename T>
__device__ __host__ ssrlcv::Quadtree<T>::Node::Node(){
  this->key = -1;
  this->dataIndex = -1;
  this->numElements = 0;
  this->center = {-1,-1};
  this->depth = -1;
  this->parent = -1;
  for(int i = 0; i < 4; ++i) this->children[i] = -1;
  for(int i = 0; i < 9; ++i) this->neighbors[i] = -1;
  for(int i = 0; i < 4; ++i) this->edges[i] = -1;
  for(int i = 0; i < 4; ++i) this->vertices[i] = -1;
}
template<typename T>
__device__ __host__ ssrlcv::Quadtree<T>::Vertex::Vertex(){
  this->loc = {-1,-1};
  for(int i = 0; i < 4; ++i) this->nodes[i] = -1;
  this->depth = -1;
}
template<typename T>
__device__ __host__ ssrlcv::Quadtree<T>::Edge::Edge(){
  this->vertices = {-1,-1};
  for(int i = 0; i < 2; ++i) this->nodes[i] = -1;
  this->depth = -1;
}

template<typename T>
ssrlcv::Quadtree<T>::Quadtree(){
  this->nodes = nullptr;
  this->data = nullptr;
  this->edges = nullptr;
  this->vertices = nullptr;
  this->size = {0,0};
}
//specifically for index full quadtree
template<>
ssrlcv::Quadtree<unsigned int>::Quadtree(uint2 size){
  this->nodes = nullptr;
  this->edges = nullptr;
  this->vertices = nullptr;
  this->size = size;
  float fullDepth = (this->size.x > this->size.y) ? this->size.x : this->size.y;
  this->depth = {0, (unsigned int)log2(fullDepth)};
  unsigned int* data_host = new unsigned int[this->size.x*this->size.y];
  for(int i = 0; i < this->size.x*this->size.y; ++i){
    data_host[i] = i;
  }
  this->data = new Unity<unsigned int>(data_host, this->size.x*this->size.y, cpu);
  this->generateLeafNodes();
  this->generateParentNodes();
}
template<typename T>
ssrlcv::Quadtree<T>::Quadtree(uint2 size, ssrlcv::Unity<T>* data){
  if(size.x*size.y != data->numElements){
    std::cout<<"ERROR in usage of ssrlcv::Quadtree<T>::Quadtree(uint2 size, ssrlcv::Unity<T>* data):"<<std::endl;
    std::cout<<"if(size.x*size.y != data->numElements) LocalizedData,int2,or float2 must be used"<<std::endl;
  }
  this->nodes = nullptr;
  this->edges = nullptr;
  this->vertices = nullptr;
  this->data = data;
  this->size = size;
  float fullDepth = (this->size.x > this->size.y) ? this->size.x : this->size.y;
  this->depth = {0, (unsigned int)log2(fullDepth)};
  this->generateLeafNodes();
  this->generateParentNodes();
}
template<typename T>
ssrlcv::Quadtree<T>::~Quadtree(){
  if(this->nodes != nullptr) delete this->nodes;
  if(this->vertices != nullptr) delete this->vertices;
  if(this->edges != nullptr) delete this->edges;
  if(this->data != nullptr) delete this->data;
  if(this->nodeDepthIndex != nullptr) delete this->nodeDepthIndex;
  if(this->vertexDepthIndex != nullptr) delete this->vertexDepthIndex;
  if(this->edgeDepthIndex != nullptr) delete this->edgeDepthIndex;
}


//TODO ensure numLeafNodes cant go over max int (conditional usage of gridDim.y)
//check if log2 will work
template<typename T>
void ssrlcv::Quadtree<T>::generateLeafNodes(){
  int* leafNodeKeys_device = nullptr;
  float2* leafNodeCenters_device = nullptr;
  unsigned int* nodeDataIndex_device = nullptr;

  unsigned long numLeafNodes = 0;
  numLeafNodes = this->data->numElements;
  CudaSafeCall(cudaMalloc((void**)&leafNodeKeys_device, numLeafNodes*sizeof(int)));
  CudaSafeCall(cudaMalloc((void**)&leafNodeCenters_device, numLeafNodes*sizeof(float2)));
  dim3 grid = {(numLeafNodes/1024) + 1,1,1};
  dim3 block = {1024,1,1};
  getKeys<<<grid,block>>>(leafNodeKeys_device, leafNodeCenters_device, this->size, this->depth.y);
  CudaCheckError();

  thrust::counting_iterator<unsigned int> iter(0);
  thrust::device_vector<unsigned int> indices(this->data->numElements);
  thrust::copy(iter, iter + this->data->numElements, indices.begin());
  thrust::device_ptr<int> kys(leafNodeKeys_device);
  thrust::sort_by_key(kys, kys + this->data->numElements, indices.begin());

  if(this->data->fore != gpu){
    this->data->transferMemoryTo(gpu);
  }

  thrust::device_ptr<float2> cnts(leafNodeCenters_device);
  thrust::device_vector<float2> sortedCnts(this->data->numElements);
  thrust::gather(indices.begin(), indices.end(), cnts, sortedCnts.begin());
  CudaSafeCall(cudaMemcpy(leafNodeCenters_device, thrust::raw_pointer_cast(sortedCnts.data()), this->data->numElements*sizeof(float2),cudaMemcpyDeviceToDevice));

  thrust::device_ptr<T> dataSorter(this->data->device);
  thrust::device_vector<T> sortedData(this->data->numElements);
  thrust::gather(indices.begin(), indices.end(), dataSorter, sortedData.begin());
  T* data_device = nullptr;
  CudaSafeCall(cudaMalloc((void**)&data_device,this->data->numElements*sizeof(T)));
  CudaSafeCall(cudaMemcpy(data_device,thrust::raw_pointer_cast(sortedData.data()), this->data->numElements*sizeof(T), cudaMemcpyDeviceToDevice));
  this->data->setData(data_device, this->data->numElements, gpu);
  this->data->transferMemoryTo(cpu);
  this->data->clear(gpu);

  thrust::pair<thrust::device_ptr<int>, thrust::device_vector<unsigned int>::iterator> new_end;//the last value of these node array

  new_end = thrust::unique_by_key(kys,kys + this->data->numElements, indices.begin());
  numLeafNodes = thrust::get<1>(new_end) - indices.begin();

  CudaSafeCall(cudaMalloc((void**)&nodeDataIndex_device, numLeafNodes*sizeof(unsigned int)));
  CudaSafeCall(cudaMemcpy(nodeDataIndex_device, thrust::raw_pointer_cast(indices.data()), numLeafNodes*sizeof(unsigned int),cudaMemcpyDeviceToDevice));


  Node* leafNodes_device = nullptr;
  CudaSafeCall(cudaMalloc((void**)&leafNodes_device, numLeafNodes*sizeof(Node)));

  grid = {(numLeafNodes/1024) + 1, 1,1};
  block = {1024,1,1};

  fillLeafNodes<T><<<grid,block>>>(numLeafNodes,leafNodes_device,leafNodeKeys_device,leafNodeCenters_device,nodeDataIndex_device);

  this->nodes = new Unity<Node>(leafNodes_device, numLeafNodes, gpu);

  CudaSafeCall(cudaFree(leafNodeKeys_device));
  CudaSafeCall(cudaFree(leafNodeCenters_device));
  CudaSafeCall(cudaFree(nodeDataIndex_device));

}

template<typename T>
void ssrlcv::Quadtree<T>::generateParentNodes(){
  if(this->nodes == nullptr || this->nodes->state == null){
    //TODO potentially develop support for bottom up growth
    throw NullUnityException("Cannot generate parent nodes before children");
  }
  Node* uniqueNodes_device;
  if(this->nodes->state == cpu){
    this->nodes->transferMemoryTo(gpu);
  }
  int numUniqueNodes = this->nodes->numElements;
  CudaSafeCall(cudaMalloc((void**)&uniqueNodes_device, this->nodes->numElements*sizeof(Node)));
  CudaSafeCall(cudaMemcpy(uniqueNodes_device, this->nodes->device, this->nodes->numElements*sizeof(Node), cudaMemcpyDeviceToDevice));
  delete this->nodes;
  this->nodes = nullptr;
  unsigned int totalNodes = 0;

  Node** nodeArray2D = new Node*[(this->depth.y - this->depth.x) + 1];

  int* nodeAddresses_device;
  int* nodeNumbers_device;

  unsigned int* nodeDepthIndex_host = new unsigned int[(this->depth.y - this->depth.x) + 1]();
  unsigned int* pointNodeIndex_device;
  CudaSafeCall(cudaMalloc((void**)&pointNodeIndex_device, this->data->numElements*sizeof(unsigned int)));

  for(int d = this->depth.y; d >= (int)this->depth.x; --d){


  }

}

/*
CUDA implementations
*/
__global__ void ssrlcv::getKeys(int* keys, float2* nodeCenters, uint2 size, int depth){
  int globalID = blockIdx.x *blockDim.x + threadIdx.x;
  if(globalID < size.x*size.y){
    float x = (globalID%size.x) + 0.5f;
    float y = (globalID/size.x) + 0.5f;
    int key = 0;
    unsigned int depth_reg = depth;
    int currentDepth = 1;
    float2 reg_size = {size.x/2.0f, size.y/2.0f};
    float2 center = reg_size;
    while((reg_size.x > 1.0f || reg_size.y > 1.0f) && depth_reg != currentDepth){
      reg_size.x /= 2.0f;
      reg_size.y /= 2.0f;
      if(x < center.x){
        key <<= 1;
        center.x -= reg_size.x;
      }
      else{
        key = (key << 1) + 1;
        center.x += reg_size.x;
      }
      if(y < center.y){
        key <<= 1;
        center.y -= reg_size.y;
      }
      else{
        key = (key << 1) + 1;
        center.y += reg_size.y;
      }
      currentDepth++;
    }
    keys[globalID] = key;
    nodeCenters[globalID] = center;
  }
}

template<typename T>
__global__ void ssrlcv::fillLeafNodes(unsigned long numLeafNodes, typename ssrlcv::Quadtree<T>::Node* leafNodes,
int* keys, float2* nodeCenters, unsigned int* nodeDataIndex){

  int globalID = blockIdx.x *blockDim.x + threadIdx.x;
  if(globalID < numLeafNodes){
    typename Quadtree<T>::Node node = typename Quadtree<T>::Node();
    node.key = keys[globalID];
    node.dataIndex = nodeDataIndex[globalID];
    node.center = nodeCenters[node.dataIndex];//centers are not compacted by key so
    leafNodes[globalID] = node;
  }
}
