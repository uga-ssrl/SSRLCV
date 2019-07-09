#include "Quadtree.cuh"

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
  this->width = 0;
  this->imageSize = {0,0};
  this->nodes = nullptr;
  this->data = nullptr;
  this->edges = nullptr;
  this->vertices = nullptr;
  this->imageSize = {0,0};
}
template<typename T>
ssrlcv::Quadtree<T>::Quadtree(uint2 imageSize, ssrlcv::Unity<T>* data){
  this->nodes = nullptr;
  this->edges = nullptr;
  this->vertices = nullptr;
  this->data = data;
  this->imageSize = imageSize;
  this->width = (imageSize.x > imageSize.y) ? imageSize.x : imageSize.y;
  if(width % 2 != 0) this->width += 1;
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

template<typename T>
void ssrlcv::Quadtree<T>::generateLeafNodes(int depth){
  Node* leafNodes_device = nullptr;
  CudaSafeCall(cudaMalloc((void**)&leafNodes_device, this->data->numElements*sizeof(Node)));
  dim3 grid = {(this->data->numElements/1024) + 1,1,1};
  dim3 block = {1024,1,1};
  fillLeafNodesDensly<<<grid,block>>>(leafNodes_device, this->width, this->imageSize, depth);
  CudaCheckError();
  this->nodes = new ssrlcv::Unity<Node>(leafNodes_device, this->data->numElements, ssrlcv::gpu);
  // if(depth != -1){
  //   //TO USE IN POINT BASED QUADTREE
  //
  //   thrust::counting_iterator<unsigned int> iter(0);
  //   thrust::device_vector<unsigned int> indices(this->data->numElements);
  //   thrust::copy(iter, iter + this->data->numElements, indices.begin());
  //
  //   unsigned int* nodePointIndex = new unsigned int[this->points->numElements]();
  //   CudaSafeCall(cudaMemcpy(nodePointIndex, thrust::raw_pointer_cast(indices.data()), this->data->numElements*sizeof(unsigned int),cudaMemcpyDeviceToHost));
  //
  //   thrust::device_ptr<int> kys(nodeKeys_device);
  //   thrust::sort_by_key(kys, kys + this->data->numElements, indices.begin());
  //
  //   if(this->data->fore != ssrlcv::gpu){
  //     this->data->transferMemoryTo(ssrlcv::gpu);
  //   }
  //
  //   thrust::device_ptr<float2> cnts(nodeCenters_device);
  //   thrust::device_vector<float2> sortedCnts(this->data->numElements);
  //   thrust::gather(indices.begin(), indices.end(), cnts, sortedCnts.begin());
  //   CudaSafeCall(cudaMemcpy(nodeCenters_device, thrust::raw_pointer_cast(sortedCnts.data()), this->data->numElements*sizeof(float2),cudaMemcpyDeviceToDevice));
  //
  //   thrust::device_ptr<T> dataSorter(this->data->device);
  //   thrust::device_vector<T> sortedData(this->data->numElements);
  //   thrust::gather(indices.begin(), indices.end(), dataSorter, sortedData.begin());
  //   //determine if this is necessary
  //   this->data->setData(thrust::raw_pointer_cast(sortedData.data()), this->data->numElements, ssrlcv::gpu);
  //   this->data->transferMemoryTo(ssrlcv::cpu);
  //   this->data->clearDevice();
  //
  //   //there may be a faster way to do this
  //   thrust::pair<int*, unsigned int*> new_end;//the last value of these node array
  //   new_end = thrust::unique_by_key(kys,kys + this->data->numElements, indices.begin());
  //
  //   //now you need to copy over all the nonredudant nodes
  //
  // }
}

template<typename T>
void ssrlcv::Quadtree<T>::generateParentNodes(){
  if(this->nodes == nullptr || this->nodes->state == ssrlcv::null){
    //TODO potentially develop support for bottom up growth
    throw ssrlcv::NullUnityException("Cannot generate parent nodes before children");
  }

}

/*
CUDA implementations
*/
template<typename T>
__global__ void ssrlcv::fillLeafNodesDensly(typename Quadtree<T>::Node* leafNodes, unsigned int width, uint2 imageSize, int depth){
  int globalID = blockIdx.x *blockDim.x + threadIdx.x;
  if(globalID < imageSize.x*imageSize.y){
    int x = globalID%imageSize.x;
    int y = globalID/imageSize.x;
    if(imageSize.x > imageSize.y){
      x += width/2;
    }
    else{
      y += width/2;
    }
    int key = 0;
    unsigned int depth_reg = depth;
    int currentDepth = 1;
    int W = width/2;
    int2 center = {W,W};
    while(W > 1 && depth_reg != currentDepth){
      W /= 2;
      if(x < center.x){
        key <<= 1;
        center.x -= W;
      }
      else{
        key = (key << 1) + 1;
        center.x += W;
      }
      if(y < center.y){
        key <<= 1;
        center.y -= W;
      }
      else{
        key = (key << 1) + 1;
        center.y += W;
      }
      currentDepth++;
    }
    typename Quadtree<T>::Node leaf = typename Quadtree<T>::Node();
    leaf.key = key;
    leaf.center = {center.x + 0.5f, center.y + 0.5f};
    leaf.depth = currentDepth;
    leafNodes[globalID] = leaf;
  }
}
