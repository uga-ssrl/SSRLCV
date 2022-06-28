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
  this->flag = false;
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
  this->border = {0,0};
}

template<typename T>
ssrlcv::Quadtree<T>::Quadtree(uint2 size, unsigned int depth, ssrlcv::Unity<LocalizedData<T>>* data, unsigned int colorDepth, int2 border){
  this->nodes = nullptr;
  this->edges = nullptr;
  this->vertices = nullptr;
  this->data = data;
  this->colorDepth = colorDepth;
  this->border = border;
  this->size = {size.x + (border.x*2),size.y + (border.y*2)};
  this->depth = depth;
  logger.info.printf("Building Quadtree with following characteristics:");
  logger.info.printf("depth = %d",this->depth);
  logger.info.printf("size = {%d,%d}",this->size.x,this->size.y);
  logger.info.printf("boirder = {%d,%d}",this->border.x,this->border.y);
  this->generateLeafNodes();
  this->generateParentNodes();
  this->fillNeighborhoods();
}

template<typename T>
ssrlcv::Quadtree<T>::~Quadtree(){
  if(this->nodes != nullptr) delete this->nodes;
  if(this->vertices != nullptr) delete this->vertices;
  if(this->edges != nullptr) delete this->edges;
  if(this->data != nullptr) delete this->data;
}


//TODO ensure numLeafNodes cant go over max int (conditional usage of gridDim.y)
//check if log2 will work
template<typename T>
void ssrlcv::Quadtree<T>::generateLeafNodes(){

  clock_t timer = clock();
  std::cout<<"generating leaf nodes for quadtree..."<<std::endl;

  unsigned long numLeafNodes = 0;
  numLeafNodes = this->data->size();
  ssrlcv::ptr::device<int> leafNodeKeys_device( numLeafNodes);
  ssrlcv::ptr::device<float2> leafNodeCenters_device( numLeafNodes);
  dim3 grid = {1,1,1};
  dim3 block = {1,1,1};
  getFlatGridBlock(numLeafNodes,grid,block,getKeys<T>);
  if(this->data->getMemoryState() != gpu) this->data->setMemoryState(gpu);
  getKeys<T><<<grid,block>>>(numLeafNodes, this->data->device.get(), leafNodeKeys_device.get(),leafNodeCenters_device.get(), this->size,this->depth);
  cudaDeviceSynchronize();
  CudaCheckError();

  thrust::counting_iterator<unsigned int> iter(0);
  thrust::device_vector<unsigned int> indices(this->data->size());
  thrust::copy(iter, iter + this->data->size(), indices.begin());
  ssrlcv::ptr::device<unsigned int> nodeDataIndex_device( numLeafNodes);
  CudaSafeCall(cudaMemcpy(nodeDataIndex_device.get(), thrust::raw_pointer_cast(indices.data()), numLeafNodes*sizeof(unsigned int),cudaMemcpyDeviceToDevice));

  thrust::device_ptr<int> kys(leafNodeKeys_device.get());
  thrust::sort_by_key(kys, kys + this->data->size(), indices.begin());

  thrust::device_ptr<float2> cnts(leafNodeCenters_device.get());
  thrust::device_vector<float2> sortedCnts(this->data->size());
  thrust::gather(indices.begin(), indices.end(), cnts, sortedCnts.begin());
  CudaSafeCall(cudaMemcpy(leafNodeCenters_device.get(), thrust::raw_pointer_cast(sortedCnts.data()), this->data->size()*sizeof(float2),cudaMemcpyDeviceToDevice));

  if(this->data->getFore() != gpu){
    this->data->transferMemoryTo(gpu);
  }

  thrust::device_ptr<T> dataSorter(this->data->device.get());
  thrust::device_vector<T> sortedData(this->data->size());
  thrust::gather(indices.begin(), indices.end(), dataSorter, sortedData.begin());
  ssrlcv::ptr::device<T> data_device(this->data->size());
  CudaSafeCall(cudaMemcpy(data_device.get(),thrust::raw_pointer_cast(sortedData.data()), this->data->size()*sizeof(T), cudaMemcpyDeviceToDevice));
  this->data->setData(data_device, this->data->size(), gpu);
  this->data->transferMemoryTo(cpu);
  this->data->clear(gpu);

  thrust::pair<thrust::device_ptr<int>, thrust::device_ptr<unsigned int>> new_end;//the last value of these node array
  thrust::device_ptr<unsigned int> compactNodeDataIndex(nodeDataIndex_device.get());
  new_end = thrust::unique_by_key(kys,kys + this->data->size(), compactNodeDataIndex);
  numLeafNodes = thrust::get<0>(new_end) - kys;

  ssrlcv::ptr::device<Node> leafNodes_device( numLeafNodes);

  grid = {1,1,1};
  block = {1,1,1};
  getFlatGridBlock(numLeafNodes,grid,block,fillLeafNodes<T>);

  fillLeafNodes<T><<<grid,block>>>(this->data->size(), numLeafNodes,leafNodes_device.get(),leafNodeKeys_device.get(),leafNodeCenters_device.get(),nodeDataIndex_device.get(),this->depth);
  cudaDeviceSynchronize();
  CudaCheckError();

  this->nodes = ssrlcv::ptr::value<ssrlcv::Unity<Node>>(leafNodes_device, numLeafNodes, gpu);

  this->nodes->setFore(gpu);

  logger.info.printf("done in %f seconds.",((float) clock() -  timer)/CLOCKS_PER_SEC);

}


template<typename T>
void ssrlcv::Quadtree<T>::generateParentNodes(){
  clock_t timer = clock();
  std::cout<<"filling coarser depths of quadtree..."<<std::endl;
  if(this->nodes == nullptr || this->nodes->getMemoryState() == null){
    //TODO potentially develop support for bottom up growth
    throw NullUnityException("Cannot generate parent nodes before children");
  }
  if(this->nodes->getMemoryState() == cpu){
    this->nodes->transferMemoryTo(gpu);
  }
  int numUniqueNodes = this->nodes->size();
  ssrlcv::ptr::device<Node> uniqueNodes_device( this->nodes->size());
  CudaSafeCall(cudaMemcpy(uniqueNodes_device.get(), this->nodes->device.get(), this->nodes->size()*sizeof(Node), cudaMemcpyDeviceToDevice));
  delete this->nodes;
  this->nodes = nullptr;
  unsigned int totalNodes = 0;

  Node** nodes2D = new Node*[this->depth + 1];


  unsigned int* nodeDepthIndex_host = new unsigned int[this->depth + 1]();
  this->nodeDepthIndex = ssrlcv::ptr::value<ssrlcv::Unity<unsigned int>>(nodeDepthIndex_host, this->depth + 1, cpu);

  dim3 grid = {1,1,1};
  dim3 block = {1,1,1};
  getFlatGridBlock(numUniqueNodes, grid, block,findAllNodes<T>);

  for(int d = this->depth; d >= 0; --d){

    ssrlcv::ptr::device<unsigned int> nodeNumbers_device( numUniqueNodes );
    ssrlcv::ptr::device<unsigned int> nodeAddresses_device( numUniqueNodes );
    //this is just to fill the arrays with 0s

    findAllNodes<T><<<grid,block>>>(numUniqueNodes, nodeNumbers_device.get(), uniqueNodes_device.get());
    cudaDeviceSynchronize();
    CudaCheckError();
    thrust::device_ptr<unsigned int> nN(nodeNumbers_device.get());
    thrust::device_ptr<unsigned int> nA(nodeAddresses_device.get());
    thrust::inclusive_scan(nN, nN + numUniqueNodes, nA);

    unsigned int numNodesAtDepth = 0;
    CudaSafeCall(cudaMemcpy(&numNodesAtDepth, nodeAddresses_device.get() + (numUniqueNodes - 1), sizeof(unsigned int), cudaMemcpyDeviceToHost));
    numNodesAtDepth = (d > 0) ? numNodesAtDepth + 4: 1;

    CudaSafeCall(cudaMalloc((void**)&nodes2D[this->depth - d], numNodesAtDepth*sizeof(Node)));
    Node* blankNodes = new Node[numNodesAtDepth]();
    CudaSafeCall(cudaMemcpy(nodes2D[this->depth - d], blankNodes, numNodesAtDepth*sizeof(Node),cudaMemcpyHostToDevice));
    delete[] blankNodes;

    fillNodesAtDepth<T><<<grid,block>>>(numUniqueNodes, nodeNumbers_device.get(), nodeAddresses_device.get(), uniqueNodes_device.get(), nodes2D[this->depth - d], d, this->depth);
    cudaDeviceSynchronize();
    CudaCheckError();

    numUniqueNodes = numNodesAtDepth / 4;
    if(d != 0){
      grid = {1,1,1};
      block = {1,1,1};
      CudaSafeCall(cudaMalloc((void**)&uniqueNodes_device, numUniqueNodes*sizeof(Node)));
      getFlatGridBlock(numUniqueNodes, grid, block,buildParentalNodes<T>);
      buildParentalNodes<T><<<grid,block>>>(numNodesAtDepth,totalNodes,nodes2D[this->depth - d],uniqueNodes_device.get(),this->size);
      cudaDeviceSynchronize();
      CudaCheckError();
    }
    this->nodeDepthIndex->host.get()[this->depth - d] = totalNodes;
    totalNodes += numNodesAtDepth;
  }
  unsigned int numRootNodes = totalNodes - this->nodeDepthIndex->host.get()[this->depth];
  ssrlcv::ptr::device<Node> nodes_device(totalNodes);
  this->nodes = ssrlcv::ptr::value<ssrlcv::Unity<Node>>(nodes_device, totalNodes, gpu);
  for(int i = 0; i <= this->depth; ++i){
    if(i < this->depth){
      CudaSafeCall(cudaMemcpy(this->nodes->device.get() + this->nodeDepthIndex->host.get()[i], nodes2D[i],
        (this->nodeDepthIndex->host.get()[i+1]-this->nodeDepthIndex->host.get()[i])*sizeof(Node), cudaMemcpyDeviceToDevice));
    }
    else{
      CudaSafeCall(cudaMemcpy(this->nodes->device.get() + this->nodeDepthIndex->host.get()[i],
        nodes2D[i], numRootNodes*sizeof(Node), cudaMemcpyDeviceToDevice));
    }
    CudaSafeCall(cudaFree(nodes2D[i]));
  }
  delete[] nodes2D;

  ssrlcv::ptr::device<unsigned int> dataNodeIndex_device( this->data->size());
  this->dataNodeIndex = ssrlcv::ptr::value<ssrlcv::Unity<unsigned int>>(dataNodeIndex_device, this->data->size(), gpu);

  unsigned int numNodesAtDepth = 1;
  unsigned int depthStartingIndex = 0;
  grid = {1,1,1};
  block = {4,1,1};
  for(int i = this->depth; i >= 0; --i){
    depthStartingIndex = this->nodeDepthIndex->host.get()[i];
    if(i != (int)this->depth){
      numNodesAtDepth = this->nodeDepthIndex->host.get()[i + 1] - depthStartingIndex;
    }
    getGrid(numNodesAtDepth, grid);
    fillParentIndex<T><<<grid, block>>>(numNodesAtDepth, depthStartingIndex, this->nodes->device.get());
    CudaCheckError();
  }

  grid = {1,1,1};
  block = {1,1,1};
  getFlatGridBlock(this->nodeDepthIndex->host.get()[1],grid,block,fillDataNodeIndex<T>);
  fillDataNodeIndex<T><<<grid,block>>>(this->nodeDepthIndex->host.get()[1],this->nodes->device.get(), this->dataNodeIndex->device.get());
  cudaDeviceSynchronize();
  CudaCheckError();
  logger.info.printf("TOTAL NODES = %d",totalNodes);
  logger.info.printf("done in %f seconds.",((float) clock() -  timer)/CLOCKS_PER_SEC);

  this->nodes->setFore(gpu);
  this->dataNodeIndex->setFore(gpu);
}

template<typename T>
void ssrlcv::Quadtree<T>::fillNeighborhoods(){
  unsigned int parentLUT[4][9] = {
    {0,1,1,3,4,4,3,4,4},
    {1,1,2,4,4,5,4,4,5},
    {3,4,4,3,4,4,6,7,7},
    {4,4,5,4,4,5,7,7,8}
  };
  unsigned int childLUT[4][9] = {
    {3,2,3,1,0,1,3,2,3},
    {2,3,2,0,1,0,2,3,2},
    {1,0,1,3,2,3,1,0,1},
    {0,1,0,2,3,2,0,1,0}
  };
  ssrlcv::ptr::device<unsigned int> parentLUT_device( 36);
  ssrlcv::ptr::device<unsigned int> childLUT_device( 36);
  for(int i = 0; i < 4; ++i){
    CudaSafeCall(cudaMemcpy(parentLUT_device.get() + i*9, &(parentLUT[i]), 9*sizeof(int), cudaMemcpyHostToDevice));
    CudaSafeCall(cudaMemcpy(childLUT_device.get() + i*9, &(childLUT[i]), 9*sizeof(int), cudaMemcpyHostToDevice));
  }

  dim3 grid = {1,1,1};
  dim3 block = {9,1,1};

  unsigned int numNodesAtDepth = 0;
  unsigned int depthStartingIndex = 0;
  for(int i = this->depth; i >= 0; --i){
    numNodesAtDepth = 1;
    depthStartingIndex = this->nodeDepthIndex->host.get()[i];
    if(i != this->depth){
      numNodesAtDepth = this->nodeDepthIndex->host.get()[i + 1] - depthStartingIndex;
    }
    getGrid(numNodesAtDepth, grid);
    computeNeighboringNodes<T><<<grid, block>>>(numNodesAtDepth, depthStartingIndex, parentLUT_device.get(), childLUT_device.get(), this->nodes->device.get());
    cudaDeviceSynchronize();
    CudaCheckError();
  }
  this->nodes->setFore(gpu);//just to ensure that it is known gpu nodes was edited last
  std::cout<<"Neighborhoods filled"<<std::endl;
}

template<typename T>
void ssrlcv::Quadtree<T>::generateVertices(){

  unsigned int numNodesAtDepth = 0;
  dim3 grid = {1,1,1};
  dim3 block = {1,1,1};
  int numVertices = 0;
  ssrlcv::ptr::device<int> atomicCounter(1);
  CudaSafeCall(cudaMemcpy(atomicCounter.get(), &numVertices, sizeof(int), cudaMemcpyHostToDevice));
  ssrlcv::ptr::host<ssrlcv::ptr::device<Vertex>> vertices2D(this->depth + 1);

  ssrlcv::ptr::host<unsigned int> vertexDepthIndex_host(this->depth + 1);

  int prevCount = 0;
  for(int i = 0; i <= this->depth; ++i){
    //reset previously allocated resources
    grid.y = 1;
    block.x = 4;
    if(i == this->depth){//WARNING MAY CAUSE ISSUE
      numNodesAtDepth = this->nodes->size() - this->nodeDepthIndex->host.get()[this->depth];
    }
    else{
      numNodesAtDepth = this->nodeDepthIndex->host.get()[i + 1] - this->nodeDepthIndex->host.get()[i];
    }

    getGrid(numNodesAtDepth,grid);

    int* ownerInidices = new int[numNodesAtDepth*4];
    for(int v = 0;v < numNodesAtDepth*4; ++v){
      ownerInidices[v] = -1;
    }
    ssrlcv::ptr::device<int> ownerInidices_device(numNodesAtDepth*4);
    ssrlcv::ptr::device<int> vertexPlacement_device(numNodesAtDepth*4);
    CudaSafeCall(cudaMemcpy(ownerInidices_device.get(), ownerInidices, numNodesAtDepth*4*sizeof(int), cudaMemcpyHostToDevice));
    CudaSafeCall(cudaMemcpy(vertexPlacement_device.get(), ownerInidices, numNodesAtDepth*4*sizeof(int), cudaMemcpyHostToDevice));
    delete[] ownerInidices;

    prevCount = numVertices;
    vertexDepthIndex_host.get()[i] = numVertices;

    findVertexOwners<T><<<grid, block>>>(numNodesAtDepth, this->nodeDepthIndex->host.get()[i], this->nodes->device.get(), atomicCounter.get(), ownerInidices_device.get(), vertexPlacement_device.get());
    CudaCheckError();
    CudaSafeCall(cudaMemcpy(&numVertices, atomicCounter.get(), sizeof(int), cudaMemcpyDeviceToHost));
    if(i == this->depth  && numVertices - prevCount != 4){
      std::cout<<"ERROR GENERATING VERTICES, vertices at depth 0 != 4 -> "<<numVertices - prevCount<<std::endl;
      exit(-1);
    }
    vertices2D.get()[i].set((numVertices - prevCount));
    ssrlcv::ptr::device<int> compactedOwnerArray_device((numVertices - prevCount));
    ssrlcv::ptr::device<int> compactedVertexPlacement_device((numVertices - prevCount));

    thrust::device_ptr<int> arrayToCompact(ownerInidices_device.get());
    thrust::device_ptr<int> arrayOut(compactedOwnerArray_device.get());
    thrust::device_ptr<int> placementToCompact(vertexPlacement_device.get());
    thrust::device_ptr<int> placementOut(compactedVertexPlacement_device.get());

    //TODO change to just remove
    thrust::copy_if(arrayToCompact, arrayToCompact + (numNodesAtDepth*4), arrayOut, is_not_neg());
    CudaCheckError();
    thrust::copy_if(placementToCompact, placementToCompact + (numNodesAtDepth*4), placementOut, is_not_neg());
    CudaCheckError();

    grid = {1,1,1};
    block = {1,1,1};
    getGrid(numVertices - prevCount, grid);

    fillUniqueVertexArray<T><<<grid, block>>>(this->nodeDepthIndex->host.get()[i], this->nodes->device.get(), numVertices - prevCount,
      vertexDepthIndex_host.get()[i], vertices2D.get()[i], this->depth - i, compactedOwnerArray_device.get(), compactedVertexPlacement_device.get(),this->size);
    CudaCheckError();

  }
  ssrlcv::ptr::device<Vertex> vertices_device( numVertices);
  for(int i = 0; i <= this->depth; ++i){
    if(i < this->depth){
      CudaSafeCall(cudaMemcpy(vertices_device.get() + vertexDepthIndex_host.get()[i], vertices2D.get()[i], (vertexDepthIndex_host.get()[i+1] - vertexDepthIndex_host.get()[i])*sizeof(Vertex), cudaMemcpyDeviceToDevice));
    }
    else{
      CudaSafeCall(cudaMemcpy(vertices_device.get() + vertexDepthIndex_host.get()[i], vertices2D.get()[i].get(), 4*sizeof(Vertex), cudaMemcpyDeviceToDevice));
    }
  }

  this->vertices = ssrlcv::ptr::value<ssrlcv::Unity<Vertex>>(vertices_device, numVertices, gpu);
  this->vertexDepthIndex = ssrlcv::ptr::value<ssrlcv::Unity<unsigned int>>(vertexDepthIndex_host, this->depth + 1, cpu);

}

template<typename T>
void ssrlcv::Quadtree<T>::generateEdges(){
  unsigned int numNodesAtDepth = 0;
  dim3 grid = {1,1,1};
  dim3 block = {1,1,1};
  int* atomicCounter;
  int numEdges = 0;
  CudaSafeCall(cudaMalloc((void**)&atomicCounter, sizeof(int)));
  CudaSafeCall(cudaMemcpy(atomicCounter, &numEdges, sizeof(int), cudaMemcpyHostToDevice));
  Edge** edges2D_device;
  CudaSafeCall(cudaMalloc((void**)&edges2D_device, (this->depth + 1)*sizeof(Edge*)));
  Edge** edges2D = new Edge*[this->depth + 1];

  unsigned int* edgeDepthIndex_host = new unsigned int[this->depth + 1];

  int prevCount = 0;
  for(int i = 0; i <= this->depth; ++i){
    //reset previously allocated resources
    grid.y = 1;
    block.x = 4;
    if(i == this->depth){//WARNING MAY CAUSE ISSUE
      numNodesAtDepth = 1;
    }
    else{
      numNodesAtDepth = this->nodeDepthIndex->host.get()[i + 1] - this->nodeDepthIndex->host.get()[i];
    }

    getGrid(numNodesAtDepth,grid);

    int* ownerInidices = new int[numNodesAtDepth*4];
    for(int v = 0;v < numNodesAtDepth*4; ++v){
      ownerInidices[v] = -1;
    }
    ssrlcv::ptr::device<int> ownerInidices_device(numNodesAtDepth*4);
    ssrlcv::ptr::device<int> edgePlacement_device(numNodesAtDepth*4);
    CudaSafeCall(cudaMemcpy(ownerInidices_device.get(), ownerInidices, numNodesAtDepth*4*sizeof(int), cudaMemcpyHostToDevice));
    CudaSafeCall(cudaMemcpy(edgePlacement_device.get(), ownerInidices, numNodesAtDepth*4*sizeof(int), cudaMemcpyHostToDevice));
    delete[] ownerInidices;

    prevCount = numEdges;
    edgeDepthIndex_host[i] = numEdges;

    findEdgeOwners<T><<<grid, block>>>(numNodesAtDepth, this->nodeDepthIndex->host.get()[i], this->nodes->device.get(), atomicCounter, ownerInidices_device.get(), edgePlacement_device.get());
    CudaCheckError();
    CudaSafeCall(cudaMemcpy(&numEdges, atomicCounter, sizeof(int), cudaMemcpyDeviceToHost));
    if(i == this->depth  && numEdges - prevCount != 4){
      std::cout<<"ERROR GENERATING EDGES, vertices at depth 0 != 4 -> "<<numEdges - prevCount<<std::endl;
      exit(-1);
    }

    CudaSafeCall(cudaMalloc((void**)&edges2D[i], (numEdges - prevCount)*sizeof(Edge)));
    ssrlcv::ptr::device<int> compactedOwnerArray_device((numEdges - prevCount));
    ssrlcv::ptr::device<int> compactedEdgePlacement_device((numEdges - prevCount));

    thrust::device_ptr<int> arrayToCompact(ownerInidices_device.get());
    thrust::device_ptr<int> arrayOut(compactedOwnerArray_device.get());
    thrust::device_ptr<int> placementToCompact(edgePlacement_device.get());
    thrust::device_ptr<int> placementOut(compactedEdgePlacement_device.get());

    thrust::copy_if(arrayToCompact, arrayToCompact + (numNodesAtDepth*4), arrayOut, is_not_neg());
    CudaCheckError();
    thrust::copy_if(placementToCompact, placementToCompact + (numNodesAtDepth*4), placementOut, is_not_neg());
    CudaCheckError();

    //reset and allocated resources
    grid = {1,1,1};
    block = {1,1,1};
    getGrid(numEdges - prevCount, grid);


    fillUniqueEdgeArray<T><<<grid, block>>>(this->nodeDepthIndex->host.get()[i], this->nodes->device.get(), numEdges - prevCount,
      edgeDepthIndex_host[i], edges2D[i], this->depth - i, compactedOwnerArray_device, compactedEdgePlacement_device);
    CudaCheckError();

  }
  ssrlcv::ptr::device<Edge> edges_device( numEdges);
  for(int i = 0; i <= this->depth; ++i){
    if(i < this->depth){
      CudaSafeCall(cudaMemcpy(edges_device.get() + edgeDepthIndex_host[i], edges2D[i], (edgeDepthIndex_host[i+1] - edgeDepthIndex_host[i])*sizeof(Edge), cudaMemcpyDeviceToDevice));
    }
    else{
      CudaSafeCall(cudaMemcpy(edges_device.get() + edgeDepthIndex_host[i], edges2D[i], 4*sizeof(Edge), cudaMemcpyDeviceToDevice));
    }
    CudaSafeCall(cudaFree(edges2D[i]));
  }
  CudaSafeCall(cudaFree(edges2D_device));

  this->edges = ssrlcv::ptr::value<ssrlcv::Unity<Edge>>(edges_device, numEdges, gpu);
  this->edgeDepthIndex = ssrlcv::ptr::value<ssrlcv::Unity<unsigned int>>(edgeDepthIndex_host, this->depth + 1, cpu);

}

template<typename T>
void ssrlcv::Quadtree<T>::generateVerticesAndEdges(){
  this->generateVertices();
  this->generateEdges();
}

template<typename T>
void ssrlcv::Quadtree<T>::setNodeFlags(ssrlcv::ptr::value<ssrlcv::Unity<bool>> hashMap, bool requireFullNeighbors, uint2 depthRange){
  if(hashMap == nullptr || hashMap->getMemoryState() == null){
    throw NullUnityException("hashMap must be filled before setFlags is called");
  }
  if(!(depthRange.x == 0 && depthRange.y == 0) && (depthRange.x > depthRange.y || depthRange.x > this->depth || this->depth > depthRange.y || this->depth < depthRange.x)){
    std::cout<<"ERROR: invalid depthRange in setFlags"<<std::endl;
    exit(-1);
  }
  MemoryState origin[3] = {hashMap->getMemoryState(),this->nodes->getMemoryState(),this->nodeDepthIndex->getMemoryState()};
  if(hashMap->getFore() == cpu) hashMap->transferMemoryTo(gpu);
  if(this->nodes->getFore() == cpu) this->nodes->transferMemoryTo(gpu);
  if(this->nodeDepthIndex->getFore() == gpu) this->nodeDepthIndex->transferMemoryTo(cpu);

  unsigned int nodeDepthIndex = 0;
  if(depthRange.y == 0){
    nodeDepthIndex = this->nodeDepthIndex->host.get()[0];
  }
  else{
    nodeDepthIndex = this->nodeDepthIndex->host.get()[this->depth - depthRange.y];
  }
  unsigned int numNodes = 0;
  if(depthRange.x == 0){
    numNodes = this->nodes->size() - nodeDepthIndex;
  }
  else{
    numNodes = this->nodeDepthIndex->host.get()[this->depth - (depthRange.x - 1)] - nodeDepthIndex;
  }

  dim3 grid = {1,1,1};
  dim3 block = {1,1,1};
  void (*fp)(unsigned int, unsigned int, typename Quadtree<T>::Node*, bool*, bool) = &applyNodeFlags<T>;
  getFlatGridBlock(numNodes, grid, block,fp);
  applyNodeFlags<T><<<grid,block>>>(numNodes,nodeDepthIndex,this->nodes->device.get(),hashMap->device.get(),requireFullNeighbors);
  cudaDeviceSynchronize();
  CudaCheckError();

  if(origin[0] != hashMap->getMemoryState()){
    hashMap->setMemoryState(origin[0]);
  }
  this->nodes->setFore(gpu);//due to editing nodes in this method
  if(origin[1] != this->nodes->getMemoryState()){
    this->nodes->setMemoryState(origin[1]);
  }
  if(origin[2] != this->nodeDepthIndex->getMemoryState()){
    this->nodeDepthIndex->setMemoryState(origin[2]);
  }
}
template<typename T>
void ssrlcv::Quadtree<T>::setNodeFlags(float2 flagBorder, bool requireFullNeighbors, uint2 depthRange){
  if(!(depthRange.x == 0 && depthRange.y == 0) && (depthRange.x > depthRange.y || depthRange.x > this->depth || this->depth > depthRange.y ||
    this->depth < depthRange.x)){
    std::cout<<"ERROR: invalid depthRange in setFlags"<<std::endl;
    exit(-1);
  }
  MemoryState origin[2] = {this->nodes->getMemoryState(),this->nodeDepthIndex->getMemoryState()};
  if(this->nodes->getFore() == cpu) this->nodes->transferMemoryTo(gpu);
  if(this->nodeDepthIndex->getFore() == gpu) this->nodeDepthIndex->transferMemoryTo(cpu);

  unsigned int nodeDepthIndex = 0;
  if(depthRange.y == 0){
    nodeDepthIndex = this->nodeDepthIndex->host.get()[0];
  }
  else{
    nodeDepthIndex = this->nodeDepthIndex->host.get()[this->depth - depthRange.y];
  }
  unsigned int numNodes = 0;
  if(depthRange.x == 0){
    numNodes = this->nodes->size() - nodeDepthIndex;
  }
  else{
    numNodes = this->nodeDepthIndex->host.get()[this->depth - (depthRange.x - 1)] - nodeDepthIndex;
  }

  dim3 grid = {1,1,1};
  dim3 block = {1,1,1};
  void (*fp)(unsigned int, unsigned int, typename Quadtree<T>::Node*, float4, bool) = &applyNodeFlags<T>;
  getFlatGridBlock(numNodes, grid, block,fp);

  if(requireFullNeighbors) {
    logger.info.printf("Setting node flags based on distance from edge = {%f,%f} while also requiring full neighbors",flagBorder.x,flagBorder.y);
  } else {
    logger.info.printf("Setting node flags based on distance from edge = {%f,%f}",flagBorder.x,flagBorder.y);
  }
  float4 bounds = {flagBorder.x, flagBorder.y, ((float)this->size.x) - flagBorder.x, ((float)this->size.y) - flagBorder.y};

  applyNodeFlags<T><<<grid,block>>>(numNodes,nodeDepthIndex,this->nodes->device.get(),bounds,requireFullNeighbors);
  cudaDeviceSynchronize();
  CudaCheckError();

  this->nodes->setFore(gpu);//due to editing nodes in this method
  if(origin[0] != this->nodes->getMemoryState()){
    this->nodes->setMemoryState(origin[0]);
  }
  if(origin[1] != this->nodeDepthIndex->getMemoryState()){
    this->nodeDepthIndex->setMemoryState(origin[1]);
  }
}

template<typename T>
void ssrlcv::Quadtree<T>::writePLY(){
  std::string newFile = "out/test_"+ std::to_string(rand())+ ".ply";
  std::cout<<"writing "<<newFile<<std::endl;
  std::ofstream plystream(newFile);
  if (plystream.is_open()) {
    int verticesToWrite = this->nodes->size();
    this->nodes->transferMemoryTo(cpu);
    std::ostringstream stringBuffer = std::ostringstream("");
    stringBuffer << "ply\nformat ascii 1.0\ncomment object: SSRL test\n";
    stringBuffer << "element vertex ";
    stringBuffer << verticesToWrite;
    stringBuffer << "\nproperty float x\nproperty float y\nproperty float z\n";
    stringBuffer << "property uchar red\nproperty uchar green\nproperty uchar blue\n";
    stringBuffer << "end_header\n";
    plystream << stringBuffer.str();
    for(int i = 0; i < verticesToWrite; ++i){
      stringBuffer = std::ostringstream("");
      stringBuffer << this->nodes->host.get()[i].center.x;
      stringBuffer << " ";
      stringBuffer << this->nodes->host.get()[i].center.y;
      stringBuffer << " 0 ";
      if(this->nodes->host.get()[i].flag){
        stringBuffer << " 0 0 0\n";
      }
      else{
        stringBuffer << " 255 255 255\n";
      }
      plystream << stringBuffer.str();
    }

    std::cout<<newFile + " has been created.\n"<<std::endl;
  }
  else{
    std::cout << "Unable to open: " + newFile<< std::endl;
    exit(1);
  }
}


/*
CUDA implementations
*/
//NOTE: THIS SHOULD ONLY BE USED FOR DENSE POINTER QUADTREE
template<typename T>
__global__ void ssrlcv::getKeys(unsigned int numLocalizedPointers, ssrlcv::LocalizedData<T>* localizedPointers, int* keys, float2* nodeCenters, uint2 size, unsigned int depth){
  unsigned int globalID = (blockIdx.y* gridDim.x+ blockIdx.x)* blockDim.x + threadIdx.x;
  if(globalID < numLocalizedPointers){
    float2 point = localizedPointers[globalID].loc;
    int key = 0;
    unsigned int depth_reg = depth;
    int currentDepth = 1;
    float2 reg_size = {((float)size.x)/2.0f, ((float)size.y)/2.0f};
    float2 center = reg_size;
    while(depth_reg >= currentDepth){
      reg_size.x /= 2.0f;
      reg_size.y /= 2.0f;
      currentDepth++;
      if(point.x < center.x){
        key <<= 1;
        center.x -= reg_size.x;
      }
      else{
        key = (key << 1) + 1;
        center.x += reg_size.x;
      }
      if(point.y < center.y){
        key <<= 1;
        center.y -= reg_size.y;
      }
      else{
        key = (key << 1) + 1;
        center.y += reg_size.y;
      }
    }
    keys[globalID] = key;
    nodeCenters[globalID] = center;
  }
}


template<typename T>
__global__ void ssrlcv::fillLeafNodes(unsigned long numDataElements, unsigned long numLeafNodes, typename ssrlcv::Quadtree<T>::Node* leafNodes,
int* keys, float2* nodeCenters, unsigned int* nodeDataIndex, unsigned int depth){
  unsigned int globalID = (blockIdx.y* gridDim.x+ blockIdx.x)* blockDim.x + threadIdx.x;
  if(globalID < numLeafNodes){
    typename Quadtree<T>::Node node = typename Quadtree<T>::Node();
    node.key = keys[globalID];
    node.dataIndex = nodeDataIndex[globalID];
    if(globalID + 1 != numLeafNodes){
      node.numElements = nodeDataIndex[globalID + 1] - node.dataIndex;
    }
    else{
      node.numElements = numDataElements - node.dataIndex;
    }
    node.center = nodeCenters[node.dataIndex];//centers are not compacted by key so
    node.depth = depth;
    leafNodes[globalID] = node;
  }
}


template<typename T>
__global__ void ssrlcv::findAllNodes(unsigned long numUniqueNodes, unsigned int* nodeNumbers, typename ssrlcv::Quadtree<T>::Node* uniqueNodes){
  unsigned int globalID = (blockIdx.y* gridDim.x+ blockIdx.x)* blockDim.x + threadIdx.x;
  int tempCurrentKey = 0;
  int tempPrevKey = 0;
  if(globalID < numUniqueNodes){
    if(globalID == 0){
      nodeNumbers[globalID] = 0;
      return;
    }
    tempCurrentKey = uniqueNodes[globalID].key>>2;
    tempPrevKey = uniqueNodes[globalID - 1].key>>2;
    if(tempPrevKey == tempCurrentKey){
      nodeNumbers[globalID] = 0;
    }
    else{
      nodeNumbers[globalID] = 4;
    }
  }
}

template<typename T>
__global__ void ssrlcv::fillNodesAtDepth(unsigned long numUniqueNodes, unsigned int* nodeNumbers, unsigned int* nodeAddresses, typename ssrlcv::Quadtree<T>::Node* existingNodes,
typename ssrlcv::Quadtree<T>::Node* allNodes, unsigned int currentDepth, unsigned int totalDepth){
  unsigned int globalID = (blockIdx.y* gridDim.x+ blockIdx.x)* blockDim.x + threadIdx.x;
  if(currentDepth != 0 && globalID < numUniqueNodes){
    typename Quadtree<T>::Node node = existingNodes[globalID];
    unsigned int address = nodeAddresses[globalID];
    allNodes[address + (node.key&0x00000003)] = node;
    if(nodeNumbers[globalID] == 4 || globalID == 0){
      int siblingKey = node.key&0xfffffffc;//will clear last 2 bits
      for(int i = 0; i < 4; ++i){
        allNodes[address + i].depth = currentDepth;
        allNodes[address + i].key = siblingKey + i;
      }
    }
  }
  else if(currentDepth == 0){
    allNodes[nodeAddresses[0]] = existingNodes[0];
  }
}

template<typename T>
__global__ void ssrlcv::buildParentalNodes(unsigned long numChildNodes, unsigned long childDepthIndex, typename ssrlcv::Quadtree<T>::Node* childNodes, 
typename ssrlcv::Quadtree<T>::Node* parentNodes, uint2 size){
  unsigned long numUniqueNodesAtParentDepth = numChildNodes / 4;
  unsigned int globalID = (blockIdx.y* gridDim.x+ blockIdx.x)* blockDim.x + threadIdx.x;
  int nodesIndex = globalID*4;
  int2 childLoc[4] = {
    {-1,-1},
    {-1,1},
    {1,-1},
    {1,1}
  };
  if(globalID < numUniqueNodesAtParentDepth){
    typename Quadtree<T>::Node node = typename Quadtree<T>::Node();//may not be necessary
    node.key = (childNodes[nodesIndex].key>>2);
    node.depth =  childNodes[nodesIndex].depth - 1;

    float2 widthOfNode = {((float)size.x)/powf(2,node.depth + 1),((float)size.y)/powf(2,node.depth + 1)};

    int chosen = 0;
    for(int i = 0; i < 4; ++i){
      if(childNodes[nodesIndex + i].dataIndex != -1){
        if(node.dataIndex == -1){
          node.dataIndex = childNodes[nodesIndex + i].dataIndex;
          node.center.x = childNodes[nodesIndex + i].center.x - (widthOfNode.x*0.5*childLoc[i].x);
          node.center.y = childNodes[nodesIndex + i].center.y - (widthOfNode.y*0.5*childLoc[i].y);
          chosen = i;
        }
        node.numElements += childNodes[nodesIndex + i].numElements;
      }
      node.children[i] = nodesIndex + childDepthIndex + i;
    }
    for(int i = 0; i < 4; ++i){
      if(childNodes[nodesIndex + i].dataIndex == -1){
        childNodes[nodesIndex + i].center.x = node.center.x + (widthOfNode.x*0.5*childLoc[i].x);
        childNodes[nodesIndex + i].center.y = node.center.y + (widthOfNode.y*0.5*childLoc[i].y);
      }
      if(childNodes[nodesIndex + i].center.x - (widthOfNode.x*0.5*childLoc[i].x) != node.center.x || childNodes[nodesIndex + i].center.y - (widthOfNode.y*0.5*childLoc[i].y) != node.center.y){
        printf("%f,%f = %f,%f (%f,%f)%d,%d\n",node.center.x,node.center.y,childNodes[nodesIndex+i].center.x,childNodes[nodesIndex+i].center.y,childNodes[nodesIndex+chosen].center.x,childNodes[nodesIndex+chosen].center.y,node.depth + 1,node.dataIndex);
        //asm("trap;");
      }
    }
    parentNodes[globalID] = node;
  }
}

//NOTE this is recursive
template<typename T>
__global__ void ssrlcv::fillParentIndex(unsigned int numNodesAtDepth, unsigned int depthStartingIndex, typename Quadtree<T>::Node* nodes){
  unsigned int blockID = blockIdx.y * gridDim.x + blockIdx.x;
  if(blockID < numNodesAtDepth && nodes[blockID + depthStartingIndex].children[threadIdx.x] != -1){
    nodes[nodes[blockID + depthStartingIndex].children[threadIdx.x]].parent = depthStartingIndex + blockID;
  }
}

template<typename T>
__global__ void ssrlcv::fillDataNodeIndex(unsigned long numLeafNodes, typename Quadtree<T>::Node* nodes, unsigned int* dataNodeIndex){
  unsigned int globalID = (blockIdx.y* gridDim.x+ blockIdx.x) * blockDim.x + threadIdx.x;
  if(globalID < numLeafNodes){//no need for depth index as leaf nodes come first in node ordering
    typename Quadtree<T>::Node node = nodes[globalID];
    for(int i = 0;node.dataIndex != -1 && i < node.numElements; ++i){
      dataNodeIndex[node.dataIndex + i] = globalID;
    }
  }
}

template<typename T>
__global__ void ssrlcv::computeNeighboringNodes(unsigned int numNodesAtDepth, unsigned int currentDepthIndex, unsigned int* parentLUT,
unsigned int* childLUT, typename Quadtree<T>::Node* nodes){
  unsigned int blockID = blockIdx.y * gridDim.x + blockIdx.x;
  if(blockID < numNodesAtDepth){
    int neighborParentIndex = 0;
    nodes[blockID + currentDepthIndex].neighbors[4] = blockID + currentDepthIndex;
    __syncthreads();//threads wait until all other threads have finished above operations
    if(nodes[blockID + currentDepthIndex].parent != -1){
      int parentIndex = nodes[blockID + currentDepthIndex].parent;
      int depthKey = nodes[blockID + currentDepthIndex].key&(0x00000003);//will clear all but last 2 bits
      int lutIndexHelper = (depthKey*9) + threadIdx.x;
      int parentLUTIndex = parentLUT[lutIndexHelper];
      int childLUTIndex = childLUT[lutIndexHelper];
      neighborParentIndex = nodes[parentIndex].neighbors[parentLUTIndex];
      if(neighborParentIndex != -1){
        nodes[blockID + currentDepthIndex].neighbors[threadIdx.x] = nodes[neighborParentIndex].children[childLUTIndex];
      }
    }
  }
}

template<typename T>
__global__ void ssrlcv::findVertexOwners(unsigned int numNodesAtDepth, unsigned int depthIndex, typename ssrlcv::Quadtree<T>::Node* nodes, int* numVertices, 
int* ownerInidices, int* vertexPlacement){
  unsigned int vertexLUT[4][3] = {
    {0,1,3},
    {1,2,5},
    {3,6,7},
    {5,7,8}
  };
  unsigned int blockID = blockIdx.y * gridDim.x + blockIdx.x;
  if(blockID < numNodesAtDepth){
    int vertexID = (blockID*4) + threadIdx.x;
    int sharesVertex = -1;
    for(int i = 0; i < 3; ++i){//iterate through neighbors that share edge
      sharesVertex = vertexLUT[threadIdx.x][i];
      if(nodes[blockID + depthIndex].neighbors[sharesVertex] != -1 && sharesVertex < 4){//less than itself
        return;
      }
    }
    //if thread reaches this point, that means that this edge is owned by the current node
    //also means owner == current node
    ownerInidices[vertexID] = blockID + depthIndex;
    vertexPlacement[vertexID] = threadIdx.x;
    atomicAdd(numVertices, 1);
  }
}

template<typename T>
__global__ void ssrlcv::fillUniqueVertexArray(unsigned int depthIndex, typename ssrlcv::Quadtree<T>::Node* nodes, unsigned long numVertices, int vertexIndex,
typename ssrlcv::Quadtree<T>::Vertex* vertices, int depth, int* ownerInidices, int* vertexPlacement, uint2 size){
  unsigned int vertexLUT[4][3] = {
    {0,1,3},
    {1,2,5},
    {3,6,7},
    {5,7,8}
  };
  int2 coordPlacementIdentity[4] = {
    {-1,-1},
    {-1,1},
    {1,-1},
    {1,1}
  };
  unsigned int globalID = (blockIdx.y* gridDim.x+ blockIdx.x)* blockDim.x + threadIdx.x;
  if(globalID < numVertices){

    int ownerNodeIndex = ownerInidices[globalID];
    int ownedIndex = vertexPlacement[globalID];

    nodes[ownerNodeIndex].vertices[ownedIndex] = globalID + vertexIndex;

    float2 depthHalfWidth = {size.x/powf(2, depth + 1),size.y/powf(2, depth + 1)};
    typename Quadtree<T>::Vertex vertex = typename Quadtree<T>::Vertex();
    vertex.loc.x = nodes[ownerNodeIndex].center.x + (depthHalfWidth.x*coordPlacementIdentity[ownedIndex].x);
    vertex.loc.y = nodes[ownerNodeIndex].center.y + (depthHalfWidth.y*coordPlacementIdentity[ownedIndex].y);

    vertex.depth = depth;
    vertex.nodes[0] = ownerNodeIndex;
    int neighborSharingVertex = -1;
    for(int i = 0; i < 3; ++i){
      neighborSharingVertex = nodes[ownerNodeIndex].neighbors[vertexLUT[ownedIndex][i]];
      vertex.nodes[i + 1] =  neighborSharingVertex;
      if(neighborSharingVertex == -1) continue;
      //WARNING CHECK THIS
      nodes[neighborSharingVertex].vertices[2 - i] = globalID + vertexIndex;
    }
    vertices[globalID] = vertex;
  }
}

template<typename T>
__global__ void ssrlcv::findEdgeOwners(unsigned int numNodesAtDepth, unsigned int depthIndex, typename ssrlcv::Quadtree<T>::Node* nodes, int* numEdges, 
int* ownerInidices, int* edgePlacement){
  unsigned int edgeLUT[4] = {1,3,5,7};
  unsigned blockID = blockIdx.y * gridDim.x + blockIdx.x;
  if(blockID < numNodesAtDepth){
    int edgeID = (blockID*4) + threadIdx.x;
    int sharesEdge = -1;
    sharesEdge = edgeLUT[threadIdx.x];
    if(nodes[blockID + depthIndex].neighbors[sharesEdge] != -1 && sharesEdge < 4){//less than itself
      return;
    }
    //if thread reaches this point, that means that this edge is owned by the current node
    //also means owner == current node
    ownerInidices[edgeID] = blockID + depthIndex;
    edgePlacement[edgeID] = threadIdx.x;
    atomicAdd(numEdges, 1);
  }
}

template<typename T>
__global__ void ssrlcv::fillUniqueEdgeArray(unsigned int depthIndex, typename ssrlcv::Quadtree<T>::Node* nodes, unsigned long numEdges, int edgeIndex,
typename ssrlcv::Quadtree<T>::Edge* edges, int depth, int* ownerInidices, int* edgePlacement){
  unsigned int edgeLUT[4] = {1,3,5,7};
  uint2 vertexEdgeIdentity[4] = {
    {0,1},
    {0,2},
    {1,3},
    {3,2}
  };
  unsigned long blockId = blockIdx.y* gridDim.x+ blockIdx.x;
  unsigned long globalID = blockId * blockDim.x + threadIdx.x;
  if(globalID < numEdges){
    int ownerNodeIndex = ownerInidices[globalID];
    int ownedIndex = edgePlacement[globalID];

    nodes[ownerNodeIndex].edges[ownedIndex] = globalID + edgeIndex;

    typename Quadtree<T>::Edge edge = typename Quadtree<T>::Edge();
    edge.vertices.x = nodes[ownerNodeIndex].vertices[vertexEdgeIdentity[ownedIndex].x];
    edge.vertices.y = nodes[ownerNodeIndex].vertices[vertexEdgeIdentity[ownedIndex].y];
    edge.depth = depth;
    edge.nodes[0] = ownerNodeIndex;

    int neighborSharingFace = -1;
    neighborSharingFace = nodes[ownerNodeIndex].neighbors[edgeLUT[ownedIndex]];
    edge.nodes[1] =  neighborSharingFace;
    if(neighborSharingFace != -1) nodes[neighborSharingFace].edges[3 - ownedIndex] = globalID + edgeIndex;
    edges[globalID] = edge;
  }
}

template<typename T>
__global__ void ssrlcv::applyNodeFlags(unsigned int numNodes, unsigned int depthIndex, typename ssrlcv::Quadtree<T>::Node* nodes, bool* hashMap, bool requireFullNeighbors){
  unsigned int globalID = (blockIdx.y* gridDim.x+ blockIdx.x)* blockDim.x + threadIdx.x;
  if(globalID < numNodes){
    typename Quadtree<T>::Node node = nodes[globalID + depthIndex];
    node.flag = hashMap[globalID];
    if(requireFullNeighbors && !node.flag){
      for(int i = 0; i < 9; ++i){
        if(node.neighbors[i] == -1){
          nodes[globalID + depthIndex].flag = false;
          return;
        }
      }
    }
  }
}
template<typename T>
__global__ void ssrlcv::applyNodeFlags(unsigned int numNodes, unsigned int depthIndex, typename ssrlcv::Quadtree<T>::Node* nodes, float4 flagBounds, bool requireFullNeighbors){
  unsigned int globalID = (blockIdx.y* gridDim.x+ blockIdx.x)* blockDim.x + threadIdx.x;
  if(globalID < numNodes){
    globalID += depthIndex;
    typename Quadtree<T>::Node node = nodes[globalID];
    if(node.center.x > flagBounds.x && node.center.y > flagBounds.y && node.center.x < flagBounds.z && node.center.y < flagBounds.w){
      if(requireFullNeighbors){
        for(int i = 0; i < 9; ++i){
          if(node.neighbors[i] == -1){
            nodes[globalID].flag = false;
            return;
          }
        }
      }
      nodes[globalID].flag = true;
    }
    else{
      nodes[globalID].flag = false;
    }
  }
}
