#include "Octree.cuh"

// =============================================================================================================
//
// Constructors and Destructors
//
// =============================================================================================================

ssrlcv::Octree::Octree(){
  this->depth = 1;
  this->points = nullptr;
  this->normals = nullptr;
  this->nodes = nullptr;
  this->vertices = nullptr;
  this->edges = nullptr;
  this->faces = nullptr;
  this->pointNodeIndex = nullptr;
  this->nodeDepthIndex = nullptr;
  this->vertexDepthIndex = nullptr;
  this->edgeDepthIndex = nullptr;
  this->faceDepthIndex = nullptr;
}

ssrlcv::Octree::Octree(int numPoints, ssrlcv::ptr::host<float3> points, int depth, bool createVEF){
  this->min = {FLT_MAX,FLT_MAX,FLT_MAX};
  this->max = {-FLT_MAX,-FLT_MAX,-FLT_MAX};
  this->points = nullptr;
  this->normals = nullptr;
  this->nodes = nullptr;
  this->vertices = nullptr;
  this->edges = nullptr;
  this->faces = nullptr;
  this->pointNodeIndex = nullptr;
  this->nodeDepthIndex = nullptr;
  this->vertexDepthIndex = nullptr;
  this->edgeDepthIndex = nullptr;
  this->faceDepthIndex = nullptr;

  bool local_debug = false;

  this->depth = depth;
  if(this->depth >= 10){
    logger.err<<"ERROR this octree currently only supports a depth of 10 at the max";
    exit(-1);
  }

  this->points = ssrlcv::ptr::value<ssrlcv::Unity<float3>>(points, numPoints, cpu);

  for(int i = 0; i < numPoints; ++i){
    if(this->min.x > points.get()[i].x) this->min.x = points.get()[i].x;
    else if(this->max.x < points.get()[i].x) this->max.x = points.get()[i].x;
    if(this->min.y > points.get()[i].y) this->min.y = points.get()[i].y;
    else if(this->max.y < points.get()[i].y) this->max.y = points.get()[i].y;
    if(this->min.z > points.get()[i].z) this->min.z = points.get()[i].z;
    else if(this->max.z < points.get()[i].z) this->max.z = points.get()[i].z;
  }

  this->center.x = (this->max.x + this->min.x)/2;
  this->center.y = (this->max.y + this->min.y)/2;
  this->center.z = (this->max.z + this->min.z)/2;

  this->width = this->max.x - this->min.x;
  if(this->width < this->max.y - this->min.y) this->width = this->max.y - this->min.y;
  if(this->width < this->max.z - this->min.z) this->width = this->max.z - this->min.z;

  this->width = ceil(this->width);
  if(((int)this->width) % 2) this->width++;
  this->width += 6.0f;
  this->max = this->center + (this->width/2);
  this->min = this->center - (this->width/2);

  if (local_debug){
    printf("\nmin = %f,%f,%f\n",this->min.x,this->min.y,this->min.z);
    printf("max = %f,%f,%f\n",this->max.x,this->max.y,this->max.z);
    printf("bounding box width = %f\n", this->width);
    printf("center = %f,%f,%f\n",this->center.x,this->center.y,this->center.z);
    printf("number of points = %lu\n\n", this->points->size());
  }

  this->createFinestNodes();
  this->fillInCoarserDepths();
  this->fillNeighborhoods();
  if(!createVEF) this->createVEFArrays();
}
ssrlcv::Octree::Octree(int numPoints, ssrlcv::ptr::host<float3> points, float deepestWidth, bool createVEF){
  this->min = {FLT_MAX,FLT_MAX,FLT_MAX};
  this->max = {-FLT_MAX,-FLT_MAX,-FLT_MAX};
  this->points = nullptr;
  this->normals = nullptr;
  this->nodes = nullptr;
  this->vertices = nullptr;
  this->edges = nullptr;
  this->faces = nullptr;
  this->pointNodeIndex = nullptr;
  this->nodeDepthIndex = nullptr;
  this->vertexDepthIndex = nullptr;
  this->edgeDepthIndex = nullptr;
  this->faceDepthIndex = nullptr;

  bool local_debug = false;

  this->points = ssrlcv::ptr::value<ssrlcv::Unity<float3>>(points, numPoints, cpu);

  for(int i = 0; i < numPoints; ++i){
    if(this->min.x > points.get()[i].x) this->min.x = points.get()[i].x;
    else if(this->max.x < points.get()[i].x) this->max.x = points.get()[i].x;
    if(this->min.y > points.get()[i].y) this->min.y = points.get()[i].y;
    else if(this->max.y < points.get()[i].y) this->max.y = points.get()[i].y;
    if(this->min.z > points.get()[i].z) this->min.z = points.get()[i].z;
    else if(this->max.z < points.get()[i].z) this->max.z = points.get()[i].z;
  }

  this->center.x = (this->max.x + this->min.x)/2;
  this->center.y = (this->max.y + this->min.y)/2;
  this->center.z = (this->max.z + this->min.z)/2;

  this->width = this->max.x - this->min.x;
  if(this->width < this->max.y - this->min.y) this->width = this->max.y - this->min.y;
  if(this->width < this->max.z - this->min.z) this->width = this->max.z - this->min.z;

  this->width = ceil(this->width);
  if(((int)this->width) % 2) this->width++;
  this->width += 6.0f;
  this->max = this->center + (this->width/2);
  this->min = this->center - (this->width/2);

  if (local_debug){
    printf("\nmin = %f,%f,%f\n",this->min.x,this->min.y,this->min.z);
    printf("max = %f,%f,%f\n",this->max.x,this->max.y,this->max.z);
    printf("bounding box width = %f\n", this->width);
    printf("center = %f,%f,%f\n",this->center.x,this->center.y,this->center.z);
    printf("number of points = %lu\n\n", this->points->size());
  }

  this->depth = 0;
  float finestWidth = this->width;
  while(finestWidth > deepestWidth){
    finestWidth /= 2.0f;
    ++this->depth;
  }
  if(this->depth >= 10){
    logger.err<<"ERROR this octree currently only supports a depth of 10 at the max";
    exit(-1);
  }

  this->createFinestNodes();
  this->fillInCoarserDepths();
  this->fillNeighborhoods();
  if(createVEF) this->createVEFArrays();
}

ssrlcv::Octree::Octree(ssrlcv::ptr::value<ssrlcv::Unity<float3>> points, int depth, bool createVEF){
  this->min = {FLT_MAX,FLT_MAX,FLT_MAX};
  this->max = {-FLT_MAX,-FLT_MAX,-FLT_MAX};
  this->normals = nullptr;
  this->nodes = nullptr;
  this->vertices = nullptr;
  this->edges = nullptr;
  this->faces = nullptr;
  this->pointNodeIndex = nullptr;
  this->nodeDepthIndex = nullptr;
  this->vertexDepthIndex = nullptr;
  this->edgeDepthIndex = nullptr;
  this->faceDepthIndex = nullptr;

  bool local_debug = false;

  this->points = points;
  if(this->points->getMemoryState() == gpu || this->points->getFore() != cpu){
    this->points->transferMemoryTo(cpu);
  }
  ssrlcv::ptr::host<float3> points_host = this->points->host;

  for(int i = 0; i < points->size(); ++i){
    if(this->min.x > points_host.get()[i].x) this->min.x = points_host.get()[i].x;
    else if(this->max.x < points_host.get()[i].x) this->max.x = points_host.get()[i].x;
    if(this->min.y > points_host.get()[i].y) this->min.y = points_host.get()[i].y;
    else if(this->max.y < points_host.get()[i].y) this->max.y = points_host.get()[i].y;
    if(this->min.z > points_host.get()[i].z) this->min.z = points_host.get()[i].z;
    else if(this->max.z < points_host.get()[i].z) this->max.z = points_host.get()[i].z;
  }


  this->max = {this->max.x,this->max.y,this->max.z};

  this->center.x = (this->max.x + this->min.x)/2;
  this->center.y = (this->max.y + this->min.y)/2;
  this->center.z = (this->max.z + this->min.z)/2;

  this->width = this->max.x - this->min.x;
  if(this->width < this->max.y - this->min.y) this->width = this->max.y - this->min.y;
  if(this->width < this->max.z - this->min.z) this->width = this->max.z - this->min.z;

  this->width = ceil(this->width);
  if(((int)this->width) % 2) this->width++;
  this->width += 6.0f;
  this->max = this->center + (this->width/2);
  this->min = this->center - (this->width/2);

  if (local_debug){
      printf("\nmin = %f,%f,%f\n",this->min.x,this->min.y,this->min.z);
      printf("max = %f,%f,%f\n",this->max.x,this->max.y,this->max.z);
      printf("bounding box width = %f\n", this->width);
      printf("center = %f,%f,%f\n",this->center.x,this->center.y,this->center.z);
      printf("number of points = %lu\n\n", this->points->size());
    }

  this->depth = depth;
  if(this->depth > 10){
    logger.err<<"ERROR this octree currently only supports a depth of 10 at the max";
    exit(-1);
  }

  this->createFinestNodes();
  this->fillInCoarserDepths();
  this->fillNeighborhoods();
  if(createVEF) this->createVEFArrays();
}
ssrlcv::Octree::Octree(ssrlcv::ptr::value<ssrlcv::Unity<float3>> points, float deepestWidth, bool createVEF){
  this->min = {FLT_MAX,FLT_MAX,FLT_MAX};
  this->max = {-FLT_MAX,-FLT_MAX,-FLT_MAX};
  this->normals = nullptr;
  this->nodes = nullptr;
  this->vertices = nullptr;
  this->edges = nullptr;
  this->faces = nullptr;
  this->pointNodeIndex = nullptr;
  this->nodeDepthIndex = nullptr;
  this->vertexDepthIndex = nullptr;
  this->edgeDepthIndex = nullptr;
  this->faceDepthIndex = nullptr;

  bool local_debug = false;

  this->points = points;
  if(this->points->getMemoryState() == gpu) this->points->transferMemoryTo(cpu);
  ssrlcv::ptr::host<float3> points_host = this->points->host;

  for(int i = 0; i < points->size(); ++i){
    if(this->min.x > points_host.get()[i].x) this->min.x = points_host.get()[i].x;
    else if(this->max.x < points_host.get()[i].x) this->max.x = points_host.get()[i].x;
    if(this->min.y > points_host.get()[i].y) this->min.y = points_host.get()[i].y;
    else if(this->max.y < points_host.get()[i].y) this->max.y = points_host.get()[i].y;
    if(this->min.z > points_host.get()[i].z) this->min.z = points_host.get()[i].z;
    else if(this->max.z < points_host.get()[i].z) this->max.z = points_host.get()[i].z;
  }

  this->max = {this->max.x,this->max.y,this->max.z};

  this->center.x = (this->max.x + this->min.x)/2;
  this->center.y = (this->max.y + this->min.y)/2;
  this->center.z = (this->max.z + this->min.z)/2;

  this->width = this->max.x - this->min.x;
  if(this->width < this->max.y - this->min.y) this->width = this->max.y - this->min.y;
  if(this->width < this->max.z - this->min.z) this->width = this->max.z - this->min.z;

  this->width = ceil(this->width);
  if(((int)this->width) % 2) this->width++;
  this->width += 6.0f;
  this->max = this->center + (this->width/2);
  this->min = this->center - (this->width/2);

  if (local_debug){
    printf("\nmin = %f,%f,%f\n",this->min.x,this->min.y,this->min.z);
    printf("max = %f,%f,%f\n",this->max.x,this->max.y,this->max.z);
    printf("bounding box width = %f\n", this->width);
    printf("center = %f,%f,%f\n",this->center.x,this->center.y,this->center.z);
    printf("number of points = %lu\n\n", this->points->size());
  }

  this->depth = 0;
  float finestWidth = this->width;
  while(finestWidth > deepestWidth){
    finestWidth /= 2.0f;
    ++this->depth;
  }
  if(this->depth >= 10){
    logger.err<<"ERROR this octree currently only supports a depth of 10 at the max";
    exit(-1);
  }

  this->createFinestNodes();
  this->fillInCoarserDepths();
  this->fillNeighborhoods();
  if(createVEF) this->createVEFArrays();
}

__device__ __host__ ssrlcv::Octree::Vertex::Vertex(){
  for(int i = 0; i < 8; ++i){
    this->nodes[i] = -1;
  }
  this->depth = -1;
  this->coord = {0.0f,0.0f,0.0f};
  this->color = {0,0,0};
}

__device__ __host__ ssrlcv::Octree::Edge::Edge(){
  for(int i = 0; i < 4; ++i){
    this->nodes[i] = -1;
  }
  this->depth = -1;
  this->v1 = -1;
  this->v2 = -1;
  this->color = {0,0,0};

}

__device__ __host__ ssrlcv::Octree::Face::Face(){
  this->nodes[0] = -1;
  this->nodes[1] = -1;
  this->depth = -1;
  this->e1 = -1;
  this->e2 = -1;
  this->e3 = -1;
  this->e4 = -1;
  this->color = {0,0,0};

}

__device__ __host__ ssrlcv::Octree::Node::Node(){
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


// =============================================================================================================
//
// Octree Host Methods
//
// =============================================================================================================

//TODO check if using iterator for nodePoint works
//TODO make sure that thrust usage is GPU
void ssrlcv::Octree::createFinestNodes(){
  this->points->transferMemoryTo(both);
  ssrlcv::ptr::host<int> finestNodeKeys(this->points->size());
  ssrlcv::ptr::host<float3> finestNodeCenters(this->points->size());

  ssrlcv::ptr::device<int> finestNodeKeys_device( this->points->size());
  ssrlcv::ptr::device<float3> finestNodeCenters_device( this->points->size());

  dim3 grid = {1,1,1};
  dim3 block = {1,1,1};
  getFlatGridBlock(this->points->size(),grid,block,getNodeKeys);

  getNodeKeys<<<grid,block>>>(this->points->device.get(), finestNodeCenters_device.get(), finestNodeKeys_device.get(), this->center, this->width, this->points->size(), this->depth);
  CudaCheckError();

  thrust::device_ptr<int> kys(finestNodeKeys_device.get());
  thrust::device_ptr<float3> cnts(finestNodeCenters_device.get());

  thrust::device_vector<float3> sortedCnts(this->points->size());

  thrust::counting_iterator<unsigned int> iter(0);
  thrust::device_vector<unsigned int> indices(this->points->size());
  thrust::copy(iter, iter + this->points->size(), indices.begin());

  ssrlcv::ptr::host<unsigned int> nodePointIndex(this->points->size());
  CudaSafeCall(cudaMemcpy(nodePointIndex.get(), thrust::raw_pointer_cast(indices.data()), this->points->size()*sizeof(unsigned int),cudaMemcpyDeviceToHost));

  thrust::sort_by_key(kys, kys + this->points->size(), indices.begin());
  CudaSafeCall(cudaMemcpy(finestNodeKeys.get(), finestNodeKeys_device.get(), this->points->size()*sizeof(int),cudaMemcpyDeviceToHost));


  thrust::device_ptr<float3> pnts(this->points->device.get());
  thrust::device_vector<float3> sortedPnts(this->points->size());
  thrust::gather(indices.begin(), indices.end(), pnts, sortedPnts.begin());
  CudaSafeCall(cudaMemcpy(this->points->host.get(), thrust::raw_pointer_cast(sortedPnts.data()), this->points->size()*sizeof(float3),cudaMemcpyDeviceToHost));

  this->points->clear(gpu);

  thrust::gather(indices.begin(), indices.end(), cnts, sortedCnts.begin());

  CudaSafeCall(cudaMemcpy(finestNodeCenters.get(), thrust::raw_pointer_cast(sortedCnts.data()), this->points->size()*sizeof(float3),cudaMemcpyDeviceToHost));

  if(this->normals != nullptr && this->normals->getMemoryState() != null && this->normals->size() != 0){
    this->normals->transferMemoryTo(both);
    thrust::device_ptr<float3> nmls(this->normals->device.get());
    thrust::device_vector<float3> sortedNmls(this->points->size());
    thrust::gather(indices.begin(), indices.end(), nmls, sortedNmls.begin());
    CudaSafeCall(cudaMemcpy(this->normals->host.get(), thrust::raw_pointer_cast(sortedNmls.data()), this->points->size()*sizeof(float3),cudaMemcpyDeviceToHost));
    this->normals->clear(gpu);
  }

  thrust::pair<int*, unsigned int*> new_end;//the last value of these node arrays
  //there shouldbe better way to do this
  new_end = thrust::unique_by_key(finestNodeKeys.get(), finestNodeKeys.get() + this->points->size(), nodePointIndex.get());

  bool foundFirst = false;
  int numUniqueNodes = 0;
  while(numUniqueNodes != this->points->size()){
    if(finestNodeKeys.get()[numUniqueNodes] == *new_end.first){
      if(foundFirst) break;
      else foundFirst = true;
    }
    numUniqueNodes++;
  }

  ssrlcv::ptr::host<Node> finestNodes(numUniqueNodes);
  for(int i = 0; i < numUniqueNodes; ++i){

    Node currentNode;
    currentNode.key = finestNodeKeys.get()[i];

    currentNode.center = finestNodeCenters.get()[nodePointIndex.get()[i]];

    currentNode.pointIndex = nodePointIndex.get()[i];
    currentNode.depth = this->depth;
    if(i + 1 != numUniqueNodes){
      currentNode.numPoints = nodePointIndex.get()[i + 1] - nodePointIndex.get()[i];
    }
    else{
      currentNode.numPoints = this->points->size() - nodePointIndex.get()[i];

    }

    finestNodes.get()[i] = currentNode;
  }
  this->nodes = ssrlcv::ptr::value<ssrlcv::Unity<Node>>(finestNodes, numUniqueNodes, cpu);
}

void ssrlcv::Octree::fillInCoarserDepths(){
  if(this->nodes == nullptr || this->nodes->getMemoryState() == null){
    logger.err<<"ERROR cannot create coarse depths before finest nodes have been built";
    exit(-1);
  }

  if(this->nodes->getMemoryState() == cpu){
    this->nodes->transferMemoryTo(gpu);
  }
  int numUniqueNodes = this->nodes->size();
  ssrlcv::ptr::device<Node> uniqueNodes_device( this->nodes->size());
  CudaSafeCall(cudaMemcpy(uniqueNodes_device.get(), this->nodes->device.get(), this->nodes->size()*sizeof(Node), cudaMemcpyDeviceToDevice));
  this->nodes = nullptr;
  unsigned int totalNodes = 0;

  Node** nodeArray2D = new Node*[this->depth + 1];


  ssrlcv::ptr::host<unsigned int> nodeDepthIndex_host(this->depth + 1);
  ssrlcv::ptr::device<unsigned int> pointNodeIndex_device( this->points->size());

  ssrlcv::ptr::device<int3> coordPlacementIdentity_device(8);
  CudaSafeCall(cudaMemcpy(coordPlacementIdentity_device.get(),coordPlacementIdentity_host,8*sizeof(int3),cudaMemcpyHostToDevice));
  dim3 grid = {1,1,1};
  dim3 block = {1,1,1};
  for(int d = this->depth; d >= 0; --d){

    ssrlcv::ptr::device<int> nodeNumbers_device( numUniqueNodes );
    ssrlcv::ptr::device<int> nodeAddresses_device( numUniqueNodes );
    //this is just to fill the arrays with 0s

    getFlatGridBlock(numUniqueNodes,grid,block,findAllNodes);
    findAllNodes<<<grid,block>>>(numUniqueNodes, nodeNumbers_device.get(), uniqueNodes_device.get());
    cudaDeviceSynchronize();
    CudaCheckError();

    thrust::device_ptr<int> nN(nodeNumbers_device.get());
    thrust::device_ptr<int> nA(nodeAddresses_device.get());
    thrust::inclusive_scan(nN, nN + numUniqueNodes, nA);

    int numNodesAtDepth = 0;
    CudaSafeCall(cudaMemcpy(&numNodesAtDepth, nodeAddresses_device.get() + (numUniqueNodes - 1), sizeof(int), cudaMemcpyDeviceToHost));

    numNodesAtDepth = (d > 0) ? numNodesAtDepth + 8: 1;

    CudaSafeCall(cudaMalloc((void**)&nodeArray2D[this->depth - d], numNodesAtDepth*sizeof(Node)));

    getFlatGridBlock(numUniqueNodes,grid,block,fillBlankNodeArray);
    fillBlankNodeArray<<<grid,block>>>(uniqueNodes_device.get(), nodeNumbers_device.get(),  nodeAddresses_device.get(), nodeArray2D[this->depth - d], numUniqueNodes, d, this->width);
    CudaCheckError();
    cudaDeviceSynchronize();

    if(this->depth == d){
      getFlatGridBlock(numUniqueNodes,grid,block,fillFinestNodeArrayWithUniques);
      fillFinestNodeArrayWithUniques<<<grid,block>>>(uniqueNodes_device.get(), nodeAddresses_device.get(),nodeArray2D[this->depth - d], numUniqueNodes, pointNodeIndex_device.get());
      CudaCheckError();
    }
    else{
      getFlatGridBlock(numUniqueNodes,grid,block,fillNodeArrayWithUniques);
      fillNodeArrayWithUniques<<<grid,block>>>(uniqueNodes_device.get(), nodeAddresses_device.get(), nodeArray2D[this->depth - d], nodeArray2D[this->depth - d - 1], numUniqueNodes);
      CudaCheckError();
    }

    numUniqueNodes = numNodesAtDepth / 8;

    //get unique nodes at next depth
    if(d > 0){
      CudaSafeCall(cudaMalloc((void**)&uniqueNodes_device, numUniqueNodes*sizeof(Node)));
      getFlatGridBlock(numUniqueNodes,grid,block,generateParentalUniqueNodes);

      generateParentalUniqueNodes<<<grid,block>>>(uniqueNodes_device.get(), nodeArray2D[this->depth - d], numNodesAtDepth, this->width,coordPlacementIdentity_device.get());
      CudaCheckError();
    }
    nodeDepthIndex_host.get()[this->depth - d] = totalNodes;
    totalNodes += numNodesAtDepth;
  }
  ssrlcv::ptr::device<Node> nodeArray_device( totalNodes);
  for(int i = 0; i <= this->depth; ++i){
    if(i < this->depth){
      CudaSafeCall(cudaMemcpy(nodeArray_device.get() + nodeDepthIndex_host.get()[i], nodeArray2D[i], (nodeDepthIndex_host.get()[i+1]-nodeDepthIndex_host.get()[i])*sizeof(Node), cudaMemcpyDeviceToDevice));
    }
    else{
      CudaSafeCall(cudaMemcpy(nodeArray_device.get() + nodeDepthIndex_host.get()[i], nodeArray2D[i], sizeof(Node), cudaMemcpyDeviceToDevice));
    }
    CudaSafeCall(cudaFree(nodeArray2D[i]));
  }
  delete[] nodeArray2D;
  logger.info.printf("TOTAL NODES = %d",totalNodes);
  this->pointNodeIndex = ssrlcv::ptr::value<ssrlcv::Unity<unsigned int>>(pointNodeIndex_device, this->points->size(), gpu);
  this->nodes = ssrlcv::ptr::value<ssrlcv::Unity<Node>>(nodeArray_device, totalNodes, gpu);
  this->nodeDepthIndex = ssrlcv::ptr::value<ssrlcv::Unity<unsigned int>>(nodeDepthIndex_host, this->depth + 1, cpu);
}

void ssrlcv::Octree::fillNeighborhoods(){
  if(this->nodes == nullptr || this->nodes->getMemoryState() == null){
    logger.err<<"ERROR cannot fill neighborhood without nodes";
    exit(-1);
  }
  int* parentLUT = new int[216];
  int* childLUT = new int[216];

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
    					parentLUT[numbLUT*27 + numb] = p[i+l][j+m][k+n];
    					childLUT[numbLUT*27 + numb] = c[i+l][j+m][k+n];
              numb++;
    				}
    			}
        }
        numbLUT++;
      }
    }
  }

  ssrlcv::ptr::device<int> parentLUT_device( 216);
  ssrlcv::ptr::device<int> childLUT_device( 216);
  CudaSafeCall(cudaMemcpy(parentLUT_device.get(), parentLUT, 216*sizeof(int), cudaMemcpyHostToDevice));
  CudaSafeCall(cudaMemcpy(childLUT_device.get(), childLUT, 216*sizeof(int), cudaMemcpyHostToDevice));
  delete[] parentLUT;
  delete[] childLUT;

  dim3 grid = {1,1,1};
  dim3 block = {27,1,1};
  int numNodesAtDepth;
  int depthStartingIndex;
  int childDepthIndex;

  if(this->nodeDepthIndex->getMemoryState() != both || this->nodeDepthIndex->getMemoryState() != cpu){
    this->nodeDepthIndex->transferMemoryTo(cpu);
  }
  ssrlcv::ptr::host<unsigned int> nodeDepthIndex_host = this->nodeDepthIndex->host;
  if(this->nodes->getMemoryState() != both || this->nodes->getMemoryState() != gpu){
    this->nodes->transferMemoryTo(gpu);
  }
  for(int i = this->depth; i >= 0 ; --i){
    numNodesAtDepth = 1;
    depthStartingIndex = nodeDepthIndex_host.get()[i];
    childDepthIndex = -1;
    if(i != this->depth){
      numNodesAtDepth = nodeDepthIndex_host.get()[i + 1] - depthStartingIndex;
    }
    if(i != 0){
      childDepthIndex = nodeDepthIndex_host.get()[i - 1];
    }

    getGrid(numNodesAtDepth,grid);
    computeNeighboringNodes<<<grid, block>>>(this->nodes->device.get(), numNodesAtDepth, depthStartingIndex, parentLUT_device.get(), childLUT_device.get(), childDepthIndex);
    cudaDeviceSynchronize();
    CudaCheckError();
  }

  if(this->nodes->getMemoryState() == both) this->nodes->transferMemoryTo(cpu);
}
void ssrlcv::Octree::computeVertexArray(){
  clock_t cudatimer;
  cudatimer = clock();

  int vertexLUT[8][7]{
    {0,1,3,4,9,10,12},
    {1,2,4,5,10,11,14},
    {3,4,6,7,12,15,16},
    {4,5,7,8,14,16,17},
    {9,10,12,18,19,21,22},
    {10,11,14,19,20,22,23},
    {12,15,16,21,22,24,25},
    {14,16,17,22,23,25,26}
  };
  ssrlcv::ptr::device<int> vertexLUT_device( 56);
  for(int i = 0; i < 8; ++i){
    CudaSafeCall(cudaMemcpy(vertexLUT_device.get() + i*7, &(vertexLUT[i]), 7*sizeof(int), cudaMemcpyHostToDevice));
  }

  int numNodesAtDepth = 0;
  dim3 grid = {1,1,1};
  dim3 block = {1,1,1};
  int* atomicCounter;
  int numVertices = 0;
  CudaSafeCall(cudaMalloc((void**)&atomicCounter, sizeof(int)));
  CudaSafeCall(cudaMemcpy(atomicCounter, &numVertices, sizeof(int), cudaMemcpyHostToDevice));
  Vertex** vertexArray2D_device;
  CudaSafeCall(cudaMalloc((void**)&vertexArray2D_device, (this->depth + 1)*sizeof(Vertex*)));
  Vertex** vertexArray2D = new Vertex*[this->depth + 1];

  if(this->nodeDepthIndex->getMemoryState() != both || this->nodeDepthIndex->getMemoryState() != cpu){
    this->nodeDepthIndex->transferMemoryTo(cpu);
  }
  ssrlcv::ptr::host<unsigned int> nodeDepthIndex_host = this->nodeDepthIndex->host;
  if(this->nodes->getMemoryState() != both || this->nodes->getMemoryState() != gpu){
    this->nodes->transferMemoryTo(gpu);
  }

  ssrlcv::ptr::host<unsigned int> vertexDepthIndex_host(this->depth + 1);

  int prevCount = 0;
  ssrlcv::ptr::device<int3> coordPlacementIdentity_device(8);
  CudaSafeCall(cudaMemcpy(coordPlacementIdentity_device.get(),coordPlacementIdentity_host,8*sizeof(int3),cudaMemcpyHostToDevice));
  for(int i = 0; i <= this->depth; ++i){
    if(i == this->depth){
      numNodesAtDepth = 1;
    }
    else{
      numNodesAtDepth = nodeDepthIndex_host.get()[i + 1] - nodeDepthIndex_host.get()[i];
    }
    //reset previously allocated resources
    block = {8,1,1};
    getGrid(numNodesAtDepth,grid);

    int* ownerInidices = new int[numNodesAtDepth*8];
    for(int v = 0;v < numNodesAtDepth*8; ++v){
      ownerInidices[v] = -1;
    }
    ssrlcv::ptr::device<int> ownerInidices_device(numNodesAtDepth*8);
    ssrlcv::ptr::device<int> vertexPlacement_device(numNodesAtDepth*8);
    CudaSafeCall(cudaMemcpy(ownerInidices_device.get(), ownerInidices, numNodesAtDepth*8*sizeof(int), cudaMemcpyHostToDevice));
    CudaSafeCall(cudaMemcpy(vertexPlacement_device.get(), ownerInidices, numNodesAtDepth*8*sizeof(int), cudaMemcpyHostToDevice));
    delete[] ownerInidices;

    prevCount = numVertices;
    vertexDepthIndex_host.get()[i] = numVertices;

    findVertexOwners<<<grid, block>>>(this->nodes->device.get(), numNodesAtDepth,
      nodeDepthIndex_host.get()[i], vertexLUT_device.get(), atomicCounter, ownerInidices_device.get(), vertexPlacement_device.get());
    CudaCheckError();
    CudaSafeCall(cudaMemcpy(&numVertices, atomicCounter, sizeof(int), cudaMemcpyDeviceToHost));
    if(i == this->depth  && numVertices - prevCount != 8){
      logger.err<<"ERROR GENERATING VERTICES, vertices at depth 0 != 8 -> " + std::to_string(numVertices - prevCount);
      exit(-1);
    }

    CudaSafeCall(cudaMalloc((void**)&vertexArray2D[i], (numVertices - prevCount)*sizeof(Vertex)));
    ssrlcv::ptr::device<int> compactedOwnerArray_device((numVertices - prevCount));
    ssrlcv::ptr::device<int> compactedVertexPlacement_device((numVertices - prevCount));

    thrust::device_ptr<int> arrayToCompact(ownerInidices_device.get());
    thrust::device_ptr<int> arrayOut(compactedOwnerArray_device.get());
    thrust::device_ptr<int> placementToCompact(vertexPlacement_device.get());
    thrust::device_ptr<int> placementOut(compactedVertexPlacement_device.get());

    thrust::copy_if(arrayToCompact, arrayToCompact + (numNodesAtDepth*8), arrayOut, is_not_neg());
    CudaCheckError();
    thrust::copy_if(placementToCompact, placementToCompact + (numNodesAtDepth*8), placementOut, is_not_neg());
    CudaCheckError();

    getFlatGridBlock(numVertices - prevCount,grid,block,fillUniqueVertexArray);

    fillUniqueVertexArray<<<grid, block>>>(this->nodes->device.get(), vertexArray2D[i],
      numVertices - prevCount, vertexDepthIndex_host.get()[i], nodeDepthIndex_host.get()[i], this->depth - i,
      this->width, vertexLUT_device.get(), compactedOwnerArray_device.get(), compactedVertexPlacement_device.get(),coordPlacementIdentity_device.get());
    CudaCheckError();

  }
  ssrlcv::ptr::device<Vertex> vertices_device( numVertices);
  for(int i = 0; i <= this->depth; ++i){
    if(i < this->depth){
      CudaSafeCall(cudaMemcpy(vertices_device.get() + vertexDepthIndex_host.get()[i], vertexArray2D[i], (vertexDepthIndex_host.get()[i+1] - vertexDepthIndex_host.get()[i])*sizeof(Vertex), cudaMemcpyDeviceToDevice));
    }
    else{
      CudaSafeCall(cudaMemcpy(vertices_device.get() + vertexDepthIndex_host.get()[i], vertexArray2D[i], 8*sizeof(Vertex), cudaMemcpyDeviceToDevice));
    }
    CudaSafeCall(cudaFree(vertexArray2D[i]));
  }
  CudaSafeCall(cudaFree(vertexArray2D_device));

  this->vertices = ssrlcv::ptr::value<ssrlcv::Unity<Vertex>>(vertices_device, numVertices, gpu);
  this->vertexDepthIndex = ssrlcv::ptr::value<ssrlcv::Unity<unsigned int>>(vertexDepthIndex_host, this->depth + 1, cpu);

  logger.info.printf("octree createVertexArray took %f seconds.", ((float) clock() - cudatimer)/CLOCKS_PER_SEC);
}
void ssrlcv::Octree::computeEdgeArray(){
  clock_t cudatimer;
  cudatimer = clock();

  int edgeLUT[12][3]{
    {1,4,10},
    {3,4,12},
    {4,5,14},
    {4,7,16},
    {9,10,12},
    {10,11,14},
    {12,15,16},
    {14,16,17},
    {10,19,22},
    {12,21,22},
    {14,22,23},
    {16,22,25}
  };

  ssrlcv::ptr::device<int> edgeLUT_device( 36);
  for(int i = 0; i < 12; ++i){
    CudaSafeCall(cudaMemcpy(edgeLUT_device.get() + i*3, &(edgeLUT[i]), 3*sizeof(int), cudaMemcpyHostToDevice));
  }

  int numNodesAtDepth = 0;
  dim3 grid = {1,1,1};
  dim3 block = {1,1,1};
  int* atomicCounter;
  int numEdges = 0;
  CudaSafeCall(cudaMalloc((void**)&atomicCounter, sizeof(int)));
  CudaSafeCall(cudaMemcpy(atomicCounter, &numEdges, sizeof(int), cudaMemcpyHostToDevice));
  Edge** edgeArray2D_device;
  CudaSafeCall(cudaMalloc((void**)&edgeArray2D_device, (this->depth + 1)*sizeof(Edge*)));
  Edge** edgeArray2D = new Edge*[this->depth + 1];

  if(this->nodeDepthIndex->getMemoryState() != both || this->nodeDepthIndex->getMemoryState() != cpu){
    this->nodeDepthIndex->transferMemoryTo(cpu);
  }
  ssrlcv::ptr::host<unsigned int> nodeDepthIndex_host = this->nodeDepthIndex->host;
  if(this->nodes->getMemoryState() != both || this->nodes->getMemoryState() != gpu){
    this->nodes->transferMemoryTo(gpu);
  }

  ssrlcv::ptr::host<unsigned int> edgeDepthIndex_host(this->depth + 1);

  int prevCount = 0;
  ssrlcv::ptr::device<int2> vertexEdgeIdentity_device(12);
  CudaSafeCall(cudaMemcpy(vertexEdgeIdentity_device.get(),vertexEdgeIdentity_host,12*sizeof(int2),cudaMemcpyHostToDevice));

  for(int i = 0; i <= this->depth; ++i){
    if(i == this->depth){
      numNodesAtDepth = 1;
    }
    else{
      numNodesAtDepth = nodeDepthIndex_host.get()[i + 1] - nodeDepthIndex_host.get()[i];
    }
    //reset previously allocated resources
    block = {12,1,1};
    getGrid(numNodesAtDepth,grid);

    int* ownerInidices = new int[numNodesAtDepth*12];
    for(int v = 0;v < numNodesAtDepth*12; ++v){
      ownerInidices[v] = -1;
    }
    ssrlcv::ptr::device<int> ownerInidices_device(numNodesAtDepth*12);
    ssrlcv::ptr::device<int> edgePlacement_device(numNodesAtDepth*12);
    CudaSafeCall(cudaMemcpy(ownerInidices_device.get(), ownerInidices, numNodesAtDepth*12*sizeof(int), cudaMemcpyHostToDevice));
    CudaSafeCall(cudaMemcpy(edgePlacement_device.get(), ownerInidices, numNodesAtDepth*12*sizeof(int), cudaMemcpyHostToDevice));
    delete[] ownerInidices;

    prevCount = numEdges;
    edgeDepthIndex_host.get()[i] = numEdges;
    findEdgeOwners<<<grid, block>>>(this->nodes->device.get(), numNodesAtDepth,
      nodeDepthIndex_host.get()[i], edgeLUT_device.get(), atomicCounter, ownerInidices_device.get(), edgePlacement_device.get());
    CudaCheckError();
    CudaSafeCall(cudaMemcpy(&numEdges, atomicCounter, sizeof(int), cudaMemcpyDeviceToHost));
    if(i == this->depth  && numEdges - prevCount != 12){
      logger.err<<"ERROR GENERATING EDGES, edges at depth 0 != 12 -> " + std::to_string(numEdges - prevCount);
      exit(-1);
    }

    CudaSafeCall(cudaMalloc((void**)&edgeArray2D[i], (numEdges - prevCount)*sizeof(Edge)));
    ssrlcv::ptr::device<int> compactedOwnerArray_device((numEdges - prevCount));
    ssrlcv::ptr::device<int> compactedEdgePlacement_device((numEdges - prevCount));

    thrust::device_ptr<int> arrayToCompact(ownerInidices_device.get());
    thrust::device_ptr<int> arrayOut(compactedOwnerArray_device.get());
    thrust::device_ptr<int> placementToCompact(edgePlacement_device.get());
    thrust::device_ptr<int> placementOut(compactedEdgePlacement_device.get());

    thrust::copy_if(arrayToCompact, arrayToCompact + (numNodesAtDepth*12), arrayOut, is_not_neg());
    CudaCheckError();
    thrust::copy_if(placementToCompact, placementToCompact + (numNodesAtDepth*12), placementOut, is_not_neg());
    CudaCheckError();

    //reset and allocated resources
    getFlatGridBlock(numEdges - prevCount,grid,block,fillUniqueEdgeArray);

    fillUniqueEdgeArray<<<grid, block>>>(this->nodes->device.get(), edgeArray2D[i],
      numEdges - prevCount, edgeDepthIndex_host.get()[i], nodeDepthIndex_host.get()[i], this->depth - i,
      this->width, edgeLUT_device.get(), compactedOwnerArray_device.get(), compactedEdgePlacement_device.get(),vertexEdgeIdentity_device.get());
    CudaCheckError();

  }
  ssrlcv::ptr::device<Edge> edgeArray_device( numEdges);
  for(int i = 0; i <= this->depth; ++i){
    if(i < this->depth){
      CudaSafeCall(cudaMemcpy(edgeArray_device.get() + edgeDepthIndex_host.get()[i], edgeArray2D[i], (edgeDepthIndex_host.get()[i+1] - edgeDepthIndex_host.get()[i])*sizeof(Edge), cudaMemcpyDeviceToDevice));
    }
    else{
      CudaSafeCall(cudaMemcpy(edgeArray_device.get() + edgeDepthIndex_host.get()[i], edgeArray2D[i], 12*sizeof(Edge), cudaMemcpyDeviceToDevice));
    }
    CudaSafeCall(cudaFree(edgeArray2D[i]));
  }
  CudaSafeCall(cudaFree(edgeArray2D_device));
  this->edges = ssrlcv::ptr::value<ssrlcv::Unity<Edge>>(edgeArray_device, numEdges, gpu);
  this->edgeDepthIndex = ssrlcv::ptr::value<ssrlcv::Unity<unsigned int>>(edgeDepthIndex_host, this->depth + 1, cpu);

  logger.info.printf("octree createEdgeArray took %f seconds.", ((float) clock() - cudatimer)/CLOCKS_PER_SEC);
}
void ssrlcv::Octree::computeFaceArray(){
  clock_t cudatimer;
  cudatimer = clock();

  int faceLUT[6] = {4,10,12,14,16,22};
  ssrlcv::ptr::device<int> faceLUT_device( 6);
  CudaSafeCall(cudaMemcpy(faceLUT_device.get(), &faceLUT, 6*sizeof(int), cudaMemcpyHostToDevice));

  int numNodesAtDepth = 0;
  dim3 grid = {1,1,1};
  dim3 block = {1,1,1};
  int* atomicCounter;
  int numFaces = 0;
  CudaSafeCall(cudaMalloc((void**)&atomicCounter, sizeof(int)));
  CudaSafeCall(cudaMemcpy(atomicCounter, &numFaces, sizeof(int), cudaMemcpyHostToDevice));
  Face** faceArray2D_device;
  CudaSafeCall(cudaMalloc((void**)&faceArray2D_device, (this->depth + 1)*sizeof(Face*)));
  Face** faceArray2D = new Face*[this->depth + 1];

  if(this->nodeDepthIndex->getMemoryState() != both || this->nodeDepthIndex->getMemoryState() != cpu){
    this->nodeDepthIndex->transferMemoryTo(cpu);
  }
  ssrlcv::ptr::host<unsigned int> nodeDepthIndex_host = this->nodeDepthIndex->host;
  if(this->nodes->getMemoryState() != both || this->nodes->getMemoryState() != gpu){
    this->nodes->transferMemoryTo(gpu);
  }

  ssrlcv::ptr::host<unsigned int> faceDepthIndex_host(this->depth + 1);

  int prevCount = 0;
  ssrlcv::ptr::device<int4> edgeFaceIdentity_device(6);
  CudaSafeCall(cudaMemcpy(edgeFaceIdentity_device.get(),edgeFaceIdentity_host,6*sizeof(int4),cudaMemcpyHostToDevice));
  for(int i = 0; i <= this->depth; ++i){
    if(i == this->depth){
      numNodesAtDepth = 1;
    }
    else{
      numNodesAtDepth = nodeDepthIndex_host.get()[i + 1] - nodeDepthIndex_host.get()[i];
    }
    //reset previously allocated resources
    block = {6,1,1};
    getGrid(numNodesAtDepth,grid);

    int* ownerInidices = new int[numNodesAtDepth*6];
    for(int v = 0;v < numNodesAtDepth*6; ++v){
      ownerInidices[v] = -1;
    }
    ssrlcv::ptr::device<int> ownerInidices_device(numNodesAtDepth*6);
    ssrlcv::ptr::device<int> facePlacement_device(numNodesAtDepth*6);
    CudaSafeCall(cudaMemcpy(ownerInidices_device.get(), ownerInidices, numNodesAtDepth*6*sizeof(int), cudaMemcpyHostToDevice));
    CudaSafeCall(cudaMemcpy(facePlacement_device.get(), ownerInidices, numNodesAtDepth*6*sizeof(int), cudaMemcpyHostToDevice));
    delete[] ownerInidices;

    prevCount = numFaces;
    faceDepthIndex_host.get()[i] = numFaces;
    findFaceOwners<<<grid, block>>>((Octree::Node*) this->nodes->device.get(), numNodesAtDepth,
      nodeDepthIndex_host.get()[i], faceLUT_device.get(), atomicCounter, ownerInidices_device.get(), facePlacement_device.get());
    CudaCheckError();
    CudaSafeCall(cudaMemcpy(&numFaces, atomicCounter, sizeof(int), cudaMemcpyDeviceToHost));
    if(i == this->depth  && numFaces - prevCount != 6){
      logger.err<<"ERROR GENERATING FACES, faces at depth 0 != 6 -> " + std::to_string(numFaces - prevCount);
      exit(-1);
    }

    CudaSafeCall(cudaMalloc((void**)&faceArray2D[i], (numFaces - prevCount)*sizeof(Face)));
    ssrlcv::ptr::device<int> compactedOwnerArray_device((numFaces - prevCount));
    ssrlcv::ptr::device<int> compactedFacePlacement_device((numFaces - prevCount));

    thrust::device_ptr<int> arrayToCompact(ownerInidices_device.get());
    thrust::device_ptr<int> arrayOut(compactedOwnerArray_device.get());
    thrust::device_ptr<int> placementToCompact(facePlacement_device.get());
    thrust::device_ptr<int> placementOut(compactedFacePlacement_device.get());

    thrust::copy_if(arrayToCompact, arrayToCompact + (numNodesAtDepth*6), arrayOut, is_not_neg());
    CudaCheckError();
    thrust::copy_if(placementToCompact, placementToCompact + (numNodesAtDepth*6), placementOut, is_not_neg());
    CudaCheckError();

    //reset and allocated resources
    getFlatGridBlock(numFaces - prevCount,grid,block,fillUniqueFaceArray);

    fillUniqueFaceArray<<<grid, block>>>(this->nodes->device.get(), faceArray2D[i],
      numFaces - prevCount, numFaces, nodeDepthIndex_host.get()[i], this->depth - i,
      this->width, faceLUT_device.get(), compactedOwnerArray_device.get(), compactedFacePlacement_device.get(),edgeFaceIdentity_device.get());
    CudaCheckError();

  }
  ssrlcv::ptr::device<Face> faceArray_device( numFaces);
  for(int i = 0; i <= this->depth; ++i){
    if(i < this->depth){
      CudaSafeCall(cudaMemcpy(faceArray_device.get() + faceDepthIndex_host.get()[i], faceArray2D[i], (faceDepthIndex_host.get()[i+1] - faceDepthIndex_host.get()[i])*sizeof(Face), cudaMemcpyDeviceToDevice));
    }
    else{
      CudaSafeCall(cudaMemcpy(faceArray_device.get() + faceDepthIndex_host.get()[i], faceArray2D[i], 6*sizeof(Face), cudaMemcpyDeviceToDevice));
    }
    CudaSafeCall(cudaFree(faceArray2D[i]));
  }
  CudaSafeCall(cudaFree(faceArray2D_device));
  this->faces = ssrlcv::ptr::value<ssrlcv::Unity<Face>>(faceArray_device, numFaces, gpu);
  this->faceDepthIndex = ssrlcv::ptr::value<ssrlcv::Unity<unsigned int>>(faceDepthIndex_host, this->depth + 1, cpu);

  logger.info.printf("octree createFaceArray took %f seconds.", ((float) clock() - cudatimer)/CLOCKS_PER_SEC);
}

void ssrlcv::Octree::checkForGeneralNodeErrors(){
  MemoryState origin;
  if(this->nodes != nullptr && this->nodes->getMemoryState() != null && this->nodes->size() != 0){
    origin = this->nodes->getMemoryState();
    this->nodes->transferMemoryTo(cpu);
  }
  else{
    logger.err<<"ERROR cannot check nodes for errors without nodes";
    exit(-1);
  }
  clock_t cudatimer;
  cudatimer = clock();
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
  ssrlcv::ptr::host<Node> nodes_host = this->nodes->host;

  for(int i = 0; i < this->nodes->size(); ++i){
    if(nodes_host.get()[i].depth < 0){
      numFuckedNodes++;
    }
    if(nodes_host.get()[i].parent != -1 && nodes_host.get()[i].depth == nodes_host.get()[nodes_host.get()[i].parent].depth){
      ++numSiblingParents;
    }
    if(nodes_host.get()[i].parent == -1 && nodes_host.get()[i].depth != 0){
      orphanNodes++;
    }
    int checkForChildren = 0;
    for(int c = 0; c < 8 && nodes_host.get()[i].depth < 10; ++c){
      if(nodes_host.get()[i].children[c] == -1){
        checkForChildren++;
      }
      if(nodes_host.get()[i].children[c] == 0 && nodes_host.get()[i].depth != this->depth - 1){
        logger.err<<"NODE THAT IS NOT AT 2nd TO FINEST DEPTH HAS A CHILD WITH INDEX 0 IN FINEST DEPTH";
      }
    }
    if(nodes_host.get()[i].numPoints == 0){
      noPoints++;
    }
    if(nodes_host.get()[i].depth != 0 && nodes_host.get()[nodes_host.get()[i].parent].children[nodes_host.get()[i].key&((1<<3)-1)] == -1){

      nodesThatCantFindChildren++;
    }
    if(checkForChildren == 8){
      nodesWithOutChildren++;
    }
    if(nodes_host.get()[i].depth == 0){
      if(nodes_host.get()[i].numPoints != this->points->size()){
        logger.err<<"DEPTH 0 DOES NOT CONTAIN ALL POINTS " + std::to_string(nodes_host.get()[i].numPoints) + "," + std::to_string(this->points->size());
        exit(-1);
      }
    }
    childNeighbor = false;
    parentNeighbor = false;
    for(int n = 0; n < 27; ++n){
      if(nodes_host.get()[i].neighbors[n] != -1){
        if(nodes_host.get()[i].depth < nodes_host.get()[nodes_host.get()[i].neighbors[n]].depth){
          childNeighbor = true;
        }
        else if(nodes_host.get()[i].depth > nodes_host.get()[nodes_host.get()[i].neighbors[n]].depth){
          parentNeighbor = true;
        }
      }
    }
    for(int v = 0; v < 8; ++v){
      if(nodes_host.get()[i].vertices[v] == -1){
        ++numVerticesMissing;
      }
    }
    for(int e = 0; e < 12; ++e){
      if(nodes_host.get()[i].edges[e] == -1){
        ++numEgesMissing;
      }
    }
    for(int f = 0; f < 6; ++f){
      if(nodes_host.get()[i].faces[f] == -1){
        ++numFacesMissing;
      }
    }
    if(parentNeighbor){
      ++numParentNeighbors;
    }
    if(childNeighbor){
      ++numChildNeighbors;
    }
    if((nodes_host.get()[i].center.x < this->min.x ||
    nodes_host.get()[i].center.y < this->min.y ||
    nodes_host.get()[i].center.z < this->min.z ||
    nodes_host.get()[i].center.x > this->max.x ||
    nodes_host.get()[i].center.y > this->max.y ||
    nodes_host.get()[i].center.z > this->max.z )){
      ++numCentersOUTSIDE;
    }
  }
  if(numCentersOUTSIDE > 0){
    logger.err.printf("ERROR %d centers outside of bounding box",numCentersOUTSIDE);
    error = true;
  }
  if(numSiblingParents > 0){
    logger.err.printf("ERROR %d NODES THINK THEIR PARENT IS IN THE SAME DEPTH AS THEMSELVES", numSiblingParents);
    error = true;
  }
  if(numChildNeighbors > 0){
    logger.err.printf("ERROR %d NODES WITH SIBLINGS AT HIGHER DEPTH", numChildNeighbors);
    error = true;
  }
  if(numParentNeighbors > 0){
    logger.err.printf("ERROR %d NODES WITH SIBLINGS AT LOWER DEPTH", numParentNeighbors);
    error = true;
  }
  if(numFuckedNodes > 0){
    logger.err.printf("ERROR IN %d NODE CONCATENATION OR GENERATION", numFuckedNodes);
    error = true;
  }
  if(orphanNodes > 0){
    logger.err.printf("ERROR THERE ARE %d ORPHAN NODES", orphanNodes);
    error = true;
  }
  if(nodesThatCantFindChildren > 0){
    logger.err.printf("ERROR %d PARENTS WITHOUT CHILDREN", nodesThatCantFindChildren);
    error = true;
  }
  if(numVerticesMissing > 0){
    logger.err.printf("ERROR %d VERTICES MISSING", numVerticesMissing);
    error = true;
  }
  if(numEgesMissing > 0){
    logger.err.printf("ERROR %d EDGES MISSING", numEgesMissing);
    error = true;
  }
  if(numFacesMissing > 0){
    logger.err.printf("ERROR %d FACES MISSING", numFacesMissing);
    error = true;
  }
  if(error) exit(-1);
  else logger.err.printf("NO ERRORS DETECTED IN OCTREE");
  logger.err<<"NODES WITHOUT POINTS = " + std::to_string(noPoints);
  logger.err<<"NODES WITH POINTS = " + std::to_string(this->nodes->size() - noPoints);

  logger.info.printf("octree checkForErrors took %f seconds.", ((float) clock() - cudatimer)/CLOCKS_PER_SEC);
  this->nodes->transferMemoryTo(origin);
}

// RUN THIS
void ssrlcv::Octree::createVEFArrays(){
  this->computeVertexArray();
  this->computeEdgeArray();
  this->computeFaceArray();
}

// =============================================================================================================
//
// Filtering Methods
//
// =============================================================================================================

/**
 * calculates the average distance from a point to N of it's neighbors
 * @param the numer of neighbors to consider
 * @return averages a unity float of the average distance for n neighbors per point
 */
ssrlcv::ptr::value<ssrlcv::Unity<float>> ssrlcv::Octree::averageNeighboorDistances(int n){

  // the number of neightbors to check
  int* d_num;
  CudaSafeCall(cudaMalloc((void**) &d_num,sizeof(int)));
  CudaSafeCall(cudaMemcpy(d_num,&n,sizeof(int),cudaMemcpyHostToDevice));

  ssrlcv::ptr::value<ssrlcv::Unity<float>> averages = ssrlcv::ptr::value<ssrlcv::Unity<float>>(nullptr,this->points->size(),gpu);

  this->points->transferMemoryTo(gpu);
  this->pointNodeIndex->transferMemoryTo(gpu);
  this->nodes->transferMemoryTo(gpu);

  dim3 grid = {1,1,1};
  dim3 block = {1,1,1};
  void (*fp)(int *, unsigned long, float3*, unsigned int *, Octree::Node *, float *) = &computeAverageNeighboorDistances;
  getFlatGridBlock(this->points->size(),grid,block,fp);

  computeAverageNeighboorDistances<<<grid,block>>>(d_num, this->pointNodeIndex->size(),this->points->device.get(), this->pointNodeIndex->device.get(), this->nodes->device.get(), averages->device.get());

  cudaDeviceSynchronize();
  CudaCheckError();

  // transfer the poitns back to the CPU
  this->points->transferMemoryTo(cpu);
  this->pointNodeIndex->transferMemoryTo(cpu);
  this->nodes->transferMemoryTo(cpu);

  averages->setFore(gpu);
  averages->transferMemoryTo(cpu);
  averages->clear(gpu);

  // clean up memory
  cudaFree(d_num);

  return averages;
}

/**
 * calculates the average distance from a point to N of it's neighbors and finds that average for all points
 * @param the numer of neighbors to consider
 * @return average the average distance from any given point to it's neighbors
 */
float ssrlcv::Octree::averageNeighboorDistance(int n){
  // the number of neightbors to check
  int* d_num;
  CudaSafeCall(cudaMalloc((void**) &d_num,sizeof(int)));
  CudaSafeCall(cudaMemcpy(d_num,&n,sizeof(int),cudaMemcpyHostToDevice));

  ssrlcv::ptr::value<ssrlcv::Unity<float>> average = ssrlcv::ptr::value<ssrlcv::Unity<float>>(nullptr,1,gpu);

  this->points->transferMemoryTo(gpu);
  this->pointNodeIndex->transferMemoryTo(gpu);
  this->nodes->transferMemoryTo(gpu);

  dim3 grid = {1,1,1};
  dim3 block = {1,1,1};
  void (*fp)(int *, unsigned long, float3*, unsigned int *, Octree::Node *, float *) = &computeAverageNeighboorDistance;
  getFlatGridBlock(this->points->size(),grid,block,fp);

  computeAverageNeighboorDistance<<<grid,block>>>(d_num, this->pointNodeIndex->size(),this->points->device.get(), this->pointNodeIndex->device.get(), this->nodes->device.get(), average->device.get());

  cudaDeviceSynchronize();
  CudaCheckError();

  // transfer the poitns back to the CPU
  this->points->transferMemoryTo(cpu);
  this->pointNodeIndex->transferMemoryTo(cpu);
  this->nodes->transferMemoryTo(cpu);

  average->setFore(gpu);
  average->transferMemoryTo(cpu);
  average->clear(gpu);

  // clean up memory
  cudaFree(d_num);

  return average->host.get()[0];
}

/**
 * finds the point indexes that should be removed this is done for each point.
 * returns a NULL index if the point does not need to be removed, returns the actual index if it does need to be
 * @param cutoff is the minimum average distance from a point to N of its neightbors
 * @param n the number of neighbor points to consider
 * @return points returns unity pf float3 points that are densly packed enough within the cutoff
 */
ssrlcv::ptr::value<ssrlcv::Unity<float3>> ssrlcv::Octree::removeLowDensityPoints(float cutoff, int n){
  // the number of neightbors to check
  int* d_num;
  CudaSafeCall(cudaMalloc((void**) &d_num,sizeof(int)));
  CudaSafeCall(cudaMemcpy(d_num,&n,sizeof(int),cudaMemcpyHostToDevice));

  float* d_cutoff;
  CudaSafeCall(cudaMalloc((void**) &d_cutoff,sizeof(float)));
  CudaSafeCall(cudaMemcpy(d_cutoff,&cutoff,sizeof(float),cudaMemcpyHostToDevice));

  ssrlcv::ptr::value<ssrlcv::Unity<float3>> indexes = ssrlcv::ptr::value<ssrlcv::Unity<float3>>(nullptr,this->points->size(),gpu);

  this->points->transferMemoryTo(gpu);
  this->pointNodeIndex->transferMemoryTo(gpu);
  this->nodes->transferMemoryTo(gpu);

  dim3 grid = {1,1,1};
  dim3 block = {1,1,1};
  void (*fp)(int *, float*, unsigned long, float3*, unsigned int *, Octree::Node *, float3 *) = &getGoodDensePoints;
  getFlatGridBlock(this->points->size(),grid,block,fp);

  getGoodDensePoints<<<grid,block>>>(d_num, d_cutoff, this->pointNodeIndex->size(),this->points->device.get(), this->pointNodeIndex->device.get(), this->nodes->device.get(), indexes->device.get());

  cudaDeviceSynchronize();
  CudaCheckError();

  // transfer the poitns back to the CPU
  this->points->transferMemoryTo(cpu);
  this->pointNodeIndex->transferMemoryTo(cpu);
  this->nodes->transferMemoryTo(cpu);

  indexes->setFore(gpu);
  indexes->transferMemoryTo(cpu);
  indexes->clear(gpu);

  // clean up memory
  cudaFree(d_num);
  cudaFree(d_cutoff);

  return indexes;
}

// =============================================================================================================
//
// Normal Caclulation Methods
//
// =============================================================================================================

/**
* Computes normals for the points within the input points cloud
* @param minNeighForNorms the minimum number of neighbors to consider for normal calculation
* @param maxNeighbors the maximum number of neightbors to consider for normal calculation
*/
void ssrlcv::Octree::computeNormals(int minNeighForNorms, int maxNeighbors){

  // enable local_debug to have local print statements
  bool local_debug = false;

  if (local_debug) std::cout << std::endl;
  clock_t cudatimer;
  cudatimer = clock();

  int numNodesAtDepth = 0;
  int currentNumNeighbors = 0;
  int currentNeighborIndex = -1;
  int maxPointsInOneNode = 0;
  int minPossibleNeighbors = std::numeric_limits<int>::max();
  int nodeDepthIndex = 0;
  int currentDepth = 0;
  MemoryState node_origin = this->nodes->getMemoryState();
  MemoryState nodeDepthIndex_origin = this->nodeDepthIndex->getMemoryState();

  if(node_origin != both || node_origin != cpu){
    this->nodes->transferMemoryTo(cpu);
  }
  ssrlcv::ptr::host<Node> nodes_host = this->nodes->host;
  if(nodeDepthIndex_origin != both || nodeDepthIndex_origin != cpu){
    this->nodes->transferMemoryTo(cpu);
  }
  ssrlcv::ptr::host<unsigned int> nodeDepthIndex_host = this->nodeDepthIndex->host;

  for(int i = 0; i < this->nodes->size(); ++i){
    currentNumNeighbors = 0;
    if(minPossibleNeighbors < minNeighForNorms){
      ++currentDepth;
      i = nodeDepthIndex_host.get()[currentDepth];
      minPossibleNeighbors = std::numeric_limits<int>::max();
      maxPointsInOneNode = 0;
    }
    if(this->depth - nodes_host.get()[i].depth != currentDepth){
      if(minPossibleNeighbors >= minNeighForNorms) break;
      ++currentDepth;
    }
    if(maxPointsInOneNode < nodes_host.get()[i].numPoints){
      maxPointsInOneNode = nodes_host.get()[i].numPoints;
    }
    for(int n = 0; n < 27; ++n){
      currentNeighborIndex = nodes_host.get()[i].neighbors[n];
      if(currentNeighborIndex != -1) currentNumNeighbors += nodes_host.get()[currentNeighborIndex].numPoints;
    }
    if(minPossibleNeighbors > currentNumNeighbors){
      minPossibleNeighbors = currentNumNeighbors;
    }
  }

  nodeDepthIndex = nodeDepthIndex_host.get()[currentDepth];
  numNodesAtDepth = nodeDepthIndex_host.get()[currentDepth + 1] - nodeDepthIndex;
  if (local_debug){
    std::cout<<"Continuing with depth "<<this->depth - currentDepth<<" nodes starting at "<<nodeDepthIndex<<" with "<<numNodesAtDepth<<" nodes"<<std::endl;
    std::cout<<"Continuing with "<<minPossibleNeighbors<<" minPossibleNeighbors"<<std::endl;
    std::cout<<"Continuing with "<<maxNeighbors<<" maxNeighborsAllowed"<<std::endl;
    std::cout<<"Continuing with "<<maxPointsInOneNode<<" maxPointsInOneNode"<<std::endl;
  }

  uint size = this->points->size()*maxNeighbors*3;
  int* numRealNeighbors = new int[this->points->size()];

  for(int i = 0; i < this->points->size(); ++i){
    numRealNeighbors[i] = 0;
  }
  int* temp = new int[size/3];
  for(int i = 0; i < size/3; ++i){
    temp[i] = -1;
  }

  ssrlcv::ptr::device<int> numRealNeighbors_device( this->points->size());
  ssrlcv::ptr::device<float> cMatrix_device( size);
  ssrlcv::ptr::device<int> neighborIndices_device( (size/3));
  CudaSafeCall(cudaMemcpy(numRealNeighbors_device.get(), numRealNeighbors, this->points->size()*sizeof(int), cudaMemcpyHostToDevice));
  CudaSafeCall(cudaMemcpy(neighborIndices_device.get(), temp, (size/3)*sizeof(int), cudaMemcpyHostToDevice));
  delete[] temp;

  dim3 grid = {1,1,1};
  dim3 block = {1,1,1};
  getGridAndBlock(numNodesAtDepth,grid,maxPointsInOneNode,block,findNormalNeighborsAndComputeCMatrix);

  MemoryState points_origin = this->points->getMemoryState();
  if(points_origin == cpu){
    this->points->transferMemoryTo(gpu);
  }
  logger.warn <<"WARNING: float3 normal calculation is currently performed only on sphere centers";
  findNormalNeighborsAndComputeCMatrix<<<grid, block>>>(numNodesAtDepth, nodeDepthIndex, maxNeighbors,
    this->nodes->device.get(), this->points->device.get(), cMatrix_device.get(), neighborIndices_device.get(), numRealNeighbors_device.get());

  CudaCheckError();
  CudaSafeCall(cudaMemcpy(numRealNeighbors, numRealNeighbors_device.get(), this->points->size()*sizeof(int), cudaMemcpyDeviceToHost));

  this->normals = ssrlcv::ptr::value<ssrlcv::Unity<float3>>(nullptr, this->points->size(), gpu);

  cusolverDnHandle_t cusolverH = nullptr;
  cublasHandle_t cublasH = nullptr;
  cublasStatus_t cublas_status = CUBLAS_STATUS_SUCCESS;
  cusolverStatus_t cusolver_status = CUSOLVER_STATUS_SUCCESS;

  float *d_A, *d_S, *d_U, *d_VT, *d_work, *d_rwork;
  int* devInfo;

  cusolver_status = cusolverDnCreate(&cusolverH);
  assert(CUSOLVER_STATUS_SUCCESS == cusolver_status);

  cublas_status = cublasCreate(&cublasH);
  assert(CUBLAS_STATUS_SUCCESS == cublas_status);

  int n = 3;
  int m = 0;
  int lwork = 0;

  //TODO changed this to gesvdjBatched (this will enable doing multiple svds at once)
  for(int p = 0; p < this->points->size(); ++p){
    m = numRealNeighbors[p];
    lwork = 0;
    if(m < minNeighForNorms){
      logger.err<<"ERROR...point does not have enough neighbors...increase min neighbors";
      exit(-1);
    }
    CudaSafeCall(cudaMalloc((void**)&d_A, m*n*sizeof(float)));
    CudaSafeCall(cudaMalloc((void**)&d_S, n*sizeof(float)));
    CudaSafeCall(cudaMalloc((void**)&d_U, m*m*sizeof(float)));
    CudaSafeCall(cudaMalloc((void**)&d_VT, n*n*sizeof(float)));
    CudaSafeCall(cudaMalloc((void**)&devInfo, sizeof(int)));
    CudaSafeCall(cudaMemcpy(d_A, cMatrix_device.get() + (p*maxNeighbors*n), m*n*sizeof(float), cudaMemcpyDeviceToDevice));
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
    setNormal<<<1, 1>>>(p, d_VT, this->normals->device.get());
    CudaCheckError();

    CudaSafeCall(cudaFree(d_A));
    CudaSafeCall(cudaFree(d_S));
    CudaSafeCall(cudaFree(d_U));
    CudaSafeCall(cudaFree(d_VT));
    CudaSafeCall(cudaFree(d_work));
    CudaSafeCall(cudaFree(devInfo));
  }
  if (local_debug) std::cout<<"normals have been estimated by use of svd"<<std::endl;
  if (cublasH) cublasDestroy(cublasH);
  if (cusolverH) cusolverDnDestroy(cusolverH);

  delete[] numRealNeighbors;

  if(points_origin != gpu) this->normals->setMemoryState(points_origin);
  if(this->points->getMemoryState() != points_origin) this->points->setMemoryState(points_origin);
  this->nodes->transferMemoryTo(node_origin);
  this->nodeDepthIndex->transferMemoryTo(nodeDepthIndex_origin);

  if (local_debug) printf("octree computeNormals took %f seconds.\n\n", ((float) clock() - cudatimer)/CLOCKS_PER_SEC);
}

/**
* Computes normals for the points within the input points cloud
* @param minNeighForNorms the minimum number of neighbors to consider for normal calculation
* @param maxNeighbors the maximum number of neightbors to consider for normal calculation
* @param numCameras the total number of cameras which resulted in the point cloud
* @param cameraPositions the x,y,z coordinates of the cameras
*/
void ssrlcv::Octree::computeNormals(int minNeighForNorms, int maxNeighbors, unsigned int numCameras, float3* cameraPositions){

  // enable local_debug to have local print statements
  bool local_debug = false;

  if (local_debug) std::cout << std::endl;
  clock_t cudatimer;
  cudatimer = clock();

  int numNodesAtDepth = 0;
  int currentNumNeighbors = 0;
  int currentNeighborIndex = -1;
  int maxPointsInOneNode = 0;
  int minPossibleNeighbors = std::numeric_limits<int>::max();
  int nodeDepthIndex = 0;
  int currentDepth = 0;
  MemoryState node_origin = this->nodes->getMemoryState();
  MemoryState nodeDepthIndex_origin = this->nodeDepthIndex->getMemoryState();

  if(node_origin != both || node_origin != cpu){
    this->nodes->transferMemoryTo(cpu);
  }
  ssrlcv::ptr::host<Node> nodes_host = this->nodes->host;
  if(nodeDepthIndex_origin != both || nodeDepthIndex_origin != cpu){
    this->nodes->transferMemoryTo(cpu);
  }
  ssrlcv::ptr::host<unsigned int> nodeDepthIndex_host = this->nodeDepthIndex->host;

  for(int i = 0; i < this->nodes->size(); ++i){
    currentNumNeighbors = 0;
    if(minPossibleNeighbors < minNeighForNorms){
      ++currentDepth;
      i = nodeDepthIndex_host.get()[currentDepth];
      minPossibleNeighbors = std::numeric_limits<int>::max();
      maxPointsInOneNode = 0;
    }
    if(this->depth - nodes_host.get()[i].depth != currentDepth){
      if(minPossibleNeighbors >= minNeighForNorms) break;
      ++currentDepth;
    }
    if(maxPointsInOneNode < nodes_host.get()[i].numPoints){
      maxPointsInOneNode = nodes_host.get()[i].numPoints;
    }
    for(int n = 0; n < 27; ++n){
      currentNeighborIndex = nodes_host.get()[i].neighbors[n];
      if(currentNeighborIndex != -1) currentNumNeighbors += nodes_host.get()[currentNeighborIndex].numPoints;
    }
    if(minPossibleNeighbors > currentNumNeighbors){
      minPossibleNeighbors = currentNumNeighbors;
    }
  }

  nodeDepthIndex = nodeDepthIndex_host.get()[currentDepth];
  numNodesAtDepth = nodeDepthIndex_host.get()[currentDepth + 1] - nodeDepthIndex;
  if (local_debug){
    std::cout<<"Continuing with depth "<<this->depth - currentDepth<<" nodes starting at "<<nodeDepthIndex<<" with "<<numNodesAtDepth<<" nodes"<<std::endl;
    std::cout<<"Continuing with "<<minPossibleNeighbors<<" minPossibleNeighbors"<<std::endl;
    std::cout<<"Continuing with "<<maxNeighbors<<" maxNeighborsAllowed"<<std::endl;
    std::cout<<"Continuing with "<<maxPointsInOneNode<<" maxPointsInOneNode"<<std::endl;
  }

  uint size = this->points->size()*maxNeighbors*3;
  int* numRealNeighbors = new int[this->points->size()];

  for(int i = 0; i < this->points->size(); ++i){
    numRealNeighbors[i] = 0;
  }
  int* temp = new int[size/3];
  for(int i = 0; i < size/3; ++i){
    temp[i] = -1;
  }

  ssrlcv::ptr::device<int> numRealNeighbors_device( this->points->size());
  ssrlcv::ptr::device<float> cMatrix_device( size);
  ssrlcv::ptr::device<int> neighborIndices_device( (size/3));
  CudaSafeCall(cudaMemcpy(numRealNeighbors_device.get(), numRealNeighbors, this->points->size()*sizeof(int), cudaMemcpyHostToDevice));
  CudaSafeCall(cudaMemcpy(neighborIndices_device.get(), temp, (size/3)*sizeof(int), cudaMemcpyHostToDevice));
  delete[] temp;

  dim3 grid = {1,1,1};
  dim3 block = {1,1,1};
  getGridAndBlock(numNodesAtDepth,grid,maxPointsInOneNode,block,findNormalNeighborsAndComputeCMatrix);

  MemoryState points_origin = this->points->getMemoryState();
  if(points_origin == cpu){
    this->points->transferMemoryTo(gpu);
  }
  findNormalNeighborsAndComputeCMatrix<<<grid, block>>>(numNodesAtDepth, nodeDepthIndex, maxNeighbors,
    this->nodes->device.get(), this->points->device.get(), cMatrix_device.get(), neighborIndices_device.get(), numRealNeighbors_device.get());

  CudaCheckError();
  CudaSafeCall(cudaMemcpy(numRealNeighbors, numRealNeighbors_device.get(), this->points->size()*sizeof(int), cudaMemcpyDeviceToHost));

  this->normals = ssrlcv::ptr::value<ssrlcv::Unity<float3>>(nullptr, this->points->size(), gpu);

  cusolverDnHandle_t cusolverH = nullptr;
  cublasHandle_t cublasH = nullptr;
  cublasStatus_t cublas_status = CUBLAS_STATUS_SUCCESS;
  cusolverStatus_t cusolver_status = CUSOLVER_STATUS_SUCCESS;

  float *d_A, *d_S, *d_U, *d_VT, *d_work, *d_rwork;
  int* devInfo;

  cusolver_status = cusolverDnCreate(&cusolverH);
  assert(CUSOLVER_STATUS_SUCCESS == cusolver_status);

  cublas_status = cublasCreate(&cublasH);
  assert(CUBLAS_STATUS_SUCCESS == cublas_status);

  int n = 3;
  int m = 0;
  int lwork = 0;

  //TODO changed this to gesvdjBatched (this will enable doing multiple svds at once)
  for(int p = 0; p < this->points->size(); ++p){
    m = numRealNeighbors[p];
    lwork = 0;
    if(m < minNeighForNorms){
      logger.err<<"ERROR...point does not have enough neighbors...increase min neighbors";
      exit(-1);
    }
    CudaSafeCall(cudaMalloc((void**)&d_A, m*n*sizeof(float)));
    CudaSafeCall(cudaMalloc((void**)&d_S, n*sizeof(float)));
    CudaSafeCall(cudaMalloc((void**)&d_U, m*m*sizeof(float)));
    CudaSafeCall(cudaMalloc((void**)&d_VT, n*n*sizeof(float)));
    CudaSafeCall(cudaMalloc((void**)&devInfo, sizeof(int)));
    CudaSafeCall(cudaMemcpy(d_A, cMatrix_device.get() + (p*maxNeighbors*n), m*n*sizeof(float), cudaMemcpyDeviceToDevice));
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
    setNormal<<<1, 1>>>(p, d_VT, this->normals->device.get());
    CudaCheckError();

    CudaSafeCall(cudaFree(d_A));
    CudaSafeCall(cudaFree(d_S));
    CudaSafeCall(cudaFree(d_U));
    CudaSafeCall(cudaFree(d_VT));
    CudaSafeCall(cudaFree(d_work));
    CudaSafeCall(cudaFree(devInfo));
  }
  if (local_debug) std::cout<<"normals have been estimated by use of svd"<<std::endl;
  if (cublasH) cublasDestroy(cublasH);
  if (cusolverH) cusolverDnDestroy(cusolverH);

  delete[] numRealNeighbors;

  //TODO add ambiguity test here

  ssrlcv::ptr::value<ssrlcv::Unity<bool>> ambiguity = ssrlcv::ptr::value<ssrlcv::Unity<bool>>(nullptr,this->points->size(),gpu);

  ssrlcv::ptr::device<float3> cameraPositions_device( numCameras);
  CudaSafeCall(cudaMemcpy(cameraPositions_device.get(), cameraPositions, numCameras*sizeof(float3), cudaMemcpyHostToDevice));

  getGridAndBlock(this->points->size(),grid,numCameras,block,checkForAmbiguity);

  checkForAmbiguity<<<grid, block>>>(this->points->size(), numCameras, this->normals->device.get(),
    this->points->device.get(), cameraPositions_device.get(), ambiguity->device.get());
  CudaCheckError();

  ambiguity->transferMemoryTo(cpu);

  int numAmbiguous = 0;
  for(int i = 0; i < this->points->size(); ++i){
    if(ambiguity->host.get()[i]) ++numAmbiguous;
  }
  if (local_debug) std::cout<<"numAmbiguous = "<<numAmbiguous<<"/"<<ambiguity->size()<<std::endl;

  if(this->points->getMemoryState() != points_origin) this->points->setMemoryState(points_origin);

  reorient<<<grid, block>>>(numNodesAtDepth, nodeDepthIndex, this->nodes->device.get(), numRealNeighbors_device.get(), maxNeighbors, this->normals->device.get(),
    neighborIndices_device.get(), ambiguity->device.get());
  CudaCheckError();

  if(points_origin != gpu) this->normals->setMemoryState(points_origin);
  this->nodes->transferMemoryTo(node_origin);
  this->nodeDepthIndex->transferMemoryTo(nodeDepthIndex_origin);

  if (local_debug) printf("octree computeNormals took %f seconds.\n\n", ((float) clock() - cudatimer)/CLOCKS_PER_SEC);
}

/**
* Computes the average normal of the input points. This is only useful if you can make a "planar" assumption about
* the input points, that is the points are mostly aligned along a plane. For use in reconstructon filtering should occur
* before one considers using this method
* @param minNeighForNorms the minimum number of neighbors to consider for normal calculation
* @param maxNeighbors the maximum number of neightbors to consider for normal calculation
* @param numCameras the total number of cameras which resulted in the point cloud
* @param cameraPositions the x,y,z coordinates of the cameras
* @returns norm the average normal vector
*/
ssrlcv::ptr::value<ssrlcv::Unity<float3>> ssrlcv::Octree::computeAverageNormal(int minNeighForNorms, int maxNeighbors, unsigned int numCameras, float3* cameraPositions){

  bool local_debug = false;

  // call the fucntion that already does this!
  computeNormals(minNeighForNorms, maxNeighbors, numCameras, cameraPositions);

  ssrlcv::ptr::value<ssrlcv::Unity<float3>> average = ssrlcv::ptr::value<ssrlcv::Unity<float3>>(nullptr,1,ssrlcv::gpu);
  this->normals->transferMemoryTo(gpu);

  // call kernel
  dim3 grid = {1,1,1};
  dim3 block = {1,1,1};
  void (*fp)(float3*, unsigned long, float3*) = &calculateCloudAverageNormal;
  getFlatGridBlock(this->normals->size(),grid,block,fp);

  calculateCloudAverageNormal<<<grid,block>>>(average->device.get(), this->normals->size(), this->normals->device.get());

  cudaDeviceSynchronize();
  CudaCheckError();

  this->normals->transferMemoryTo(cpu);
  this->normals->clear(gpu);
  this->writeNormalPLY();
  average->transferMemoryTo(cpu);
  average->clear(gpu);

  if (local_debug) std::cout << average->host.get()[0].x << ", " << average->host.get()[0].y << ", " << average->host.get()[0].z << std::endl;

  // normalize the average
  float mag = sqrtf((average->host.get()[0].x * average->host.get()[0].x) + (average->host.get()[0].y * average->host.get()[0].y) + (average->host.get()[0].z * average->host.get()[0].z));
  average->host.get()[0] /= mag;

  if (local_debug) {
    std::cout << "mag: " << mag << std::endl;
    std::cout << average->host.get()[0].x << ", " << average->host.get()[0].y << ", " << average->host.get()[0].z << std::endl;
  }

  return average;
}

// =============================================================================================================
//
// PLY writers
//
// =============================================================================================================


void ssrlcv::Octree::writeVertexPLY(bool binary){
  MemoryState origin = this->vertices->getMemoryState();
  if(origin != cpu && this->vertices->getFore() != cpu){
    this->vertices->transferMemoryTo(cpu);
  }
  std::vector<float3> vertices_data;
  for(int i = 0; i < this->vertices->size(); ++i){
    vertices_data.push_back(this->vertices->host.get()[i].coord);
  }
  this->vertices->transferMemoryTo(origin);

  tinyply::PlyFile ply;
  ply.get_comments().push_back("SSRL Test");
  ply.add_properties_to_element("vertex",{"x","y","z"},tinyply::Type::FLOAT32, vertices_data.size(), reinterpret_cast<uint8_t*>(vertices_data.data()), tinyply::Type::INVALID, 0);

  std::filebuf fb_binary;
  if(this->name.length() == 0) this->name = this->pathToFile.substr(this->pathToFile.find_last_of("/") + 1,this->pathToFile.length() - 4);
  std::string newFile = "out/" + this->name + "_vertices_" + std::to_string(this->depth)+ ".ply";

  if(binary){
    fb_binary.open(newFile, std::ios::out | std::ios::binary);
    std::ostream outstream_binary(&fb_binary);
    if (outstream_binary.fail()) throw std::runtime_error("failed to open " + newFile);
    ply.write(outstream_binary, true);
  }
  else{
    std::filebuf fb_ascii;
  	fb_ascii.open(newFile, std::ios::out);
  	std::ostream outstream_ascii(&fb_ascii);
  	if (outstream_ascii.fail()) throw std::runtime_error("failed to open " + newFile);
    ply.write(outstream_ascii, false);
  }


}
void ssrlcv::Octree::writeEdgePLY(bool binary){
  std::vector<int2> edges_data;
  MemoryState origin[2] = {this->vertices->getMemoryState(), this->edges->getMemoryState()};
  if(origin[0] != cpu && this->vertices->getFore() != cpu){
    this->vertices->transferMemoryTo(cpu);
  }
  std::vector<float3> vertices_data;
  for(int i = 0; i < this->vertices->size(); ++i){
    vertices_data.push_back(this->vertices->host.get()[i].coord);
  }
  this->vertices->transferMemoryTo(origin[0]);
  if(origin[1] != cpu && this->edges->getFore() != cpu){
    this->edges->transferMemoryTo(cpu);
  }
  for(int i = 0; i < this->edges->size(); ++i){
    edges_data.push_back({this->edges->host.get()[i].v1,this->edges->host.get()[i].v2});
  }
  this->edges->transferMemoryTo(origin[1]);

  tinyply::PlyFile ply;
  ply.get_comments().push_back("SSRL Test");
  ply.add_properties_to_element("vertex",{"x","y","z"},tinyply::Type::FLOAT32, vertices_data.size(), reinterpret_cast<uint8_t*>(vertices_data.data()), tinyply::Type::INVALID, 0);
  ply.add_properties_to_element("edge",{"vertex1","vertex2"},tinyply::Type::INT32, edges_data.size(), reinterpret_cast<uint8_t*>(edges_data.data()), tinyply::Type::INVALID, 0);

  std::filebuf fb_binary;
  if(this->name.length() == 0) this->name = this->pathToFile.substr(this->pathToFile.find_last_of("/") + 1,this->pathToFile.length() - 4);
  std::string newFile = "out/" + this->name + "_edges_" + std::to_string(this->depth)+ ".ply";

  if(binary){
    fb_binary.open(newFile, std::ios::out | std::ios::binary);
    std::ostream outstream_binary(&fb_binary);
    if (outstream_binary.fail()) throw std::runtime_error("failed to open " + newFile);
    ply.write(outstream_binary, true);
  }
  else{
    std::filebuf fb_ascii;
  	fb_ascii.open(newFile, std::ios::out);
  	std::ostream outstream_ascii(&fb_ascii);
  	if (outstream_ascii.fail()) throw std::runtime_error("failed to open " + newFile);
    ply.write(outstream_ascii, false);
  }


}
void ssrlcv::Octree::writeCenterPLY(bool binary){

  MemoryState origin = this->nodes->getMemoryState();
  if(origin != cpu && this->nodes->getFore() != cpu){
    this->nodes->transferMemoryTo(cpu);
  }

  std::vector<float3> vertices_data;
  for(int i = 0; i < this->nodes->size(); ++i){
    vertices_data.push_back(this->nodes->host.get()[i].center);
  }
  this->nodes->transferMemoryTo(origin);

  tinyply::PlyFile ply;
  ply.get_comments().push_back("SSRL Test");
  ply.add_properties_to_element("vertex",{"x","y","z"},tinyply::Type::FLOAT32, vertices_data.size(), reinterpret_cast<uint8_t*>(vertices_data.data()), tinyply::Type::INVALID, 0);

  std::filebuf fb_binary;
  if(this->name.length() == 0) this->name = this->pathToFile.substr(this->pathToFile.find_last_of("/") + 1,this->pathToFile.length() - 4);
  std::string newFile = "out/" + this->name + "_nodecenter_" + std::to_string(this->depth)+ ".ply";

  if(binary){
    fb_binary.open(newFile, std::ios::out | std::ios::binary);
    std::ostream outstream_binary(&fb_binary);
    if (outstream_binary.fail()) throw std::runtime_error("failed to open " + newFile);
    ply.write(outstream_binary, true);
  }
  else{
    std::filebuf fb_ascii;
  	fb_ascii.open(newFile, std::ios::out);
  	std::ostream outstream_ascii(&fb_ascii);
  	if (outstream_ascii.fail()) throw std::runtime_error("failed to open " + newFile);
    ply.write(outstream_ascii, false);
  }



}
void ssrlcv::Octree::writeNormalPLY(bool binary){

  MemoryState origin[2] = {this->points->getMemoryState(), this->normals->getMemoryState()};
  if(origin[0] != cpu && this->points->getFore() != cpu){
    this->points->transferMemoryTo(cpu);
  }
  if(origin[1] != cpu && this->normals->getFore() != cpu){
    this->normals->transferMemoryTo(cpu);
  }

  tinyply::PlyFile ply;
  ply.get_comments().push_back("SSRL Test");
  ply.add_properties_to_element("vertex",{"x","y","z"},tinyply::Type::FLOAT32, this->points->size(), reinterpret_cast<uint8_t*>(this->points->host.get()), tinyply::Type::INVALID, 0);
  ply.add_properties_to_element("vertex",{"nx","ny","nz"},tinyply::Type::FLOAT32, this->normals->size(), reinterpret_cast<uint8_t*>(this->normals->host.get()), tinyply::Type::INVALID, 0);

  std::filebuf fb_binary;
  if(this->name.length() == 0) this->name = this->pathToFile.substr(this->pathToFile.find_last_of("/") + 1,this->pathToFile.length() - 4);
  std::string newFile = "out/" + this->name + "_normals_" + std::to_string(this->depth)+ ".ply";
  if(binary){
    fb_binary.open(newFile, std::ios::out | std::ios::binary);
    std::ostream outstream_binary(&fb_binary);
    if (outstream_binary.fail()) throw std::runtime_error("failed to open " + newFile);
    ply.write(outstream_binary, true);
  }
  else{
    std::filebuf fb_ascii;
  	fb_ascii.open(newFile, std::ios::out);
  	std::ostream outstream_ascii(&fb_ascii);
  	if (outstream_ascii.fail()) throw std::runtime_error("failed to open " + newFile);
    ply.write(outstream_ascii, false);
  }
  this->points->transferMemoryTo(origin[0]);
  this->normals->transferMemoryTo(origin[1]);
}
void ssrlcv::Octree::writeDepthPLY(int d, bool binary){
  MemoryState origin[5] = {
    this->vertices->getMemoryState(),
    this->edges->getMemoryState(),
    this->vertexDepthIndex->getMemoryState(),
    this->edgeDepthIndex->getMemoryState(),
    this->faceDepthIndex->getMemoryState()
  };
  if(origin[0] != cpu && this->vertices->getFore() != cpu){
    this->vertices->transferMemoryTo(cpu);
  }
  std::vector<float3> vertices_data;
  for(int i = 0; i < this->vertices->size(); ++i){
    vertices_data.push_back(this->vertices->host.get()[i].coord);
  }
  this->vertices->transferMemoryTo(origin[0]);
  std::vector<int2> edges_data;
  if(origin[1] != cpu && this->edges->getFore() != cpu){
    this->edges->transferMemoryTo(cpu);
  }
  for(int i = 0; i < this->edges->size(); ++i){
    edges_data.push_back({this->edges->host.get()[i].v1,this->edges->host.get()[i].v2});
  }
  this->edges->transferMemoryTo(origin[1]);

  if(origin[2] != cpu && this->vertexDepthIndex->getFore() != cpu){
    this->vertexDepthIndex->transferMemoryTo(cpu);
  }
  if(origin[3] != cpu && this->edgeDepthIndex->getFore() != cpu){
    this->edgeDepthIndex->transferMemoryTo(cpu);
  }
  if(origin[4] != cpu && this->faceDepthIndex->getFore() != cpu){
    this->faceDepthIndex->transferMemoryTo(cpu);
  }

  if(d < 0 || d > this->depth){
    logger.err<<"ERROR DEPTH FOR WRITEDEPTHPLY IS OUT OF BOUNDS";
    exit(-1);
  }
  if(this->name.length() == 0) this->name = this->pathToFile.substr(this->pathToFile.find_last_of("/") + 1,this->pathToFile.length() - 4);
  std::string newFile = "out/" + this->name +
  "_nodes_" + std::to_string(d) + "_"+ std::to_string(this->depth)+ ".ply";

  tinyply::PlyFile ply;
  std::vector<uint4> faces_data;
  int verticesToWrite = (depth != 0) ? this->vertexDepthIndex->host.get()[this->depth - d + 1] : this->vertices->size();
  int facesToWrite = (depth != 0) ? this->faceDepthIndex->host.get()[this->depth - d + 1] - this->faceDepthIndex->host.get()[this->depth - d] : 6;
  int faceStartingIndex = this->faceDepthIndex->host.get()[this->depth - d];
  this->vertexDepthIndex->transferMemoryTo(origin[2]);
  this->edgeDepthIndex->transferMemoryTo(origin[3]);
  this->faceDepthIndex->transferMemoryTo(origin[4]);
  for(int i = 0; i < verticesToWrite; ++i){
    vertices_data.push_back(this->vertices->host.get()[i].coord);
  }
  for(int i = faceStartingIndex; i < facesToWrite + faceStartingIndex; ++i){
    faces_data.push_back({
      (unsigned int) edges_data[this->faces->host.get()[i].e1].x,
      (unsigned int) edges_data[this->faces->host.get()[i].e1].y,
      (unsigned int) edges_data[this->faces->host.get()[i].e4].y,
      (unsigned int) edges_data[this->faces->host.get()[i].e4].x
    });
  }
  ply.get_comments().push_back("SSRL Test");
  ply.add_properties_to_element("vertex",{"x","y","z"},tinyply::Type::FLOAT32, vertices_data.size(), reinterpret_cast<uint8_t*>(vertices_data.data()), tinyply::Type::INVALID, 0);
  ply.add_properties_to_element("face",{"vertex_indices"},tinyply::Type::INT32, faces_data.size(), reinterpret_cast<uint8_t*>(faces_data.data()), tinyply::Type::INT32, 4);

  std::filebuf fb_binary;

  if(binary){
    fb_binary.open(newFile, std::ios::out | std::ios::binary);
    std::ostream outstream_binary(&fb_binary);
    if (outstream_binary.fail()) throw std::runtime_error("failed to open " + newFile);
    ply.write(outstream_binary, true);
  }
  else{
    std::filebuf fb_ascii;
    fb_ascii.open(newFile, std::ios::out);
    std::ostream outstream_ascii(&fb_ascii);
    if (outstream_ascii.fail()) throw std::runtime_error("failed to open " + newFile);
    ply.write(outstream_ascii, false);
  }

}


// =============================================================================================================
//
// Device Kernels
//
// =============================================================================================================

__device__ __host__ float3 ssrlcv::getVoidCenter(const Octree::Node &node, int neighbor){
  float3 center = node.center;
  center.x += node.width*((neighbor/9) - 1);
  center.y += node.width*(((neighbor%9)/3) - 1);
  center.z += node.width*((neighbor%3) - 1);
  return center;
}
__device__ __host__ float3 ssrlcv::getVoidChildCenter(const Octree::Node &parent, int child){
  float3 center = parent.center;
  float dist = parent.width/4;
  if((1 << 2) & child) center.x += dist;
  if((1 << 1) & child) center.y += dist;
  if(1 & child) center.z += dist;
  return center;
}

__device__ __forceinline__ int ssrlcv::floatToOrderedInt(float floatVal){
 int intVal = __float_as_int( floatVal );
 return (intVal >= 0 ) ? intVal : intVal ^ 0x7FFFFFFF;
}
__device__ __forceinline__ float ssrlcv::orderedIntToFloat(int intVal){
 return __int_as_float( (intVal >= 0) ? intVal : intVal ^ 0x7FFFFFFF);
}

__global__ void ssrlcv::getNodeKeys(float3* points, float3* nodeCenters, int* nodeKeys, float3 c, float W, int numPoints, int D){
  int globalID = blockIdx.x * blockDim.x + threadIdx.x;
  if(globalID < numPoints){
    float x = points[globalID].x;
    float y = points[globalID].y;
    float z = points[globalID].z;
    int key = 0;
    int depth = 1;
    W /= 2.0f;
    float3 center = c;
    while(depth <= D){
      W /= 2.0f;
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
      if(z < center.z){
        key <<= 1;
        center.z -= W;
      }
      else{
        key = (key << 1) + 1;
        center.z += W;
      }
      depth++;
    }
    nodeKeys[globalID] = key;
    nodeCenters[globalID] = center;
    //printf("%f,%f,%f\n",c.x,c.y,c.z);

  }
}

//createFinalNodeArray kernels
__global__ void ssrlcv::findAllNodes(int numUniqueNodes, int* nodeNumbers, Octree::Node* uniqueNodes){
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
__global__ void ssrlcv::fillBlankNodeArray(Octree::Node* uniqueNodes, int* nodeNumbers, int* nodeAddresses, Octree::Node* outputNodeArray, int numUniqueNodes, int currentDepth, float totalWidth){
  int globalID = blockIdx.x * blockDim.x + threadIdx.x;
  int address = 0;
  if(currentDepth != 0 && globalID < numUniqueNodes && (globalID == 0 || nodeNumbers[globalID] == 8)){
    int siblingKey = uniqueNodes[globalID].key;
    uchar3 color = uniqueNodes[globalID].color;
    siblingKey &= 0xfffffff8;//will clear last 3 bits
    for(int i = 0; i < 8; ++i){
      address = nodeAddresses[globalID] + i;
      outputNodeArray[address] = Octree::Node();
      outputNodeArray[address].color = color;
      outputNodeArray[address].depth = currentDepth;
      outputNodeArray[address].key = siblingKey + i;
    }
  }
  else if(currentDepth == 0){
    address = nodeAddresses[0];
    outputNodeArray[address] = Octree::Node();
    outputNodeArray[address].color = {255,255,255};
    outputNodeArray[address].depth = currentDepth;
    outputNodeArray[address].key = 0;
  }
}
__global__ void ssrlcv::fillFinestNodeArrayWithUniques(Octree::Node* uniqueNodes, int* nodeAddresses, Octree::Node* outputNodeArray, int numUniqueNodes, unsigned int* pointNodeIndex){
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
__global__ void ssrlcv::fillNodeArrayWithUniques(Octree::Node* uniqueNodes, int* nodeAddresses, Octree::Node* outputNodeArray, Octree::Node* childNodeArray,int numUniqueNodes){
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
__global__ void ssrlcv::generateParentalUniqueNodes(Octree::Node* uniqueNodes, Octree::Node* nodeArrayD, int numNodesAtDepth, float totalWidth, const int3* __restrict__ coordPlacementIdentity){
  int numUniqueNodesAtParentDepth = numNodesAtDepth / 8;
  int globalID = blockIdx.x * blockDim.x + threadIdx.x;
  int nodeArrayIndex = globalID*8;
  if(globalID < numUniqueNodesAtParentDepth){
    uniqueNodes[globalID] = Octree::Node();//may not be necessary
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
__global__ void ssrlcv::computeNeighboringNodes(Octree::Node* nodeArray, int numNodes, int depthIndex,int* parentLUT, int* childLUT, int childDepthIndex){
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

// calculates the average normal
__global__ void ssrlcv::calculateCloudAverageNormal(float3* average, unsigned long num, float3* normals){
  // get ready to do the stuff local memory space
  // this will later be added back to a global memory space
  __shared__ float3 localSum;
  if (threadIdx.x == 0) localSum = {0.0f,0.0f,0.0f};
  __syncthreads();

  unsigned long globalID = (blockIdx.y* gridDim.x+ blockIdx.x)*blockDim.x + threadIdx.x;
  if (globalID > (num-1)) return;

  // if the normal has been normalized then we don't need  to do this part
  float mag = sqrtf((normals[globalID].x * normals[globalID].x) + (normals[globalID].y * normals[globalID].y) + (normals[globalID].z * normals[globalID].z));
  float3 localValue;

  localValue.x = normals[globalID].x / (num * mag);
  localValue.y = normals[globalID].y / (num * mag);
  localValue.z = normals[globalID].z / (num * mag);

  atomicAdd(&localSum.x,localValue.x);
  atomicAdd(&localSum.y,localValue.y);
  atomicAdd(&localSum.z,localValue.z);
  __syncthreads();
  if (!threadIdx.x) {
    atomicAdd(&average[0].x,localSum.x);
    atomicAdd(&average[0].y,localSum.y);
    atomicAdd(&average[0].z,localSum.z);
  }
}

// calculates average distance to N neighbors
__global__ void ssrlcv::computeAverageNeighboorDistances(int* n, unsigned long numpoints, float3* points, unsigned int* pointNodeIndex, Octree::Node* nodes, float* averages){
  unsigned long globalID = (blockIdx.y* gridDim.x+ blockIdx.x)*blockDim.x + threadIdx.x;
  if (globalID > (numpoints-1)) return;

   // the point we want neighbors for!
   float3 P = points[globalID];
   // the index of the node in the octree containing the point
   unsigned int nodeIndex = pointNodeIndex[globalID];
   // the node containing point P from which we can search for neighbors
   Octree::Node node = nodes[nodeIndex];

   int neighborsFound = 0;
   float sum = 0.0f;

   // and nodes at this leaf depth are considered close neighbors
   for (unsigned long i = node.pointIndex; i < (node.pointIndex + node.numPoints); i++){
     if (i != globalID){ // if not self
       float3 A = points[i];
       float dist = sqrtf((P.x - A.x)*(P.x - A.x) + (P.y - A.y)*(P.y - A.y) + (P.z - A.z)*(P.z - A.z));
       sum += dist;
       neighborsFound++;
     }
     if (neighborsFound == *n){ // then we have found the max
       averages[globalID] = (sum / (float) neighborsFound);
       return;
     }
   }

   // now search neighbor nodes for neighbor points
   for (unsigned long i = 0; i < 27; i++){
     if (node.neighbors[i] > 0 && i != 13){
      // see if there is a point in that
      node = nodes[i];
      // and nodes at this leaf depth are considered close neighbors
      for (unsigned long j = node.pointIndex; j < (node.pointIndex + node.numPoints); j++){
        if (j != globalID){ // if not self
          float3 A = points[j];
          float dist = sqrtf((P.x - A.x)*(P.x - A.x) + (P.y - A.y)*(P.y - A.y) + (P.z - A.z)*(P.z - A.z));
          sum += dist;
          neighborsFound++;
        }
        if (neighborsFound == *n){ // then we have found the max
          averages[globalID] = (sum / (float) neighborsFound);
          return;
        }
      }
     }
     if (neighborsFound == *n){ // then we have found the max
       averages[globalID] = (sum / (float) neighborsFound);
       return;
     }
   }

   // otherwise you tried your best (sort of, traversal can garuntee n is reached), just return what you have!
   if (!neighborsFound) {
     averages[globalID] = 100000.0f; // just give it something bad
   } else {
     averages[globalID] = (sum / (float) neighborsFound); // TEMP fill
   }
}

// calculates average distance to N neighbors
__global__ void ssrlcv::computeAverageNeighboorDistance(int* n, unsigned long numpoints, float3* points, unsigned int* pointNodeIndex, Octree::Node* nodes, float* average){

  __shared__ float localSum;
  if (threadIdx.x == 0) localSum = 0;
  __syncthreads();

  unsigned long globalID = (blockIdx.y* gridDim.x+ blockIdx.x)*blockDim.x + threadIdx.x;
  if (globalID > (numpoints-1)) return;

   // the point we want neighbors for!
   float3 P = points[globalID];
   // the index of the node in the octree containing the point
   unsigned int nodeIndex = pointNodeIndex[globalID];
   // the node containing point P from which we can search for neighbors
   Octree::Node node = nodes[nodeIndex];

   int neighborsFound = 0;
   float sum = 0.0f;
   float local_avg = 0.0f;

   // and nodes at this leaf depth are considered close neighbors
   for (unsigned long i = node.pointIndex; i < (node.pointIndex + node.numPoints); i++){
     if (i != globalID){ // if not self
       float3 A = points[i];
       float dist = sqrtf((P.x - A.x)*(P.x - A.x) + (P.y - A.y)*(P.y - A.y) + (P.z - A.z)*(P.z - A.z));
       sum += dist;
       neighborsFound++;
     }
     if (neighborsFound == *n){ // then we have found the max
       // averages[globalID] = (sum / (float) neighborsFound);
       break;
     }
   }

   // now search neighbor nodes for neighbor points
   for (unsigned long i = 0; i < 27; i++){
     if (node.neighbors[i] > 0 && i != 13){
      // see if there is a point in that
      node = nodes[i];
      // and nodes at this leaf depth are considered close neighbors
      for (unsigned long j = node.pointIndex; j < (node.pointIndex + node.numPoints); j++){
        if (j != globalID){ // if not self
          float3 A = points[j];
          float dist = sqrtf((P.x - A.x)*(P.x - A.x) + (P.y - A.y)*(P.y - A.y) + (P.z - A.z)*(P.z - A.z));
          sum += dist;
          neighborsFound++;
        }
        if (neighborsFound == *n){ // then we have found the max
          // averages[globalID] = (sum / (float) neighborsFound);
          break;
        }
      }
     }
     if (neighborsFound == *n){ // then we have found the max
       // averages[globalID] = (sum / (float) neighborsFound);
       break;
     }
   }

   // otherwise you tried your best (sort of, traversal can garuntee n is reached), just return what you have!
   if (!neighborsFound) {
     local_avg = 0.0f; // just give it something bad
     //printf("WARNING: average neightbor error is being skewed due to lack of neighbors at certain depth\n");
   } else {
     local_avg = (sum / (float) neighborsFound) / numpoints; // TEMP fill
   }

   atomicAdd(&localSum,local_avg);
   __syncthreads();
   if (!threadIdx.x) atomicAdd(average,localSum);
}

// gives back good indexes
__global__ void ssrlcv::getGoodDensePoints(int* n, float* cutoff, unsigned long numpoints, float3* points, unsigned int* pointNodeIndex, Octree::Node* nodes, float3* goodPoints){
  unsigned long globalID = (blockIdx.y* gridDim.x+ blockIdx.x)*blockDim.x + threadIdx.x;
  if (globalID > (numpoints-1)) return;

   // the point we want neighbors for!
   float3 P = points[globalID];
   // the index of the node in the octree containing the point
   unsigned int nodeIndex = pointNodeIndex[globalID];
   // the node containing point P from which we can search for neighbors
   Octree::Node node = nodes[nodeIndex];

   int neighborsFound = 0;
   float sum = 0.0f;
   float local_avg = 0.0f;

   // and nodes at this leaf depth are considered close neighbors
   for (unsigned long i = node.pointIndex; i < (node.pointIndex + node.numPoints); i++){
     if (i != globalID){ // if not self
       float3 A = points[i];
       float dist = sqrtf((P.x - A.x)*(P.x - A.x) + (P.y - A.y)*(P.y - A.y) + (P.z - A.z)*(P.z - A.z));
       sum += dist;
       neighborsFound++;
     }
     if (neighborsFound == *n){ // then we have found the max
       // averages[globalID] = (sum / (float) neighborsFound);
       break;
     }
   }

   // now search neighbor nodes for neighbor points
   for (unsigned long i = 0; i < 27; i++){
     if (node.neighbors[i] > 0 && i != 13){
      // see if there is a point in that
      node = nodes[i];
      // and nodes at this leaf depth are considered close neighbors
      for (unsigned long j = node.pointIndex; j < (node.pointIndex + node.numPoints); j++){
        if (j != globalID){ // if not self
          float3 A = points[j];
          float dist = sqrtf((P.x - A.x)*(P.x - A.x) + (P.y - A.y)*(P.y - A.y) + (P.z - A.z)*(P.z - A.z));
          sum += dist;
          neighborsFound++;
        }
        if (neighborsFound == *n){ // then we have found the max
          // averages[globalID] = (sum / (float) neighborsFound);
          break;
        }
      }
     }
     if (neighborsFound == *n){ // then we have found the max
       // averages[globalID] = (sum / (float) neighborsFound);
       break;
     }
   }

   // otherwise you tried your best (sort of, traversal can garuntee n is reached), just return what you have!
   if (!neighborsFound) { // this case means it's prob an outlier
     float nanboi = 0.0f / 0.0f;
     goodPoints[globalID] = {nanboi,nanboi,nanboi};
     return;
     //printf("WARNING: average neightbor error is being skewed due to lack of neighbors at certain depth\n");
   }

   local_avg = (sum / (float) neighborsFound); // TEMP fill

   if (local_avg > *cutoff){
     float nanboi = 0.0f / 0.0f;
     goodPoints[globalID] = {nanboi,nanboi,nanboi}; // bad
     return;
   } else {
     goodPoints[globalID] = P;
   }

}

__global__ void ssrlcv::findNormalNeighborsAndComputeCMatrix(int numNodesAtDepth, int depthIndex, int maxNeighbors, Octree::Node* nodeArray, float3* points, float* cMatrix, int* neighborIndices, int* numNeighbors){
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
__global__ void ssrlcv::transposeFloatMatrix(int m, int n, float* matrix){
  int globalID = blockIdx.x * blockDim.x + threadIdx.x;
  if(globalID < m*n){
    int2 regLocation = {globalID/n,globalID%n};
    float regPastValue = matrix[globalID];
    __syncthreads();
    matrix[regLocation.y*m + regLocation.x] = regPastValue;
  }
}
__global__ void ssrlcv::setNormal(int currentPoint, float* vt, float3* normals){
  normals[currentPoint] = {vt[2],vt[5],vt[8]};
}
__global__ void ssrlcv::checkForAmbiguity(int numPoints, int numCameras, float3* normals, float3* points, float3* cameraPositions, bool* ambiguous){
  int blockID = blockIdx.y * gridDim.x + blockIdx.x;
  if(blockID < numPoints){
    float3 coord = points[blockID];
    float3 norm = normals[blockID];
    float3 regCameraPosition = {0.0f,0.0f,0.0f};
    __shared__ int directionCheck;
    directionCheck = 0;
    __syncthreads();
    for(int c = threadIdx.x; c < numCameras && !ambiguous[blockID]; c+=blockDim.x){
      regCameraPosition = cameraPositions[c];
      coord = {regCameraPosition.x - coord.x,regCameraPosition.y - coord.y,regCameraPosition.z - coord.z};
      if(dotProduct(coord,norm) < 0.0f) atomicSub(&directionCheck,1);
      else atomicAdd(&directionCheck,1);
    }
    __syncthreads();
    if(!threadIdx.x) return;
    if(abs(directionCheck) == numCameras){
      if(directionCheck > 0){//normal vector from camera to point and normal should be in opposite directions
        normals[blockID] = {-1.0f*norm.x,-1.0f*norm.y,-1.0f*norm.z};
      }
      ambiguous[blockID] = false;
    }
    else{
      ambiguous[blockID] = true;
    }
  }
}
__global__ void ssrlcv::reorient(int numNodesAtDepth, int depthIndex, Octree::Node* nodeArray, int* numNeighbors, int maxNeighbors, float3* normals, int* neighborIndices, bool* ambiguous){
  int blockID = blockIdx.y * gridDim.x + blockIdx.x;
  bool local_debug = false;
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
    if(numPointsInNode == 0) return;
    while(ambiguityExists){
      ambiguityExists = false;
      __syncthreads();
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
          ambiguous[regPointIndex + threadID] = false;
          if(directionCounter.x < directionCounter.y){
            normals[regPointIndex + threadID] = {-1.0f*norm.x,-1.0f*norm.y,-1.0f*norm.z};
          }
        }
        else{
          ambiguityExists = true;
        }
      }
      __syncthreads();
    }
    if(!threadIdx.x && local_debug) printf("%d reoriented\n",numPointsInNode);
  }
}

//vertex edge and face array kernels
__global__ void ssrlcv::findVertexOwners(Octree::Node* nodeArray, int numNodes, int depthIndex, int* vertexLUT, int* numVertices, int* ownerInidices, int* vertexPlacement){
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
__global__ void ssrlcv::fillUniqueVertexArray(Octree::Node* nodeArray, Octree::Vertex* vertexArray, int numVertices, int vertexIndex,int depthIndex, int depth, float width, int* vertexLUT, int* ownerInidices, int* vertexPlacement, const int3* __restrict__ coordPlacementIdentity){
  int globalID = blockIdx.x * blockDim.x + threadIdx.x;
  if(globalID < numVertices){

    int ownerNodeIndex = ownerInidices[globalID];
    int ownedIndex = vertexPlacement[globalID];

    nodeArray[ownerNodeIndex].vertices[ownedIndex] = globalID + vertexIndex;

    float depthHalfWidth = width/powf(2, depth + 1);
    Octree::Vertex vertex = Octree::Vertex();
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
__global__ void ssrlcv::findEdgeOwners(Octree::Node* nodeArray, int numNodes, int depthIndex, int* edgeLUT, int* numEdges, int* ownerInidices, int* edgePlacement){
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
__global__ void ssrlcv::fillUniqueEdgeArray(Octree::Node* nodeArray, Octree::Edge* edgeArray, int numEdges, int edgeIndex, int depthIndex, int depth, float width, int* edgeLUT, int* ownerInidices, int* edgePlacement, const int2* __restrict__ vertexEdgeIdentity){
  int globalID = blockIdx.x * blockDim.x + threadIdx.x;
  if(globalID < numEdges){
    int ownerNodeIndex = ownerInidices[globalID];
    int ownedIndex = edgePlacement[globalID];
    nodeArray[ownerNodeIndex].edges[ownedIndex] = globalID + edgeIndex;

    float depthHalfWidth = width/powf(2, depth + 1);
    Octree::Edge edge = Octree::Edge();
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
__global__ void ssrlcv::findFaceOwners(Octree::Node* nodeArray, int numNodes, int depthIndex, int* faceLUT, int* numFaces, int* ownerInidices, int* facePlacement){
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
__global__ void ssrlcv::fillUniqueFaceArray(Octree::Node* nodeArray, Octree::Face* faceArray, int numFaces, int faceIndex, int depthIndex, int depth, float width, int* faceLUT, int* ownerInidices, int* facePlacement, const int4* __restrict__ edgeFaceIdentity){
  int globalID = blockIdx.x * blockDim.x + threadIdx.x;
  if(globalID < numFaces){

    int ownerNodeIndex = ownerInidices[globalID];
    int ownedIndex = facePlacement[globalID];

    nodeArray[ownerNodeIndex].faces[ownedIndex] = globalID + faceIndex;

    float depthHalfWidth = width/powf(2, depth + 1);
    Octree::Face face = Octree::Face();
    int4 edge = edgeFaceIdentity[ownedIndex];

    face.e1 = nodeArray[ownerNodeIndex].edges[edge.x];
    face.e2 = nodeArray[ownerNodeIndex].edges[edge.y];
    face.e3 = nodeArray[ownerNodeIndex].edges[edge.z];
    face.e4 = nodeArray[ownerNodeIndex].edges[edge.w];
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






//
