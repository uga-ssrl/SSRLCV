#include "MeshFactory.cuh"

// =============================================================================================================
//
// Constructors and Destructors
//
// =============================================================================================================

ssrlcv::MeshFactory::MeshFactory(){

}
ssrlcv::MeshFactory::~MeshFactory(){

}

ssrlcv::MeshFactory::MeshFactory(Octree* octree){
  this->octree = octree;
  if(this->octree->normals == nullptr || this->octree->normals->getMemoryState() == null){
    this->octree->computeNormals(3, 20);
  }

}

// =============================================================================================================
//
// Mesh Loading Methods
//
// =============================================================================================================

/**
 * loads a mesh from a file into
 * currently only ASCII encoded PLY files are supported
 * @param filePath the filepath, relative to the install location
 */
void ssrlcv::MeshFactory::loadMesh(const char* filePath){
  // TODO perhaps move some of this into io_util

  bool local_debug = true;

  // temp storage
  std::vector<float3> tempPoints;
  std::vector<int> tempFaces;
  std::ifstream input(filePath);
  int numPoints = 0;
  int numFaces  = 0;
  int numEdges  = 0;
  bool inData   = false;

  // assuming ASCII encoding
  std::string line;
  while (std::getline(input, line)){
    std::istringstream iss(line);

    if (!inData){ // parse the header

      std::string tag;
      iss >> tag;

      //
      // Handle elements here
      //
      if (!tag.compare("element")){
        if(local_debug) std::cout << "element found" << std::endl;
        // temp vars for strings
        std::string elem;
        std::string type;
        int num;

        iss >> type;
        iss >> num;

        // set the correct value
        if (!type.compare("vertex")){
          numPoints = num;
          if(local_debug) std::cout << "detected " << num << " Points" << std::endl;
        } else if (!type.compare("face")) {
          numFaces = num;
          if(local_debug) std::cout << "detected " << num << " Faces" << std::endl;
        } else if (!type.compare("edge")) {
          // TODO read in edges if desired
          std::cout << "\tWARN: edge reading is not currently supported in MeshFactory" << std::endl;
          if(local_debug) std::cout << "detected " << num << " Edges" << std::endl;
        }

      }

      // header is ending
      if (!tag.compare("end_header"){
        inData = true;
      }
    } else {

      //
      // Handle the Data reading here
      //

      if (tempPoints.size() < numPoints && numPoints) {
        //
        // add the point
        //

        float3 point;
        iss >> point.x;
        iss >> point.y;
        iss >> point.z;
        tempPoints.push_back();
        if (local_debug) std::cout << "\t" << points.x << ", " << point.y << ", " << point.z << std::endl;
      } else if (tempFaces.size() < numFaces && numFaces) {
        //
        // add the face
        //

        // set the face encoding
        if (!tempFaces.size()) {
          iss >> this->faceEncoding;
          numFaces *= this->faceEncoding; // because they are not stored as int3 or int4 yet
          if (local_debug) {
            std::cout << "face encoding set to: " << this->faceEncoding << std::endl;
            std::cout << "faceNum updated to:   " << numFaces << "\t from " << (numFace / this->faceEncoding) << std::endl;
          }
          if (this->faceEncoding != 3 || this->faceEncoding != 4){
            std::cerr << "ERROR: error with reading mesh PLY face encoding" << std::endl;
          }
        } else {
          int throwAway;
          iss >> throwAway;
          if (local_debug) << std::cout < "\t face encoding: " << throwAway << " ";
        }

        // either quad or triangle
        for (int i = 0; i < (int) (this->faceEncoding); i++){
          int index;
          iss >> index;
          tempFaces.push_back(index);
          if (local_debug) std::cout << index << ", ";
        }
        if (local_debug) std::cout << std::endl;

      } // end face reading

    } // end data reading

  } // end while

  input.close(); // close the stream

}

/**
 * saves a PLY encoded Mesh as a given filename to the out directory
 * @param filename the filename
 */
void ssrlcv::MeshFactory::saveMesh(const char* filename){
  // TODO

}

// =============================================================================================================
//
// Other MeshFactory Methods
//
// =============================================================================================================

void ssrlcv::MeshFactory::computeVertexImplicitJAX(int focusDepth){
  clock_t timer;
  timer = clock();

  float* easyVertexImplicit = new float[this->octree->nodes->size()];
  int numConsideredVertices = -1;
  MemoryState origin[5] = {
    this->octree->vertexDepthIndex->getMemoryState(),
    this->octree->vertices->getMemoryState(),
    this->octree->points->getMemoryState(),
    this->octree->normals->getMemoryState(),
    this->octree->nodes->getMemoryState()
  };
  if(focusDepth == 0){
    numConsideredVertices = this->octree->vertices->size();
  }
  else{
    if(origin[0] != cpu && this->octree->vertexDepthIndex->getFore() != cpu){
      this->octree->vertexDepthIndex->transferMemoryTo(cpu);
    }
    numConsideredVertices = this->octree->vertexDepthIndex->host[this->octree->depth - focusDepth + 1];
  }
  if(origin[1] != gpu && this->octree->vertices->getFore() != gpu){
    this->octree->vertices->transferMemoryTo(gpu);
  }
  if(origin[2] != gpu && this->octree->points->getFore() != gpu){
    this->octree->points->transferMemoryTo(gpu);
  }
  if(origin[3] != gpu && this->octree->normals->getFore() != gpu){
    this->octree->normals->transferMemoryTo(gpu);
  }
  if(origin[4] != gpu && this->octree->nodes->getFore() != gpu){
    this->octree->nodes->transferMemoryTo(gpu);
  }
  CudaSafeCall(cudaMalloc((void**)&this->vertexImplicitDevice, numConsideredVertices*sizeof(float)));

  dim3 grid = {1,1,1};
  dim3 block = {8,1,1};
  if(numConsideredVertices < 65535) grid.x = (unsigned int) numConsideredVertices;
  else{
    grid.x = 65535;
    while(grid.x*grid.y < numConsideredVertices){
      ++grid.y;
    }
    while(grid.x*grid.y > numConsideredVertices){
      --grid.x;
    }
    if(grid.x*grid.y < numConsideredVertices){
      ++grid.x;
    }
  }
  vertexImplicitFromNormals<<<grid,block>>>(numConsideredVertices, this->octree->vertices->device, this->octree->nodes->device, this->octree->normals->device, this->octree->points->device, this->vertexImplicitDevice);
  cudaDeviceSynchronize();//may not be necessary
  CudaCheckError();
  this->octree->vertexDepthIndex->transferMemoryTo(origin[0]);
  if(origin[0] == gpu){
    this->octree->vertexDepthIndex->clear(cpu);
  }
  this->octree->vertices->transferMemoryTo(origin[1]);
  if(origin[1] == cpu){
    this->octree->vertices->clear(gpu);
  }
  this->octree->points->transferMemoryTo(origin[2]);
  if(origin[2] == cpu){
    this->octree->points->clear(gpu);
  }
  this->octree->normals->transferMemoryTo(origin[3]);
  if(origin[3] == cpu){
    this->octree->normals->clear(gpu);
  }
  this->octree->nodes->transferMemoryTo(origin[4]);
  if(origin[4] == cpu){
    this->octree->nodes->clear(gpu);
  }

  timer = clock() - timer;
  printf("Computing Vertex Implicit Values with normals took a total of %f seconds.\n\n",((float) timer)/CLOCKS_PER_SEC);
}
void ssrlcv::MeshFactory::adaptiveMarchingCubes(){
  this->computeVertexImplicitJAX(0);
  clock_t timer;
  timer = clock();

  MemoryState origin[4] = {
    this->octree->edges->getMemoryState(),
    this->octree->nodes->getMemoryState(),
    this->octree->nodeDepthIndex->getMemoryState(),
    this->octree->vertices->getMemoryState()
  };
  if(origin[0] != gpu && this->octree->edges->getFore() != gpu) this->octree->edges->transferMemoryTo(gpu);
  if(origin[1] != gpu && this->octree->nodes->getFore() != gpu) this->octree->nodes->transferMemoryTo(gpu);
  if(origin[2] != cpu && this->octree->nodeDepthIndex->getFore() != cpu){
    this->octree->nodeDepthIndex->transferMemoryTo(cpu);
  }
  int* vertexNumbersDevice;
  CudaSafeCall(cudaMalloc((void**)&vertexNumbersDevice, this->octree->edges->size()*sizeof(int)));
  dim3 gridEdge = {1,1,1};
  dim3 blockEdge = {1,1,1};
  if(this->octree->edges->size() < 65535) gridEdge.x = (unsigned int) this->octree->edges->size();
  else{
    gridEdge.x = 65535;
    while(gridEdge.x*blockEdge.x < this->octree->edges->size()){
      ++blockEdge.x;
    }
    while(gridEdge.x*blockEdge.x > this->octree->edges->size()){
      --gridEdge.x;
    }
    if(gridEdge.x*blockEdge.x < this->octree->edges->size()){
      ++gridEdge.x;
    }
  }
  calcVertexNumbers<<<gridEdge,blockEdge>>>(this->octree->edges->size(), 0, this->octree->edges->device, this->vertexImplicitDevice, vertexNumbersDevice);
  cudaDeviceSynchronize();
  CudaCheckError();
  CudaSafeCall(cudaFree(this->vertexImplicitDevice));

  /*Triangles*/
  //surround vertices with values less than 0

  int* triangleNumbersDevice;
  int* cubeCategoryDevice;
  CudaSafeCall(cudaMalloc((void**)&triangleNumbersDevice, this->octree->nodes->size()*sizeof(int)));
  CudaSafeCall(cudaMalloc((void**)&cubeCategoryDevice, this->octree->nodes->size()*sizeof(int)));

  categorizeCubesRecursively<<<1,8>>>(this->octree->nodeDepthIndex->host[this->octree->depth - 1], this->octree->edges->device, this->octree->nodes->device, vertexNumbersDevice, cubeCategoryDevice, triangleNumbersDevice);
  cudaDeviceSynchronize();
  CudaCheckError();

  this->octree->nodeDepthIndex->transferMemoryTo(origin[2]);
  if(origin[2] == gpu){
    this->octree->nodeDepthIndex->clear(cpu);
  }

  dim3 gridEdge2 = {1,1,1};
  dim3 blockEdge2 = {4,1,1};
  if(this->octree->edges->size() < 65535) gridEdge2.x = (unsigned int) this->octree->edges->size();
  else{
    gridEdge2.x = 65535;
    while(gridEdge2.x*gridEdge2.y < this->octree->edges->size()){
      ++gridEdge2.y;
    }
    while(gridEdge2.x*gridEdge2.y > this->octree->edges->size()){
      --gridEdge2.x;
    }
    if(gridEdge2.x*gridEdge2.y < this->octree->edges->size()){
      ++gridEdge2.x;
    }
  }

  minimizeVertices<<<gridEdge2, blockEdge2>>>(this->octree->edges->size(), this->octree->edges->device, this->octree->nodes->device, cubeCategoryDevice, vertexNumbersDevice);
  cudaDeviceSynchronize();
  CudaCheckError();

  int* vertexAddressesDevice;
  CudaSafeCall(cudaMalloc((void**)&vertexAddressesDevice, this->octree->edges->size()*sizeof(int)));
  thrust::device_ptr<int> vN(vertexNumbersDevice);
  thrust::device_ptr<int> vA(vertexAddressesDevice);
  thrust::inclusive_scan(vN, vN + this->octree->edges->size(), vA);
  cudaDeviceSynchronize();

  int* triangleAddressesDevice;
  CudaSafeCall(cudaMalloc((void**)&triangleAddressesDevice, this->octree->nodes->size()*sizeof(int)));
  thrust::device_ptr<int> tN(triangleNumbersDevice);
  thrust::device_ptr<int> tA(triangleAddressesDevice);
  thrust::inclusive_scan(tN, tN + this->octree->nodes->size(), tA);
  cudaDeviceSynchronize();

  this->numSurfaceVertices = 0;
  this->numSurfaceTriangles = 0;

  CudaSafeCall(cudaMemcpy(&this->numSurfaceVertices, vertexAddressesDevice + (this->octree->edges->size() - 1), sizeof(int), cudaMemcpyDeviceToHost));
  CudaSafeCall(cudaMemcpy(&this->numSurfaceTriangles, triangleAddressesDevice + (this->octree->nodes->size() - 1), sizeof(int), cudaMemcpyDeviceToHost));

  printf("%d vertices and %d triangles from %lu finestNodes\n",this->numSurfaceVertices, this->numSurfaceTriangles, this->octree->nodes->size());
  CudaSafeCall(cudaFree(triangleNumbersDevice));

  float3* surfaceVerticesDevice;
  CudaSafeCall(cudaMalloc((void**)&surfaceVerticesDevice, this->numSurfaceVertices*sizeof(float3)));

  if(origin[3] != gpu && this->octree->vertices->getFore() != gpu){
    this->octree->vertices->transferMemoryTo(gpu);
  }

  /* generate vertices */
  generateSurfaceVertices<<<gridEdge,blockEdge>>>(this->octree->edges->size(), 0, this->octree->edges->device, this->octree->vertices->device, vertexNumbersDevice, vertexAddressesDevice, surfaceVerticesDevice);
  CudaCheckError();
  this->surfaceVertices = new float3[this->numSurfaceVertices];
  CudaSafeCall(cudaMemcpy(this->surfaceVertices, surfaceVerticesDevice, this->numSurfaceVertices*sizeof(float3),cudaMemcpyDeviceToHost));
  CudaSafeCall(cudaFree(surfaceVerticesDevice));
  CudaSafeCall(cudaFree(vertexNumbersDevice));
  this->octree->edges->transferMemoryTo(origin[0]);
  if(origin[0] == cpu){
    this->octree->edges->clear(gpu);
  }
  this->octree->vertices->transferMemoryTo(origin[3]);
  if(origin[3] == cpu){
    this->octree->vertices->clear(gpu);
  }
  int3* surfaceTrianglesDevice;

  CudaSafeCall(cudaMalloc((void**)&surfaceTrianglesDevice, this->numSurfaceTriangles*sizeof(int3)));

  /* generate triangles */
  //grid is already numFinestNodes
  dim3 grid = {1,1,1};
  dim3 block = {5,1,1};
  if(this->octree->nodes->size() < 65535) grid.x = (unsigned int) this->octree->nodes->size();
  else{
    grid.x = 65535;
    while(grid.x*grid.y < this->octree->nodes->size()){
      ++grid.y;
    }
    while(grid.x*grid.y > this->octree->nodes->size()){
      --grid.x;
    }
    if(grid.x*grid.y < this->octree->nodes->size()){
      ++grid.x;
    }
  }
  generateSurfaceTriangles<<<grid,block>>>(this->octree->nodes->size(), 0, 0, this->octree->nodes->device, vertexAddressesDevice, triangleAddressesDevice, cubeCategoryDevice, surfaceTrianglesDevice);
  CudaCheckError();

  this->surfaceTriangles = new int3[this->numSurfaceTriangles];
  CudaSafeCall(cudaMemcpy(this->surfaceTriangles, surfaceTrianglesDevice, this->numSurfaceTriangles*sizeof(int3),cudaMemcpyDeviceToHost));
  this->octree->nodes->transferMemoryTo(origin[1]);
  if(origin[1] == cpu){
    this->octree->nodes->clear(gpu);
  }
  CudaSafeCall(cudaFree(surfaceTrianglesDevice));
  CudaSafeCall(cudaFree(vertexAddressesDevice));
  CudaSafeCall(cudaFree(triangleAddressesDevice));
  CudaSafeCall(cudaFree(cubeCategoryDevice));
  timer = clock() - timer;
  printf("Marching cubes took a total of %f seconds.\n\n",((float) timer)/CLOCKS_PER_SEC);
  this->generateMesh();
}
void ssrlcv::MeshFactory::marchingCubes(){
  this->computeVertexImplicitJAX(this->octree->depth);
  clock_t timer;
  timer = clock();

  MemoryState origin[5] = {
    this->octree->edges->getMemoryState(),
    this->octree->nodes->getMemoryState(),
    this->octree->nodeDepthIndex->getMemoryState(),
    this->octree->vertices->getMemoryState(),
    this->octree->edgeDepthIndex->getMemoryState()
  };
  if(origin[0] != gpu && this->octree->edges->getFore() != gpu) this->octree->edges->transferMemoryTo(gpu);
  if(origin[1] != gpu && this->octree->nodes->getFore() != gpu) this->octree->nodes->transferMemoryTo(gpu);
  if(origin[2] != cpu && this->octree->nodeDepthIndex->getFore() != cpu){
    this->octree->nodeDepthIndex->transferMemoryTo(cpu);
  }
  if(origin[4] != cpu && this->octree->edgeDepthIndex->getFore() != cpu){
    this->octree->edgeDepthIndex->transferMemoryTo(cpu);
  }
  int numFinestEdges = this->octree->edgeDepthIndex->host[1];
  this->octree->edgeDepthIndex->transferMemoryTo(origin[4]);
  if(origin[4] == gpu){
    this->octree->edgeDepthIndex->clear(cpu);
  }
  int* vertexNumbersDevice;
  CudaSafeCall(cudaMalloc((void**)&vertexNumbersDevice, numFinestEdges*sizeof(int)));
  dim3 gridEdge = {1,1,1};
  dim3 blockEdge = {1,1,1};
  if(numFinestEdges < 65535) gridEdge.x = (unsigned int) numFinestEdges;
  else{
    gridEdge.x = 65535;
    while(gridEdge.x*blockEdge.x < numFinestEdges){
      ++blockEdge.x;
    }
    while(gridEdge.x*blockEdge.x > numFinestEdges){
      --gridEdge.x;
    }
    if(gridEdge.x*blockEdge.x < numFinestEdges){
      ++gridEdge.x;
    }
  }
  calcVertexNumbers<<<gridEdge,blockEdge>>>(numFinestEdges, 0, this->octree->edges->device, this->vertexImplicitDevice, vertexNumbersDevice);
  cudaDeviceSynchronize();
  CudaCheckError();
  CudaSafeCall(cudaFree(this->vertexImplicitDevice));
  int* vertexAddressesDevice;
  CudaSafeCall(cudaMalloc((void**)&vertexAddressesDevice, numFinestEdges*sizeof(int)));
  thrust::device_ptr<int> vN(vertexNumbersDevice);
  thrust::device_ptr<int> vA(vertexAddressesDevice);
  thrust::inclusive_scan(vN, vN + numFinestEdges, vA);
  cudaDeviceSynchronize();

  /*Triangles*/
  //surround vertices with values less than 0

  int numFinestNodes = this->octree->nodeDepthIndex->host[1];
  this->octree->nodeDepthIndex->transferMemoryTo(origin[2]);
  if(origin[2] == gpu){
    this->octree->nodeDepthIndex->clear(cpu);
  }
  int* triangleNumbersDevice;
  int* cubeCategoryDevice;
  CudaSafeCall(cudaMalloc((void**)&triangleNumbersDevice, numFinestNodes*sizeof(int)));
  CudaSafeCall(cudaMalloc((void**)&cubeCategoryDevice, numFinestNodes*sizeof(int)));

  dim3 grid = {1,1,1};
  dim3 block = {1,1,1};
  if(numFinestNodes < 65535) grid.x = (unsigned int) numFinestNodes;
  else{
    grid.x = 65535;
    while(grid.x*block.x < numFinestNodes){
      ++block.x;
    }
    while(grid.x*block.x > numFinestNodes){
      --grid.x;
    }
    if(grid.x*block.x < numFinestNodes){
      ++grid.x;
    }
  }
  determineCubeCategories<<<grid,block>>>(numFinestNodes, 0, 0, this->octree->nodes->device, vertexNumbersDevice, cubeCategoryDevice, triangleNumbersDevice);
  cudaDeviceSynchronize();
  CudaCheckError();

  int* triangleAddressesDevice;
  CudaSafeCall(cudaMalloc((void**)&triangleAddressesDevice, numFinestNodes*sizeof(int)));
  thrust::device_ptr<int> tN(triangleNumbersDevice);
  thrust::device_ptr<int> tA(triangleAddressesDevice);
  thrust::inclusive_scan(tN, tN + numFinestNodes, tA);
  cudaDeviceSynchronize();

  this->numSurfaceVertices = 0;
  this->numSurfaceTriangles = 0;

  CudaSafeCall(cudaMemcpy(&this->numSurfaceVertices, vertexAddressesDevice + (numFinestEdges - 1), sizeof(int), cudaMemcpyDeviceToHost));
  CudaSafeCall(cudaMemcpy(&this->numSurfaceTriangles, triangleAddressesDevice + (numFinestNodes - 1), sizeof(int), cudaMemcpyDeviceToHost));

  printf("%d vertices and %d triangles from %d finestNodes\n",this->numSurfaceVertices, this->numSurfaceTriangles, numFinestNodes);
  CudaSafeCall(cudaFree(triangleNumbersDevice));

  float3* surfaceVerticesDevice;
  CudaSafeCall(cudaMalloc((void**)&surfaceVerticesDevice, this->numSurfaceVertices*sizeof(float3)));


  if(origin[3] != gpu && this->octree->vertices->getFore() != gpu){
    this->octree->vertices->transferMemoryTo(gpu);
  }

  /* generate vertices */
  generateSurfaceVertices<<<gridEdge,blockEdge>>>(numFinestEdges, 0, this->octree->edges->device, this->octree->vertices->device, vertexNumbersDevice, vertexAddressesDevice, surfaceVerticesDevice);
  CudaCheckError();
  this->surfaceVertices = new float3[this->numSurfaceVertices];
  CudaSafeCall(cudaMemcpy(this->surfaceVertices, surfaceVerticesDevice, this->numSurfaceVertices*sizeof(float3),cudaMemcpyDeviceToHost));
  CudaSafeCall(cudaFree(surfaceVerticesDevice));
  CudaSafeCall(cudaFree(vertexNumbersDevice));
  this->octree->edges->transferMemoryTo(origin[0]);
  if(origin[0] == cpu){
    this->octree->edges->clear(gpu);
  }
  this->octree->vertices->transferMemoryTo(origin[3]);
  if(origin[3] == cpu){
    this->octree->vertices->clear(gpu);
  }

  int3* surfaceTrianglesDevice;

  CudaSafeCall(cudaMalloc((void**)&surfaceTrianglesDevice, this->numSurfaceTriangles*sizeof(int3)));

  /* generate triangles */
  //grid is already numFinestNodes
  if(numFinestNodes < 65535) grid.x = (unsigned int) numFinestNodes;
  else{
    grid.x = 65535;
    while(grid.x*grid.y < numFinestNodes){
      ++grid.y;
    }
    while(grid.x*grid.y > numFinestNodes){
      --grid.x;
    }
    if(grid.x*grid.y < numFinestNodes){
      ++grid.x;
    }
  }
  block = {5,1,1};
  generateSurfaceTriangles<<<grid,block>>>(numFinestNodes, 0, 0, this->octree->nodes->device, vertexAddressesDevice, triangleAddressesDevice, cubeCategoryDevice, surfaceTrianglesDevice);
  CudaCheckError();

  this->surfaceTriangles = new int3[this->numSurfaceTriangles];
  CudaSafeCall(cudaMemcpy(this->surfaceTriangles, surfaceTrianglesDevice, this->numSurfaceTriangles*sizeof(int3),cudaMemcpyDeviceToHost));
  this->octree->nodes->transferMemoryTo(origin[1]);
  if(origin[1] == cpu){
    this->octree->nodes->clear(gpu);
  }
  CudaSafeCall(cudaFree(surfaceTrianglesDevice));
  CudaSafeCall(cudaFree(vertexAddressesDevice));
  CudaSafeCall(cudaFree(triangleAddressesDevice));
  CudaSafeCall(cudaFree(cubeCategoryDevice));
  timer = clock() - timer;
  printf("Marching cubes took a total of %f seconds.\n\n",((float) timer)/CLOCKS_PER_SEC);
  this->generateMesh(true);

}

// =============================================================================================================
//
// Mesh Generation Methods
//
// =============================================================================================================

void ssrlcv::MeshFactory::jaxMeshing(){
  //TODO make this not necessary
  clock_t timer;
  timer = clock();
  MemoryState origin[5] = {
    this->octree->edges->getMemoryState(),
    this->octree->edgeDepthIndex->getMemoryState(),
    this->octree->nodes->getMemoryState(),
    this->octree->nodeDepthIndex->getMemoryState(),
    this->octree->vertices->getMemoryState()
  };
  if(origin[0] != gpu && this->octree->edges->getFore() != gpu) this->octree->edges->transferMemoryTo(gpu);
  if(origin[1] != cpu && this->octree->edgeDepthIndex->getFore() != cpu){
    this->octree->edgeDepthIndex->transferMemoryTo(cpu);
  }
  if(origin[2] != both && this->octree->nodes->getFore() != both) this->octree->nodes->transferMemoryTo(both);
  if(origin[3] != cpu && this->octree->nodeDepthIndex->getFore() != cpu){
    this->octree->nodeDepthIndex->transferMemoryTo(cpu);
  }

  bool foundSurfaceDepth = false;
  int numNodesAtDepth = 0;
  int currentDepthIndex = -1;
  int surfaceDepth = -1;
  bool hadNeighborsWithPoints = false;
  int currentNeighbor = -1;
  int numNodesWithPointNeighbors = 0;
  for(int d = 0; d < this->octree->depth; ++d){
    numNodesAtDepth = this->octree->nodeDepthIndex->host[d + 1] - this->octree->nodeDepthIndex->host[d];
    currentDepthIndex = this->octree->nodeDepthIndex->host[d];
    foundSurfaceDepth = true;
    numNodesWithPointNeighbors = 0;
    for(int n = currentDepthIndex; n < numNodesAtDepth + currentDepthIndex; ++n){
      if(this->octree->nodes->host[n].numPoints == 0) continue;
      hadNeighborsWithPoints = false;
      for(int neigh = 0; neigh < 27; ++neigh){
        if(neigh == 13) continue;
        currentNeighbor = this->octree->nodes->host[n].neighbors[neigh];
        if(currentNeighbor != -1 && this->octree->nodes->host[currentNeighbor].numPoints != 0){
          hadNeighborsWithPoints = true;
          break;
        }
      }
      if(!hadNeighborsWithPoints){
        foundSurfaceDepth = false;
        break;
      }
      else{
        ++numNodesWithPointNeighbors;
      }
    }
    if(foundSurfaceDepth){
      surfaceDepth = d;
      break;
    }
  }
  //this->octree->writeDepthPLY(this->octree->depth - surfaceDepth);
  printf("%d is the depth at which the surface is surrounded by nodes without holes\n",this->octree->depth - surfaceDepth);
  this->computeVertexImplicitJAX(this->octree->depth - surfaceDepth);

  //MARCHING CUBES ON

  int numMarchingEdges = this->octree->edgeDepthIndex->host[surfaceDepth + 1] - this->octree->edgeDepthIndex->host[surfaceDepth];
  int* vertexNumbersDevice;
  CudaSafeCall(cudaMalloc((void**)&vertexNumbersDevice, numMarchingEdges*sizeof(int)));
  dim3 gridEdge = {1,1,1};
  dim3 blockEdge = {1,1,1};
  if(numMarchingEdges < 65535) gridEdge.x = (unsigned int) numMarchingEdges;
  else{
    gridEdge.x = 65535;
    while(gridEdge.x*blockEdge.x < numMarchingEdges){
      ++blockEdge.x;
    }
    while(gridEdge.x*blockEdge.x > numMarchingEdges){
      --gridEdge.x;
    }
    if(gridEdge.x*blockEdge.x < numMarchingEdges){
      ++gridEdge.x;
    }
  }
  calcVertexNumbers<<<gridEdge,blockEdge>>>(numMarchingEdges, this->octree->edgeDepthIndex->host[surfaceDepth], this->octree->edges->device, this->vertexImplicitDevice, vertexNumbersDevice);
  cudaDeviceSynchronize();
  CudaCheckError();
  CudaSafeCall(cudaFree(this->vertexImplicitDevice));
  int* vertexAddressesDevice;
  CudaSafeCall(cudaMalloc((void**)&vertexAddressesDevice, numMarchingEdges*sizeof(int)));
  thrust::device_ptr<int> vN(vertexNumbersDevice);
  thrust::device_ptr<int> vA(vertexAddressesDevice);
  thrust::inclusive_scan(vN, vN + numMarchingEdges, vA);
  cudaDeviceSynchronize();

  /*Triangles*/
  //surround vertices with values less than 0

  int numMarchingNodes = this->octree->nodeDepthIndex->host[surfaceDepth + 1] - this->octree->nodeDepthIndex->host[surfaceDepth];
  int* triangleNumbersDevice;
  int* cubeCategoryDevice;
  CudaSafeCall(cudaMalloc((void**)&triangleNumbersDevice, numMarchingNodes*sizeof(int)));
  CudaSafeCall(cudaMalloc((void**)&cubeCategoryDevice, numMarchingNodes*sizeof(int)));

  dim3 grid = {1,1,1};
  dim3 block = {1,1,1};
  if(numMarchingNodes < 65535) grid.x = (unsigned int) numMarchingNodes;
  else{
    grid.x = 65535;
    while(grid.x*block.x < numMarchingNodes){
      ++block.x;
    }
    while(grid.x*block.x > numMarchingNodes){
      --grid.x;
    }
    if(grid.x*block.x < numMarchingNodes){
      ++grid.x;
    }
  }
  determineCubeCategories<<<grid,block>>>(numMarchingNodes, this->octree->nodeDepthIndex->host[surfaceDepth],
    this->octree->edgeDepthIndex->host[surfaceDepth], this->octree->nodes->device, vertexNumbersDevice,
    cubeCategoryDevice, triangleNumbersDevice);
  cudaDeviceSynchronize();
  CudaCheckError();

  int* triangleAddressesDevice;
  CudaSafeCall(cudaMalloc((void**)&triangleAddressesDevice, numMarchingNodes*sizeof(int)));
  thrust::device_ptr<int> tN(triangleNumbersDevice);
  thrust::device_ptr<int> tA(triangleAddressesDevice);
  thrust::inclusive_scan(tN, tN + numMarchingNodes, tA);
  cudaDeviceSynchronize();

  this->numSurfaceVertices = 0;
  this->numSurfaceTriangles = 0;

  CudaSafeCall(cudaMemcpy(&this->numSurfaceVertices, vertexAddressesDevice + (numMarchingEdges - 1), sizeof(int), cudaMemcpyDeviceToHost));
  CudaSafeCall(cudaMemcpy(&this->numSurfaceTriangles, triangleAddressesDevice + (numMarchingNodes - 1), sizeof(int), cudaMemcpyDeviceToHost));

  printf("%d vertices and %d triangles from %d finestNodes\n",this->numSurfaceVertices, this->numSurfaceTriangles, numMarchingNodes);
  CudaSafeCall(cudaFree(triangleNumbersDevice));

  float3* surfaceVerticesDevice;
  CudaSafeCall(cudaMalloc((void**)&surfaceVerticesDevice, this->numSurfaceVertices*sizeof(float3)));


  if(origin[4] != gpu && this->octree->vertices->getFore() != gpu){
    this->octree->vertices->transferMemoryTo(gpu);
  }
  /* generate vertices */
  generateSurfaceVertices<<<gridEdge,blockEdge>>>(numMarchingEdges, this->octree->edgeDepthIndex->host[surfaceDepth],
    this->octree->edges->device, this->octree->vertices->device, vertexNumbersDevice, vertexAddressesDevice, surfaceVerticesDevice);
  CudaCheckError();
  this->surfaceVertices = new float3[this->numSurfaceVertices];
  CudaSafeCall(cudaMemcpy(this->surfaceVertices, surfaceVerticesDevice, this->numSurfaceVertices*sizeof(float3),cudaMemcpyDeviceToHost));
  CudaSafeCall(cudaFree(surfaceVerticesDevice));
  CudaSafeCall(cudaFree(vertexNumbersDevice));
  this->octree->edges->transferMemoryTo(origin[0]);
  if(origin[0] == cpu){
    this->octree->edges->clear(gpu);
  }
  this->octree->vertices->transferMemoryTo(origin[4]);
  if(origin[4] == cpu){
    this->octree->vertices->clear(gpu);
  }

  int3* surfaceTrianglesDevice;

  CudaSafeCall(cudaMalloc((void**)&surfaceTrianglesDevice, this->numSurfaceTriangles*sizeof(int3)));

  /* generate triangles */
  //grid is already numMarchingNodes
  if(numMarchingNodes < 65535) grid.x = (unsigned int) numMarchingNodes;
  else{
    grid.x = 65535;
    while(grid.x*grid.y < numMarchingNodes){
      ++grid.y;
    }
    while(grid.x*grid.y > numMarchingNodes){
      --grid.x;
    }
    if(grid.x*grid.y < numMarchingNodes){
      ++grid.x;
    }
  }
  block = {5,1,1};
  generateSurfaceTriangles<<<grid,block>>>(numMarchingNodes, this->octree->nodeDepthIndex->host[surfaceDepth],
    this->octree->edgeDepthIndex->host[surfaceDepth], this->octree->nodes->device, vertexAddressesDevice,
    triangleAddressesDevice, cubeCategoryDevice, surfaceTrianglesDevice);
  CudaCheckError();

  this->surfaceTriangles = new int3[this->numSurfaceTriangles];
  CudaSafeCall(cudaMemcpy(this->surfaceTriangles, surfaceTrianglesDevice, this->numSurfaceTriangles*sizeof(int3),cudaMemcpyDeviceToHost));
  this->octree->edgeDepthIndex->transferMemoryTo(origin[1]);
  if(origin[1] == gpu){
    this->octree->edgeDepthIndex->clear(cpu);
  }
  this->octree->nodes->transferMemoryTo(origin[2]);
  if(origin[2] == cpu){
    this->octree->nodes->clear(gpu);
  }
  this->octree->nodeDepthIndex->transferMemoryTo(origin[3]);
  if(origin[3] == gpu){
    this->octree->nodeDepthIndex->clear(cpu);
  }

  CudaSafeCall(cudaFree(surfaceTrianglesDevice));
  CudaSafeCall(cudaFree(vertexAddressesDevice));
  CudaSafeCall(cudaFree(triangleAddressesDevice));
  CudaSafeCall(cudaFree(cubeCategoryDevice));

  timer = clock() - timer;
  printf("Jax meshing took a total of %f seconds.\n\n",((float) timer)/CLOCKS_PER_SEC);
  this->generateMesh();

}
void ssrlcv::MeshFactory::generateMesh(bool binary){

  tinyply::PlyFile ply;
  ply.get_comments().push_back("SSRL Test");
  ply.add_properties_to_element("vertex",{"x","y","z"},tinyply::Type::FLOAT32, this->numSurfaceVertices, reinterpret_cast<uint8_t*>(this->surfaceVertices), tinyply::Type::INVALID, 0);
  ply.add_properties_to_element("face",{"vertex_indices"},tinyply::Type::INT32, this->numSurfaceTriangles, reinterpret_cast<uint8_t*>(this->surfaceTriangles), tinyply::Type::INT32, 3);

  std::filebuf fb_binary;
  if(this->octree->name.length() == 0) this->octree->name = std::to_string(clock());
  std::string newFile = "out/" + this->octree->name + "_mesh_march_" + std::to_string(this->octree->depth)+ ".ply";

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
void ssrlcv::MeshFactory::generateMesh(){
  if(this->octree->name.length() == 0) this->octree->name = std::to_string(clock());
  std::string newFile = "out/" + this->octree->name + "_mesh_march_" + std::to_string(this->octree->depth)+ ".ply";
  std::ofstream plystream(newFile);

  if (plystream.is_open()) {
    std::ostringstream stringBuffer = std::ostringstream("");
    stringBuffer << "ply\nformat ascii 1.0\ncomment object: SSRL test\n";
    stringBuffer << "element vertex ";
    stringBuffer << this->numSurfaceVertices;
    stringBuffer << "\nproperty float x\nproperty float y\nproperty float z\n";
    stringBuffer << "element face ";
    stringBuffer << this->numSurfaceTriangles;
    stringBuffer << "\nproperty list uchar int vertex_index\n";
    stringBuffer << "end_header\n";
    plystream << stringBuffer.str();
    for(int i = 0; i < this->numSurfaceVertices; ++i){
      stringBuffer = std::ostringstream("");
      stringBuffer << this->surfaceVertices[i].x;
      stringBuffer << " ";
      stringBuffer << this->surfaceVertices[i].y;
      stringBuffer << " ";
      stringBuffer << this->surfaceVertices[i].z;
      stringBuffer << "\n";
      plystream << stringBuffer.str();
    }
    for(int i = 0; i < this->numSurfaceTriangles; ++i){
      stringBuffer = std::ostringstream("");
      stringBuffer << "3 ";
      stringBuffer << this->surfaceTriangles[i].x;
      stringBuffer << " ";
      stringBuffer << this->surfaceTriangles[i].y;
      stringBuffer << " ";
      stringBuffer << this->surfaceTriangles[i].z;
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
void ssrlcv::MeshFactory::generateMeshWithFinestEdges(){
  if(this->octree->name.length() == 0) this->octree->name = this->octree->pathToFile.substr(this->octree->pathToFile.find_last_of("/") + 1,this->octree->pathToFile.length() - 4);
  std::string newFile = "out/" + this->octree->name + "_meshwedges_" + std::to_string(this->octree->depth)+ ".ply";
  std::ofstream plystream(newFile);
  MemoryState origin[4] = {
    this->octree->vertices->getMemoryState(),
    this->octree->vertexDepthIndex->getMemoryState(),
    this->octree->edges->getMemoryState(),
    this->octree->edgeDepthIndex->getMemoryState()
  };
  if(origin[0] != cpu || this->octree->vertices->getFore() != cpu){
    this->octree->vertices->transferMemoryTo(cpu);
  }
  if(origin[1] != cpu || this->octree->vertexDepthIndex->getFore() != cpu){
    this->octree->vertexDepthIndex->transferMemoryTo(cpu);
  }
  if(origin[2] != cpu || this->octree->edges->getFore() != cpu){
    this->octree->edges->transferMemoryTo(cpu);
  }
  if(origin[3] != cpu || this->octree->edgeDepthIndex->getFore() != cpu){
    this->octree->edgeDepthIndex->transferMemoryTo(cpu);
  }
  if (plystream.is_open()) {
    std::ostringstream stringBuffer = std::ostringstream("");
    stringBuffer << "ply\nformat ascii 1.0\ncomment object: SSRL test\n";
    stringBuffer << "element vertex ";
    stringBuffer << (this->numSurfaceVertices + this->octree->vertexDepthIndex->host[1]);
    stringBuffer << "\nproperty float x\nproperty float y\nproperty float z\n";
    stringBuffer << "element face ";
    stringBuffer << this->numSurfaceTriangles;
    stringBuffer << "\nproperty list uchar int vertex_index\n";
    stringBuffer << "element edge ";
    stringBuffer <<  this->octree->edgeDepthIndex->host[1];
    stringBuffer << "\nproperty int vertex1\nproperty int vertex2\n";
    stringBuffer << "property uchar red\nproperty uchar green\nproperty uchar blue\n";
    stringBuffer << "end_header\n";
    plystream << stringBuffer.str();
    for(int i = 0; i < this->numSurfaceVertices; ++i){
      stringBuffer = std::ostringstream("");
      stringBuffer << this->surfaceVertices[i].x;
      stringBuffer << " ";
      stringBuffer << this->surfaceVertices[i].y;
      stringBuffer << " ";
      stringBuffer << this->surfaceVertices[i].z;
      stringBuffer << "\n";
      plystream << stringBuffer.str();
    }
    for(int i = 0; i < this->octree->vertexDepthIndex->host[1]; ++i){
      stringBuffer = std::ostringstream("");
      stringBuffer << this->octree->vertices->host[i].coord.x;
      stringBuffer << " ";
      stringBuffer << this->octree->vertices->host[i].coord.y;
      stringBuffer << " ";
      stringBuffer << this->octree->vertices->host[i].coord.z;
      stringBuffer << "\n";
      plystream << stringBuffer.str();
    }
    for(int i = 0; i < this->numSurfaceTriangles; ++i){
      stringBuffer = std::ostringstream("");
      stringBuffer << "3 ";
      stringBuffer << this->surfaceTriangles[i].x;
      stringBuffer << " ";
      stringBuffer << this->surfaceTriangles[i].y;
      stringBuffer << " ";
      stringBuffer << this->surfaceTriangles[i].z;
      stringBuffer << "\n";
      plystream << stringBuffer.str();
    }
    for(int i = 0; i < this->octree->edgeDepthIndex->host[1]; ++i){
      stringBuffer = std::ostringstream("");
      stringBuffer << (this->octree->edges->host[i].v1 + this->numSurfaceVertices);
      stringBuffer << " ";
      stringBuffer << (this->octree->edges->host[i].v2 + this->numSurfaceVertices);
      stringBuffer << " 255 255 255\n";
      plystream << stringBuffer.str();
    }
    std::cout<<newFile + " has been created.\n"<<std::endl;
  }
  else{
    std::cout << "Unable to open: " + newFile<< std::endl;
    exit(1);
  }
  this->octree->vertices->transferMemoryTo(origin[0]);
  if(origin[0] == gpu){
    this->octree->vertices->clear(cpu);
  }
  this->octree->vertexDepthIndex->transferMemoryTo(origin[1]);
  if(origin[1] == gpu){
    this->octree->vertexDepthIndex->clear(cpu);
  }
  this->octree->edges->transferMemoryTo(origin[2]);
  if(origin[2] == gpu){
    this->octree->edges->clear(cpu);
  }
  this->octree->edgeDepthIndex->transferMemoryTo(origin[3]);
  if(origin[3] == gpu){
    this->octree->edgeDepthIndex->clear(cpu);
  }
}

/*
CUDA implementations
*/
/*
my edges - everyone elses
0-0
1-8
2-9
3-4
4-3
5-1
6-7
7-5
8-2
9-11
10-10
11-6
*/

// =============================================================================================================
//
// Device Kernels
//
// =============================================================================================================

__constant__ int ssrlcv::cubeCategoryTrianglesFromEdges[256][15] = {
  {-1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
  {0, 1, 4, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
  {0, 5, 2, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
  {5, 1, 4, 2, 1, 5, -1, -1, -1, -1, -1, -1, -1, -1, -1},
  {5, 8, 10, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
  {0, 1, 4, 5, 8, 10, -1, -1, -1, -1, -1, -1, -1, -1, -1},
  {2, 8, 10, 0, 8, 2, -1, -1, -1, -1, -1, -1, -1, -1, -1},
  {8, 1, 4, 8, 10, 1, 10, 2, 1, -1, -1, -1, -1, -1, -1},
  {4, 9, 8, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
  {0, 9, 8, 1, 9, 0, -1, -1, -1, -1, -1, -1, -1, -1, -1},
  {5, 2, 0, 8, 4, 9, -1, -1, -1, -1, -1, -1, -1, -1, -1},
  {5, 9, 8, 5, 2, 9, 2, 1, 9, -1, -1, -1, -1, -1, -1},
  {4, 10, 5, 9, 10, 4, -1, -1, -1, -1, -1, -1, -1, -1, -1},
  {0, 10, 5, 0, 1, 10, 1, 9, 10, -1, -1, -1, -1, -1, -1},
  {4, 2, 0, 4, 9, 2, 9, 10, 2, -1, -1, -1, -1, -1, -1},
  {2, 1, 10, 10, 1, 9, -1, -1, -1, -1, -1, -1, -1, -1, -1},
  {3, 6, 1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
  {3, 4, 0, 6, 4, 3, -1, -1, -1, -1, -1, -1, -1, -1, -1},
  {0, 5, 2, 1, 3, 6, -1, -1, -1, -1, -1, -1, -1, -1, -1},
  {3, 5, 2, 3, 6, 5, 6, 4, 5, -1, -1, -1, -1, -1, -1},
  {5, 8, 10, 1, 3, 6, -1, -1, -1, -1, -1, -1, -1, -1, -1},
  {4, 3, 6, 4, 0, 3, 5, 8, 10, -1, -1, -1, -1, -1, -1},
  {2, 8, 10, 2, 0, 8, 1, 3, 6, -1, -1, -1, -1, -1, -1},
  {8, 10, 2, 8, 2, 6, 8, 6, 4, 6, 2, 3, -1, -1, -1},
  {1, 3, 6, 4, 9, 8, -1, -1, -1, -1, -1, -1, -1, -1, -1},
  {9, 3, 6, 9, 8, 3, 8, 0, 3, -1, -1, -1, -1, -1, -1},
  {2, 0, 5, 1, 3, 6, 8, 4, 9, -1, -1, -1, -1, -1, -1},
  {3, 6, 9, 2, 3, 9, 2, 9, 8, 2, 8, 5, -1, -1, -1},
  {4, 10, 5, 4, 9, 10, 6, 1, 3, -1, -1, -1, -1, -1, -1},
  {5, 9, 10, 5, 3, 9, 5, 0, 3, 6, 9, 3, -1, -1, -1},
  {3, 6, 1, 2, 0, 9, 2, 9, 10, 9, 0, 4, -1, -1, -1},
  {3, 6, 9, 3, 9, 2, 2, 9, 10, -1, -1, -1, -1, -1, -1},
  {2, 7, 3, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
  {2, 7, 3, 0, 1, 4, -1, -1, -1, -1, -1, -1, -1, -1, -1},
  {0, 7, 3, 5, 7, 0, -1, -1, -1, -1, -1, -1, -1, -1, -1},
  {1, 7, 3, 1, 4, 7, 4, 5, 7, -1, -1, -1, -1, -1, -1},
  {5, 8, 10, 2, 7, 3, -1, -1, -1, -1, -1, -1, -1, -1, -1},
  {4, 0, 1, 5, 8, 10, 3, 2, 7, -1, -1, -1, -1, -1, -1},
  {7, 8, 10, 7, 3, 8, 3, 0, 8, -1, -1, -1, -1, -1, -1},
  {8, 10, 7, 4, 8, 7, 4, 7, 3, 4, 3, 1, -1, -1, -1},
  {2, 7, 3, 8, 4, 9, -1, -1, -1, -1, -1, -1, -1, -1, -1},
  {0, 9, 8, 0, 1, 9, 3, 2, 7, -1, -1, -1, -1, -1, -1},
  {0, 7, 3, 0, 5, 7, 8, 4, 9, -1, -1, -1, -1, -1, -1},
  {8, 5, 7, 8, 7, 1, 8, 1, 9, 3, 1, 7, -1, -1, -1},
  {10, 4, 9, 10, 5, 4, 2, 7, 3, -1, -1, -1, -1, -1, -1},
  {3, 2, 7, 0, 1, 5, 1, 10, 5, 1, 9, 10, -1, -1, -1},
  {7, 3, 0, 7, 0, 9, 7, 9, 10, 9, 0, 4, -1, -1, -1},
  {7, 3, 1, 7, 1, 10, 10, 1, 9, -1, -1, -1, -1, -1, -1},
  {2, 6, 1, 7, 6, 2, -1, -1, -1, -1, -1, -1, -1, -1, -1},
  {2, 4, 0, 2, 7, 4, 7, 6, 4, -1, -1, -1, -1, -1, -1},
  {0, 6, 1, 0, 5, 6, 5, 7, 6, -1, -1, -1, -1, -1, -1},
  {5, 7, 4, 4, 7, 6, -1, -1, -1, -1, -1, -1, -1, -1, -1},
  {2, 6, 1, 2, 7, 6, 10, 5, 8, -1, -1, -1, -1, -1, -1},
  {10, 5, 8, 2, 7, 0, 7, 4, 0, 7, 6, 4, -1, -1, -1},
  {1, 0, 8, 1, 8, 7, 1, 7, 6, 10, 7, 8, -1, -1, -1},
  {8, 10, 7, 8, 7, 4, 4, 7, 6, -1, -1, -1, -1, -1, -1},
  {6, 2, 7, 6, 1, 2, 4, 9, 8, -1, -1, -1, -1, -1, -1},
  {2, 7, 6, 2, 6, 8, 2, 8, 0, 8, 6, 9, -1, -1, -1},
  {8, 4, 9, 0, 5, 1, 5, 6, 1, 5, 7, 6, -1, -1, -1},
  {9, 8, 5, 9, 5, 6, 6, 5, 7, -1, -1, -1, -1, -1, -1},
  {2, 7, 1, 1, 7, 6, 10, 5, 4, 10, 4, 9, -1, -1, -1},
  {7, 6, 0, 7, 0, 2, 6, 9, 0, 5, 0, 10, 9, 10, 0},
  {9, 10, 0, 9, 0, 4, 10, 7, 0, 1, 0, 6, 7, 6, 0},
  {9, 10, 7, 6, 9, 7, -1, -1, -1, -1, -1, -1, -1, -1, -1},
  {10, 11, 7, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
  {0, 1, 4, 7, 10, 11, -1, -1, -1, -1, -1, -1, -1, -1, -1},
  {2, 0, 5, 7, 10, 11, -1, -1, -1, -1, -1, -1, -1, -1, -1},
  {5, 1, 4, 5, 2, 1, 7, 10, 11, -1, -1, -1, -1, -1, -1},
  {5, 11, 7, 8, 11, 5, -1, -1, -1, -1, -1, -1, -1, -1, -1},
  {5, 11, 7, 5, 8, 11, 4, 0, 1, -1, -1, -1, -1, -1, -1},
  {2, 11, 7, 2, 0, 11, 0, 8, 11, -1, -1, -1, -1, -1, -1},
  {7, 2, 1, 7, 1, 8, 7, 8, 11, 4, 8, 1, -1, -1, -1},
  {8, 4, 9, 10, 11, 7, -1, -1, -1, -1, -1, -1, -1, -1, -1},
  {9, 0, 1, 9, 8, 0, 10, 11, 7, -1, -1, -1, -1, -1, -1},
  {0, 5, 2, 8, 4, 9, 7, 10, 11, -1, -1, -1, -1, -1, -1},
  {7, 10, 11, 5, 2, 8, 2, 9, 8, 2, 1, 9, -1, -1, -1},
  {11, 4, 9, 11, 7, 4, 7, 5, 4, -1, -1, -1, -1, -1, -1},
  {0, 1, 9, 0, 9, 7, 0, 7, 5, 7, 9, 11, -1, -1, -1},
  {4, 9, 11, 0, 4, 11, 0, 11, 7, 0, 7, 2, -1, -1, -1},
  {11, 7, 2, 11, 2, 9, 9, 2, 1, -1, -1, -1, -1, -1, -1},
  {7, 10, 11, 3, 6, 1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
  {3, 4, 0, 3, 6, 4, 11, 7, 10, -1, -1, -1, -1, -1, -1},
  {5, 2, 0, 7, 10, 11, 1, 3, 6, -1, -1, -1, -1, -1, -1},
  {10, 11, 7, 5, 2, 6, 5, 6, 4, 6, 2, 3, -1, -1, -1},
  {11, 5, 8, 11, 7, 5, 3, 6, 1, -1, -1, -1, -1, -1, -1},
  {5, 8, 7, 7, 8, 11, 4, 0, 3, 4, 3, 6, -1, -1, -1},
  {1, 3, 6, 2, 0, 7, 0, 11, 7, 0, 8, 11, -1, -1, -1},
  {6, 4, 2, 6, 2, 3, 4, 8, 2, 7, 2, 11, 8, 11, 2},
  {4, 9, 8, 6, 1, 3, 10, 11, 7, -1, -1, -1, -1, -1, -1},
  {7, 10, 11, 3, 6, 8, 3, 8, 0, 8, 6, 9, -1, -1, -1},
  {0, 5, 2, 3, 6, 1, 8, 4, 9, 7, 10, 11, -1, -1, -1},
  {2, 8, 5, 2, 9, 8, 2, 3, 9, 6, 9, 3, 7, 10, 11},
  {1, 3, 6, 4, 9, 7, 4, 7, 5, 7, 9, 11, -1, -1, -1},
  {7, 5, 9, 7, 9, 11, 5, 0, 9, 6, 9, 3, 0, 3, 9},
  {0, 7, 2, 0, 11, 7, 0, 4, 11, 9, 11, 4, 1, 3, 6},
  {11, 7, 2, 11, 2, 9, 3, 6, 2, 6, 9, 2, -1, -1, -1},
  {10, 3, 2, 11, 3, 10, -1, -1, -1, -1, -1, -1, -1, -1, -1},
  {3, 10, 11, 3, 2, 10, 0, 1, 4, -1, -1, -1, -1, -1, -1},
  {10, 0, 5, 10, 11, 0, 11, 3, 0, -1, -1, -1, -1, -1, -1},
  {1, 4, 5, 1, 5, 11, 1, 11, 3, 11, 5, 10, -1, -1, -1},
  {5, 3, 2, 5, 8, 3, 8, 11, 3, -1, -1, -1, -1, -1, -1},
  {4, 0, 1, 5, 8, 2, 8, 3, 2, 8, 11, 3, -1, -1, -1},
  {0, 8, 3, 3, 8, 11, -1, -1, -1, -1, -1, -1, -1, -1, -1},
  {1, 4, 8, 1, 8, 3, 3, 8, 11, -1, -1, -1, -1, -1, -1},
  {10, 3, 2, 10, 11, 3, 9, 8, 4, -1, -1, -1, -1, -1, -1},
  {0, 1, 8, 8, 1, 9, 3, 2, 10, 3, 10, 11, -1, -1, -1},
  {4, 9, 8, 0, 5, 11, 0, 11, 3, 11, 5, 10, -1, -1, -1},
  {11, 3, 5, 11, 5, 10, 3, 1, 5, 8, 5, 9, 1, 9, 5},
  {2, 11, 3, 2, 4, 11, 2, 5, 4, 9, 11, 4, -1, -1, -1},
  {1, 9, 5, 1, 5, 0, 9, 11, 5, 2, 5, 3, 11, 3, 5},
  {4, 9, 11, 4, 11, 0, 0, 11, 3, -1, -1, -1, -1, -1, -1},
  {11, 3, 1, 9, 11, 1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
  {6, 10, 11, 6, 1, 10, 1, 2, 10, -1, -1, -1, -1, -1, -1},
  {0, 6, 4, 0, 10, 6, 0, 2, 10, 11, 6, 10, -1, -1, -1},
  {10, 11, 6, 5, 10, 6, 5, 6, 1, 5, 1, 0, -1, -1, -1},
  {10, 11, 6, 10, 6, 5, 5, 6, 4, -1, -1, -1, -1, -1, -1},
  {5, 8, 11, 5, 11, 1, 5, 1, 2, 1, 11, 6, -1, -1, -1},
  {8, 11, 2, 8, 2, 5, 11, 6, 2, 0, 2, 4, 6, 4, 2},
  {6, 1, 0, 6, 0, 11, 11, 0, 8, -1, -1, -1, -1, -1, -1},
  {6, 4, 8, 11, 6, 8, -1, -1, -1, -1, -1, -1, -1, -1, -1},
  {8, 4, 9, 10, 11, 1, 10, 1, 2, 1, 11, 6, -1, -1, -1},
  {8, 0, 6, 8, 6, 9, 0, 2, 6, 11, 6, 10, 2, 10, 6},
  {5, 1, 0, 5, 6, 1, 5, 10, 6, 11, 6, 10, 8, 4, 9},
  {9, 8, 5, 9, 5, 6, 10, 11, 5, 11, 6, 5, -1, -1, -1},
  {1, 2, 11, 1, 11, 6, 2, 5, 11, 9, 11, 4, 5, 4, 11},
  {0, 2, 5, 9, 11, 6, -1, -1, -1, -1, -1, -1, -1, -1, -1},
  {6, 1, 0, 6, 0, 11, 4, 9, 0, 9, 11, 0, -1, -1, -1},
  {6, 9, 11, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
  {6, 11, 9, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
  {4, 0, 1, 9, 6, 11, -1, -1, -1, -1, -1, -1, -1, -1, -1},
  {0, 5, 2, 9, 6, 11, -1, -1, -1, -1, -1, -1, -1, -1, -1},
  {1, 5, 2, 1, 4, 5, 9, 6, 11, -1, -1, -1, -1, -1, -1},
  {10, 5, 8, 11, 9, 6, -1, -1, -1, -1, -1, -1, -1, -1, -1},
  {5, 8, 10, 4, 0, 1, 11, 9, 6, -1, -1, -1, -1, -1, -1},
  {8, 2, 0, 8, 10, 2, 11, 9, 6, -1, -1, -1, -1, -1, -1},
  {11, 9, 6, 8, 10, 4, 10, 1, 4, 10, 2, 1, -1, -1, -1},
  {6, 8, 4, 11, 8, 6, -1, -1, -1, -1, -1, -1, -1, -1, -1},
  {6, 0, 1, 6, 11, 0, 11, 8, 0, -1, -1, -1, -1, -1, -1},
  {8, 6, 11, 8, 4, 6, 0, 5, 2, -1, -1, -1, -1, -1, -1},
  {5, 11, 8, 5, 1, 11, 5, 2, 1, 1, 6, 11, -1, -1, -1},
  {10, 6, 11, 10, 5, 6, 5, 4, 6, -1, -1, -1, -1, -1, -1},
  {10, 6, 11, 5, 6, 10, 5, 1, 6, 5, 0, 1, -1, -1, -1},
  {0, 4, 6, 0, 6, 10, 0, 10, 2, 11, 10, 6, -1, -1, -1},
  {6, 11, 10, 6, 10, 1, 1, 10, 2, -1, -1, -1, -1, -1, -1},
  {11, 1, 3, 9, 1, 11, -1, -1, -1, -1, -1, -1, -1, -1, -1},
  {4, 11, 9, 4, 0, 11, 0, 3, 11, -1, -1, -1, -1, -1, -1},
  {1, 11, 9, 1, 3, 11, 2, 0, 5, -1, -1, -1, -1, -1, -1},
  {2, 3, 11, 2, 11, 4, 2, 4, 5, 9, 4, 11, -1, -1, -1},
  {11, 1, 3, 11, 9, 1, 8, 10, 5, -1, -1, -1, -1, -1, -1},
  {5, 8, 10, 4, 0, 9, 0, 11, 9, 0, 3, 11, -1, -1, -1},
  {3, 9, 1, 3, 11, 9, 0, 8, 2, 8, 10, 2, -1, -1, -1},
  {10, 2, 4, 10, 4, 8, 2, 3, 4, 9, 4, 11, 3, 11, 4},
  {1, 8, 4, 1, 3, 8, 3, 11, 8, -1, -1, -1, -1, -1, -1},
  {0, 3, 8, 3, 11, 8, -1, -1, -1, -1, -1, -1, -1, -1, -1},
  {5, 2, 0, 8, 4, 3, 8, 3, 11, 3, 4, 1, -1, -1, -1},
  {5, 2, 3, 5, 3, 8, 8, 3, 11, -1, -1, -1, -1, -1, -1},
  {1, 5, 4, 1, 11, 5, 1, 3, 11, 11, 10, 5, -1, -1, -1},
  {10, 5, 0, 10, 0, 11, 11, 0, 3, -1, -1, -1, -1, -1, -1},
  {3, 11, 4, 3, 4, 1, 11, 10, 4, 0, 4, 2, 10, 2, 4},
  {10, 2, 3, 11, 10, 3, -1, -1, -1, -1, -1, -1, -1, -1, -1},
  {3, 2, 7, 6, 11, 9, -1, -1, -1, -1, -1, -1, -1, -1, -1},
  {0, 1, 4, 3, 2, 7, 9, 6, 11, -1, -1, -1, -1, -1, -1},
  {7, 0, 5, 7, 3, 0, 6, 11, 9, -1, -1, -1, -1, -1, -1},
  {9, 6, 11, 1, 4, 3, 4, 7, 3, 4, 5, 7, -1, -1, -1},
  {2, 7, 3, 10, 5, 8, 6, 11, 9, -1, -1, -1, -1, -1, -1},
  {11, 9, 6, 5, 8, 10, 0, 1, 4, 3, 2, 7, -1, -1, -1},
  {6, 11, 9, 7, 3, 10, 3, 8, 10, 3, 0, 8, -1, -1, -1},
  {4, 3, 1, 4, 7, 3, 4, 8, 7, 10, 7, 8, 9, 6, 11},
  {6, 8, 4, 6, 11, 8, 7, 3, 2, -1, -1, -1, -1, -1, -1},
  {2, 7, 3, 0, 1, 11, 0, 11, 8, 11, 1, 6, -1, -1, -1},
  {4, 11, 8, 4, 6, 11, 5, 7, 0, 7, 3, 0, -1, -1, -1},
  {11, 8, 1, 11, 1, 6, 8, 5, 1, 3, 1, 7, 5, 7, 1},
  {2, 7, 3, 10, 5, 11, 5, 6, 11, 5, 4, 6, -1, -1, -1},
  {5, 11, 10, 5, 6, 11, 5, 0, 6, 1, 6, 0, 2, 7, 3},
  {3, 0, 10, 3, 10, 7, 0, 4, 10, 11, 10, 6, 4, 6, 10},
  {6, 11, 10, 6, 10, 1, 7, 3, 10, 3, 1, 10, -1, -1, -1},
  {11, 2, 7, 11, 9, 2, 9, 1, 2, -1, -1, -1, -1, -1, -1},
  {4, 11, 9, 0, 11, 4, 0, 7, 11, 0, 2, 7, -1, -1, -1},
  {0, 9, 1, 0, 7, 9, 0, 5, 7, 7, 11, 9, -1, -1, -1},
  {11, 9, 4, 11, 4, 7, 7, 4, 5, -1, -1, -1, -1, -1, -1},
  {5, 8, 10, 2, 7, 9, 2, 9, 1, 9, 7, 11, -1, -1, -1},
  {0, 9, 4, 0, 11, 9, 0, 2, 11, 7, 11, 2, 5, 8, 10},
  {9, 1, 7, 9, 7, 11, 1, 0, 7, 10, 7, 8, 0, 8, 7},
  {11, 9, 4, 11, 4, 7, 8, 10, 4, 10, 7, 4, -1, -1, -1},
  {7, 1, 2, 7, 8, 1, 7, 11, 8, 4, 1, 8, -1, -1, -1},
  {2, 7, 11, 2, 11, 0, 0, 11, 8, -1, -1, -1, -1, -1, -1},
  {5, 7, 1, 5, 1, 0, 7, 11, 1, 4, 1, 8, 11, 8, 1},
  {5, 7, 11, 8, 5, 11, -1, -1, -1, -1, -1, -1, -1, -1, -1},
  {5, 4, 11, 5, 11, 10, 4, 1, 11, 7, 11, 2, 1, 2, 11},
  {10, 5, 0, 10, 0, 11, 2, 7, 0, 7, 11, 0, -1, -1, -1},
  {0, 4, 1, 7, 11, 10, -1, -1, -1, -1, -1, -1, -1, -1, -1},
  {10, 7, 11, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
  {9, 7, 10, 6, 7, 9, -1, -1, -1, -1, -1, -1, -1, -1, -1},
  {9, 7, 10, 9, 6, 7, 1, 4, 0, -1, -1, -1, -1, -1, -1},
  {7, 9, 6, 7, 10, 9, 5, 2, 0, -1, -1, -1, -1, -1, -1},
  {10, 6, 7, 10, 9, 6, 2, 1, 5, 1, 4, 5, -1, -1, -1},
  {9, 5, 8, 9, 6, 5, 6, 7, 5, -1, -1, -1, -1, -1, -1},
  {0, 1, 4, 5, 8, 6, 5, 6, 7, 6, 8, 9, -1, -1, -1},
  {2, 6, 7, 2, 8, 6, 2, 0, 8, 8, 9, 6, -1, -1, -1},
  {6, 7, 8, 6, 8, 9, 7, 2, 8, 4, 8, 1, 2, 1, 8},
  {8, 7, 10, 8, 4, 7, 4, 6, 7, -1, -1, -1, -1, -1, -1},
  {1, 8, 0, 1, 7, 8, 1, 6, 7, 10, 8, 7, -1, -1, -1},
  {2, 0, 5, 7, 10, 4, 7, 4, 6, 4, 10, 8, -1, -1, -1},
  {2, 1, 8, 2, 8, 5, 1, 6, 8, 10, 8, 7, 6, 7, 8},
  {5, 4, 7, 4, 6, 7, -1, -1, -1, -1, -1, -1, -1, -1, -1},
  {0, 1, 6, 0, 6, 5, 5, 6, 7, -1, -1, -1, -1, -1, -1},
  {2, 0, 4, 2, 4, 7, 7, 4, 6, -1, -1, -1, -1, -1, -1},
  {2, 1, 6, 7, 2, 6, -1, -1, -1, -1, -1, -1, -1, -1, -1},
  {7, 1, 3, 7, 10, 1, 10, 9, 1, -1, -1, -1, -1, -1, -1},
  {7, 0, 3, 7, 9, 0, 7, 10, 9, 9, 4, 0, -1, -1, -1},
  {0, 5, 2, 1, 3, 10, 1, 10, 9, 10, 3, 7, -1, -1, -1},
  {10, 9, 3, 10, 3, 7, 9, 4, 3, 2, 3, 5, 4, 5, 3},
  {8, 7, 5, 8, 1, 7, 8, 9, 1, 3, 7, 1, -1, -1, -1},
  {0, 3, 9, 0, 9, 4, 3, 7, 9, 8, 9, 5, 7, 5, 9},
  {0, 8, 7, 0, 7, 2, 8, 9, 7, 3, 7, 1, 9, 1, 7},
  {2, 3, 7, 8, 9, 4, -1, -1, -1, -1, -1, -1, -1, -1, -1},
  {8, 7, 10, 4, 7, 8, 4, 3, 7, 4, 1, 3, -1, -1, -1},
  {7, 10, 8, 7, 8, 3, 3, 8, 0, -1, -1, -1, -1, -1, -1},
  {4, 10, 8, 4, 7, 10, 4, 1, 7, 3, 7, 1, 0, 5, 2},
  {7, 10, 8, 7, 8, 3, 5, 2, 8, 2, 3, 8, -1, -1, -1},
  {1, 3, 7, 1, 7, 4, 4, 7, 5, -1, -1, -1, -1, -1, -1},
  {0, 3, 7, 5, 0, 7, -1, -1, -1, -1, -1, -1, -1, -1, -1},
  {1, 3, 7, 1, 7, 4, 2, 0, 7, 0, 4, 7, -1, -1, -1},
  {2, 3, 7, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
  {3, 9, 6, 3, 2, 9, 2, 10, 9, -1, -1, -1, -1, -1, -1},
  {0, 1, 4, 3, 2, 6, 2, 9, 6, 2, 10, 9, -1, -1, -1},
  {5, 10, 9, 5, 9, 3, 5, 3, 0, 6, 3, 9, -1, -1, -1},
  {4, 5, 3, 4, 3, 1, 5, 10, 3, 6, 3, 9, 10, 9, 3},
  {3, 9, 6, 2, 9, 3, 2, 8, 9, 2, 5, 8, -1, -1, -1},
  {2, 6, 3, 2, 9, 6, 2, 5, 9, 8, 9, 5, 0, 1, 4},
  {9, 6, 3, 9, 3, 8, 8, 3, 0, -1, -1, -1, -1, -1, -1},
  {9, 6, 3, 9, 3, 8, 1, 4, 3, 4, 8, 3, -1, -1, -1},
  {8, 2, 10, 8, 6, 2, 8, 4, 6, 6, 3, 2, -1, -1, -1},
  {2, 10, 6, 2, 6, 3, 10, 8, 6, 1, 6, 0, 8, 0, 6},
  {4, 6, 10, 4, 10, 8, 6, 3, 10, 5, 10, 0, 3, 0, 10},
  {5, 10, 8, 1, 6, 3, -1, -1, -1, -1, -1, -1, -1, -1, -1},
  {3, 2, 5, 3, 5, 6, 6, 5, 4, -1, -1, -1, -1, -1, -1},
  {3, 2, 5, 3, 5, 6, 0, 1, 5, 1, 6, 5, -1, -1, -1},
  {3, 0, 4, 6, 3, 4, -1, -1, -1, -1, -1, -1, -1, -1, -1},
  {3, 1, 6, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
  {2, 10, 1, 10, 9, 1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
  {4, 0, 2, 4, 2, 9, 9, 2, 10, -1, -1, -1, -1, -1, -1},
  {0, 5, 10, 0, 10, 1, 1, 10, 9, -1, -1, -1, -1, -1, -1},
  {4, 5, 10, 9, 4, 10, -1, -1, -1, -1, -1, -1, -1, -1, -1},
  {5, 8, 9, 5, 9, 2, 2, 9, 1, -1, -1, -1, -1, -1, -1},
  {4, 0, 2, 4, 2, 9, 5, 8, 2, 8, 9, 2, -1, -1, -1},
  {0, 8, 9, 1, 0, 9, -1, -1, -1, -1, -1, -1, -1, -1, -1},
  {4, 8, 9, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
  {8, 4, 1, 8, 1, 10, 10, 1, 2, -1, -1, -1, -1, -1, -1},
  {2, 10, 8, 0, 2, 8, -1, -1, -1, -1, -1, -1, -1, -1, -1},
  {8, 4, 1, 8, 1, 10, 0, 5, 1, 5, 10, 1, -1, -1, -1},
  {5, 10, 8, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
  {5, 4, 1, 2, 5, 1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
  {0, 2, 5, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
  {0, 4, 1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
  {-1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1}
};
__constant__ int ssrlcv::cubeCategoryEdgeIdentity[256] = {0, 19, 37, 54, 1312, 1331, 1285, 1302,
  784, 771, 821, 806, 1584, 1571, 1557, 1542, 74, 89, 111, 124, 1386, 1401, 1359, 1372,
  858, 841, 895, 876, 1658, 1641, 1631, 1612, 140, 159, 169, 186, 1452, 1471, 1417, 1434,
  924, 911, 953, 938, 1724, 1711, 1689, 1674, 198, 213, 227, 240, 1510, 1525, 1475, 1488,
  982, 965, 1011, 992, 1782, 1765, 1747, 1728, 3200, 3219, 3237, 3254, 2464, 2483, 2437,
  2454, 3984, 3971, 4021, 4006, 2736, 2723, 2709, 2694, 3274, 3289, 3311, 3324, 2538,
  2553, 2511, 2524, 4058, 4041, 4095, 876, 2810, 2793, 2709, 2764, 3084, 3103, 3113,
  3130, 2348, 2367, 2313, 2330, 3868, 3855, 3897, 3882, 2620, 2607, 2585, 2570, 3142,
  3157, 3171, 3184, 2406, 2421, 2371, 2384, 3926, 3909, 3171, 3936, 2678, 2661, 2643,
  2624, 2624, 2643, 2661, 2678, 3936, 3955, 3909, 3926, 2384, 2371, 2421, 2406, 3184,
  3171, 3157, 3142, 2570, 2585, 2607, 2620, 3882, 3897, 3855, 3868, 2330, 2313, 2367,
  2348, 3130, 3113, 3103, 3084, 2764, 2783, 2793, 2810, 4076, 4095, 4041, 1434, 2524,
  2511, 2553, 2538, 3324, 3171, 3289, 3274, 2694, 2709, 2723, 2736, 4006, 2709, 3971,
  3984, 2454, 2437, 2483, 2464, 3254, 3237, 3219, 3200, 1728, 1747, 1765, 1782, 992,
  1011, 965, 982, 1488, 1475, 1525, 1510, 240, 227, 213, 198, 1674, 1689, 1711, 1724,
  938, 953, 911, 924, 1434, 1417, 1434, 1452, 186, 169, 159, 140, 1612, 1631, 1641,
  1658, 876, 876, 841, 858, 1372, 1359, 1401, 1386, 124, 111, 89, 74, 1542, 1557,
  1571, 1584, 806, 821, 771, 784, 1302, 1285, 1331, 1312, 54, 37, 19, 0};
__constant__ int ssrlcv::numTrianglesInCubeCategory[256] = {0, 1, 1, 2, 1, 2, 2, 3, 1, 2,
  2, 3, 2, 3, 3, 2, 1, 2, 2, 3, 2, 3, 3, 4, 2, 3, 3, 4, 3, 4, 4, 3, 1, 2, 2, 3, 2,
  3, 3, 4, 2, 3, 3, 4, 3, 4, 4, 3, 2, 3, 3, 2, 3, 4, 4, 3, 3, 4, 4, 3, 4, 5, 5, 2,
  1, 2, 2, 3, 2, 3, 3, 4, 2, 3, 3, 4, 3, 4, 4, 3, 2, 3, 3, 4, 3, 4, 4, 5, 3, 4, 4,
  5, 4, 5, 5, 4, 2, 3, 3, 4, 3, 4, 2, 3, 3, 4, 4, 5, 4, 5, 3, 2, 3, 4, 4, 3, 4, 5,
  3, 2, 4, 5, 5, 4, 5, 2, 4, 1, 1, 2, 2, 3, 2, 3, 3, 4, 2, 3, 3, 4, 3, 4, 4, 3, 2,
  3, 3, 4, 3, 4, 4, 5, 3, 2, 4, 3, 4, 3, 5, 2, 2, 3, 3, 4, 3, 4, 4, 5, 3, 4, 4, 5,
  4, 5, 5, 4, 3, 4, 4, 3, 4, 5, 5, 4, 4, 3, 5, 2, 5, 4, 2, 1, 2, 3, 3, 4, 3, 4, 4,
  5, 3, 4, 4, 5, 2, 3, 3, 2, 3, 4, 4, 5, 4, 5, 5, 2, 4, 3, 5, 4, 3, 2, 4, 1, 3, 4,
  4, 5, 4, 5, 3, 4, 4, 5, 5, 2, 3, 4, 2, 1, 2, 3, 3, 2, 3, 4, 2, 1, 3, 2, 4, 1, 2,
  1, 1, 0};

  // =============================================================================================================
  //
  // Device Kernels
  //
  // =============================================================================================================

__global__ void ssrlcv::vertexImplicitFromNormals(int numVertices, Octree::Vertex* vertexArray, Octree::Node* nodeArray, float3* normals, float3* points, float* vertexImplicit){
  int blockID = blockIdx.y * gridDim.x + blockIdx.x;
  if(blockID < numVertices){
    int node = -1;
    int nodes[8] = {0};
    for(int i = 0; i < 8; ++i) nodes[i] = vertexArray[blockID].nodes[i];
    float3 vertex = vertexArray[blockID].coord;
    int numPoints = 0;
    int pointIndex = -1;
    float3 currentNormal = {0.0f,0.0f,0.0f};
    float3 currentVector = {0.0f,0.0f,0.0f};
    float smallestDistanceSq = FLT_MAX;
    float currentDistanceSq = 0.0f;
    int closestPoint = -1;
    while(closestPoint == -1){
      for(int nd = 0; nd < 8; ++nd){
        node = nodes[nd];
        if(node == -1) continue;
        numPoints = nodeArray[node].numPoints;
        pointIndex = nodeArray[node].pointIndex;
        for(int p = pointIndex; p < pointIndex + numPoints; ++p){
          currentDistanceSq = dotProduct(vertex - points[p],vertex - points[p]);
          if(smallestDistanceSq > currentDistanceSq){
            smallestDistanceSq = currentDistanceSq;
            closestPoint = p;
          }
        }
        nodes[nd] = nodeArray[nodes[nd]].parent;
      }
    }
    currentNormal = normals[closestPoint];
    currentNormal = currentNormal/sqrtf(dotProduct(currentNormal,currentNormal));
    currentVector = vertex - points[closestPoint];
    currentVector = currentVector/sqrtf(dotProduct(currentVector,currentVector));
    vertexImplicit[blockID] = dotProduct(currentNormal,currentVector);
  }
}
__global__ void ssrlcv::calcVertexNumbers(int numEdges, int depthIndex, Octree::Edge* edgeArray, float* vertexImplicit, int* vertexNumbers){
  int globalID = blockIdx.x * blockDim.x + threadIdx.x;
  if(globalID < numEdges){
    float impV1 = 0;
    float impV2 = 0;
    impV1 = vertexImplicit[edgeArray[globalID + depthIndex].v1];
    impV2 = vertexImplicit[edgeArray[globalID + depthIndex].v2];
    if(impV1 > 0.0f && impV2 < 0.0f || impV1 < 0.0f && impV2 > 0.0f || impV1 == 0.0f || impV2 == 0.0f){
      vertexNumbers[globalID] = 1;
    }
    else{
      vertexNumbers[globalID] = 0;
    }
  }
}

//adaptive Marching cubes
__global__ void ssrlcv::categorizeCubesRecursively_child(int parent, int parentCategory, Octree::Edge* edgeArray, Octree::Node* nodeArray, int* vertexNumbers, int* cubeCategory, int* triangleNumbers){
  __shared__ int numTrianglesFromChildren;
  numTrianglesFromChildren = 0;
  __syncthreads();
  int childIndex = nodeArray[parent].children[threadIdx.x];
  if(childIndex == -1) return;
  int edgeBasedCategory = 0;
  int regEdge = 0;
  int category = 0;
  for(int i = 11; i >= 0; --i){
    regEdge = nodeArray[childIndex].edges[i];
    if(vertexNumbers[regEdge]){
      edgeBasedCategory = (edgeBasedCategory << 1) + 1;
    }
    else{
      edgeBasedCategory <<= 1;
    }
  }
  for(int i = 0; i < 256; ++i){
    if(edgeBasedCategory == cubeCategoryEdgeIdentity[i]){
      category = i;
      atomicAdd(&numTrianglesFromChildren, numTrianglesInCubeCategory[i]);
      break;
    }
  }
  __syncthreads();
  if(numTrianglesFromChildren < numTrianglesInCubeCategory[parentCategory]) return;
  triangleNumbers[parent] = 0;
  cubeCategory[parent] = 0;
  triangleNumbers[childIndex] = numTrianglesInCubeCategory[category];
  cubeCategory[childIndex] = category;
  categorizeCubesRecursively_child<<<1,8>>>(childIndex, category, edgeArray, nodeArray, vertexNumbers, cubeCategory, triangleNumbers);
  cudaDeviceSynchronize();
}
__global__ void ssrlcv::categorizeCubesRecursively(int firstChildrenIndex, Octree::Edge* edgeArray, Octree::Node* nodeArray, int* vertexNumbers, int* cubeCategory, int* triangleNumbers){
  int edgeBasedCategory = 0;
  int regEdge = 0;
  int category = 0;
  for(int i = 11; i >= 0; --i){
    regEdge = nodeArray[firstChildrenIndex + threadIdx.x].edges[i];
    if(vertexNumbers[regEdge]){
      edgeBasedCategory = (edgeBasedCategory << 1) + 1;
    }
    else{
      edgeBasedCategory <<= 1;
    }
  }
  for(int i = 0; i < 256; ++i){
    if(edgeBasedCategory == cubeCategoryEdgeIdentity[i]){
      category = i;
      break;
    }
  }
  triangleNumbers[firstChildrenIndex + threadIdx.x] = numTrianglesInCubeCategory[category];
  cubeCategory[firstChildrenIndex + threadIdx.x] = category;
  categorizeCubesRecursively_child<<<1,8>>>(firstChildrenIndex + threadIdx.x, category, edgeArray, nodeArray, vertexNumbers, cubeCategory, triangleNumbers);
  cudaDeviceSynchronize();

}
__global__ void ssrlcv::minimizeVertices(int numEdges, Octree::Edge* edgeArray, Octree::Node* nodeArray, int* cubeCategory, int* vertexNumbers){
  int blockID = blockIdx.y * gridDim.x + blockIdx.x;
  if(blockID < numEdges){
    vertexNumbers[blockID] = 0;
    __syncthreads();
    int nodeIndex = edgeArray[blockID].nodes[threadIdx.x];
    int edgeOfNode = -1;
    if(nodeIndex == -1) return;
    for(int i = 0; i < 12; ++i){
      if(nodeArray[nodeIndex].edges[i] == blockID){
        edgeOfNode = i;
      }
    }
    int category = cubeCategory[nodeIndex];
    if(category <= 0 || category == 255){
      return;
    }
    for(int i = 0; i < 15; ++i){
      if(edgeOfNode == cubeCategoryTrianglesFromEdges[category][i]){
        vertexNumbers[blockID] = 1;
        return;
      }
    }
  }
}

//Marching cubes
__global__ void ssrlcv::determineCubeCategories(int numNodes, int nodeIndex, int edgeIndex, Octree::Node* nodeArray, int* vertexNumbers, int* cubeCategory, int* triangleNumbers){
  int globalID = blockIdx.x * blockDim.x + threadIdx.x;
  if(globalID < numNodes){
    int edgeBasedCategory = 0;
    int regEdge = 0;
    for(int i = 11; i >= 0; --i){
      regEdge = nodeArray[globalID + nodeIndex].edges[i];
      if(vertexNumbers[regEdge - edgeIndex]){
        edgeBasedCategory = (edgeBasedCategory << 1) + 1;
      }
      else{
        edgeBasedCategory <<= 1;
      }
    }
    triangleNumbers[globalID] = 0;
    for(int i = 0; i < 256; ++i){
      if(edgeBasedCategory == cubeCategoryEdgeIdentity[i]){
        triangleNumbers[globalID] = numTrianglesInCubeCategory[i];
        cubeCategory[globalID] = i;
        break;
      }
    }
  }
}
__global__ void ssrlcv::generateSurfaceVertices(int numEdges, int depthIndex, Octree::Edge* edgeArray, Octree::Vertex* vertexArray, int* vertexNumbers, int* vertexAddresses, float3* surfaceVertices){
  int globalID = blockIdx.x * blockDim.x + threadIdx.x;
  if(globalID < numEdges){
    if(vertexNumbers[globalID] == 1){
      int v1 = edgeArray[globalID + depthIndex].v1;
      int v2 = edgeArray[globalID + depthIndex].v2;
      float3 midPoint = vertexArray[v1].coord + vertexArray[v2].coord;
      midPoint = midPoint/2.0f;
      int vertAddress = (globalID == 0) ? 0 : vertexAddresses[globalID - 1];
      surfaceVertices[vertAddress] = midPoint;
    }
  }
}
__global__ void ssrlcv::generateSurfaceTriangles(int numNodes, int nodeIndex, int edgeIndex, Octree::Node* nodeArray, int* vertexAddresses, int* triangleAddresses, int* cubeCategory, int3* surfaceTriangles){
  int blockID = blockIdx.y * gridDim.x + blockIdx.x;
  if(blockID < numNodes){
    int3 nodeTriangle = {cubeCategoryTrianglesFromEdges[cubeCategory[blockID]][threadIdx.x*3],
      cubeCategoryTrianglesFromEdges[cubeCategory[blockID]][threadIdx.x*3 + 1],
      cubeCategoryTrianglesFromEdges[cubeCategory[blockID]][threadIdx.x*3 + 2]};
    if(nodeTriangle.x != -1){
      int3 surfaceTriangle = {nodeArray[blockID + nodeIndex].edges[nodeTriangle.x] - edgeIndex,
        nodeArray[blockID + nodeIndex].edges[nodeTriangle.y] - edgeIndex,
        nodeArray[blockID + nodeIndex].edges[nodeTriangle.z] - edgeIndex};
      int triAddress = (blockID == 0) ? threadIdx.x: triangleAddresses[blockID - 1] + threadIdx.x;
      int3 vertAddress = {-1,-1,-1};
      vertAddress.x = (surfaceTriangle.x == 0) ? 0 : vertexAddresses[surfaceTriangle.x - 1];
      vertAddress.y = (surfaceTriangle.y == 0) ? 0 : vertexAddresses[surfaceTriangle.y - 1];
      vertAddress.z = (surfaceTriangle.z == 0) ? 0 : vertexAddresses[surfaceTriangle.z - 1];
      surfaceTriangles[triAddress] = vertAddress;
    }
  }
}
