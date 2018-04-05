#include "common_includes.h"
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

void printBits(size_t const size, void const * const ptr){
    unsigned char *b = (unsigned char*) ptr;
    unsigned char byte;
    int i, j;

    for (i=size-1;i>=0;i--)
    {
        for (j=7;j>=0;j--)
        {
            byte = (b[i] >> j) & 1;
            printf("%u", byte);
        }
    }
    puts("");
}


int main(int argc, char *argv[]){
  try{
    if(argc == 2){
      string filePath = argv[1];
      clock_t totalTimer = clock();

      //if we want further depth than 10 our nodeKeys will need to then be long or long long
      int depth = 10;
      Octree octree = Octree(filePath, depth);

      //this will be temporary due to the normals needing to be facing inward
      for(int i = 0; i < octree.numPoints; ++i){
        octree.normals[i].x = octree.normals[i].x * -1;
        octree.normals[i].y = octree.normals[i].y * -1;
        octree.normals[i].z = octree.normals[i].z * -1;
      }

      dim3 grid = {1,1,1};
      dim3 block = {1,1,1};

      //this is set for getNodeKeys
      if(octree.numPoints < 65535) grid.x = (unsigned int) octree.numPoints;
      else{
        grid.x = 65535;
        while(grid.x*block.x < octree.numPoints){
          ++block.x;
        }
        while(grid.x*block.x > octree.numPoints){
          --grid.x;
        }
      }

      clock_t cudatimer;

      cudatimer = clock();
      CudaSafeCall(cudaMalloc((void**)&octree.pointsDevice, octree.numPoints * sizeof(float3)));
      CudaSafeCall(cudaMalloc((void**)&octree.normalsDevice, octree.numPoints * sizeof(float3)));
      CudaSafeCall(cudaMalloc((void**)&octree.finestNodeCentersDevice, octree.numPoints * sizeof(float3)));
      CudaSafeCall(cudaMalloc((void**)&octree.finestNodeKeysDevice, octree.numPoints * sizeof(int)));
      CudaSafeCall(cudaMalloc((void**)&octree.finestNodePointIndexesDevice, octree.numPoints * sizeof(int)));
      cudatimer = clock() - cudatimer;
      printf("initial allocation took %f seconds.\n\n",((float) cudatimer)/CLOCKS_PER_SEC);

      /*
      Gets unique key for each node that houses points.
      */
      cudatimer = clock();
      octree.copyPointsToDevice();
      octree.copyFinestNodeCentersToDevice();
      octree.copyFinestNodeKeysToDevice();
      octree.copyNormalsToDevice();

      cudatimer = clock() - cudatimer;
      printf("cudaMemcpyHostToDevice for key retrieval took %f seconds.\n",((float) cudatimer)/CLOCKS_PER_SEC);

      printf("\ngetNodeKeys KERNEL: grid = {%d,%d,%d} - block = {%d,%d,%d}\n",grid.x, grid.y, grid.z, block.x, block.y, block.z);
      cudatimer = clock();
      octree.executeKeyRetrieval(grid, block);//get keys at lowest depth
      cudatimer = clock() - cudatimer;
      printf("getnodeKeys kernel took %f seconds.\n\n",((float) cudatimer)/CLOCKS_PER_SEC);

      /*
      Sort all arrays on host by key
      */
      cudatimer = clock();
      octree.sortByKey();
      cudatimer = clock() - cudatimer;
      printf("octree sort_by_key took %f seconds.\n\n",((float) cudatimer)/CLOCKS_PER_SEC);

      //copy sorted general variables back over
      cudatimer = clock();
      octree.copyFinestNodeCentersToHost();
      octree.copyFinestNodeKeysToHost();
      octree.copyPointsToHost();
      octree.copyNormalsToHost();
      cudatimer = clock() - cudatimer;
      printf("cudaMemcpyDeviceToHost took %f seconds.\n\n",((float) cudatimer)/CLOCKS_PER_SEC);

      /*
      Compact the nodeKeys, point indexes, nodeCenters so that we have
      the unique nodes at the finest levelD.
      */
      cudatimer = clock();
      octree.compactData();
      cudatimer = clock() - cudatimer;
      printf("octree unique_by_key took %f seconds.\n\n",((float) cudatimer)/CLOCKS_PER_SEC);

      octree.fillUniqueNodesAtFinestLevel();
      octree.createFinalNodeArray();
      octree.fillLUTs();
      octree.printLUTs();

      printf("TOTAL NODES = %d",octree.totalNodes);

      //this should be a parent table and child table
      //octree.fillNeighborhoods();

      //octree.computeVertexArray();
      //octree.computeEdgeArray();
      //octree.computeFaceArray();S

      CudaSafeCall(cudaFree(octree.finestNodeKeysDevice));
      CudaSafeCall(cudaFree(octree.finestNodeCentersDevice));
      CudaSafeCall(cudaFree(octree.pointsDevice));
      CudaSafeCall(cudaFree(octree.normalsDevice));
      CudaSafeCall(cudaFree(octree.finestNodePointIndexesDevice));
      totalTimer = clock() - totalTimer;
      printf("\nRECONSTRUCTION TOOK %f seconds.\n\n",((float) totalTimer)/CLOCKS_PER_SEC);


      return 0;
    }
    else{
      cout<<"LACK OF PLY INPUT...goodbye"<<endl;
      exit(1);
    }
  }
  catch (const std::exception &e){
      std::cerr << "Caught exception: " << e.what() << '\n';
      std::exit(1);
  }
  catch (...){
      std::cerr << "Caught unknown exception\n";
      std::exit(1);
  }

}
