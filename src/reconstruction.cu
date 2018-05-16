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
        octree.normals[i].x *= -1;
        octree.normals[i].y *= -1;
        octree.normals[i].z *= -1;
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
      //missing parameters when comparing to hoppes algorithm on: numNodesD and currentNodeDepth
      //https://devtalk.nvidia.com/default/topic/609551/teaching-and-curriculum-support/
      //my-cuda-programming-lecture-and-teaching-of-poisson-parallel-surface-reconstruction-in-a-summer-scho/

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

      //missing parameters when comparing to hoppes algorithm on: numNodesD and currentNodeDepth
    //https://devtalk.nvidia.com/default/topic/609551/teaching-and-curriculum-support/
    //my-cuda-programming-lecture-and-teaching-of-poisson-parallel-surface-reconstruction-in-a-summer-scho/
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

      printf("TOTAL NODES = %d\n\n",octree.totalNodes);
      octree.fillLUTs();
      octree.fillNeighborhoods();

      octree.computeVertexArray();
      octree.computeEdgeArray();
      //octree.computeFaceArray();

      totalTimer = clock() - totalTimer;
      printf("\nOCTREE BUILD TOOK %f seconds.\n\n",((float) totalTimer)/CLOCKS_PER_SEC);
      //this will destroy all memory on the GPU
      //should place more cudaFrees throughout the program
      cudaDeviceReset();

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
