#include "common_includes.h"
#include "octree.cuh"
#include <thrust/scan.h>
#include <thrust/device_ptr.h>
#include <thrust/sort.h>
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
      CudaSafeCall(cudaMalloc((void**)&octree.nodeCentersDevice, octree.numPoints * sizeof(float3)));
      CudaSafeCall(cudaMalloc((void**)&octree.nodeKeysDevice, octree.numPoints * sizeof(int)));
      CudaSafeCall(cudaMalloc((void**)&octree.nodePointIndexesDevice, octree.numPoints * sizeof(int)));
      cudatimer = clock() - cudatimer;
      printf("initial allocation took %f seconds.\n\n",((float) cudatimer)/CLOCKS_PER_SEC);

      /*
      Gets unique key for each node that houses points.
      */
      cudatimer = clock();
      octree.copyPointsToDevice();
      octree.copyNodeCentersToDevice();
      octree.copyNodeKeysToDevice();
      octree.copyNormalsToDevice();

      cudatimer = clock() - cudatimer;
      printf("cudaMemcpyHostToDevice for key retrieval took %f seconds.\n",((float) cudatimer)/CLOCKS_PER_SEC);

      printf("\ngetNodeKeys KERNEL: grid = {%d,%d,%d} - block = {%d,%d,%d}\n",grid.x, grid.y, grid.z, block.x, block.y, block.z);
      cudatimer = clock();
      octree.executeKeyRetrieval(grid, block);
      cudatimer = clock() - cudatimer;
      printf("getnodeKeys kernel took %f seconds.\n\n",((float) cudatimer)/CLOCKS_PER_SEC);


      /*
      Sort all arrays on host by key
      */
      octree.copyNodeKeysToHost();
      int* sortCheck = new int[octree.numPoints];
      for(int i = 0; i < octree.numPoints; ++i){
        sortCheck[i] = octree.nodeKeys[i];
      }
      clock_t ctimer = clock();
      sort(sortCheck, sortCheck + octree.numPoints);
      ctimer = clock() - ctimer;
      printf("sort with c++ took %f seconds.\n\n",((float) ctimer)/CLOCKS_PER_SEC);

      cudatimer = clock();
      octree.sortByKey();
      cudatimer = clock() - cudatimer;
      printf("octree sort_by_key took %f seconds.\n\n",((float) cudatimer)/CLOCKS_PER_SEC);

      //copy sorted general variables back over




      cudatimer = clock();
      octree.copyNodeCentersToHost();
      octree.copyNodeKeysToHost();
      octree.copyPointsToHost();
      octree.copyNormalsToHost();
      cudatimer = clock() - cudatimer;
      printf("cudaMemcpyDeviceToHost took %f seconds.\n\n",((float) cudatimer)/CLOCKS_PER_SEC);
      for(int i = 0; i < octree.numPoints; ++i){
        //printf("sorting with cuda:%d - sorting with c++ sort() %d\n", octree.nodeKeys[i], sortCheck[i]);
      }
      /*
      Compact the nodeKeys, point indexes, nodeCenters so that we have
      the unique nodes at the finest levelD.
      */
      cudatimer = clock();
      octree.compactData();
      cudatimer = clock() - cudatimer;
      printf("octree unique_by_key took %f seconds.\n\n",((float) cudatimer)/CLOCKS_PER_SEC);


      octree.fillInUniqueNodes();

      //for(int i = 0; i < octree.numUniqueNodes; ++i){
      //  printBits(sizeof(int), &octree.nodeKeys[i]);
      //}

      /*
      Now we have numUniqueNodes, sorted pointIndexes for nodes, and sorted
      unique keys per node, and sorted nodeCenters,octree.nodeAddresses
      */

      printf("NUMBER OF UNIQUE NODES = %d\n", octree.numUniqueNodes);
      Node* uniqueNodeArrayDevice;
      CudaSafeCall(cudaMalloc((void**)&uniqueNodeArrayDevice, octree.numUniqueNodes * sizeof(Node)));
      CudaSafeCall(cudaMemcpy(octree.uniqueNodeArray, uniqueNodeArrayDevice,  octree.numUniqueNodes * sizeof(Node), cudaMemcpyDeviceToHost));

      //this is about to get hairy
      Node** nodesInEachDepthArrayDevice;
      CudaSafeCall(cudaMalloc((void**)& nodesInEachDepthArrayDevice, (octree.depth + 1)*sizeof(Node*)));
      Node** nodesInEachDepthArray = new Node*[octree.depth + 1];
      CudaSafeCall(cudaMemcpy(nodesInEachDepthArray, nodesInEachDepthArrayDevice, (octree.depth + 1)*sizeof(Node*), cudaMemcpyDeviceToHost));
      //octree.totalNodes = octree.nodeAddresses[octree.numUniqueNodes - 1];//?????
      int* nodeBaseAddressDevice;
      int* depthStartingIndex = new int[octree.depth + 1];
      int currentUniqueNodeNum = octree.numUniqueNodes;
      int nodesInDepth = 0;
      int currentNodeDepth = 0;//what is this?????

      //octree.finalNodeArray = finalNodeArray
      //octree.totalNodes = totalNodes
      /*
      for(int d = octree.depth; d >= 0; --d){
        CudaSafeCall(cudaMalloc((void**)&nodeBaseAddressDevice, currentUniqueNodeNum*sizeof(int)));
        //this is set for fillNodes

        grid = {1,1,1};
        block = {1,1,1};

        if(currentUniqueNodeNum < 65535) grid.x = (unsigned int) currentUniqueNodeNum;
        else{
          grid.x = 65535;
          while(grid.x*block.x < currentUniqueNodeNum){
            ++block.x;
          }
          while(grid.x*block.x > currentUniqueNodeNum){
            --grid.x;
          }
        }
        octree.executeFindAllNodes(grid, block,currentUniqueNodeNum,nodeBaseAddressDevice, uniqueNodeArrayDevice);//this is helping find all nodes that are contained in a parent
        thrust::device_ptr<int> scannedAddresses = thrust::device_pointer_cast(nodeBaseAddressDevice);
        thrust::inclusive_scan(scannedAddresses, scannedAddresses + currentUniqueNodeNum, scannedAddresses);


        int numNodesInDepth = d > 0 ? scannedAddresses[currentUniqueNodeNum - 1] + 8 : 1;
        CudaSafeCall(cudaMalloc((void**)&nodesInEachDepthArray[octree.depth - d], numNodesInDepth*sizeof(Node)));

        //populate_node_array_from_base_address(duniqueNodes, supportNode[D-d], dnodeBaseAddress, numbUniqueNodes, nodesInDepth, currentNodeDepth);//this is where numNodesD is set

        CudaSafeCall(cudaFree(uniqueNodeArrayDevice));
        CudaSafeCall(cudaFree(nodeBaseAddressDevice));


        currentUniqueNodeNum = nodesInDepth / 8;
    		if (d > 0){
    			CudaSafeCall(cudaMalloc((void**)&uniqueNodeArrayDevice, currentUniqueNodeNum*sizeof(Node)));
    			//construct_next_level(supportNode[D-d], duniqueNodes, numbNodesd);//what is this doing
    		}
    		depthStartingIndex[octree.depth-d] = octree.totalNodes;
    		octree.totalNodes += nodesInDepth;
        Octree
    		currentNodeDepth *= 2.0;
      }

      octree.finalNodeArray = new Node[octree.totalNodes];
      CudaSafeCall(cudaMalloc((void**)&octree.finalNodeArrayDevice, octree.totalNodes*sizeof(Node)));
      for (int i = 0; i <= D; i++){
        if (i < D)	CudaSafeCall(cudaMemcpy(octree.finalNodeArrayDevice+depthStartingIndex[i], nodesInEachDepthArray[i], (depthStartingIndex[i+1]-depthStartingIndex[i])*sizeof(Node), cudaMemcpyDeviceToDevice));
        else
        	CudaSafeCall(cudaMemcpy(octree.finalNodeArrayDevice+depthStartingIndex[i], nodesInEachDepthArray[i], sizeof(Node), cudaMemcpyDeviceToDevice);
        CudaSafeCall(cudaFree(nodesInEachDepthArray[i]));
      }
      CudaSafeCall(cudaFree(nodesInEachDepthArray));
      delete[] nodesInEachDepthArray;
      CudaSafeCall(cudaMemcpy(octree.finalNodeArray, octree.finalNodeArrayDevice,  octree.totalNodes * sizeof(Node), cudaMemcpyDeviceToHost));

      //FILL NODE array
      printf("\nget node array KERNEL: grid = {%d,%d,%d} - block = {%d,%d,%d}\n",grid.x, grid.y, grid.z, block.x, block.y, block.z);
      cudatimer = clock();
      //filled node array method octree.getFinalNodeArray(grid,block); //does nothing right now
      cudatimer = clock() - cudatimer;

      */
      //THEN WILL FILL VERTEX AND EDGE ARRAY




      CudaSafeCall(cudaFree(octree.nodeKeysDevice));
      CudaSafeCall(cudaFree(octree.nodeCentersDevice));
      CudaSafeCall(cudaFree(octree.pointsDevice));
      CudaSafeCall(cudaFree(octree.normalsDevice));
      CudaSafeCall(cudaFree(octree.nodePointIndexesDevice));
      totalTimer = clock() - totalTimer;
      printf("\nRECONSTRUCTION TOOK %f seconds.\n\n",((float) totalTimer)/CLOCKS_PER_SEC);

      long* testArray = new long[100000000];
      for(int i = 0; i < 100000000; ++i){
        testArray[i] = rand() % 100000 + 1;

      }
      long* cudaHost = new long[100000000];
      long* cudaDevice;
      CudaSafeCall(cudaMalloc((void**)&cudaDevice,100000000*sizeof(long)));
      CudaSafeCall(cudaMemcpy(cudaDevice, testArray, 100000000*sizeof(long), cudaMemcpyHostToDevice));
      thrust::device_ptr<long> cudaSortable(cudaDevice);

      clock_t sortTester = clock();

      thrust::sort(cudaSortable, cudaSortable + 100000000);
      sortTester = clock() - sortTester;

      printf("cuda sort took %f seconds.\n\n",((float) sortTester)/CLOCKS_PER_SEC);

      CudaSafeCall(cudaMemcpy(cudaHost, cudaDevice, 100000000*sizeof(long), cudaMemcpyDeviceToHost));

      sortTester = clock();

      sort(testArray, testArray + 100000000);
      sortTester = clock() - sortTester;

      printf("c++ sort took %f seconds.\n\n",((float) sortTester)/CLOCKS_PER_SEC);


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
