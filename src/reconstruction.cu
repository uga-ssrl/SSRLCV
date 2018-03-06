#include "common_includes.h"
#include "octree.cuh"
using namespace std;

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
      octree.allocateDeviceVariablesGENERAL();
      octree.allocateDeviceVariablesNODES(false);
      cudatimer = clock() - cudatimer;
      printf("initial allocation took %f seconds.\n\n",((float) cudatimer)/CLOCKS_PER_SEC);

      /*
      Gets unique key for each node that houses points.
      */
      cudatimer = clock();
      octree.copyArraysHostToDeviceGENERAL(true, false);
      octree.copyArraysHostToDeviceNODES(true, true, false, false, false);
      cudatimer = clock() - cudatimer;
      printf("cudaMemcpyHostToDevice for key retrieval took %f seconds.\n",((float) cudatimer)/CLOCKS_PER_SEC);

      printf("\ngetNodeKeys KERNEL: grid = {%d,%d,%d} - block = {%d,%d,%d}\n",grid.x, grid.y, grid.z, block.x, block.y, block.z);
      cudatimer = clock();
      octree.executeKeyRetrieval(grid, block);
      cudatimer = clock() - cudatimer;
      printf("getnodeKeys kernel took %f seconds.\n\n",((float) cudatimer)/CLOCKS_PER_SEC);

      cudatimer = clock();
      octree.copyArraysDeviceToHostNODES(true, true, false, false, false);
      cudatimer = clock() - cudatimer;
      printf("cudaMemcpyDeviceToHost after key retrieval took %f seconds.\n\n",((float) cudatimer)/CLOCKS_PER_SEC);

      /*
      Sort all arrays on host by key
      */
      cudatimer = clock();
      octree.sortByKey();
      cudatimer = clock() - cudatimer;
      octree.copyArraysHostToDeviceGENERAL(true, true);//copy sorted general variables back over
      printf("octree sort_by_key took %f seconds.\n\n",((float) cudatimer)/CLOCKS_PER_SEC);

      /*
      Compact the nodeKeys, point indexes, nodeCenters so that we have
      the unique nodes at the finest levelD.
      */
      cudatimer = clock();
      octree.compactData();
      cudatimer = clock() - cudatimer;
      printf("octree unique_by_key took %f seconds.\n\n",((float) cudatimer)/CLOCKS_PER_SEC);

      /*
      Now we have numNodes, sorted pointIndexes for nodes, and sorted
      unique keys per node, and sorted nodeCenters,
      */
      cudatimer = clock();
      octree.allocateDeviceVariablesNODES(true);
      cudatimer = clock() - cudatimer;
      printf("further node allocation took %f seconds.\n\n",((float) cudatimer)/CLOCKS_PER_SEC);

      grid = {1,1,1};
      block = {1,1,1};

      //this is set for fillNodes
      if(octree.numNodes < 65535) grid.x = (unsigned int) octree.numNodes;
      else{
        grid.x = 65535;
        while(grid.x*block.x < octree.numNodes){
          ++block.x;
        }
        while(grid.x*block.x > octree.numNodes){
          --grid.x;
        }
      }

      //copy device variables that have changed or need to be copied
      cudatimer = clock();
      octree.copyArraysHostToDeviceNODES(false, true, false, true, false);
      cudatimer = clock() - cudatimer;
      printf("cudaMemcpyHostToDevice for filling nodes took %f seconds.\n",((float) cudatimer)/CLOCKS_PER_SEC);

      /*
      Find all nodes even if they do not contain points this will give children
      */
      printf("\nfindall KERNEL: grid = {%d,%d,%d} - block = {%d,%d,%d}\n",grid.x, grid.y, grid.z, block.x, block.y, block.z);
      cudatimer = clock();
      octree.executeFindAllNodes(grid, block);//this is helping find all nodes that are contained in a parent
      cudatimer = clock() - cudatimer;
      printf("find all nodes kernel took %f seconds.\n\n",((float) cudatimer)/CLOCKS_PER_SEC);

      cudatimer = clock();
      octree.copyArraysDeviceToHostNODES(false, false, false, true, false);
      cudatimer = clock() - cudatimer;
      printf("cudaMemcpyDeviceToHost after finding all possible nodes took %f seconds.\n\n",((float) cudatimer)/CLOCKS_PER_SEC);

      cudatimer = clock();
      cudatimer = clock() - cudatimer;
      octree.inclusiveScanForNodeAddresses();
      printf("inclusive scan of nodeNumbers took %f seconds.\n\n",((float) cudatimer)/CLOCKS_PER_SEC);

      for(int i = 0; i < octree.numNodes; ++i){
        printf("%d children...%d incSum\n",octree.nodeNumbers[i], octree.nodeAddresses[i]);
      }

      octree.cudaFreeMemory();
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
