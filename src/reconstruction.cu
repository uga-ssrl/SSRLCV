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
      //0. find mins and maxs {minX,minY,minZ} {maxX, maxY, maxZ}
      //1. find keys
      //2. sort keys, points, normals, and device_launch_parameters
      //3. compact the keys

      string filePath = argv[1];
      clock_t totalTimer = clock();

      int depth = 12;//this number is a placeholder
      Octree octree = Octree(filePath, depth);

      //this will be temporary due to the normals needing to be facing inward
      for(int i = 0; i < octree.numPoints; ++i){
        octree.normals[i].x = octree.normals[i].x * -1;
        octree.normals[i].y = octree.normals[i].y * -1;
        octree.normals[i].z = octree.normals[i].z * -1;
      }
      bool transferPoints;
      bool transfernodeCenters;
      bool transferNormals;
      bool transferKeys;

      dim3 grid = {1,1,1};
      dim3 block = {1,1,1};

      //this is set for getKeys
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
      octree.allocateDeviceVariables();
      cudatimer = clock() - cudatimer;
      printf("allocation took %f seconds.\n\n",((float) cudatimer)/CLOCKS_PER_SEC);

      transferPoints = true;transfernodeCenters = true;transferNormals = false;transferKeys = true;
      cudatimer = clock();
      octree.copyArraysHostToDevice(transferPoints, transfernodeCenters, transferNormals, transferKeys);
      cudatimer = clock() - cudatimer;
      printf("cudaMemcpyHostToDevice for key retrieval took %f seconds.\n",((float) cudatimer)/CLOCKS_PER_SEC);

      /*
      Gets unique key for each node that houses points.
      */
      //grid = {2,1,1};
      //block = {1024,1,1};
      printf("\nGETKEYS KERNEL: grid = {%d,%d,%d} - block = {%d,%d,%d}\n",grid.x, grid.y, grid.z, block.x, block.y, block.z);
      cudatimer = clock();
      octree.executeKeyRetrieval(grid, block);
      cudatimer = clock() - cudatimer;
      printf("getKeys kernel took %f seconds.\n\n",((float) cudatimer)/CLOCKS_PER_SEC);
      cudatimer = clock();

      transferPoints = false;//transfernodeCenters = true;transferNormals = false;transferKeys = true;
      cudatimer = clock();
      octree.copyArraysDeviceToHost(transferPoints, transfernodeCenters, transferNormals, transferKeys);
      cudatimer = clock() - cudatimer;
      printf("cudaMemcpyDeviceToHost after key retrieval took %f seconds.\n\n",((float) cudatimer)/CLOCKS_PER_SEC);

      /*
      //this is just for checking key retrieval and or printing points and normals
      for(int i = 0; i < octree.numPoints; ++i){
        //printBits(sizeof(int), &octree.keys[i]);
        //cout<<octree.keys[i]<<endl;
        //printf("point = {%f,%f,%f} - norm = {%f,%f,%f}\n",octree.points[i].x,octree.points[i].y,octree.points[i].z,octree.normals[i].x,octree.normals[i].y,octree.normals[i].z);
        printf("center = {%f,%f,%f}\n",octree.nodeCenters[i].x,octree.nodeCenters[i].y,octree.nodeCenters[i].z);
      }
      */

      /*
      Sort all arrays on host by key
      */
      cudatimer = clock();
      octree.sortByKey();
      cudatimer = clock() - cudatimer;
      printf("octree sort_by_key took %f seconds.\n\n",((float) cudatimer)/CLOCKS_PER_SEC);

      /*
      Compact the keys, point indexes, nodeCenters so that we have
      the unique nodes at the finest levelD.
      */
      cudatimer = clock();
      octree.compactData();
      cudatimer = clock() - cudatimer;
      printf("octree unique_by_key took %f seconds.\n\n",((float) cudatimer)/CLOCKS_PER_SEC);

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
