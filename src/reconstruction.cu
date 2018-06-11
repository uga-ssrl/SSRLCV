#include "common_includes.h"
#include "octree.cuh"
#include "poisson.cuh"
using namespace std;


//TODO across all methods in octree and poisson use const __restrict__ to enable
//https://stackoverflow.com/questions/31344454/can-a-const-restrict-increase-cuda-register-usage

//TODO use __local__ memory when possible for predefined arrays like thread specific LUTs

//TODO remove all namespace std and use std:: on relavent methods

//TODO to have further depth make octree node keys 64 bit instead of current 32 bit int

//TODO think of a better way to spread out color

int main(int argc, char *argv[]){
  try{
    if(argc > 1 && argc < 4){
      string filePath = argv[1];
      int depth = 10;

      if(argc == 2) depth = 9;
      if(argc == 3) depth = stoi(argv[2]);
      clock_t totalTimer = clock();
      clock_t partialTimer = clock();
      cout<<"COMPUTING OCTREE\n"<<endl;
      Octree octree = Octree(filePath, depth);
      octree.init_octree_gpu();
      octree.generateKeys();
      octree.sortByKey();
      octree.compactData();
      octree.fillUniqueNodesAtFinestLevel();
      octree.createFinalNodeArray();
      octree.freePrereqArrays();
      octree.fillLUTs();
      octree.printLUTs();
      octree.fillNeighborhoods();
      octree.checkForGeneralNodeErrors();
      octree.computeVertexArray();
      octree.computeEdgeArray();
      octree.computeFaceArray();
      partialTimer = clock() - partialTimer;
      printf("OCTREE BUILD TOOK %f seconds.\n",((float) partialTimer)/CLOCKS_PER_SEC);
      cout<<"---------------------------------------------------"<<endl<<endl;
      partialTimer = clock();

      if(!octree.normalsComputed){
        cout<<"COMPUTING NORMALS"<<endl<<endl;
        octree.computeNormals();
        partialTimer = clock() - partialTimer;
        printf("COMPUTING NORMALS TOOK %f seconds.\n",((float) partialTimer)/CLOCKS_PER_SEC);
        cout<<"---------------------------------------------------"<<endl<<endl;
        partialTimer = clock();
      }

      cout<<"PERFORMING POISSON RECONSTRUCTION WITH OCTREE"<<endl;
      Poisson poisson = Poisson(&octree);
      //poisson.computeLUTs();
      //poisson.computeDivergenceVector();
      //poisson.computeImplicitFunction();
      //poisson.marchingCubes();
      //poisson.isosurfaceExtraction();
      partialTimer = clock() - partialTimer;
      printf("\nPOISSON RECONSTRUCTION TOOK %f seconds.\n",((float) partialTimer)/CLOCKS_PER_SEC);
      cout<<"---------------------------------------------------"<<endl<<endl;
      partialTimer = clock();

      cout<<"WRITING DERIVED PLY FILES\n"<<endl;
      octree.writeNormalPLY();
      //octree.writeEdgePLY();
      partialTimer = clock() - partialTimer;
      printf("\nWRITING PLY FILES TOOK %f seconds.\n",((float) partialTimer)/CLOCKS_PER_SEC);
      cout<<"---------------------------------------------------"<<endl<<endl;

      totalTimer = clock() - totalTimer;
      printf("\nTOTAL TIME = %f seconds.\n\n",((float) totalTimer)/CLOCKS_PER_SEC);

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
