#include "common_includes.h"
#include "octree.cuh"
#include "poisson.cuh"
using namespace std;


//TODO across all methods in octree and poisson use const __restrict__ to enable
//https://stackoverflow.com/questions/31344454/can-a-const-restrict-increase-cuda-register-usage

//TODO remove all namespace std and use std:: on relavent methods

//TODO to have further depth make octree node keys a long

//TODO think of a better way to spread out color

//TODO convert as many LUTs to be constant as possible, use __local__, __constant__, and __shared__

//TODO add timers to copy methods?

//TODO make method for getting grid and block dimensions

//TODO use overload operators for cuda vector arithmetic in octree.cu

//TODO implement octree.computeNormals()

//TODO make octree a class not a struct


int main(int argc, char *argv[]){
  try{
    if(argc > 1 && argc < 4){
      string filePath = argv[1];
      int depth = 10;

      if(argc == 2) depth = 9;
      if(argc == 3) depth = stoi(argv[2]);
      clock_t totalTimer = clock();
      clock_t partialTimer = clock();
      cout<<"---------------------------------------------------"<<endl;
      cout<<"COMPUTING OCTREE\n"<<endl;

      Octree octree = Octree(filePath, depth);
      octree.init_octree_gpu();
      octree.generateKeys();
      octree.prepareFinestUniquNodes();
      octree.createFinalNodeArray();
      octree.freePrereqArrays();
      octree.fillLUTs();
      octree.fillNeighborhoods();
      octree.computeVertexArray();
      octree.computeEdgeArray();
      octree.computeFaceArray();

      partialTimer = clock() - partialTimer;
      printf("OCTREE BUILD TOOK %f seconds.\n",((float) partialTimer)/CLOCKS_PER_SEC);
      cout<<"---------------------------------------------------"<<endl;
      partialTimer = clock();

      if(!octree.normalsComputed){
        cout<<"COMPUTING NORMALS\n"<<endl;

        octree.computeNormals();

        partialTimer = clock() - partialTimer;
        printf("COMPUTING NORMALS TOOK %f seconds.\n",((float) partialTimer)/CLOCKS_PER_SEC);
        cout<<"---------------------------------------------------"<<endl;
        partialTimer = clock();
      }

      // cout<<"PERFORMING POISSON RECONSTRUCTION WITH OCTREE\n"<<endl;
      //
      // Poisson poisson = Poisson(&octree);
      // poisson.computeLUTs();
      // poisson.computeDivergenceVector();
      // poisson.computeImplicitFunction();
      // // poisson.marchingCubes();
      //
      // partialTimer = clock() - partialTimer;
      // printf("POISSON RECONSTRUCTION TOOK %f seconds.\n",((float) partialTimer)/CLOCKS_PER_SEC);
      // cout<<"---------------------------------------------------"<<endl;
      // partialTimer = clock();

      // cout<<"WRITING DERIVED PLY FILES\n"<<endl;
      //
      // octree.copyNodesToHost();//this is only necessary for writeCenterPLY
      // octree.writeVertexPLY();
      // octree.writeEdgePLY();
      // octree.writeCenterPLY();
      // octree.writeNormalPLY();
      //
      // partialTimer = clock() - partialTimer;
      // printf("WRITING PLY FILES TOOK %f seconds.\n",((float) partialTimer)/CLOCKS_PER_SEC);
      // cout<<"---------------------------------------------------"<<endl;

      totalTimer = clock() - totalTimer;
      printf("\nTOTAL TIME = %f seconds.\n\n",((float) totalTimer)/CLOCKS_PER_SEC);

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
