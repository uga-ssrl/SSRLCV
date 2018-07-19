#include "common_includes.h"
#include "octree.cuh"
#include "poisson.cuh"

//TODO across all methods in octree and poisson use const __restrict__ to enable
//https://stackoverflow.com/questions/31344454/can-a-const-restrict-increase-cuda-register-usage

//TODO to have further depth make octree node keys a long

//TODO think of a better way to spread out color

//TODO convert as many LUTs to be constant as possible, use __local__, __constant__, and __shared__

//TODO add timers to copy methods?

//TODO make method for getting grid and block dimensions

//TODO use overload operators for cuda vector arithmetic in octree.cu

//TODO implement octree.computeNormals()

//TODO make octree a class not a struct with private members and functions

//TODO optimize atomics with cooperative_groups (warp aggregated)

//TODO find all temporary device result arrays and remove redundant initial cudaMemcpyHostToDevice


int main(int argc, char *argv[]){
  try{
    if(argc > 1 && argc < 4){
      std::string filePath = argv[1];
      int depth = 10;

      if(argc == 2) depth = 9;
      if(argc == 3) depth = std::stoi(argv[2]);

      clock_t totalTimer = clock();

      std::cout<<"---------------------------------------------------"<<std::endl;
      std::cout<<"COMPUTING OCTREE\n"<<std::endl;

      Octree octree = Octree(filePath, depth);
      octree.init_octree_gpu();
      octree.generateKeys();
      octree.prepareFinestUniquNodes();
      octree.createFinalNodeArray();
      octree.freePrereqArrays();
      octree.fillLUTs();
      octree.fillNeighborhoods();
      if(!octree.normalsComputed){
        octree.computeNormals(octree.width*0.015);
      }
      octree.computeVertexArray();
      octree.computeEdgeArray();
      octree.computeFaceArray();
      std::cout<<"---------------------------------------------------"<<std::endl;

      // std::cout<<"PERFORMING POISSON RECONSTRUCTION WITH OCTREE\n"<<std::endl;
      // partialTimer = clock();
      // Poisson poisson = Poisson(&octree);
      // poisson.computeLUTs();
      // poisson.computeDivergenceVector();
      // //poisson.computeImplicitFunction();
      // //poisson.computeImplicitMagma();
      // //poisson.computeImplicitCuSPSolver();
      // //poisson.marchingCubes();
      // std::cout<<"---------------------------------------------------"<<std::endl;
      //
      // std::cout<<"WRITING DERIVED PLY FILES\n"<<std::endl;
      //
      // octree.copyNodesToHost();//this is only necessary for writeCenterPLY
      // octree.writeVertexPLY();
      octree.writeEdgePLY();
      // octree.writeCenterPLY();
      octree.writeNormalPLY();
      // std::cout<<"---------------------------------------------------"<<std::endl;

      totalTimer = clock() - totalTimer;
      printf("\nTOTAL TIME = %f seconds.\n\n",((float) totalTimer)/CLOCKS_PER_SEC);

      return 0;
    }
    else{
      std::cout<<"LACK OF PLY INPUT...goodbye"<<std::endl;
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
