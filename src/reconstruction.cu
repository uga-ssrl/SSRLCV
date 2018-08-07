#include "common_includes.h"
#include "surface.cuh"

//TODO across all methods in octree and surface use const __restrict__ to enable
//https://stackoverflow.com/questions/31344454/can-a-const-restrict-increase-cuda-register-usage

//TODO to have further depth make octree node keys a long

//TODO think of a better way to spread out color

//TODO convert as many LUTs to be constant as possible, use __local__, __constant__, and __shared__

//TODO add timers to copy methods?

//TODO make method for getting grid and block dimensions

//TODO use overload operators for cuda vector arithmetic in octree.cu

//TODO make octree a class not a struct with private members and functions

//TODO optimize atomics with cooperative_groups (warp aggregated)

//TODO find all temporary device result arrays and remove redundant initial cudaMemcpyHostToDevice

//TODO make normals unit vectors?


int main(int argc, char *argv[]){
  try{
    if(argc > 1 && argc < 4){
      std::string filePath = argv[1];
      int depth = 10;

      if(argc == 2) depth = 8;
      if(argc == 3) depth = std::stoi(argv[2]);

      clock_t totalTimer = clock();
      std::cout<<"depth = "<<depth<<std::endl;

      Surface surface = Surface(filePath, depth);
      //surface.computeImplicitFunction();
      //surface.computeImplicitMagma();
      //surface.computeImplicitCuSPSolver();
      ///surface.computeVertexImplicit();
      surface.computeImplicitEasy();
      surface.marchingCubes();
      std::cout<<"---------------------------------------------------"<<std::endl;

      std::cout<<"WRITING DERIVED PLY FILES\n"<<std::endl;
      surface.generateMesh();
      surface.octree->writeEdgePLY();
      surface.octree->writeNormalPLY();
      // surface.octree->copyNodesToHost();//this is only necessary for writeCenterPLY
      // surface.octree->writeCenterPLY();
      // surface.octree->writeVertexPLY();

      std::cout<<"---------------------------------------------------"<<std::endl;

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
