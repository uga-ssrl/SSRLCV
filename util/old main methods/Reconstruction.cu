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

//TODO cudaFree(constant memory)????????


int main(int argc, char *argv[]){
  try{
    if(argc > 1 && argc < 5){

      std::string filePath = argv[1];
      int depth = 10;

      std::string test = "surf";

      if(argc == 2) depth = 8;
      if(argc > 2) depth = std::stoi(argv[2]);
      if(argc == 4){
        test = argv[3];
      }

      clock_t totalTimer = clock();

      std::cout<<"depth = "<<depth<<std::endl;
      Octree octree = Octree(filePath, depth);
      if(test != "norm" && test != "surf"){
        octree.writeVertexPLY(true);
        octree.writeEdgePLY(true);
        octree.writeCenterPLY(true);
      }
      if(test != "octree" && !octree.normalsComputed){
        octree.computeNormals(3, 20);
        octree.writeNormalPLY(true);
      }
      if(test == "surf"){
        octree.writeDepthPLY(depth - 1, true);
        Surface surface = Surface(&octree);
        surface.marchingCubes();
      }

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
