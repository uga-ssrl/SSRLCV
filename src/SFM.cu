#include "common_includes.h"
#include "Image.cuh"
#include "FeatureFactory.cuh"
#include "MatchFactory.cuh"
#include "reprojection.cuh"
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

//TODO typedef structs

//TODO delete methods in images


int main(int argc, char *argv[]){
  try{
    //CUDA INITIALIZATION
    cuInit(0);
    clock_t totalTimer = clock();
    clock_t partialTimer = clock();

    //ARG PARSING
    if(argc < 2 || argc > 4){
      std::cout<<"USAGE ./bin/dsift_parallel </path/to/image/directory/> <optional:numorientations>"<<std::endl;
      exit(-1);
    }
    std::string path = argv[1];
    std::vector<std::string> imagePaths = findFiles(path);

    int numImages = (int) imagePaths.size();

    /*
    DENSE SIFT
    */

    SIFT_FeatureFactory featureFactory = SIFT_FeatureFactory(1);
    Image* images = new Image[numImages];
    MemoryState pixFeatureDescriptorMemoryState[3] = {gpu,gpu,gpu};
    for(int i = 0; i < numImages; ++i){
      images[i] = Image(imagePaths[i], i);
      images[i].convertToBW();

    }
    std::cout<<"image features are set"<<std::endl;



    return 0;
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
