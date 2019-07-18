#include "common_includes.h"
#include "Image.cuh"
#include "io_util.h"
#include "FeatureFactory.cuh"
#include "MatchFactory.cuh"
#include "reprojection.cuh"
#include "MeshFactory.cuh"

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
      std::cout<<"USAGE ./bin/dsift_parallel </path/to/image/directory/>"<<std::endl;
      exit(-1);
    }
    std::string path = argv[1];
    std::vector<std::string> imagePaths = ssrlcv::findFiles(path);

    int numImages = (int) imagePaths.size();

    /*
    DENSE SIFT
    */

    ssrlcv::SIFT_FeatureFactory featureFactory = ssrlcv::SIFT_FeatureFactory();
    std::vector<ssrlcv::Image> images;
    std::vector<ssrlcv::Unity<ssrlcv::Feature<ssrlcv::SIFT_Descriptor>>*> features;
    int2 border = {100,100};
    for(int i = 0; i < numImages; ++i){
      images.push_back(ssrlcv::Image(imagePaths[i], i));
      images[i].convertToBW();
      ssrlcv::Quadtree<unsigned int>* quadtree = nullptr;
      quadtree = new ssrlcv::Quadtree<unsigned int>(images[i].descriptor.size,border);
      quadtree->generateVerticesAndEdges();
      images[i].quadtree = quadtree;
      images[i].quadtree->writePLY(images[i].pixels);
      exit(0);
    }

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
