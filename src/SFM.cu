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

//TODO go through all global and block ID calculations in kernels to ensure there will be no overflow


//TODO ADD AND USE MEMORY CONSTAINED VARIABLE TO FACTORIES - gives permission to delete parameters once used
//use this or go back and make sure a unity never has a state of both

//TODO go back and make sure thrust:: calls are all device

//TODO in octree and quadtree change thrust::copy_if to thrust::remove

//TODO go back and change all Unity<T>.transferMemoryTo(),Unity<T>.clear() to Unity<T>.setMemoryState

//TODO look for locations that cudaDeviceSynchronize is used before a cudaFree, cudaMalloc, or cudaMemcpy and think about removing them as they are blockers themselves\\

//TODO go back and ensure that fore is being set based on previously edited

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
    std::vector<ssrlcv::Image*> images;
    std::vector<ssrlcv::Unity<ssrlcv::Feature<ssrlcv::SIFT_Descriptor>>*> allFeatures;
    unsigned int convertColorDepthTo = 1;
    for(int i = 0; i < numImages; ++i){
      ssrlcv::Image* image = new ssrlcv::Image(imagePaths[i],convertColorDepthTo,i);
      //sift border is 24 due to 1xbin would normally be 12
      image->quadtree->setNodeFlags({24.0f+image->quadtree->border.x,24.0f+image->quadtree->border.y},true);
      image->quadtree->writePLY();
      ssrlcv::Unity<ssrlcv::Feature<ssrlcv::SIFT_Descriptor>>* features = featureFactory.generateFeaturesDensly(image,1);
      allFeatures.push_back(features);
      images.push_back(image);
    }
    ssrlcv::MatchFactory matchFactory = ssrlcv::MatchFactory();
    //ssrlcv::Unity<ssrlcv::Match>* matches = matchFactory.generateMatchesBruteForce(images[0],allFeatures[0],images[1],allFeatures[1]);
    return 0;
  }
  catch (const std::exception &e){
      std::cerr << "Caught exception: " << e.what() << '\n';
      std::exit(1);
  }
  catch (const ssrlcv::UnityException &e){
      std::cerr << "Caught exception: " << e.what() << '\n';
      std::exit(1);
  }
  catch (...){
      std::cerr << "Caught unknown exception\n";
      std::exit(1);
  }

}
