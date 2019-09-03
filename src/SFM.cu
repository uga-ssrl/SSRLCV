#include "common_includes.h"
#include "Image.cuh"
#include "io_util.h"
#include "SIFT_FeatureFactory.cuh"
#include "MatchFactory.cuh"
#include "PointCloudFactory.cuh"
#include "MeshFactory.cuh"

//TODO to have further depth make octree node keys a long

//TODO convert as many LUTs to be constant as possible, use __local__, __constant__, and __shared__

//TODO add timers to copy methods?

//TODO make method for getting grid and block dimensions

//TODO make octree a class not a struct with private members and functions

//TODO optimize atomics with cooperative_groups (warp aggregated)

//TODO find all temporary device result arrays and remove redundant initial cudaMemcpyHostToDevice

//TODO make normals unit vectors?

//TODO cudaFree(constant memory)????????

//TODO typedef structs

//TODO delete methods in images

//TODO go through all global and block ID calculations in kernels to ensure there will be no overflow

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
      std::cout<<"USAGE ./bin/SFM </path/to/image/directory/>"<<std::endl;
      exit(-1);
    }
    std::string path = argv[1];
    std::vector<std::string> imagePaths = ssrlcv::findFiles(path);

    int numImages = (int) imagePaths.size();

    /*
    DENSE SIFT
    */

    ssrlcv::SIFT_FeatureFactory featureFactory = ssrlcv::SIFT_FeatureFactory(true,1);
    std::vector<ssrlcv::Image*> images;
    std::vector<ssrlcv::Unity<ssrlcv::Feature<ssrlcv::SIFT_Descriptor>>*> allFeatures;
    featureFactory.setDescriptorContribWidth(6.0f);
    featureFactory.setOrientationContribWidth(1.5f);
    for(int i = 0; i < numImages; ++i){
      ssrlcv::Image* image = new ssrlcv::Image(imagePaths[i],i);
      ssrlcv::Unity<ssrlcv::Feature<ssrlcv::SIFT_Descriptor>>* features = featureFactory.generateFeatures(image);
      allFeatures.push_back(features);
      images.push_back(image);
    }
    if(allFeatures[0]->numElements != allFeatures[1]->numElements){
      std::cerr<<"identical images do not have same number of features"<<std::endl;
    }
    int numWrongDescriptor = 0;
    int numWrongTheta = 0;
    allFeatures[0]->transferMemoryTo(ssrlcv::cpu);
    allFeatures[1]->transferMemoryTo(ssrlcv::cpu);
    for(int i = 0; i < allFeatures[0]->numElements; ++i){
      if(allFeatures[0]->host[i].descriptor.theta != allFeatures[1]->host[i].descriptor.theta){
        printf("feature @ {%f,%f} has different thetas",allFeatures[0]->host[i].loc.x,allFeatures[0]->host[i].loc.y);
        std::cout<<"\t[0] = "<<(int)allFeatures[0]->host[i].descriptor.theta<<" ";
        std::cout<<"[1] = "<<(int)allFeatures[1]->host[i].descriptor.theta<<std::endl;
        numWrongTheta++;
      }
      for(int d = 0; d < 128; ++d){
        if(allFeatures[0]->host[i].descriptor.values[d] != allFeatures[1]->host[i].descriptor.values[d]){
          printf("feature @ {%f,%f} has different descriptors @ %d",allFeatures[0]->host[i].loc.x,allFeatures[0]->host[i].loc.y,d);
          std::cout<<"\t[0] = "<<(int)allFeatures[0]->host[i].descriptor.values[d]<<" ";
          std::cout<<"[1] = "<<(int)allFeatures[1]->host[i].descriptor.values[d]<<std::endl;
          numWrongDescriptor++;
        }
      }
    }
    std::cout<<"number of mismatched feature descriptors = "<<numWrongDescriptor<<std::endl;
    std::cout<<"number of mismatched thetas = "<<numWrongTheta<<std::endl;

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
