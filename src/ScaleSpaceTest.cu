#include "common_includes.h"
#include "Image.cuh"
#include "io_util.h"
#include "SIFT_FeatureFactory.cuh"
#include "MatchFactory.cuh"
#include "PointCloudFactory.cuh"
#include "MeshFactory.cuh"

int main(int argc, char *argv[]){
  try{

    //CUDA INITIALIZATION
    cuInit(0);
    clock_t totalTimer = clock();
    clock_t partialTimer = clock();

    //ARG PARSING
    if(argc < 2 || argc > 4){
      std::cout<<"USAGE ./bin/StereoDisparity </path/to/image/directory/>"<<std::endl;
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
    uint2 scaleSpaceDim = {4,6};
    int nspo = scaleSpaceDim.y - 3;//num scale space/octave in dog made from a {4,6} ScaleSpace
    float noiseThreshold = 0.015*(powf(2,1.0f/nspo)-1)/(powf(2,1.0f/3.0f)-1);
    float edgeThreshold = 12.1f;//12.1 = (10.0f + 1)^2 / 10.0f
    for(int i = 0; i < numImages; ++i){
      ssrlcv::Image* image = new ssrlcv::Image(imagePaths[i],i);
      ssrlcv::FeatureFactory::DOG* dog = new ssrlcv::FeatureFactory::DOG(image,-1,scaleSpaceDim,sqrtf(2.0f)/2,{2,sqrtf(2.0f)},{8,8});
      dog->convertToDOG();      
      dog->findKeyPoints(noiseThreshold,edgeThreshold,true);      
      delete dog;
      images.push_back(image);
    }

    for(int i = 0; i < imagePaths.size(); ++i){
        delete images[i];
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
