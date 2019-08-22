#include "common_includes.h"
#include "Image.cuh"
#include "io_util.h"
#include "FeatureFactory.cuh"
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
      std::cout<<"USAGE ./bin/SFM </path/to/image/directory/>"<<std::endl;
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
    ssrlcv::Unity<ssrlcv::Match>* matches = matchFactory.generateMatchesBruteForce(images[0],allFeatures[0],images[1],allFeatures[1]);

    //






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
