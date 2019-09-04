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
    ssrlcv::MatchFactory<ssrlcv::SIFT_Descriptor> matchFactory = ssrlcv::MatchFactory<ssrlcv::SIFT_Descriptor>();
    std::cout << "Starting matching, this will take a while ..." << std::endl;
    ssrlcv::Unity<ssrlcv::DMatch>* distanceMatches = matchFactory.generateDistanceMatches(images[0],allFeatures[0],images[1],allFeatures[1]);

    matchFactory.refineMatches(distanceMatches, 0.25);
    if(distanceMatches->state != ssrlcv::gpu) distanceMatches->setMemoryState(ssrlcv::gpu);

    matchFactory.sortMatches(distanceMatches);
    distanceMatches->transferMemoryTo(ssrlcv::cpu);
    std::cout << "starting sort..." << std::endl << "Top Sorted:" << std::endl;
    for (int i = 0; i < 15; i++){
      std::cout << "[" << i << "]\t" << distanceMatches->host[i].distance << std::endl;
    }
    distanceMatches->setMemoryState(ssrlcv::gpu);

    ssrlcv::Unity<ssrlcv::Match>* matches = matchFactory.getRawMatches(distanceMatches);
    delete distanceMatches;

    ssrlcv::PointCloudFactory demPoints = ssrlcv::PointCloudFactory();
    ssrlcv::Unity<float3>* points = demPoints.stereo_disparity(matches,64.0f);

    delete matches;
    delete allFeatures[0];
    delete allFeatures[1];

    ssrlcv::writePLY("out/test.ply",points);

    delete points;

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
