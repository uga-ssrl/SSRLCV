#include "common_includes.h"
#include "Image.cuh"
#include "io_util.h"
#include "FeaturFactory.cuh"
#include "MatchFactory.cuh"
#include "PointCloudFactory.cuh"

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

    ssrlcv::SIFT_FeatureFactory featureFactory = ssrlcv::FeatureFactory();
    ssrlcv::MatchFactory<ssrlcv::Window_35x35> matchFactory = ssrlcv::MatchFactory<ssrlcv::Window_35x35>(0.6f,250.0f);

    /*
    FEATURE EXTRACTION
    */
    std::vector<ssrlcv::Image*> images;
    std::vector<ssrlcv::Unity<ssrlcv::Window_35x35>*> allFeatures;
    for(int i = 0; i < numImages; ++i){
      ssrlcv::Image* image = new ssrlcv::Image(imagePaths[i],i);
      ssrlcv::Unity<ssrlcv::Window_35x35>* features = featureFactory.generate35x35Windows(image);
      images.push_back(image);
      allFeatures.push_back(features);
    }
    
    /*
    MATCHING
    */
    //seeding with false photo

    std::cout << "Starting matching..." << std::endl;
    ssrlcv::Unity<ssrlcv::Match>* dmatches = matchFactory.generateMatches(images[0],allFeatures[0],images[1],allFeatures[1]);
    
    std::string delimiter = "/";
    std::string matchFile = imagePaths[0].substr(0,imagePaths[0].rfind(delimiter)) + "/matches.txt";
    ssrlcv::writeMatchFile(matches, matchFile);
    
    /*
    STEREODISPARITY
    */
    ssrlcv::PointCloudFactory demPoints = ssrlcv::PointCloudFactory();
    ssrlcv::Unity<float3>* points = demPoints.stereo_disparity(matches,64.0f);
    std::string disparityFile = imagePaths[0].substr(0,imagePaths[0].rfind(delimiter));
    disparityFile = disparityFile.substr(0,disparityFile.rfind(delimiter))  + "/disparity.png";
    ssrlcv::writeDisparityImage(points,63,disparityFile);

    delete matches;
    ssrlcv::writePLY("out/test.ply",points);
    delete points;

    for(int i = 0; i < imagePaths.size(); ++i){
      delete images[i];
      delete allFeatures[i];
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
