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

    ssrlcv::SIFT_FeatureFactory featureFactory = ssrlcv::SIFT_FeatureFactory(1.5f,6.0f);
    std::vector<ssrlcv::Image*> images;
    std::vector<ssrlcv::Unity<ssrlcv::Feature<ssrlcv::SIFT_Descriptor>>*> allFeatures;
    for(int i = 0; i < numImages; ++i){
      ssrlcv::Image* image = new ssrlcv::Image(imagePaths[i],i);
      ssrlcv::Unity<ssrlcv::Feature<ssrlcv::SIFT_Descriptor>>* features = featureFactory.generateFeatures(image,false,1,0.8);
      images.push_back(image);
      allFeatures.push_back(features);
      features->transferMemoryTo(ssrlcv::cpu);
      for(int f = 0; f < features->numElements; ++f){
        printf("%d,%f,%f,{%f,%f}\n",f,features->host[f].descriptor.sigma,features->host[f].descriptor.theta,features->host[f].loc.x,features->host[f].loc.y);
        // for(int x = 0,d = 0; x < 4; ++x){
        //   std::cout<<std::endl;
        //   for(int y = 0; y < 4; ++y){
        //     std::cout<<"  ";
        //     for(int a = 0; a < 8; ++a){
        //       printf("%d",(int) features->host[f].descriptor.values[d++]);
        //       if(a < 8) std::cout<<",";
        //     }
        //   }
        // }
        // printf("\n\n");
      }
    }
    ssrlcv::MatchFactory<ssrlcv::SIFT_Descriptor> matchFactory = ssrlcv::MatchFactory<ssrlcv::SIFT_Descriptor>();
    std::cout << "Starting matching, this will take a while ..." << std::endl;
    ssrlcv::Unity<ssrlcv::DMatch>* distanceMatches = matchFactory.generateDistanceMatches(images[0],allFeatures[0],images[1],allFeatures[1]);
    distanceMatches->transferMemoryTo(ssrlcv::cpu);
    float2 minMax = {0.0f,0.0f};
    for(int i = 0; i < distanceMatches->numElements;++i){
      if(distanceMatches->host[i].distance < minMax.x) minMax.x = distanceMatches->host[i].distance;
      if(distanceMatches->host[i].distance > minMax.y) minMax.y = distanceMatches->host[i].distance;
    }
    matchFactory.refineMatches(distanceMatches,((300.0f+minMax.x)/(minMax.y-minMax.x)));

    if(distanceMatches->state != ssrlcv::gpu) distanceMatches->setMemoryState(ssrlcv::gpu);
    ssrlcv::Unity<ssrlcv::Match>* matches = matchFactory.getRawMatches(distanceMatches);

    delete distanceMatches;

    std::string newFile = "./data/img/everest1024/everest1024_matches.txt";
    std::ofstream matchstream(newFile);
    matches->transferMemoryTo(ssrlcv::cpu);
    if(matchstream.is_open()){
      std::string line;
      for(int i = 0; i < matches->numElements; ++i){
        line = std::to_string(matches->host[i].keyPoints[0].loc.x) + ",";
        line += std::to_string(matches->host[i].keyPoints[0].loc.y) + ",";
        line += std::to_string(matches->host[i].keyPoints[1].loc.x) + ",";
        line += std::to_string(matches->host[i].keyPoints[1].loc.y) + "\n";
        matchstream << line;
        //std::cout<<line;
      }
    }

    ssrlcv::PointCloudFactory demPoints = ssrlcv::PointCloudFactory();
    ssrlcv::Unity<float3>* points = demPoints.stereo_disparity(matches,64.0f);

    delete matches;
    delete allFeatures[0];
    delete allFeatures[1];

    ssrlcv::writePLY("out/test.ply",points);

    delete points;

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
