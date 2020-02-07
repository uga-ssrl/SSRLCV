#include "common_includes.h"
#include "Image.cuh"
#include "io_util.h"
#include "SIFT_FeatureFactory.cuh"
#include "MatchFactory.cuh"
#include "PointCloudFactory.cuh"
#include "MeshFactory.cuh"

//TODO fix gaussian operators - currently creating very low values


int main(int argc, char *argv[]){
  try{

    //CUDA INITIALIZATION
    cuInit(0);
    clock_t totalTimer = clock();
    clock_t partialTimer = clock();

    //ARG PARSING
    
    std::map<std::string,ssrlcv::arg*> args = ssrlcv::parseArgs(argc,argv);
    if(args.find("dir") == args.end()){
      std::cerr<<"ERROR: SFM executable requires a directory of images"<<std::endl;
      exit(-1);
    }
    ssrlcv::SIFT_FeatureFactory featureFactory = ssrlcv::SIFT_FeatureFactory(1.5f,6.0f);
    ssrlcv::MatchFactory<ssrlcv::SIFT_Descriptor> matchFactory = ssrlcv::MatchFactory<ssrlcv::SIFT_Descriptor>(0.8f,250.0f*250.0f);
    bool seedProvided = false;
    ssrlcv::Unity<ssrlcv::Feature<ssrlcv::SIFT_Descriptor>>* seedFeatures = nullptr;
    if(args.find("seed") != args.end()){
      seedProvided = true;
      std::string seedPath = ((ssrlcv::img_arg*)args["seed"])->path;
      ssrlcv::Image* seed = new ssrlcv::Image(seedPath,-1);
      seedFeatures = featureFactory.generateFeatures(seed,false,3,0.8);
      matchFactory.setSeedFeatures(seedFeatures);
      delete seed;
    }
    std::vector<std::string> imagePaths = ((ssrlcv::img_dir_arg*)args["dir"])->paths;
    int numImages = (int) imagePaths.size();
    std::cout<<"found "<<numImages<<" in directory given"<<std::endl;

    std::vector<ssrlcv::Image*> images;
    std::vector<ssrlcv::Unity<ssrlcv::Feature<ssrlcv::SIFT_Descriptor>>*> allFeatures;
    for(int i = 0; i < numImages; ++i){
      ssrlcv::Image* image = new ssrlcv::Image(imagePaths[i],i);
      ssrlcv::Unity<ssrlcv::Feature<ssrlcv::SIFT_Descriptor>>* features = featureFactory.generateFeatures(image,false,3,0.8);
      features->transferMemoryTo(ssrlcv::cpu);
      images.push_back(image);
      allFeatures.push_back(features);
    }


    /*
    MATCHING
    */
    //seeding with false photo

    std::cout << "Starting matching..." << std::endl;

    if(numImages <= 2){
      ssrlcv::Unity<float>* seedDistances = (seedProvided) ? matchFactory.getSeedDistances(allFeatures[0]) : nullptr;
      ssrlcv::Unity<ssrlcv::DMatch>* distanceMatches = matchFactory.generateDistanceMatches(images[0],allFeatures[0],images[1],allFeatures[1],seedDistances);
      if(seedDistances != nullptr) delete seedDistances;

      distanceMatches->transferMemoryTo(ssrlcv::cpu);
      float maxDist = 0.0f;
      for(int i = 0; i < distanceMatches->numElements; ++i){
        if(maxDist < distanceMatches->host[i].distance) maxDist = distanceMatches->host[i].distance;
      }
      printf("max euclidean distance between features = %f\n",maxDist);
      if(distanceMatches->state != ssrlcv::gpu) distanceMatches->setMemoryState(ssrlcv::gpu);
      ssrlcv::Unity<ssrlcv::Match>* matches = matchFactory.getRawMatches(distanceMatches);
      delete distanceMatches;
      std::string delimiter = "/";
      std::string matchFile = imagePaths[0].substr(0,imagePaths[0].rfind(delimiter)) + "/matches.txt";
      ssrlcv::writeMatchFile(matches, matchFile);
      /*
      STEREODISPARITY
      */
      ssrlcv::PointCloudFactory demPoints = ssrlcv::PointCloudFactory();
      ssrlcv::Unity<float3>* points = demPoints.stereo_disparity(matches,8.0);

      delete matches;
      ssrlcv::writePLY("out/test.ply",points);
      delete points;
    }
    else{
      ssrlcv::MatchSet multiviewMatches = matchFactory.generateMatchesExaustive(images,allFeatures);
      if(multiviewMatches.matches->state != ssrlcv::cpu) multiviewMatches.matches->setMemoryState(ssrlcv::cpu);
      if(multiviewMatches.keyPoints->state != ssrlcv::cpu) multiviewMatches.keyPoints->setMemoryState(ssrlcv::cpu);
      // for(int i = 0; i < multiviewMatches.matches->numElements; ++i){
      //   for(int j = multiviewMatches.matches->host[i].index; j < multiviewMatches.matches->host[i].numKeyPoints + multiviewMatches.matches->host[i].index; ++j){
      //     printf("{%u,{%f,%f}} ",multiviewMatches.keyPoints->host[j].parentId,multiviewMatches.keyPoints->host[j].loc.x,multiviewMatches.keyPoints->host[j].loc.y);
      //   }
      //   std::cout<<std::endl;
      // }
      ssrlcv::writeMatchFile(multiviewMatches, "data/img/multiview_test/matches.txt");
      if(multiviewMatches.keyPoints != nullptr) delete multiviewMatches.keyPoints;
      if(multiviewMatches.matches != nullptr) delete multiviewMatches.matches;
    }
    
    if(seedFeatures != nullptr) delete seedFeatures;
    for(int i = 0; i < numImages; ++i){
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
