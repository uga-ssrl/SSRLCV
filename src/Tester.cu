#include "common_includes.h"
#include "Image.cuh"
#include "io_util.h"
#include "SIFT_FeatureFactory.cuh"
#include "MatchFactory.cuh"
#include "PointCloudFactory.cuh"
#include "MeshFactory.cuh"

//TODO fix gaussian operators - currently creating very low values

std::vector<float4> checkEquivanlenceSIFT(std::vector<ssrlcv::Image*> images, std::vector<ssrlcv::Unity<ssrlcv::Feature<ssrlcv::SIFT_Descriptor>>*> features, std::vector<float> rotations){
  ssrlcv::Image* base = images[0];
  ssrlcv::Unity<ssrlcv::Feature<ssrlcv::SIFT_Descriptor>>* base_features = features[0];
  if(base_features->getMemoryState() != ssrlcv::cpu) base_features->setMemoryState(ssrlcv::cpu);
  float rotation = 0.0f;
  std::vector<float4> correctness;//{thetaIncorrect,minDist,maxDist,identical}
  float2 newCoord = {0.0f,0.0f};
  float2 coord = {0.0f,0.0f};
  ssrlcv::Unity<ssrlcv::Feature<ssrlcv::SIFT_Descriptor>>* rotated_features = nullptr; 
  ssrlcv::Image* rotated_image = nullptr;
  ssrlcv::Feature<ssrlcv::SIFT_Descriptor>* base_feature = nullptr;
  ssrlcv::Feature<ssrlcv::SIFT_Descriptor>* rot_feature = nullptr;
  int thetasIncorrect = 0;
  float2 minMax = {FLT_MAX,-FLT_MAX};
  int perfectMatches = 0;
  for(int i = 1; i < images.size(); ++i){
    rotation = rotations[i-1];
    rotated_image = images[i];
    rotated_features = features[i];
    if(rotated_features->getMemoryState() != ssrlcv::cpu) rotated_features->setMemoryState(ssrlcv::cpu);
    if(rotated_features->size() != base_features->size()){
      std::cerr<<"ERROR: not the same number of features"<<std::endl;
      exit(-1);
    }
    float2 size = {(float)base->size.x/2.0f,(float)base->size.y/2.0f};
    int index = 0;
    float rotationError = 0.0f;
    for(int f = 0; f < base_features->size(); ++f){
      base_feature = &base_features->host[f];
      coord = base_feature->loc;
      coord = coord - size;
      newCoord = {
        roundf((coord.x*cos(rotation))+(coord.y*sin(rotation))),
        roundf(-(coord.x*sin(rotation))+(coord.y*cos(rotation)))
      };
      newCoord = newCoord + size;
      newCoord.x -= 1;
      coord = coord + size;
      index = (newCoord.y-12)*(rotated_image->size.x-24) + newCoord.x - 12;
      if(index >= rotated_features->size()){
        std::cerr<<"ERROR: rotation incorrect"<<std::endl;
        exit(-1);
      }
      rot_feature = &rotated_features->host[index];
      rotationError = fmodf((rotation + (2.0f*M_PI))-(base_feature->descriptor.theta-rot_feature->descriptor.theta+ (2.0f*M_PI)), 2.0f*M_PI);
      
      if(abs(rotationError)>0.00001){
        thetasIncorrect++;
      }
      float4 equivalence;
      equivalence.x = sqrtf(base_feature->descriptor.distProtocol(rot_feature->descriptor,FLT_MAX));
      if(minMax.x > equivalence.x) minMax.x = equivalence.x;
      if(minMax.y < equivalence.x) minMax.y = equivalence.x;
      if(equivalence.x < 0.00001) perfectMatches++;
      correctness.push_back(equivalence);
      // printf("|%f,%f,(theta %f)|%f|%f,%f(theta %f)|dist=%f|",
      //   coord.x,coord.y,base_feature->descriptor.theta,
      //   rotationError,
      //   rot_feature->loc.x,rot_feature->loc.y,rot_feature->descriptor.theta,
      //   ,equivalence.x
      // );
      if(f == 2500){
        base_feature->descriptor.print();
        std::cout<<"\n\n";
        rot_feature->descriptor.print();
      }
    }
  }
  printf("|rotation error = %f%|min distance = %f|max dist = %f|perfect matches = %f%|\n",
    (float)thetasIncorrect*100.0f/base_features->size(),
    minMax.x,minMax.y,(float)perfectMatches*100.0f/base_features->size()
  );
  return correctness;
}


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
    ssrlcv::MatchFactory<ssrlcv::SIFT_Descriptor> matchFactory = ssrlcv::MatchFactory<ssrlcv::SIFT_Descriptor>(0.6f,200.0f*200.0f);
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
    if(numImages < 2){
      std::cerr<<"ERROR: this exectuable needs atleast 2 images other than seed"<<std::endl;
      exit(0);
    }
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

    std::cout << "Starting matching..." << std::endl;

    if(numImages <= 2){
      ssrlcv::Unity<float>* seedDistances = (seedProvided) ? matchFactory.getSeedDistances(allFeatures[0]) : nullptr;
      ssrlcv::Unity<ssrlcv::DMatch>* distanceMatches = matchFactory.generateDistanceMatches(images[0],allFeatures[0],images[1],allFeatures[1],seedDistances);
      if(seedDistances != nullptr) delete seedDistances;

      distanceMatches->transferMemoryTo(ssrlcv::cpu);
      float maxDist = 0.0f;
      for(int i = 0; i < distanceMatches->size(); ++i){
        if(maxDist < distanceMatches->host[i].distance) maxDist = distanceMatches->host[i].distance;
      }
      printf("max euclidean distance between features = %f\n",maxDist);
      if(distanceMatches->getMemoryState() != ssrlcv::gpu) distanceMatches->setMemoryState(ssrlcv::gpu);
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
      if(multiviewMatches.matches->getMemoryState() != ssrlcv::cpu) multiviewMatches.matches->setMemoryState(ssrlcv::cpu);
      if(multiviewMatches.keyPoints->getMemoryState() != ssrlcv::cpu) multiviewMatches.keyPoints->setMemoryState(ssrlcv::cpu);
      // for(int i = 0; i < multiviewMatches.matches->size(); ++i){
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

























































// yeet
