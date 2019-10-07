#include "common_includes.h"
#include "Image.cuh"
#include "io_util.h"
#include "SIFT_FeatureFactory.cuh"
#include "MatchFactory.cuh"
#include "PointCloudFactory.cuh"
#include "MeshFactory.cuh"

//TODO fix gaussian operators - currently creating very low values

void compareEquivalentFeatures(std::vector<ssrlcv::Unity<ssrlcv::Feature<ssrlcv::SIFT_Descriptor>>*> allFeatures){
  if(allFeatures[0]->numElements != allFeatures[1]->numElements){
    std::cerr<<"Equivalent images do not have the same number of features"<<std::endl;
    printf("features[0]->numElements = %d != features[1]->numElements = %d\n",allFeatures[0]->numElements,allFeatures[1]->numElements);
    exit(-1);
  }
  allFeatures[0]->setMemoryState(ssrlcv::cpu);
  allFeatures[1]->setMemoryState(ssrlcv::cpu);
  int4 numIncorrect = {0,0,0,0};
  ssrlcv::Feature<ssrlcv::SIFT_Descriptor>* feature1 = nullptr;
  ssrlcv::Feature<ssrlcv::SIFT_Descriptor>* feature2 = nullptr;
  for(int f = 0; f < allFeatures[0]->numElements; ++f){
    feature1 = &allFeatures[0]->host[f];
    feature2 = &allFeatures[1]->host[f];
    if(feature1->loc.x != feature2->loc.x || feature1->loc.y != feature2->loc.y){
      numIncorrect.x++;
    }
    if(feature1->descriptor.sigma != feature2->descriptor.sigma){
      numIncorrect.y++;
    }
    if(feature1->descriptor.theta != feature2->descriptor.theta){
      numIncorrect.z++;
    }
    for(int d = 0; d < 128; ++d){
      if(feature1->descriptor.values[d] != feature2->descriptor.values[d]){
        numIncorrect.w++;
      }
    }
  }
  printf("errors in feature generatation between 2 identical images with %d features:\n",(int)allFeatures[0]->numElements);
  printf("location errors = %d\n",numIncorrect.x);
  printf("sigma errors = %d\n",numIncorrect.y);
  printf("theta errors = %d\n",numIncorrect.z);
  printf("SIFT_Descriptor errors = %d\n",numIncorrect.w);
}

void printSIFTFeature(ssrlcv::Feature<ssrlcv::SIFT_Descriptor> feature){
  printf("%f,%f,{%f,%f}\n",feature.descriptor.sigma,feature.descriptor.theta,feature.loc.x,feature.loc.y);
  for(int x = 0,d = 0; x < 4; ++x){
    std::cout<<std::endl;
    for(int y = 0; y < 4; ++y){
      std::cout<<"  ";
      for(int a = 0; a < 8; ++a){
          printf("%d",(int) feature.descriptor.values[d++]);
          if(a < 8) std::cout<<",";
      }
    }
  }
}

int main(int argc, char *argv[]){
  try{

    //CUDA INITIALIZATION
    cuInit(0);
    clock_t totalTimer = clock();
    clock_t partialTimer = clock();

    //these images are equivalent and are for testing feature matching and generation
    // std::vector<std::string> imagePaths;
    // imagePaths.push_back("data/feature_test/ev00.png");
    // imagePaths.push_back("data/feature_test/ev01.png");

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
        printSIFTFeature(features->host[i]);
        printf("\n\n");
      }
    }
    ssrlcv::MatchFactory<ssrlcv::SIFT_Descriptor> matchFactory = ssrlcv::MatchFactory<ssrlcv::SIFT_Descriptor>();
    std::cout << "Starting matching, this will take a while ..." << std::endl;
    ssrlcv::Unity<ssrlcv::DMatch>* distanceMatches = matchFactory.generateDistanceMatches(images[0],allFeatures[0],images[1],allFeatures[1]);
    matchFactory.refineMatches(distanceMatches,200.0f);
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
      }
    }
    ssrlcv::PointCloudFactory demPoints = ssrlcv::PointCloudFactory();
    ssrlcv::Unity<float3>* points = demPoints.stereo_disparity(matches,64.0f);

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
