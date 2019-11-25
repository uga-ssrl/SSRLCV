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
  printf("\n\n");
}

int main(int argc, char *argv[]){
  try{

    //CUDA INITIALIZATION
    cuInit(0);
    clock_t totalTimer = clock();
    clock_t partialTimer = clock();

    //ARG PARSING
    if(argc < 2 || argc > 4){
      std::cout<<"USAGE ./bin/StereoDisparity </path/to/image/directory/> </path/to/optional/seedimage.png>"<<std::endl;
      exit(-1);
    }
    std::string path = argv[1];
    std::vector<std::string> imagePaths = ssrlcv::findFiles(path);

    int numImages = (int) imagePaths.size();

    ssrlcv::SIFT_FeatureFactory featureFactory = ssrlcv::SIFT_FeatureFactory(1.5f,6.0f);
    ssrlcv::MatchFactory<ssrlcv::SIFT_Descriptor> matchFactory = ssrlcv::MatchFactory<ssrlcv::SIFT_Descriptor>(0.6f,250.0f*250.0f);

    /*
    FEATURE EXTRACTION
    */
    //seed features extraction

    ssrlcv::Unity<ssrlcv::Feature<ssrlcv::SIFT_Descriptor>>* seedFeatures = nullptr;
    if(argc == 3){
      std::string seedPath = argv[2];
      ssrlcv::Image* seed = new ssrlcv::Image(seedPath,-1);
      seedFeatures = featureFactory.generateFeatures(seed,false,2,0.8); 
      matchFactory.setSeedFeatures(seedFeatures);
      delete seed;
    } 

    std::vector<ssrlcv::Image*> images;
    std::vector<ssrlcv::Unity<ssrlcv::Feature<ssrlcv::SIFT_Descriptor>>*> allFeatures;
    for(int i = 0; i < numImages; ++i){
      ssrlcv::Image* image = new ssrlcv::Image(imagePaths[i],i);
      ssrlcv::Unity<ssrlcv::Feature<ssrlcv::SIFT_Descriptor>>* features = featureFactory.generateFeatures(image,false,2,0.8);
      images.push_back(image);
      allFeatures.push_back(features);
    }
    
    /*
    MATCHING
    */
    //seeding with false photo

    std::cout << "Starting matching..." << std::endl;
    ssrlcv::Unity<float>* seedDistances = (argc == 3) ? matchFactory.getSeedDistances(allFeatures[0]) : nullptr;    
    ssrlcv::Unity<ssrlcv::DMatch>* distanceMatches = matchFactory.generateDistanceMatches(images[0],allFeatures[0],images[1],allFeatures[1],seedDistances);
    if(seedDistances != nullptr) delete seedDistances;

    distanceMatches->transferMemoryTo(ssrlcv::cpu);
    float maxDist = 0.0f;
    for(int i = 0; i < distanceMatches->numElements; ++i){
      if(maxDist < distanceMatches->host[i].distance) maxDist = distanceMatches->host[i].distance;
    }
    printf("%f\n",maxDist);
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
    ssrlcv::Unity<float3>* points = demPoints.stereo_disparity(matches,64.0f);
    std::string disparityFile = imagePaths[0].substr(0,imagePaths[0].rfind(delimiter));
    disparityFile = disparityFile.substr(0,disparityFile.rfind(delimiter))  + "/disparity.png";
    ssrlcv::writeDisparityImage(points,0,disparityFile);

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
