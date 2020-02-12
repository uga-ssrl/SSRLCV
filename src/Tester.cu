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
    if(rotated_features->numElements != base_features->numElements){
      std::cerr<<"ERROR: not the same number of features"<<std::endl;
      exit(-1);
    }
    float2 size = {(float)base->size.x/2.0f,(float)base->size.y/2.0f};
    int index = 0;
    float rotationError = 0.0f;
    for(int f = 0; f < base_features->numElements; ++f){
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
      if(index >= rotated_features->numElements){
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
    (float)thetasIncorrect*100.0f/base_features->numElements,
    minMax.x,minMax.y,(float)perfectMatches*100.0f/base_features->numElements
  );
  return correctness;
}


int main(int argc, char *argv[]){
  try{

    //CUDA INITIALIZATION
    cuInit(0);
    clock_t totalTimer = clock();
    clock_t partialTimer = clock();

    std::cout << "=========================== TEST 01 ===========================" << std::endl;
    std::cout << "Creating Fake Images" << std::endl;
    std::vector<ssrlcv::Image*> images_vec;
    
    ssrlcv::Image* image0 = new ssrlcv::Image();
    ssrlcv::Image* image1 = new ssrlcv::Image();
    ssrlcv::Image* image2 = new ssrlcv::Image();
    images_vec.push_back(image0);
    images_vec.push_back(image1);
    images_vec.push_back(image1);
    
    // Test Camera Parameters
    std::cout << "Filling in Test Camera Params ..." << std::endl;
    images_vec[0]->id = 0;
    images_vec[0]->camera.size = {2,2};
    images_vec[0]->camera.cam_pos = {1.0,0.0,0.0};
    images_vec[0]->camera.cam_rot = {-1.0,1.0,0.5};
    images_vec[0]->camera.fov = {5,10};
    images_vec[0]->camera.foc = 0.25;
    
    images_vec[1]->id = 1;
    images_vec[1]->camera.size = {2,2};
    images_vec[1]->camera.cam_pos = {-1.0,0.0,0.0};
    images_vec[1]->camera.cam_rot = {0.5, 0.75, 0.5};
    images_vec[1]->camera.fov = {5,10};
    images_vec[1]->camera.foc = 0.25;

    images_vec[1]->id = 1;
    images_vec[1]->camera.size = {2,2};
    images_vec[1]->camera.cam_pos = {0.0,0.0,1.0};
    images_vec[1]->camera.cam_rot = {0.5, -0.80, -0.5};
    images_vec[1]->camera.fov = {5,10};
    images_vec[1]->camera.foc = 0.25;

    // Test Match Points
    std::cout << "Filling in Matches ..." << std::endl;
    ssrlcv::MatchSet matchSet;
    matchSet.matches = new ssrlcv::Unity<ssrlcv::MultiMatch>(nullptr,2,ssrlcv::cpu);
    matchSet.keyPoints = new ssrlcv::Unity<ssrlcv::KeyPoint>(nullptr,(3+2),ssrlcv::cpu);
 
    matchSet.matches->host[0] = {3,0};
    matchSet.keyPoints->host[0] = {{0}, {1.0, 1.0}};
    matchSet.keyPoints->host[1] = {{1}, {1.0, 1.0}};
    matchSet.keyPoints->host[2] = {{2}, {1.0, 1.0}};
    matchSet.matches->host[1] = {2,3};
    matchSet.keyPoints->host[3] = {{0}, {1.0, 1.0}};
    matchSet.keyPoints->host[4] = {{1}, {1.0, 1.0}};

    // Line Generation Test
    ssrlcv::PointCloudFactory demPoints = ssrlcv::PointCloudFactory();

    std::cout << "Bundles: " << std::endl;
    ssrlcv::BundleSet bundleSet = demPoints.generateBundles(&matchSet,images_vec);
    std::cout << "<lines start>" << std::endl;
    for(int i = 0; i < bundleSet.bundles->numElements; i ++){
      for (int j = bundleSet.bundles->host[i].index; j < bundleSet.bundles->host[i].index + bundleSet.bundles->host[i].numLines; j++){
        std::cout << "(" << bundleSet.lines->host[j].pnt.x << "," << bundleSet.lines->host[j].pnt.y << "," << bundleSet.lines->host[j].pnt.z << ")    ";
        std::cout << "<" << bundleSet.lines->host[j].vec.x << "," << bundleSet.lines->host[j].vec.y << "," << bundleSet.lines->host[j].vec.z << ">" << std::endl;
      }
      std::cout << std::endl;
    }

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

