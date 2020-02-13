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

    // If you want to test N many matched keypoints, you need at least N many fake cameras
    ssrlcv::Image* image0 = new ssrlcv::Image();
    ssrlcv::Image* image1 = new ssrlcv::Image();
    ssrlcv::Image* image2 = new ssrlcv::Image();
    ssrlcv::Image* image3 = new ssrlcv::Image();
    ssrlcv::Image* image4 = new ssrlcv::Image();
    images_vec.push_back(image0);
    images_vec.push_back(image1);
    images_vec.push_back(image2);
    images_vec.push_back(image3);
    images_vec.push_back(image4);

    // Test Camera Parameters
    // If you want to test N many matched keypoints, you need at least N many fake cameras
    // be sure that you index them correctly in the images_vec
    // be sure that the id matches the index
    std::cout << "Filling in Test Camera Params ..." << std::endl;
    images_vec[0]->id = 0;
    images_vec[0]->camera.size = {2,2};
    images_vec[0]->camera.cam_pos = {1.0,0.0,0.0};
    images_vec[0]->camera.cam_rot = {M_PI/2.0,0.0,0.0};
    images_vec[0]->camera.fov = {10.0 * (M_PI/180.0), 10.0 * (M_PI/180.0))}; // 10 degrees x and y fov
    images_vec[0]->camera.foc = 0.25;

    images_vec[1]->id = 1;
    images_vec[1]->camera.size = {2,2};
    images_vec[1]->camera.cam_pos = {-1.0,0.0,0.0};
    images_vec[1]->camera.cam_rot = {0.5, 0.75, 0.5};
    images_vec[1]->camera.fov = {5,10};
    images_vec[1]->camera.foc = 0.25;

    images_vec[2]->id = 2;
    images_vec[2]->camera.size = {2,2};
    images_vec[2]->camera.cam_pos = {0.0,0.0,1.0};
    images_vec[2]->camera.cam_rot = {0.5, -0.80, -0.5};
    images_vec[2]->camera.fov = {5,10};
    images_vec[2]->camera.foc = 0.25;

    images_vec[3]->id = 3;
    images_vec[3]->camera.size = {2,2};
    images_vec[3]->camera.cam_pos = {0.0,0.0,1.0};
    images_vec[3]->camera.cam_rot = {0.5, -0.80, -0.5};
    images_vec[3]->camera.fov = {5,10};
    images_vec[3]->camera.foc = 0.25;

    images_vec[4]->id = 4;
    images_vec[4]->camera.size = {2,2};
    images_vec[4]->camera.cam_pos = {0.0,0.0,1.0};
    images_vec[4]->camera.cam_rot = {0.5, -0.80, -0.5};
    images_vec[4]->camera.fov = {5,10};
    images_vec[4]->camera.foc = 0.25;

    // Test Match Points
    std::cout << "Filling in Matches ..." << std::endl;
    ssrlcv::MatchSet matchSet;

    // lets say we want the following:
    //    4 sets of matches, so we need to have a count of that
    int matchesnum = 4;
    // then, in terms of our matches we want:
    //    * first match connects 2 keypoints
    //    * second match connects 3 keypoints
    //    * third match connets 2 keypoints
    //    * forth match connets 5 keypoints
    // we need to calcualte how many keypoitns that is, so:
    int keypointnum = 2 + 3 + 2 + 5;

    // next we need to allocate memory for these guys

    // matches contain groups of keypoints, which are just the R2 coordinates tha correspond
    // here is where we use matches num
    matchSet.matches = new ssrlcv::Unity<ssrlcv::MultiMatch>(nullptr,matchesnum,ssrlcv::cpu);
    // this is the list of R2 coodrindates
    // here we want to use the keypoint num
    matchSet.keyPoints = new ssrlcv::Unity<ssrlcv::KeyPoint>(nullptr,keypointnum,ssrlcv::cpu);

    // now we need to fill in what our matches are



    // note that   *->host[#] is the memory location of the match information, that is sequential in RAM and counts up
    matchSet.matches->host[0] = {2,0}; // here we say the number of matches and the starting index of those matches in the keypoints
                                       // that ends up looking like = {number keypoints in the match, where those keyPoints start}
                                       // for us we have 2 matches and the matches start at index 0
    // note that the *->host[#] is the memory location, that's sequential in RAM, so those always count up
    matchSet.keyPoints->host[0] = {{0}, {1.0, 1.0}}; // { {image number}, {x-y match location} }
    matchSet.keyPoints->host[1] = {{1}, {1.0, 1.0}};



    // note that   *->host[#] is the memory location of the match information, that is sequential in RAM and counts up
    matchSet.matches->host[1] = {3,2}; // here we say the number of matches and the starting index of those matches in the keypoints
                                       // that ends up looking like = {number keypoints in the match, where those keyPoints start}
                                       // for us we have 3 matches and the matches start at index 3
    // note that the *->host[#] is the memory location, that's sequential in RAM, so those always count up
    matchSet.keyPoints->host[2] = {{0}, {1.0, 1.0}};
    matchSet.keyPoints->host[3] = {{1}, {1.0, 1.0}};
    matchSet.keyPoints->host[4] = {{2}, {1.0, 1.0}};



    // note that   *->host[#] is the memory location of the match information, that is sequential in RAM and counts up
    matchSet.matches->host[2] = {2,5}; // here we say the number of matches and the starting index of those matches in the keypoints
                                       // that ends up looking like = {number keypoints in the match, where those keyPoints start}
                                       // for us we have 2 matches and the matches start at index 5
    // note that the *->host[#] is the memory location, that's sequential in RAM, so those always count up
                                       // let's say we want to match the last 2 images this time, let's use image indexes 3 and 4
    matchSet.keyPoints->host[5] = {{3}, {1.0, 1.0}}; // { {image number}, {x-y match location} }
    matchSet.keyPoints->host[6] = {{4}, {1.0, 1.0}};



    // note that   *->host[#] is the memory location of the match information, that is sequential in RAM and counts up
    matchSet.matches->host[0] = {5,7}; // here we say the number of matches and the starting index of those matches in the keypoints
                                       // that ends up looking like = {number keypoints in the match, where those keyPoints start}
                                       // for us we have 5 matches and the matches start at index 7
    // note that the *->host[#] is the memory location, that's sequential in RAM, so those always count up
    matchSet.keyPoints->host[7]  = {{0}, {1.0, 1.0}}; // { {image number}, {x-y match location} }
    matchSet.keyPoints->host[8]  = {{1}, {1.0, 1.0}};
    matchSet.keyPoints->host[9]  = {{1}, {1.0, 1.0}};
    matchSet.keyPoints->host[10] = {{1}, {1.0, 1.0}};
    matchSet.keyPoints->host[11] = {{1}, {1.0, 1.0}};

    // now we can try to make lines

    // Line Generation Test
    ssrlcv::PointCloudFactory demPoints = ssrlcv::PointCloudFactory();

    std::cout << "Bundles: " << std::endl;
    ssrlcv::BundleSet bundleSet = demPoints.generateBundles(&matchSet,images_vec);

    // prints out generated lines in point vector format
    std::cout << "<lines start>" << std::endl;
    for(int i = 0; i < bundleSet.bundles->numElements; i ++){
      for (int j = bundleSet.bundles->host[i].index; j < bundleSet.bundles->host[i].index + bundleSet.bundles->host[i].numLines; j++){
        std::cout << "(" << bundleSet.lines->host[j].pnt.x << "," << bundleSet.lines->host[j].pnt.y << "," << bundleSet.lines->host[j].pnt.z << ")    ";
        std::cout << "<" << bundleSet.lines->host[j].vec.x << "," << bundleSet.lines->host[j].vec.y << "," << bundleSet.lines->host[j].vec.z << ">" << std::endl;
      }
      std::cout << std::endl;
    }

    // after the bundle generation, you should have some lines in point vector format

    // TODO
    // should clear out memory here


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
