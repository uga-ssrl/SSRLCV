#include "common_includes.h"
#include "Image.cuh"
#include "io_util.h"
#include "SIFT_FeatureFactory.cuh"
#include "MatchFactory.cuh"
#include "PointCloudFactory.cuh"
#include "MeshFactory.cuh"

//TODO fix gaussian operators - currently creating very low values
ssrlcv::Unity<float>* errors;

int main(int argc, char *argv[]){
  try{
    //CUDA INITIALIZATION
    cuInit(0);
    clock_t totalTimer = clock();
    clock_t partialTimer = clock();


    // test bundle adjustment here

    std::cout << "=========================== TEST 01 ===========================" << std::endl;
    std::cout << "Making fake image guys ..." << std::endl;
    std::vector<ssrlcv::Image*> images_vec;

    ssrlcv::Image* image0 = new ssrlcv::Image();
    ssrlcv::Image* image1 = new ssrlcv::Image();
    images_vec.push_back(image0);
    images_vec.push_back(image1);

    // fill the test camera params
    std::cout << "Filling in Test Camera Params ..." << std::endl;
    images_vec[0]->id = 0;

    images_vec[0]->camera.size = {2,2};
    images_vec[0]->camera.cam_pos = {-2.0,2.0,2.0};
    images_vec[0]->camera.cam_rot = {(M_PI/2),0.0f,0.0f};
    images_vec[0]->camera.fov = {(M_PI/8),(M_PI/8)};
    images_vec[0]->camera.foc = 0.25;

    images_vec[1]->id = 1;
    images_vec[1]->camera.size = {2,2};
    images_vec[1]->camera.cam_pos = {2.0,-2.0,2.0};
    images_vec[1]->camera.cam_rot = {(M_PI/2), 0.0, (M_PI/2)};
    images_vec[1]->camera.fov = {(M_PI/8),(M_PI/8)};
    images_vec[1]->camera.foc = 0.25;

    // fill the test match points
    std::cout << "Filling in Matches ..." << std::endl;
    ssrlcv::Match* matches_host = new ssrlcv::Match[1];
    ssrlcv::Unity<ssrlcv::Match>* matches = new ssrlcv::Unity<ssrlcv::Match>(matches_host, 1, ssrlcv::cpu);
    matches->host[0].keyPoints[0].parentId = 0;
    matches->host[0].keyPoints[1].parentId = 1;
    matches->host[0].keyPoints[0].loc = {1.0,1.0}; // at the center
    matches->host[0].keyPoints[1].loc = {1.0,1.0}; // at the center

    // start testing reprojection
    ssrlcv::PointCloudFactory demPoints = ssrlcv::PointCloudFactory();

    //match interpolation method will take the place of this here.
    ssrlcv::MatchSet matchSet;
    matchSet.keyPoints = new ssrlcv::Unity<ssrlcv::KeyPoint>(nullptr,matches->size()*2,ssrlcv::cpu);
    matchSet.matches = new ssrlcv::Unity<ssrlcv::MultiMatch>(nullptr,matches->size(),ssrlcv::cpu);
    for(int i = 0; i < matches->size(); ++i){
      matchSet.keyPoints->host[i*2] = matches->host[i].keyPoints[0];
      matchSet.keyPoints->host[i*2 + 1] = matches->host[i].keyPoints[1];
      matchSet.matches->host[i] = {2,i*2};
    }

    // test the prefect case
    std::cout << "Testing perfect case ..." << std::endl;

    errors      = new ssrlcv::Unity<float>(nullptr,matchSet->matches->size(),ssrlcv::cpu);
    float* linearError                = (float*) malloc(sizeof(float));
    float* linearErrorCutoff          = (float*) malloc(sizeof(float));
    *linearError                      = 0;
    *linearErrorCutoff                = 9001;
    ssrlcv::BundleSet bundleSet       = demPoints.generateBundles(&matchSet,images_vec);
    ssrlcv::Unity<float3>* test_point = demPoints.twoViewTriangulate(bundleSet, errors, linearError, linearErrorCutoff);

    std::cout << "Prefect point: ( " << test_point->host[0].x << ",  " << test_point->host[0].y << ", " << test_point->host[0].z << " )" << std::endl;

    //ARG PARSING

    // // ========================== REAL BUNDLE ADJUSTMENT ATTEMPT START
    // std::map<std::string,ssrlcv::arg*> args = ssrlcv::parseArgs(argc,argv);
    // if(args.find("dir") == args.end()){
    //   std::cerr<<"ERROR: SFM executable requires a directory of images"<<std::endl;
    //   exit(-1);
    // }
    // ssrlcv::SIFT_FeatureFactory featureFactory = ssrlcv::SIFT_FeatureFactory(1.5f,6.0f);
    // ssrlcv::MatchFactory<ssrlcv::SIFT_Descriptor> matchFactory = ssrlcv::MatchFactory<ssrlcv::SIFT_Descriptor>(0.6f,250.0f*250.0f);
    // bool seedProvided = false;
    // ssrlcv::Unity<ssrlcv::Feature<ssrlcv::SIFT_Descriptor>>* seedFeatures = nullptr;
    // if(args.find("seed") != args.end()){
    //   seedProvided = true;
    //   std::string seedPath = ((ssrlcv::img_arg*)args["seed"])->path;
    //   ssrlcv::Image* seed = new ssrlcv::Image(seedPath,-1);
    //   seedFeatures = featureFactory.generateFeatures(seed,false,2,0.8);
    //   matchFactory.setSeedFeatures(seedFeatures);
    //   delete seed;
    // }
    // std::vector<std::string> imagePaths = ((ssrlcv::img_dir_arg*)args["dir"])->paths;
    // int numImages = (int) imagePaths.size();
    // std::cout<<"found "<<numImages<<" in directory given"<<std::endl;
    //
    // std::vector<ssrlcv::Image*> images;
    // std::vector<ssrlcv::Unity<ssrlcv::Feature<ssrlcv::SIFT_Descriptor>>*> allFeatures;
    // for(int i = 0; i < numImages; ++i){
    //   ssrlcv::Image* image = new ssrlcv::Image(imagePaths[i],i);
    //   ssrlcv::Unity<ssrlcv::Feature<ssrlcv::SIFT_Descriptor>>* features = featureFactory.generateFeatures(image,false,2,0.8);
    //   features->transferMemoryTo(ssrlcv::cpu);
    //   images.push_back(image);
    //   allFeatures.push_back(features);
    // }
    //
    // /*
    // MATCHING
    // */
    // //seeding with false photo
    //
    // std::cout << "Starting matching..." << std::endl;
    // ssrlcv::Unity<float>* seedDistances = (seedProvided) ? matchFactory.getSeedDistances(allFeatures[0]) : nullptr;
    // ssrlcv::Unity<ssrlcv::DMatch>* distanceMatches = matchFactory.generateDistanceMatches(images[0],allFeatures[0],images[1],allFeatures[1],seedDistances);
    // if(seedDistances != nullptr) delete seedDistances;
    //
    // distanceMatches->transferMemoryTo(ssrlcv::cpu);
    // float maxDist = 0.0f;
    // for(int i = 0; i < distanceMatches->size(); ++i){
    //   if(maxDist < distanceMatches->host[i].distance) maxDist = distanceMatches->host[i].distance;
    // }
    // printf("max euclidean distance between features = %f\n",maxDist);
    // if(distanceMatches->getMemoryState() != ssrlcv::gpu) distanceMatches->setMemoryState(ssrlcv::gpu);
    // ssrlcv::Unity<ssrlcv::Match>* matches = matchFactory.getRawMatches(distanceMatches);
    // delete distanceMatches;
    // std::string delimiter = "/";
    // std::string matchFile = imagePaths[0].substr(0,imagePaths[0].rfind(delimiter)) + "/matches.txt";
    // ssrlcv::writeMatchFile(matches, matchFile);
    //
    // // HARD CODED FOR 2 VIEW
    // // Need to fill into to MatchSet boi
    // std::cout << "Generating MatchSet ..." << std::endl;
    // ssrlcv::MatchSet matchSet;
    // matchSet.keyPoints = new ssrlcv::Unity<ssrlcv::KeyPoint>(nullptr,matches->size()*2,ssrlcv::cpu);
    // matchSet.matches = new ssrlcv::Unity<ssrlcv::MultiMatch>(nullptr,matches->size(),ssrlcv::cpu);
    // matches->setMemoryState(ssrlcv::cpu);
    // for(int i = 0; i < matchSet.matches->size(); i++){
    //   matchSet.keyPoints->host[i*2] = matches->host[i].keyPoints[0];
    //   matchSet.keyPoints->host[i*2 + 1] = matches->host[i].keyPoints[1];
    //   matchSet.matches->host[i] = {2,i*2};
    // }
    // std::cout << "Generated MatchSet ..." << std::endl << "Total Matches: " << matches->size() << std::endl << std::endl;
    //
    // /*
    // attempted bundle adjustment
    // */
    //
    // ssrlcv::PointCloudFactory pc = ssrlcv::PointCloudFactory();
    //
    //
    // ssrlcv::Unity<float3>* points = pc.BundleAdjustTwoView(&matchSet,images);
    //
    // ssrlcv::writePLY("out/bundleAdjustedPoints.ply",points);
    // // points->clear();
    // // ========================== REAL BUNDLE ADJUSTMENT ATTEMPT START


    /*
    2 View Reprojection
    */
    // ssrlcv::PointCloudFactory demPoints = ssrlcv::PointCloudFactory();
    //
    // // bunlde adjustment loop would be here. images_vec woudl be modified to minimize the boi
    // unsigned long long int* linearError = (unsigned long long int*) malloc(sizeof(unsigned long long int));
    // float* linearErrorCutoff = (float*) malloc(sizeof(float));
    // ssrlcv::BundleSet bundleSet = demPoints.generateBundles(&matchSet,images);
    //
    // // the version that will be used normally
    // ssrlcv::Unity<float3>* points = demPoints.twoViewTriangulate(bundleSet, linearError);
    // std::cout << "Total Linear Error: " << *linearError << std::endl;
    //
    // // here is a version that will give me individual linear errors
    // ssrlcv::Unity<float>* errors = new ssrlcv::Unity<float>(nullptr,matches->numElements,ssrlcv::cpu);
    // *linearErrorCutoff = 620.0;
    // ssrlcv::Unity<float3>* points2 = demPoints.twoViewTriangulate(bundleSet, errors, linearError, linearErrorCutoff);
    // // then I write them to a csv to see what to heck is goin on
    // ssrlcv::writeCSV(errors->host, (int) errors->numElements, "individualLinearErrors");

    // optional stereo disparity here
    // /*
    // STEREODISPARITY
    // */
    // ssrlcv::PointCloudFactory demPoints = ssrlcv::PointCloudFactory();
    // ssrlcv::Unity<float3>* points = demPoints.stereo_disparity(matches,8.0);
    //

    // delete matches;
    // ssrlcv::writePLY("out/unfiltered.ply",points);
    // delete points;
    // ssrlcv::writePLY("out/filtered.ply",points2);
    // delete points2;

    // clean up the images
    // for(int i = 0; i < imagePaths.size(); ++i){
    //   delete images[i];
    //   delete allFeatures[i];
    // }

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
