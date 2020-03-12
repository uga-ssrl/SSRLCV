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


    // test bundle adjustment here

    // ====================== FOR MANUAL TESTING

    std::cout << "=========================== TEST 01 ===========================" << std::endl;
    /*
    std::cout << "Making fake image guys ..." << std::endl;

    std::vector<ssrlcv::Image*> images;

    ssrlcv::Image* image0 = new ssrlcv::Image();
    ssrlcv::Image* image1 = new ssrlcv::Image();
    images.push_back(image0);
    images.push_back(image1);
    */

    // fill the test camera params
    std::cout << "Filling in Test Camera Params ..." << std::endl;

    // ===== IF loading real image stuff

    /*
    std::map<std::string,ssrlcv::arg*> args = ssrlcv::parseArgs(argc,argv);
    if(args.find("dir") == args.end()){
      std::cerr<<"ERROR: SFM executable requires a directory of images"<<std::endl;
      exit(-1);
    }

    std::vector<std::string> imagePaths = ((ssrlcv::img_dir_arg*)args["dir"])->paths;
    int numImages = (int) imagePaths.size();
    std::cout<<"found "<<numImages<<" in directory given"<<std::endl;

    std::vector<ssrlcv::Image*> images;
    std::vector<ssrlcv::Unity<ssrlcv::Feature<ssrlcv::SIFT_Descriptor>>*> allFeatures;
    for(int i = 0; i < numImages; ++i){
      ssrlcv::Image* image = new ssrlcv::Image(imagePaths[i],i);
      images.push_back(image);
    }
    */

    //  X        Y      Z        rot_x    rot_y      rot_z    fov_x           fov_y         foc  dpix  dpix   time        x   y
    // 781.417, 0.0,  4436.30,  0.0,  0.1745329252, 0.0,  0.19933754453,  0.19933754453,  0.16, 0.4,  0.4,  1580766557, 1024,1024,
    // 0.0,     0.0,  4500.0,   0.0,  0.0,          0.0,  0.19933754453,  0.19933754453,  0.16, 0.4,  0.4,  1580766557, 1024,1024,

    /*

    images[0]->id = 0;
    images[0]->camera.size = {1024,1024};
    images[0]->camera.cam_pos = {0.000000000000,0.000000000000,-20.000000000000};
    images[0]->camera.cam_rot = {0.0, 0.0, 0.0};
    images[0]->camera.fov = {0.174532925199,0.174532925199};
    images[0]->camera.foc = 0.160000000000;
    images[1]->id = 1;
    images[1]->camera.size = {1024,1024};
    images[1]->camera.cam_pos = {0.000000000000,14.142135623731,-14.142135623731};
    images[1]->camera.cam_rot = {0.785398163397, 0.0, 0.0};
    images[1]->camera.fov = {0.174532925199,0.174532925199};
    images[1]->camera.foc = 0.160000000000;

    // fill the test match points
    std::cout << "Filling in Matches ..." << std::endl;

    ssrlcv::Match* matches_host = new ssrlcv::Match[9];
    ssrlcv::Unity<ssrlcv::Match>* matches = new ssrlcv::Unity<ssrlcv::Match>(matches_host, 9, ssrlcv::cpu);
    matches->host[0].keyPoints[0].parentId = 0;
    matches->host[0].keyPoints[1].parentId = 1;
    matches->host[0].keyPoints[0].loc = {203.990169526,820.009830474};
    matches->host[0].keyPoints[1].loc = {219.390661049,925.812095621};
    matches->host[1].keyPoints[0].parentId = 0;
    matches->host[1].keyPoints[1].parentId = 1;
    matches->host[1].keyPoints[0].loc = {820.009830474,820.009830474};
    matches->host[1].keyPoints[1].loc = {804.609338951,925.812095621};
    matches->host[2].keyPoints[0].parentId = 0;
    matches->host[2].keyPoints[1].parentId = 1;
    matches->host[2].keyPoints[0].loc = {820.009830474,203.990169526};
    matches->host[2].keyPoints[1].loc = {826.874315308,512.0};
    matches->host[3].keyPoints[0].parentId = 0;
    matches->host[3].keyPoints[1].parentId = 1;
    matches->host[3].keyPoints[0].loc = {203.990169526,203.990169526};
    matches->host[3].keyPoints[1].loc = {197.125684692,512.0};
    matches->host[4].keyPoints[0].parentId = 0;
    matches->host[4].keyPoints[1].parentId = 1;
    matches->host[4].keyPoints[0].loc = {233.324439095,790.675560905};
    matches->host[4].keyPoints[1].loc = {238.714840031,512.0};
    matches->host[5].keyPoints[0].parentId = 0;
    matches->host[5].keyPoints[1].parentId = 1;
    matches->host[5].keyPoints[0].loc = {790.675560905,790.675560905};
    matches->host[5].keyPoints[1].loc = {785.285159969,512.0};
    matches->host[6].keyPoints[0].parentId = 0;
    matches->host[6].keyPoints[1].parentId = 1;
    matches->host[6].keyPoints[0].loc = {790.675560905,233.324439095};
    matches->host[6].keyPoints[1].loc = {804.609338951,98.1879043789};
    matches->host[7].keyPoints[0].parentId = 0;
    matches->host[7].keyPoints[1].parentId = 1;
    matches->host[7].keyPoints[0].loc = {233.324439095,233.324439095};
    matches->host[7].keyPoints[1].loc = {219.390661049,98.1879043789};
    matches->host[8].keyPoints[0].parentId = 0;
    matches->host[8].keyPoints[1].parentId = 1;
    matches->host[8].keyPoints[0].loc = {512.0,512.0};
    matches->host[8].keyPoints[1].loc = {512.0,512.0};

    */

    // ====================== END FOR MANUAL TESTING

    // ====================== FOR TESTING WITH INPUT IMAGES


    std::map<std::string,ssrlcv::arg*> args = ssrlcv::parseArgs(argc,argv);
    if(args.find("dir") == args.end()){
      std::cerr<<"ERROR: SFM executable requires a directory of images"<<std::endl;
      exit(-1);
    }
    ssrlcv::SIFT_FeatureFactory featureFactory = ssrlcv::SIFT_FeatureFactory(1.5f,6.0f);
    ssrlcv::MatchFactory<ssrlcv::SIFT_Descriptor> matchFactory = ssrlcv::MatchFactory<ssrlcv::SIFT_Descriptor>(0.6f,250.0f*250.0f);
    bool seedProvided = false;
    ssrlcv::Unity<ssrlcv::Feature<ssrlcv::SIFT_Descriptor>>* seedFeatures = nullptr;
    if(args.find("seed") != args.end()){
      seedProvided = true;
      std::string seedPath = ((ssrlcv::img_arg*)args["seed"])->path;
      ssrlcv::Image* seed = new ssrlcv::Image(seedPath,-1);
      seedFeatures = featureFactory.generateFeatures(seed,false,2,0.8);
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
      ssrlcv::Unity<ssrlcv::Feature<ssrlcv::SIFT_Descriptor>>* features = featureFactory.generateFeatures(image,false,2,0.8);
      features->transferMemoryTo(ssrlcv::cpu);
      images.push_back(image);
      allFeatures.push_back(features);
    }

    // MATCHING

    //seeding with false photo

    std::cout << "Starting matching..." << std::endl;
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


    // ====================== END FOR TESTING WITH INPUT IMAGES

    // HARD CODED FOR 2 VIEW
    // Need to fill into to MatchSet boi
    std::cout << "Generating MatchSet ..." << std::endl;
    ssrlcv::MatchSet matchSet;
    matchSet.keyPoints = new ssrlcv::Unity<ssrlcv::KeyPoint>(nullptr,matches->size()*2,ssrlcv::cpu);
    matchSet.matches = new ssrlcv::Unity<ssrlcv::MultiMatch>(nullptr,matches->size(),ssrlcv::cpu);
    matches->setMemoryState(ssrlcv::cpu);
    for(int i = 0; i < matchSet.matches->size(); i++){
      matchSet.keyPoints->host[i*2] = matches->host[i].keyPoints[0];
      matchSet.keyPoints->host[i*2 + 1] = matches->host[i].keyPoints[1];
      matchSet.matches->host[i] = {2,i*2};
    }
    std::cout << "Generated MatchSet ..." << std::endl << "Total Matches: " << matches->size() << std::endl << std::endl;

    // start testing reprojection
    ssrlcv::PointCloudFactory demPoints = ssrlcv::PointCloudFactory();

    // test the prefect case
    std::cout << "Testing perfect case ..." << std::endl;

    ssrlcv::Unity<float>* errors       = new ssrlcv::Unity<float>(nullptr,matchSet.matches->size(),ssrlcv::cpu);
    float* linearError                 = (float*) malloc(sizeof(float));
    float* linearErrorCutoff           = (float*) malloc(sizeof(float));
    *linearError                       = 0;
    *linearErrorCutoff                 = 9001;
    ssrlcv::BundleSet bundleSet        = demPoints.generateBundles(&matchSet,images);
    ssrlcv::Unity<float3>* test_points = demPoints.twoViewTriangulate(bundleSet, errors, linearError, linearErrorCutoff);

    ssrlcv::writePLY("out/test_points.ply",test_points);

    // now start the bundle adjustment on the cube

    // saves a colored debug cloud! (hope this works on the first try)
    demPoints.saveDebugCloud(test_points, bundleSet, images);

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
