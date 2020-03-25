
// _______________________________________________________________________________________________________________
//  _____/\\\\\\\\\\\_______/\\\\\\\\\\\______/\\\\\\\\\______/\\\____________________/\\\\\\\\\__/\\\________/\\\_
//   ___/\\\/////////\\\___/\\\/////////\\\__/\\\///////\\\___\/\\\_________________/\\\////////__\/\\\_______\/\\\_
//    __\//\\\______\///___\//\\\______\///__\/\\\_____\/\\\___\/\\\_______________/\\\/___________\//\\\______/\\\__
//     ___\////\\\___________\////\\\_________\/\\\\\\\\\\\/____\/\\\______________/\\\______________\//\\\____/\\\___
//      ______\////\\\___________\////\\\______\/\\\//////\\\____\/\\\_____________\/\\\_______________\//\\\__/\\\____
//       _________\////\\\___________\////\\\___\/\\\____\//\\\___\/\\\_____________\//\\\_______________\//\\\/\\\_____
//        __/\\\______\//\\\___/\\\______\//\\\__\/\\\_____\//\\\__\/\\\______________\///\\\______________\//\\\\\______
//         _\///\\\\\\\\\\\/___\///\\\\\\\\\\\/___\/\\\______\//\\\_\/\\\\\\\\\\\\\\\____\////\\\\\\\\\______\//\\\_______
//          ___\///////////_______\///////////_____\///________\///__\///////////////________\/////////________\///________
//           _______________________________________________________________________________________________________________



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

    // fake 2 images

    ssrlcv::Image* image0 = new ssrlcv::Image();
    ssrlcv::Image* image1 = new ssrlcv::Image();
    images.push_back(image0);
    images.push_back(image1);

    images[0]->id = 0;
    images[0]->camera.size = {1024,1024};
    images[0]->camera.cam_pos = {0.000000000000,0.000000000000,-20.000000000000};
    images[0]->camera.cam_rot = {0.0, 0.0, 0.0};
    images[0]->camera.fov = {0.174532925199,0.174532925199};
    images[0]->camera.foc = 0.160000000000;
    images[0]->id = 1;
    images[0]->camera.size = {1024,1024};
    images[0]->camera.cam_pos = {0.000000000000,3.472963553339,-19.696155060244};
    images[0]->camera.cam_rot = {0.174532925199, 0.0, 0.0};
    images[0]->camera.fov = {0.174532925199,0.174532925199};
    images[0]->camera.foc = 0.160000000000;

    // fake 2-view cube

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

    //
    // 2 View Case
    //
    matchSet.keyPoints = new ssrlcv::Unity<ssrlcv::KeyPoint>(nullptr,matches->size()*2,ssrlcv::cpu);
    matchSet.matches = new ssrlcv::Unity<ssrlcv::MultiMatch>(nullptr,matches->size(),ssrlcv::cpu);
    matches->setMemoryState(ssrlcv::cpu);
    matchSet.matches->setMemoryState(ssrlcv::cpu);
    matchSet.keyPoints->setMemoryState(ssrlcv::cpu);
    for(int i = 0; i < matchSet.matches->size(); i++){
      matchSet.keyPoints->host[i*2] = matches->host[i].keyPoints[0];
      matchSet.keyPoints->host[i*2 + 1] = matches->host[i].keyPoints[1];
      matchSet.matches->host[i] = {2,i*2};
    }
    std::cout << "Generated MatchSet ..." << std::endl << "Total Matches: " << matches->size() << std::endl << std::endl;

    // the point boi
    ssrlcv::PointCloudFactory demPoints = ssrlcv::PointCloudFactory();
    ssrlcv::Unity<float3>* points;
    ssrlcv::BundleSet bundleSet;


    //
    // 2-view test
    //
    std::cout << "Attempting 2-view Triangulation" << std::endl;

    float* linearError = (float*)malloc(sizeof(float));

    bundleSet = demPoints.generateBundles(&matchSet,images);
    points = demPoints.twoViewTriangulate(bundleSet, linearError);

    std::cout << "initial linearError: " << *linearError << std::endl;

    //
    // now start a test of bundle adjustment
    //

    // start by messing up the initial paramters


    // now start the bundle adjustment 2-view loop

    // cleanup
    delete points;
    delete matches;
    delete matchSet.matches;
    delete matchSet.keyPoints;
    delete bundleSet.bundles;
    delete bundleSet.lines;
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

























































// yeet
