
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

    std::vector<ssrlcv::Image*> images;
    ssrlcv::Image* image0 = new ssrlcv::Image();
    images.push_back(image0);
    ssrlcv::Image* image1 = new ssrlcv::Image();
    images.push_back(image1);

    // cube camera params
    /*
    images[0]->id = 0;
    images[0]->camera.size = {1024,1024};
    images[0]->camera.cam_pos = {0.000000000000,0.000000000000,-400.000000000000};
    images[0]->camera.cam_rot = {0.0, 0.0, 0.0};
    images[0]->camera.fov = {0.0593411945678,0.0593411945678};
    images[0]->camera.foc = 0.160000000000;
    images[1]->id = 1;
    images[1]->camera.size = {1024,1024};
    images[1]->camera.cam_pos = {0.000000000000,69.459271066772,-393.923101204883};
    images[1]->camera.cam_rot = {0.174532925199, 0.0, 0.0};
    images[1]->camera.fov = {0.0593411945678,0.0593411945678};
    images[1]->camera.foc = 0.160000000000;
    */

    // point tests params
    images[0]->id = 0;
    images[0]->camera.size = {1024,1024};
    images[0]->camera.cam_pos = {0.000000000000,0.000000000000,-400.000000000000};
    images[0]->camera.cam_rot = {0.0, 0.0, 0.0};
    images[0]->camera.fov = {0.0593411945678,0.0593411945678};
    images[0]->camera.foc = 0.160000000000;
    images[1]->id = 1;
    images[1]->camera.size = {1024,1024};
    images[1]->camera.cam_pos = {0.000000000000,69.459271066772,-393.923101204883};
    images[1]->camera.cam_rot = {0.174532925199, 0.0, 0.0};
    images[1]->camera.fov = {0.0593411945678,0.0593411945678};
    images[1]->camera.foc = 0.160000000000;


    // fake 2-view cube

    ssrlcv::Match* matches_host = new ssrlcv::Match[9];
    ssrlcv::Unity<ssrlcv::Match>* matches = new ssrlcv::Unity<ssrlcv::Match>(matches_host, 9, ssrlcv::cpu);
    matches->host[0].keyPoints[0].parentId = 0;
    matches->host[0].keyPoints[0].loc = {468.764219112,555.235780888};
    matches->host[0].keyPoints[1].parentId = 1;
    matches->host[0].keyPoints[1].loc = {468.784672247,562.063052731};
    matches->host[1].keyPoints[0].parentId = 0;
    matches->host[1].keyPoints[0].loc = {555.235780888,555.235780888};
    matches->host[1].keyPoints[1].parentId = 1;
    matches->host[1].keyPoints[1].loc = {555.215327753,562.063052731};
    matches->host[2].keyPoints[0].parentId = 0;
    matches->host[2].keyPoints[0].loc = {555.235780888,468.764219112};
    matches->host[2].keyPoints[1].parentId = 1;
    matches->host[2].keyPoints[1].loc = {555.25295805,476.914948916};
    matches->host[3].keyPoints[0].parentId = 0;
    matches->host[3].keyPoints[0].loc = {468.764219112,468.764219112};
    matches->host[3].keyPoints[1].parentId = 1;
    matches->host[3].keyPoints[1].loc = {468.74704195,476.914948916};
    matches->host[4].keyPoints[0].parentId = 0;
    matches->host[4].keyPoints[0].loc = {468.979858917,555.020141083};
    matches->host[4].keyPoints[1].parentId = 1;
    matches->host[4].keyPoints[1].loc = {468.996851695,546.882415518};
    matches->host[5].keyPoints[0].parentId = 0;
    matches->host[5].keyPoints[0].loc = {555.020141083,555.020141083};
    matches->host[5].keyPoints[1].parentId = 1;
    matches->host[5].keyPoints[1].loc = {555.003148305,546.882415518};
    matches->host[6].keyPoints[0].parentId = 0;
    matches->host[6].keyPoints[0].loc = {555.020141083,468.979858917};
    matches->host[6].keyPoints[1].parentId = 1;
    matches->host[6].keyPoints[1].loc = {555.040409834,462.139581969};
    matches->host[7].keyPoints[0].parentId = 0;
    matches->host[7].keyPoints[0].loc = {468.979858917,468.979858917};
    matches->host[7].keyPoints[1].parentId = 1;
    matches->host[7].keyPoints[1].loc = {468.959590166,462.139581969};
    matches->host[8].keyPoints[0].parentId = 0;
    matches->host[8].keyPoints[0].loc = {512.0,512.0};
    matches->host[8].keyPoints[1].parentId = 1;
    matches->host[8].keyPoints[1].loc = {512.0,512.0};
    */

    // center point tests
    /*
    ssrlcv::Match* matches_host = new ssrlcv::Match[1];
    ssrlcv::Unity<ssrlcv::Match>* matches = new ssrlcv::Unity<ssrlcv::Match>(matches_host, 1, ssrlcv::cpu);
    matches->host[0].keyPoints[0].parentId = 0;
    matches->host[0].keyPoints[0].loc = {512.0,512.0};
    matches->host[0].keyPoints[1].parentId = 1;
    matches->host[0].keyPoints[1].loc = {512.0,512.0};
    */

    //
    // 2 View Case
    //
    ssrlcv::MatchSet matchSet;
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
    std::cout << "\t writing initial PLY ..." << std::endl;
    demPoints.saveDebugCloud(points, bundleSet, images, "initial");

    //
    // now start a test of bundle adjustment
    //

    // start by messing up the initial paramters
    // test moving the camera slightly
    images[1]->camera.cam_pos.y += 10.0;
    bundleSet = demPoints.generateBundles(&matchSet,images);
    points = demPoints.twoViewTriangulate(bundleSet, linearError);
    std::cout << "simulated with noise linearError: " << *linearError << std::endl;
    std::cout << "\t writing noisy PLY ..." << std::endl;
    demPoints.saveDebugCloud(points, bundleSet, images, "noisey");

    std::cout << "Starting Bundle Adjustment Loop ..." << std::endl;
    // now start the bundle adjustment 2-view loop
    points = demPoints.BundleAdjustTwoView(&matchSet,images);

    // cleanup
    delete points;
    delete matches;
    delete matchSet.matches;
    delete matchSet.keyPoints;
    delete bundleSet.bundles;
    delete bundleSet.lines;
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
