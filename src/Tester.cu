#include "common_includes.h"
#include "Image.cuh"
#include "io_util.h"
#include "SIFT_FeatureFactory.cuh"
#include "MatchFactory.cuh"
#include "PointCloudFactory.cuh"
#include "MeshFactory.cuh"

int main(int argc, char *argv[]){
  try{
    //CUDA INITIALIZATION
    cuInit(0);
    clock_t totalTimer = clock();
    clock_t partialTimer = clock();

    //ARG PARSING
    if(argc < 2 || argc > 4){
      std::cout<<"USAGE ./bin/Tester </path/to/image/directory/>"<<std::endl;
      exit(-1);
    }
    std::string path = argv[1];
    std::vector<std::string> imagePaths = ssrlcv::findFiles(path);

    int numImages = (int) imagePaths.size();

    std::cout << "test code running... " << std::endl;
    // ===================================================================
    // test code goes below here
    // Code within these comment blocks can always be deteled and
    // should only be used when you're testing your new stuff
    // ===================================================================

    // fake cameras
    // doesn't matter what the images are
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
    images_vec[0]->camera.cam_pos = {0.0,0.0,0.0};
    images_vec[0]->camera.cam_vec = {1.0,1.0,1.0};
    images_vec[0]->camera.axangle = 0.0;
    images_vec[0]->camera.fov = (10.0 * (M_PI/180.0));//30.0;
    images_vec[0]->camera.foc = 0.25;
    images_vec[1]->id = 1;
    images_vec[1]->camera.size = {2,2};
    images_vec[1]->camera.cam_pos = {0.0,0.0,0.0};
    images_vec[1]->camera.cam_vec = {-1.0, -1.0, -1.0};
    images_vec[1]->camera.axangle = M_PI/2.0;
    images_vec[1]->camera.fov = (10.0 * (M_PI/180.0));//30.0;
    images_vec[1]->camera.foc = 0.25;
    // ssrlcv::Unity<ssrlcv::Image>* images = new ssrlcv::Unity<ssrlcv::Image>(images_vec, 2, ssrlcv::cpu);

    // fill the test match points
    std::cout << "Filling in Matches ..." << std::endl;
    ssrlcv::Match* matches_host = new ssrlcv::Match[1];
    ssrlcv::Unity<ssrlcv::Match>* matches = new ssrlcv::Unity<ssrlcv::Match>(matches_host, 1, ssrlcv::cpu);
    matches->host[0].keyPoints[0].parentId = 0;
    matches->host[0].keyPoints[1].parentId = 1;
    matches->host[0].keyPoints[0].loc = {1.0,1.0}; // at the center!
    matches->host[0].keyPoints[1].loc = {1.0,1.0}; // in the corner

    // test the line gen method
    ssrlcv::PointCloudFactory demPoints = ssrlcv::PointCloudFactory();

    //match interpolation method will take the place of this here.
    ssrlcv::MatchSet matchSet;
    matchSet.keyPoints = new ssrlcv::Unity<ssrlcv::KeyPoint>(nullptr,matches->numElements*2,ssrlcv::cpu);
    matchSet.matches = new ssrlcv::Unity<ssrlcv::MultiMatch>(nullptr,matches->numElements,ssrlcv::cpu);
    for(int i = 0; i < matches->numElements; ++i){
      matchSet.keyPoints->host[i*2] = matches->host[i].keyPoints[0];
      matchSet.keyPoints->host[i*2 + 1] = matches->host[i].keyPoints[1];
      matchSet.matches->host[i] = {2,i*2};
    }

    std::cout << "WOW!! look at these bundles: " << std::endl;
    ssrlcv::BundleSet bundleSet = demPoints.generateBundles(&matchSet,images_vec);
    std::cout << "<lines start>" << std::endl;
    for(int i = 0; i < bundleSet.bundles->numElements; i ++){
      for (int j = bundleSet.bundles->host[i].index; j < bundleSet.bundles->host[i].index + bundleSet.bundles->host[i].numLines; j++){
        std::cout << "(" << bundleSet.lines->host[j].pnt.x << "," << bundleSet.lines->host[j].pnt.y << "," << bundleSet.lines->host[j].pnt.z << ")\t\t";
        std::cout << "<" << bundleSet.lines->host[j].vec.x << "," << bundleSet.lines->host[j].vec.y << "," << bundleSet.lines->host[j].vec.z << ">" << std::endl;
      }
      std::cout << std::endl;
    }
    std::cout << "</lines end>" << std::endl;

    std::cout << "=========================== TEST 02 ===========================" << std::endl;
    std::cout << "Making fake image guys ..." << std::endl;
    std::vector<ssrlcv::Image*> images_vec_2;

    ssrlcv::Image* image0_2 = new ssrlcv::Image();
    ssrlcv::Image* image1_2 = new ssrlcv::Image();
    ssrlcv::Image* image2_2 = new ssrlcv::Image();
    images_vec_2.push_back(image0_2);
    images_vec_2.push_back(image1_2);
    images_vec_2.push_back(image2_2);

    // fill the test camera params
    std::cout << "Filling in Test Camera Params ..." << std::endl;
    images_vec_2[0]->id = 0;
    images_vec_2[0]->camera.size = {2,2};
    images_vec_2[0]->camera.cam_pos = {0.0,0.0,0.0};
    images_vec_2[0]->camera.cam_vec = {M_PI/4.0,0.0,0.0};
    // images_vec_2[0]->camera.axangle = 0.0;
    images_vec_2[0]->camera.fov = (10.0 * (M_PI/180.0));//30.0;
    images_vec_2[0]->camera.foc = 1.0; // for easy testing
    images_vec_2[1]->id = 1;
    images_vec_2[1]->camera.size = {2,2};
    images_vec_2[1]->camera.cam_pos = {1.0,0.0,0.0};
    images_vec_2[1]->camera.cam_vec = {0.0,M_PI/4.0,0.0};
    // images_vec_2[1]->camera.axangle = M_PI/2.0;
    images_vec_2[1]->camera.fov = (10.0 * (M_PI/180.0));//30.0;
    images_vec_2[1]->camera.foc = 1.0;
    images_vec_2[2]->id = 2;
    images_vec_2[2]->camera.size = {2,2};
    images_vec_2[2]->camera.cam_pos = {2.0,0.0,0.0};
    images_vec_2[2]->camera.cam_vec = {0.0, 0.0, M_PI/4.0};
    // images_vec_2[2]->camera.axangle = M_PI;
    images_vec_2[2]->camera.fov = (10.0 * (M_PI/180.0));//30.0;
    images_vec_2[2]->camera.foc = 1.0;
    // ssrlcv::Unity<ssrlcv::Image>* images = new ssrlcv::Unity<ssrlcv::Image>(images_vec, 2, ssrlcv::cpu);

    // fill the test match points
    std::cout << "Filling in Matches ..." << std::endl;

    //match interpolation method will take the place of this here.
    ssrlcv::MatchSet matchSet_2;
    // 2 sets of keyPoints total
    // 1 set of 3 keyPoints
    // 1 set of 2 keypoints
    matchSet_2.matches = new ssrlcv::Unity<ssrlcv::MultiMatch>(nullptr,2,ssrlcv::cpu);
    matchSet_2.keyPoints = new ssrlcv::Unity<ssrlcv::KeyPoint>(nullptr,(3+2),ssrlcv::cpu);

    // define the guys
    matchSet_2.matches->host[0] = {3,0};
    // that set of matches
    matchSet_2.keyPoints->host[0] = {{0},{1.0,1.0}};
    matchSet_2.keyPoints->host[1] = {{1},{1.0,1.0}};
    matchSet_2.keyPoints->host[2] = {{2},{1.0,1.0}};
    // define the guys
    matchSet_2.matches->host[1] = {2,3};
    // that set of matches
    matchSet_2.keyPoints->host[3] = {{0},{1.0,1.0}};
    matchSet_2.keyPoints->host[4] = {{1},{1.0,1.0}};

    // test the line gen method
    ssrlcv::PointCloudFactory demPoints_2 = ssrlcv::PointCloudFactory();

    std::cout << "WOW!! look at these bundles: " << std::endl;
    ssrlcv::BundleSet bundleSet_2 = demPoints_2.generateBundles(&matchSet_2,images_vec_2);
    std::cout << "<lines start>" << std::endl;
    for(int i = 0; i < bundleSet_2.bundles->numElements; i ++){
      for (int j = bundleSet_2.bundles->host[i].index; j < bundleSet_2.bundles->host[i].index + bundleSet_2.bundles->host[i].numLines; j++){
        std::cout << "(" << bundleSet_2.lines->host[j].pnt.x << "," << bundleSet_2.lines->host[j].pnt.y << "," << bundleSet_2.lines->host[j].pnt.z << ")    ";
        std::cout << "<" << bundleSet_2.lines->host[j].vec.x << "," << bundleSet_2.lines->host[j].vec.y << "," << bundleSet_2.lines->host[j].vec.z << ">" << std::endl;
      }
      std::cout << std::endl;
    }
    std::cout << "</lines end>" << std::endl;




    // ===================================================================
    // test code goes ends above here
    // ===================================================================
    std::cout << "done running test code ... " << std::endl;

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
