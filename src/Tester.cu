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
    std::cout << "Making fake image guys ..." << std::endl;

    std::vector<ssrlcv::Image*> images;

    ssrlcv::Image* image0 = new ssrlcv::Image();
    ssrlcv::Image* image1 = new ssrlcv::Image();
    images.push_back(image0);
    images.push_back(image1);

    // fill the test camera params
    std::cout << "Filling in Test Camera Params ..." << std::endl;

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
    images[0]->id = 2;
    images[0]->camera.size = {1024,1024};
    images[0]->camera.cam_pos = {0.000000000000,6.840402866513,-18.793852415718};
    images[0]->camera.cam_rot = {0.349065850399, 0.0, 0.0};
    images[0]->camera.fov = {0.174532925199,0.174532925199};
    images[0]->camera.foc = 0.160000000000;

    // fill the test match points
    std::cout << "Filling in Matches ..." << std::endl;

    ssrlcv::MatchSet matchSet;
    matchSet.matches->host[0] = {3,0};
    matchSet.keyPoints->host[0] = {{0},{203.990169526,820.009830474}};
    matchSet.keyPoints->host[1] = {{1},{207.021343161,865.304333746}};
    matchSet.keyPoints->host[2] = {{2},{210.377075007,898.593952912}};
    matchSet.matches->host[1] = {3,3};
    matchSet.keyPoints->host[3] = {{0},{820.009830474,820.009830474}};
    matchSet.keyPoints->host[4] = {{1},{816.978656839,865.304333746}};
    matchSet.keyPoints->host[5] = {{2},{813.622924993,898.593952912}};
    matchSet.matches->host[2] = {3,6};
    matchSet.keyPoints->host[6] = {{0},{820.009830474,203.990169526}};
    matchSet.keyPoints->host[7] = {{1},{822.600169364,260.053698516}};
    matchSet.keyPoints->host[8] = {{2},{824.645420239,325.140437119}};
    matchSet.matches->host[3] = {3,9};
    matchSet.keyPoints->host[9] = {{0},{203.990169526,203.990169526}};
    matchSet.keyPoints->host[10] = {{1},{201.399830636,260.053698516}};
    matchSet.keyPoints->host[11] = {{2},{199.354579761,325.140437119}};
    matchSet.matches->host[4] = {3,12};
    matchSet.keyPoints->host[12] = {{0},{233.324439095,790.675560905}};
    matchSet.keyPoints->host[13] = {{1},{235.411443718,736.357455859}};
    matchSet.keyPoints->host[14] = {{2},{237.01335565,676.351948997}};
    matchSet.matches->host[5] = {3,15};
    matchSet.keyPoints->host[15] = {{0},{790.675560905,790.675560905}};
    matchSet.keyPoints->host[16] = {{1},{788.588556282,736.357455859}};
    matchSet.keyPoints->host[17] = {{2},{786.98664435,676.351948997}};
    matchSet.matches->host[6] = {3,18};
    matchSet.keyPoints->host[18] = {{0},{790.675560905,233.324439095}};
    matchSet.keyPoints->host[19] = {{1},{793.204262445,186.237254438}};
    matchSet.keyPoints->host[20] = {{2},{796.118838447,147.841258238}};
    matchSet.matches->host[7] = {3,21};
    matchSet.keyPoints->host[21] = {{0},{233.324439095,233.324439095}};
    matchSet.keyPoints->host[22] = {{1},{230.795737555,186.237254438}};
    matchSet.keyPoints->host[23] = {{2},{227.881161553,147.841258238}};
    matchSet.matches->host[8] = {3,24};
    matchSet.keyPoints->host[24] = {{0},{512.0,512.0}};
    matchSet.keyPoints->host[25] = {{1},{512.0,512.0}};
    matchSet.keyPoints->host[26] = {{2},{512.0,512.0}};

    // ====================== END FOR MANUAL TESTING

    std::cout << "Generated MatchSet ..." << std::endl << "Total Matches: " << matcheSet.matches->size() << std::endl << std::endl;

    // start testing reprojection
    ssrlcv::PointCloudFactory demPoints = ssrlcv::PointCloudFactory();



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
