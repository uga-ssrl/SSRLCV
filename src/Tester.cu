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
    std::cout << "Making fake image guys ..." << std::endl;
    std::vector<ssrlcv::Image*> images;
    ssrlcv::Image* image0 = new ssrlcv::Image();
    ssrlcv::Image* image1 = new ssrlcv::Image();
    images.push_back(image0);
    images.push_back(image1);

    // fill the test camera params
    std::cout << "Filling in test Camera Params ..." << std::endl;
    images[0]->camera.cam_pos = {0.0,0.0,0.0};
    images[0]->camera.cam_vec = {1.0,1.0,1.0};
    images[0]->camera.fov = 30.0;
    images[0]->camera.foc = 0.25;
    images[1]->camera.cam_pos = {0.0,-1.0,0.0};
    images[1]->camera.cam_vec = {1.0, 0.0,0.0};
    images[1]->camera.fov = 30.0;
    images[1]->camera.foc = 0.25;


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
  catch (const ssrlcv::UnityException &e){
      std::cerr << "Caught exception: " << e.what() << '\n';
      std::exit(1);
  }
  catch (...){
      std::cerr << "Caught unknown exception\n";
      std::exit(1);
  }

}
