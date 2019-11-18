#include "common_includes.h"
#include "Image.cuh"
#include "io_util.h"
#include "FeatureFactory.cuh"
#include "MatchFactory.cuh"
#include "PointCloudFactory.cuh"

int main(int argc, char *argv[]){
  try{

    //CUDA INITIALIZATION
    cuInit(0);
    clock_t totalTimer = clock();
    clock_t partialTimer = clock();

    /*
    IMAGE IO
    */
    float baseline = 285.3f;
    float doffset = 121.745f;
    ssrlcv::Image image0 = ssrlcv::Image("data/img/Australia/im0.png",1,0);
    image0.camera.cam_pos = {198.164f,241.995f,0.0f};
    image0.camera.foc = 1133.249f;
    image0.camera.cam_vec = {(doffset/2.0f)-(baseline/2.0f),0.0f,0.0f};
    ssrlcv::Image image1 = ssrlcv::Image("data/img/Australia/im1.png",1,1);
    image1.camera.cam_pos = {319.909f,241.995f,0.0f};
    image1.camera.foc = 1133.249f;
    image1.camera.cam_vec = {(baseline/2.0f)-(doffset/2.0f),0.0f,0.0f};

    // float cam0[3][3] = {
    //   {1133.249,0,198.164},
    //   {0,1133.249,241.995},
    //   {0,0,1}
    // };

    // float cam1[3][3] = {
    //   {1133.249,0,319.909},
    //   {0,1133.249,241.995},
    //   {0,0,1}
    // };

    /*
    FEATURE EXTRACTION
    */
    ssrlcv::FeatureFactory featureFactory = ssrlcv::FeatureFactory();
    ssrlcv::Unity<ssrlcv::Feature<ssrlcv::Window_9x9>>* features0 = featureFactory.generate9x9Windows(&image0);
    ssrlcv::Unity<ssrlcv::Feature<ssrlcv::Window_9x9>>* features1 = featureFactory.generate9x9Windows(&image1);

    /*
    MATCHING
    */
    ssrlcv::MatchFactory<ssrlcv::Window_9x9> matchFactory = ssrlcv::MatchFactory<ssrlcv::Window_9x9>(0.0f,250.0f);
    ssrlcv::Unity<ssrlcv::Match>* matches = matchFactory.generateMatchesConstrained(&image0,features0,&image1,features1,9.0f);
    ssrlcv::Unity<ssrlcv::Match>* matches = ssrlcv::readMatchFile("data/img/Australia/matches.txt");

    /*
    STEREODISPARITY
    */
    ssrlcv::PointCloudFactory demPoints = ssrlcv::PointCloudFactory();
    ssrlcv::Unity<float3>* points = demPoints.stereo_disparity(matches,1133.249f,258.3f,121.745f);

    /*
    OUTPUT FILE IO
    */
    //ssrlcv::writeMatchFile(matches, "data/img/Australia/matches.txt");
    ssrlcv::writePLY("data/img/Australia/Australia.ply",points);
    ssrlcv::writeDisparityImage(points,255,"data/img/Australia/disparity.png");

    delete features0;
    delete features1;
    delete matches;
    delete points;

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
