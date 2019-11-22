#include "common_includes.h"
#include "Image.cuh"
#include "io_util.h"
#include "FeatureFactory.cuh"
#include "MatchFactory.cuh"
#include "PointCloudFactory.cuh"
#include "matrix_util.cuh"

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
    ssrlcv::Image image[2] = {
      ssrlcv::Image("data/img/Australia/im0.png",1,0),
      ssrlcv::Image("data/img/Australia/im1.png",1,1)
    };
    image[0].camera.cam_pos = {198.164f,241.995f,0.0f};
    image[0].camera.foc = 1133.249f;
    image[0].camera.cam_vec = {0.0f,0.0f,0.0f};
    image[1].camera.cam_pos = {319.909f,241.995f,0.0f};
    image[1].camera.foc = 1133.249f;
    image[1].camera.cam_vec = {0.0f,0.0f,0.0f};

    // float leftCamera[3][3] = {//K for left camera
    //   {1133.249,0,198.164},
    //   {0,1133.249,241.995},
    //   {0,0,1}
    // };
    // float rightCamera[3][3] = {//K for right camera
    //   {1133.249,0,319.909},
    //   {0,1133.249,241.995},
    //   {0,0,1}
    // };
    // float R[3][3] = {0.0f};//rotation matrix
    // float S[3][3] = {//translation matrix or cross product T
    //   {0.0f,0.0f,0.0f},
    //   {0.0f,0.0f,-doffset},
    //   {0.0f,doffset,0.0f}
    // };
    float fundamental[3][3] = {
      {0.0f,0.0f,0.0f},
      {0.0f,0.0f,-1.0f},
      {0.0f,1.0f,0.0f}
    };

    // //inverse and transpose right camera matrix
    // ssrlcv::inverse(leftCamera,leftCamera);
    // ssrlcv::transpose(leftCamera,leftCamera);
    // //multiply by R then S
    // ssrlcv::multiply(leftCamera,R,leftCamera);
    // ssrlcv::multiply(leftCamera,S,leftCamera);
    // //multiply by inverse of left camera matrix
    // ssrlcv::inverse(rightCamera,rightCamera);
    // ssrlcv::multiply(leftCamera,rightCamera,fundamental);

    /*
    FEATURE EXTRACTION
    */
    // ssrlcv::FeatureFactory featureFactory = ssrlcv::FeatureFactory();
    // ssrlcv::Unity<ssrlcv::Feature<ssrlcv::Window_25x25>>* features[2] = {
    //   featureFactory.generate25x25Windows(&image[0]),
    //   featureFactory.generate25x25Windows(&image[1])
    // };

    /*
    MATCHING
    */
    //ssrlcv::MatchFactory<ssrlcv::Window_25x25> matchFactory = ssrlcv::MatchFactory<ssrlcv::Window_25x25>(0.0f,FLT_MAX);
    //ssrlcv::Unity<ssrlcv::Match>* matches = matchFactory.generateMatchesConstrained(&image[0],features[0],&image[1],features[1],0.5f,fundamental);
    //ssrlcv::Unity<ssrlcv::Match>* matches = ssrlcv::readMatchFile("data/img/Australia/matches.txt");

    ssrlcv::Unity<ssrlcv::Match>* matches = ssrlcv::generateDiparityMatches(image[0].size,image[0].pixels,image[1].size,image[1].pixels,fundamental,3,{0,0});

    /*
    STEREODISPARITY
    */
    ssrlcv::PointCloudFactory demPoints = ssrlcv::PointCloudFactory();
    ssrlcv::Unity<float3>* points = demPoints.stereo_disparity(matches,1133.249f,258.3f,121.745f);

    /*
    OUTPUT FILE IO
    */
    ssrlcv::writeMatchFile(matches, "data/img/Australia/matches.txt");
    ssrlcv::writePLY("data/img/Australia/Australia.ply",points);
    ssrlcv::writeDisparityImage(points,255,"data/img/Australia/disparity.png");

    //delete features[0];
    //delete features[1];
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
