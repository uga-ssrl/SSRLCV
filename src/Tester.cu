#include "common_includes.h"
#include "Image.cuh"
#include "io_util.h"
#include "SIFT_FeatureFactory.cuh"
#include "MatchFactory.cuh"
#include "PointCloudFactory.cuh"
#include "MeshFactory.cuh"
#include "matrix_util.cuh"

float3 nView(int numLines, ssrlcv::Bundle::Line* lines){
  //Initializing Variables
  float3 S [3];
  float3 C;
  S[0] = {0,0,0};
  S[1] = {0,0,0};
  S[2] = {0,0,0};
  C = {0,0,0};
  //Iterating through the Lines in a Bundle
  for(int i = 0; i < numLines; i++){
    ssrlcv::Bundle::Line L1 = lines[i];
    float3 tmp [3];
    ssrlcv::normalize(L1.vec);
    ssrlcv::matrixProduct(L1.vec, tmp);
    //Subtracting the 3x3 Identity Matrix from tmp
    tmp[0].x -= 1;
    tmp[1].y -= 1;
    tmp[2].z -= 1;
    //Adding tmp to S
    S[0] = S[0] + tmp[0];
    S[1] = S[1] + tmp[1];
    S[2] = S[2] + tmp[2];
    //Adding tmp * pnt to C
    float3 vectmp;
    ssrlcv::multiply(tmp, L1.pnt, vectmp);
    C = C + vectmp;
  }
  /**
   * If all of the directional vectors are skew and not parallel, then I think S is nonsingular.
   * However, I will look into this some more. This may have to use a pseudo-inverse matrix if that
   * is not the case.
   */
  float3 Inverse [3];  
  if(ssrlcv::inverse(S, Inverse)){
    float3 point;
    ssrlcv::multiply(Inverse, C, point);
    return point;
  }else{
    std::cout << "Error" << std::endl;
    exit(1);
  }
}

int main(int argc, char *argv[]){
  try{
    //CUDA INITIALIZATION
    cuInit(0);
    clock_t totalTimer = clock();
    clock_t partialTimer = clock();

    //Declare Lines
    ssrlcv::Bundle::Line * lp = new ssrlcv::Bundle::Line[3];
    ssrlcv::Bundle::Line l1;
    ssrlcv::Bundle::Line l2;
    ssrlcv::Bundle::Line l3;
    l1.pnt = {1, 0, 0};
    l1.vec = {0, 1, 0};
    l2.pnt = {-1, 0, 0};
    l2.vec = {0, 1, 0};
    l3.pnt = {0, 0, 0};
    l3.vec = {0, 0, 1};
    lp[0] = l1;
    lp[1] = l2;
    lp[2] = l3;

    //Test
    float3 point = nView(3, lp);
    std::cout << point.x << " " << point.y << " " << point.z << std::endl;
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
