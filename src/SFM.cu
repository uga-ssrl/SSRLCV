
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

#include "common_includes.hpp"
#include "Pipeline.cuh"
#include "Image.cuh"
#include "io_util.hpp"
#include "SIFT_FeatureFactory.cuh"
#include "MatchFactory.cuh"
#include "PointCloudFactory.cuh"
#include "MeshFactory.cuh"
#include "PoseEstimator.cuh"

/**
 * \brief Example of safe shutdown method caused by a signal.
 * \details the safe shutdown methods is initiated when a SIGINT is captured, but can be extended
 * to many other types of exeption handleing. Here we should makes sure that
 * memory is safely shutting down, CPU threads are killed, and whatever else is desired.
 * \note ssrlcv::Unity<T>::checkpoint() is a great way to keep progress, but the Unity must be 
 * global to call this in any signal capturing method
 */
void safeShutdown(int sig){
  logger.info << "Safely Ending SSRLCV ...";
  logger.logState("safeShutdown");
  logger.stopBackgroundLogging();
  exit(sig); // exit with the same signal
}

int main(int argc, char *argv[]){
  try{

    // register the SIGINT safe shutdown
    std::signal(SIGINT, safeShutdown);

    // CUDA INITIALIZATION
    cuInit(0);
    clock_t totalTimer = clock();
    clock_t partialTimer = clock();

    // initialize the logger, this should ONLY HAPPEN ONCE
    // the logger requires that a "safes shutdown" signal handler is created
    // so that the logger.shutdown() method can be called.
    logger.logState("start"); // these can be used to time parts of the pipeline afterwards and correlate it with ofther stuff
    logger.startBackgoundLogging(1); // write a voltage, current, power log every 5 seconds

    // ARG PARSING

    std::map<std::string, ssrlcv::arg*> args = ssrlcv::parseArgs(argc, argv);

    if(args.find("dir") == args.end()){
      std::cerr << "ERROR: SFM executable requires a directory of images" << std::endl;
      exit(-1);
    }

    std::string seedPath;
    if(args.find("seed") != args.end()){
      seedPath = ((ssrlcv::img_arg *)args["seed"])->path;
    }
    std::vector<std::string> imagePaths = ((ssrlcv::img_dir_arg *)args["dir"])->paths;
    int numImages = (int) imagePaths.size();
    logger.info.printf("Found %d images in directory given", numImages);
    logger.logState("SEED");

    // off-precision distance for epipolar matching (in pixels)
    float epsilon = 5.0;
    if (args.find("epsilon") != args.end()) {
      epsilon = ((ssrlcv::flt_arg *)args["epsilon"])->val;

      logger.info.printf("Setting epsilon (for epipolar geometry) to %f pixels.", epsilon);
    }

    // off-precision distance for orbital SfM (in kilometers)
    float delta = 0.0;
    if (args.find("delta") != args.end()) {
      delta = ((ssrlcv::flt_arg *)args["delta"])->val;

      logger.info.printf("Setting delta (for earth-centered epipolar geometry) to %f kilometers.", delta);
    }


    ssrlcv::ptr::value<ssrlcv::Image> image0 = ssrlcv::ptr::value<ssrlcv::Image>(imagePaths[0], 0);
    ssrlcv::ptr::value<ssrlcv::Image> image1 = ssrlcv::ptr::value<ssrlcv::Image>(imagePaths[1], 1);
 
    ssrlcv::MatchSet matchSet {
      std::string("tmp/0_N6ssrlcv8KeyPointE.uty"),
      std::string("tmp/0_N6ssrlcv10MultiMatchE.uty")
    };

    ssrlcv::PoseEstimator estim(image0, image1, matchSet.keyPoints);

    estim.estimatePoseRANSAC();

       /*

    //std::cout << matchSet.keyPoints->size() << std::endl;

    float2 locSum1 {0.0f, 0.0f};
    float2 locSum2 {0.0f, 0.0f};

    // get average position
    for (int i = 0; i < matchSet.keyPoints->size(); i ++) {
      if (i % 2 == 0) {
        locSum1 += matchSet.keyPoints->host.get()[i].loc;
        std::cout << matchSet.keyPoints->host.get()[i].loc.x << "," << matchSet.keyPoints->host.get()[i].loc.y << ",";
      } else {
        locSum2 += matchSet.keyPoints->host.get()[i].loc;
        std::cout << matchSet.keyPoints->host.get()[i].loc.x << "," << matchSet.keyPoints->host.get()[i].loc.y << std::endl;
      }
    }


    locSum1 /= (float)(matchSet.keyPoints->size() / 2);
    locSum2 /= (float)(matchSet.keyPoints->size() / 2);

    float dist1 = 0.0f;
    float dist2 = 0.0f;

    // update average position to center and compute average distance from center
    for (int i = 0; i < matchSet.keyPoints->size(); i ++) {
      if (i % 2 == 0) {
        matchSet.keyPoints->host.get()[i].loc -= locSum1;
        dist1 += dotProduct(matchSet.keyPoints->host.get()[i].loc, matchSet.keyPoints->host.get()[i].loc);
      } else {
        matchSet.keyPoints->host.get()[i].loc -= locSum2;
        dist2 += dotProduct(matchSet.keyPoints->host.get()[i].loc, matchSet.keyPoints->host.get()[i].loc);
      }
    }

    float scale1 = sqrtf(2.0) /  sqrtf(dist1 / (matchSet.keyPoints->size() / 2));
    float scale2 = sqrtf(2.0) /  sqrtf(dist2 / (matchSet.keyPoints->size() / 2));

    // scale so points are a RMS of sqrt(2) dist. to new center
    for (int i = 0; i < matchSet.keyPoints->size(); i ++) {
      if (i % 2 == 0) {
        matchSet.keyPoints->host.get()[i].loc *= scale1;
      } else {
        matchSet.keyPoints->host.get()[i].loc *= scale2;
      }
    }

    ssrlcv::ptr::host<float> A(matchSet.keyPoints->size() / 2 * 9);

    for (int i = 0; i < matchSet.keyPoints->size(); i += 2) {
      float *row = A.get() + (i / 2 * 9);
      float2 loc1 = matchSet.keyPoints->host.get()[i].loc;
      float2 loc2 = matchSet.keyPoints->host.get()[i+1].loc;
      row[0] = loc2.x * loc1.x;
      row[1] = loc2.x * loc1.y;
      row[2] = loc2.x;
      row[3] = loc2.y * loc1.x;
      row[4] = loc2.y * loc1.y;
      row[5] = loc2.y;
      row[6] = loc1.x;
      row[7] = loc1.y;
      row[8] = 1;
    }

    cusolverDnHandle_t cusolverH = nullptr;
    cublasHandle_t cublasH = nullptr;
    cublasStatus_t cublas_status = CUBLAS_STATUS_SUCCESS;
    cusolverStatus_t cusolver_status = CUSOLVER_STATUS_SUCCESS;

    cusolver_status = cusolverDnCreate(&cusolverH);
    assert(CUSOLVER_STATUS_SUCCESS == cusolver_status);

    cublas_status = cublasCreate(&cublasH);
    assert(CUBLAS_STATUS_SUCCESS == cublas_status);

    int m = matchSet.keyPoints->size() / 2;
    int n = 9;
    int lwork = 0;

    ssrlcv::ptr::device<float> d_A(m * n);
    ssrlcv::ptr::device<float> d_S(n);
    ssrlcv::ptr::device<float> d_U(m * m);
    ssrlcv::ptr::device<float> d_VT(n * n);
    ssrlcv::ptr::device<int> devInfo(1);
    ssrlcv::ptr::device<float> d_rwork(8);

    CudaSafeCall(cudaMemcpy(d_A.get(), A.get(), m*n*sizeof(float), cudaMemcpyHostToDevice));

    cusolver_status = cusolverDnSgesvd_bufferSize(cusolverH, m, n, &lwork);
    assert(cusolver_status == CUSOLVER_STATUS_SUCCESS);

    ssrlcv::ptr::device<float> d_work(lwork);

    cusolver_status = cusolverDnSgesvd(cusolverH, 'A', 'A', m, n,
      d_A.get(), m, d_S.get(), d_U.get(), m, d_VT.get(), n, d_work.get(), lwork, d_rwork.get(), devInfo.get());
    cudaDeviceSynchronize();
    assert(cusolver_status == CUSOLVER_STATUS_SUCCESS);

    float f_hat[3][3];

    CudaSafeCall(cudaMemcpy(&f_hat, d_VT.get() + 9 * 8, 9*sizeof(float), cudaMemcpyDeviceToHost));

    ssrlcv::ptr::device<float> d_f_hat(9);
    CudaSafeCall(cudaMemcpy(d_f_hat.get(), &f_hat, 9*sizeof(float), cudaMemcpyHostToDevice));
    m = 3;
    n = 3;

    cusolver_status = cusolverDnSgesvd_bufferSize(cusolverH, m, n, &lwork);
    assert(cusolver_status == CUSOLVER_STATUS_SUCCESS);

    ssrlcv::ptr::device<float> d_work2(lwork);

    cusolver_status = cusolverDnSgesvd(cusolverH, 'A', 'A', m, n,
      d_f_hat.get(), m, d_S.get(), d_U.get(), m, d_VT.get(), n, d_work.get(), lwork, d_rwork.get(), devInfo.get());
    cudaDeviceSynchronize();
    assert(cusolver_status == CUSOLVER_STATUS_SUCCESS);

    float S[3];
    CudaSafeCall(cudaMemcpy(&S, d_S.get(), 3*sizeof(float), cudaMemcpyDeviceToHost));
    float U[3][3];
    CudaSafeCall(cudaMemcpy(&U, d_U.get(), 9*sizeof(float), cudaMemcpyDeviceToHost));
    float VT[3][3];
    CudaSafeCall(cudaMemcpy(&VT, d_VT.get(), 9*sizeof(float), cudaMemcpyDeviceToHost));
    float diag[3][3] = {
      {S[0], 0, 0},
      {0, S[1], 0},
      {0, 0, 0}
    };
    float U_diag[3][3];
    ssrlcv::multiply(U, diag, U_diag);
    float f_hat_prime[3][3];
    ssrlcv::multiply(U_diag, VT, f_hat_prime);

    float T1[3][3] = {
      {scale1, 0, -scale1 * locSum1.x},
      {0, scale1, -scale1 * locSum1.y},
      {0, 0, 1}
    };

    float T2[3][3] = {
      {scale2, 0, -scale2 * locSum2.x},
      {0, scale2, -scale2 * locSum2.y},
      {0, 0, 1}
    };

    float f_almost[3][3];

    float T2T[3][3];

    ssrlcv::transpose(T2, T2T);

    ssrlcv::multiply(T2T, f_hat_prime, f_almost);

    float f[3][3];

    ssrlcv::multiply(f_almost, T1, f);


    std::cout << f[0][0] << "," << f[0][1] << "," << f[0][2] << std::endl;
    std::cout << f[1][0] << "," << f[1][1] << "," << f[1][2] << std::endl;
    std::cout << f[2][0] << "," << f[2][1] << "," << f[2][2] << std::endl;

    std::cout << std::endl;

    ssrlcv::ptr::value<ssrlcv::Image> image0 = ssrlcv::ptr::value<ssrlcv::Image>(imagePaths[0], 0);
    ssrlcv::ptr::value<ssrlcv::Image> image1 = ssrlcv::ptr::value<ssrlcv::Image>(imagePaths[1], 1);

    float3 F0[3];

    float p1[3] = {200, 300, 1};
    float p2[3];

    ssrlcv::multiply(f, p1, p2);

    std::cout << p2[0] << "," << p2[1] << "," << p2[2] << std::endl << std::endl;
    */

    
/*
    calcFundamentalMatrix_2View(image1.get(), image0.get(), F0);

    std::cout << F0[0].x << "," << F0[0].y << "," << F0[0].z << std::endl;
    std::cout << F0[1].x << "," << F0[1].y << "," << F0[1].z << std::endl;
    std::cout << F0[2].x << "," << F0[2].y << "," << F0[2].z << std::endl << std::endl;


    float F[3][3] = {
      {F0[0].x, F0[0].y, F0[0].z},
      {F0[1].x, F0[1].y, F0[1].z},
      {F0[2].x, F0[2].y, F0[2].z}
    };
    ssrlcv::multiply(F, p1, p2);
    std::cout << p2[0] << "," << p2[1] << "," << p2[2] << std::endl << std::endl;
*/
/*
    float3 K[3];
    K[0] = {image0->camera.foc / image0->camera.dpix.x, 0, image0->size.x / 2.0f};
    K[1] = {0, image0->camera.foc / image0->camera.dpix.y, image0->size.y / 2.0f};
    K[2] = {0, 0, 1};
    float3 KT[3], KTF[3], E[3];
    ssrlcv::transpose(K, KT);
    float3 F[3] = {
      {f[0][0], f[0][1], f[0][2]},
      {f[1][0], f[1][1], f[1][2]},
      {f[2][0], f[2][1], f[2][2]}
    };
    ssrlcv::multiply(KT, F, KTF);
    ssrlcv::multiply(KTF, K, E);



    std::cout << E[0].x << "," << E[0].y << "," << E[0].z << std::endl;
    std::cout << E[1].x << "," << E[1].y << "," << E[1].z << std::endl;
    std::cout << E[2].x << "," << E[2].y << "," << E[2].z << std::endl << std::endl;

*/

    // cleanup
    for (ssrlcv::arg_pair p : args) {
      delete p.second; 
    }

    logger.logState("end");
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
