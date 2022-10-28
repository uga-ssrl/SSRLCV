/** \file This file contains methods and kernels for pose estimation
*
*/
#pragma once
#ifndef POSEESTIMATOR_CUH
#define POSEESTIMATOR_CUH

#include "common_includes.hpp"
#include "MatchFactory.cuh"
#include "Image.cuh"

namespace ssrlcv{

    struct FMatrixInliers {
        float fmatrix[3][3];
        unsigned long inliers;
        bool valid;
    };

    struct Pose {
      float roll; // x-rotation, in radians
      float pitch; // y-rotation, in radians
      float yaw; // z-rotation, in radians
      float x; // x-position, kilometers
      float y; // y-position, kilometers
      float z; // z-position, kilometers
    };

  /**
  * \brief pose estimator for cameras
  */
  class PoseEstimator{

  private:
    ssrlcv::ptr::value<ssrlcv::Unity<ssrlcv::KeyPoint>> keyPoints; // every two keypoints should be matches (query, target, query, target, ...)
    ssrlcv::ptr::value<ssrlcv::Image> queryImage;
    ssrlcv::ptr::value<ssrlcv::Image> targetImage;

    ssrlcv::ptr::value<ssrlcv::Unity<float>> A; // for equation A F = 0

  public:
    /*
    * Sets up pose estimator to adjust target image
    */
    PoseEstimator(ssrlcv::ptr::value<ssrlcv::Image> queryImage, ssrlcv::ptr::value<ssrlcv::Image> targetImage, ssrlcv::ptr::value<ssrlcv::Unity<ssrlcv::KeyPoint>> keyPoints);

    void estimateFundamentalRANSAC(float (&F_out)[3][3]);

    ssrlcv::Pose getPose(const float (&F)[3][3], bool relative=false);

    void LM_optimize(ssrlcv::Pose pose) {
      
    }

    float getCost();
    
  private:

    void fillA();

};

__global__ void computeFMatrixAndInliers(KeyPoint *keyPoints, int numKeypoints, float *V, unsigned long N, ssrlcv::FMatrixInliers *matricesAndInliers);

}

#endif
