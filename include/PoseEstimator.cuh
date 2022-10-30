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
    ssrlcv::ptr::value<ssrlcv::Unity<ssrlcv::Match>> matches; //keypoints should be (query, target)
    ssrlcv::ptr::value<ssrlcv::Image> query;
    ssrlcv::ptr::value<ssrlcv::Image> target;

    ssrlcv::ptr::value<ssrlcv::Unity<float>> A; // for equation A F = 0

  public:
    /*
    * Sets up pose estimator to adjust target image
    */
    PoseEstimator(ssrlcv::ptr::value<ssrlcv::Image> query, ssrlcv::ptr::value<ssrlcv::Image> target, ssrlcv::ptr::value<ssrlcv::Unity<ssrlcv::Match>> matches);

    ssrlcv::Pose estimatePoseRANSAC();

    void LM_optimize(ssrlcv::Pose pose);
    
  private:

    ssrlcv::Pose getRelativePose(const float (&F)[3][3]);

    bool LM_iteration(ssrlcv::Pose *pose, float *lambda);

    void fillA();

};

__device__ __host__ float4 getResidual(ssrlcv::Pose pose, ssrlcv::Image::Camera *query, ssrlcv::Image::Camera *target, float2 q_loc, float2 t_loc);

__global__ void computeFMatrixAndInliers(ssrlcv::Match *matches, int numMatches, float *V, unsigned long N, ssrlcv::FMatrixInliers *matricesAndInliers);

__global__ void computeOutliers(ssrlcv::Match *matches, int numMatches, float *F);

__global__ void computeResidualsAndJacobian(ssrlcv::Match *matches, int numMatches, ssrlcv::Pose pose, ssrlcv::Image::Camera query, ssrlcv::Image::Camera target, float *residuals, float *jacobian);

__global__ void computeJTJ(float *jacobian, unsigned long rows, float *output);

__global__ void computeJTf(float *jacobian, float *f, unsigned long rows, float *output);

__global__ void computeCost(ssrlcv::Match *matches, int numMatches, ssrlcv::Pose pose, ssrlcv::Image::Camera query, ssrlcv::Image::Camera target, float *cost);
}

#endif
