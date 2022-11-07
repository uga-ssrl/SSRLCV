#ifndef PIPELINE_CUH
#define PIPELINE_CUH

#include "common_includes.hpp"
#include "SIFT_FeatureFactory.cuh"
#include "MatchFactory.cuh"
#include "PointCloudFactory.cuh"
#include "MeshFactory.cuh"
#include "PoseEstimator.cuh"

namespace ssrlcv {

    //
    // FEATURE GENERATION
    //

    struct FeatureGenerationInput {
        const std::string seedPath;
        const std::vector<std::string> imagePaths;
        const int numImages;
    };

    struct FeatureGenerationOutput {
        ssrlcv::ptr::value<ssrlcv::Unity<ssrlcv::Feature<ssrlcv::SIFT_Descriptor>>> seedFeatures;
        std::vector<ssrlcv::ptr::value<ssrlcv::Unity<ssrlcv::Feature<ssrlcv::SIFT_Descriptor>>>> allFeatures;
        std::vector<ssrlcv::ptr::value<ssrlcv::Image>> images;
    };

    void doFeatureGeneration(ssrlcv::FeatureGenerationInput *in, ssrlcv::FeatureGenerationOutput *out);


    //
    // POSE ESTIMATION
    //

    struct PoseEstimationInput {
        ssrlcv::ptr::value<ssrlcv::Unity<ssrlcv::Feature<ssrlcv::SIFT_Descriptor>>> seedFeatures;
        const std::vector<ssrlcv::ptr::value<ssrlcv::Unity<ssrlcv::Feature<ssrlcv::SIFT_Descriptor>>>> allFeatures;
        const std::vector<ssrlcv::ptr::value<ssrlcv::Image>> images;
    };

    
    void doPoseEstimation(ssrlcv::PoseEstimationInput *in);

    //
    // FEATURE MATCHING
    //

    struct FeatureMatchingInput {
        ssrlcv::ptr::value<ssrlcv::Unity<ssrlcv::Feature<ssrlcv::SIFT_Descriptor>>> seedFeatures;
        const std::vector<ssrlcv::ptr::value<ssrlcv::Unity<ssrlcv::Feature<ssrlcv::SIFT_Descriptor>>>> allFeatures;
        const std::vector<ssrlcv::ptr::value<ssrlcv::Image>> images;
        float epsilon; // pixel buffer around 2D epipolar line
        float delta; // kilometer buffer above and below line segment in 3D space
    };

    struct FeatureMatchingOutput {
        ssrlcv::MatchSet matchSet;
    };

    void doFeatureMatching(ssrlcv::FeatureMatchingInput *in, ssrlcv::FeatureMatchingOutput *out);

    //
    // TRIANGULATION
    //

    struct TriangulationInput {
        ssrlcv::MatchSet matchSet;
        const std::vector<ssrlcv::ptr::value<ssrlcv::Image>> images;
    };

    struct TriangulationOutput {
        ssrlcv::ptr::value<ssrlcv::Unity<float3>> points;
    };

    void doTriangulation(ssrlcv::TriangulationInput *in, ssrlcv::TriangulationOutput *out);

    //
    // FILTERING
    //

    struct FilteringInput {
        ssrlcv::MatchSet matchSet;
        const std::vector<ssrlcv::ptr::value<ssrlcv::Image>> images;
    };

    struct FilteringOutput {
        ssrlcv::ptr::value<ssrlcv::Unity<float3>> points;
    };

    void doFiltering(ssrlcv::FilteringInput *in, ssrlcv::FilteringOutput *out);

    //
    // BUNDLE ADJUSTMENT
    //

    struct BundleAdjustInput {
        ssrlcv::MatchSet matchSet;
        const std::vector<ssrlcv::ptr::value<ssrlcv::Image>> images;
    };

    struct BundleAdjustOutput {
        ssrlcv::ptr::value<ssrlcv::Unity<float3>> points;
    };

    void doBundleAdjust(ssrlcv::BundleAdjustInput *in, ssrlcv::BundleAdjustOutput *out);
}

#endif