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
        std::vector<ssrlcv::ptr::value<ssrlcv::Unity<ssrlcv::Feature<ssrlcv::SIFT_Descriptor>>>> allFeatures;
        std::vector<ssrlcv::ptr::value<ssrlcv::Image>> images;

        void fromCheckpoint(std::string featureGenDir, int numImages);
        void fromPreviousStage(FeatureGenerationOutput *featureGenOutput);
    };

    struct PoseEstimationOutput {
        ssrlcv::ptr::value<ssrlcv::Unity<float>> seedDistances = nullptr;
    };

    
    void doPoseEstimation(ssrlcv::PoseEstimationInput *in, ssrlcv::PoseEstimationOutput *out);

    //
    // FEATURE MATCHING
    //

    struct FeatureMatchingInput {
        ssrlcv::ptr::value<ssrlcv::Unity<ssrlcv::Feature<ssrlcv::SIFT_Descriptor>>> seedFeatures;
        std::vector<ssrlcv::ptr::value<ssrlcv::Unity<ssrlcv::Feature<ssrlcv::SIFT_Descriptor>>>> allFeatures;
        std::vector<ssrlcv::ptr::value<ssrlcv::Image>> images;
        ssrlcv::ptr::value<ssrlcv::Unity<float>> seedDistances;
        float epsilon; // pixel buffer around 2D epipolar line
        float delta; // kilometer buffer above and below line segment in 3D space

        void fromCheckpoint(std::string featureGenDirectory, std::string poseDirectory, int numImages, float epsilon, float delta);
        void fromPreviousStage(PoseEstimationInput *poseInput, PoseEstimationOutput *poseOutput, float epsilon, float delta);
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
        std::vector<ssrlcv::ptr::value<ssrlcv::Image>> images;

        void fromCheckpoint(std::string featureGenDir, std::string featureMatchDir, int numImages);
        void fromPreviousStage(FeatureMatchingInput *featureMatchingInput, FeatureMatchingOutput *featureMatchingOutput);
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
        std::vector<ssrlcv::ptr::value<ssrlcv::Image>> images;

        void fromCheckpoint(std::string featureGenDir, std::string featureMatchDir, int numImages);
        void fromPreviousStage(TriangulationInput *triangulationInput);
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
        std::vector<ssrlcv::ptr::value<ssrlcv::Image>> images;

        void fromCheckpoint(std::string featureGenDir, std::string filteringDir, int numImages);
        void fromPreviousStage(FilteringInput *filteringInput);
    };

    struct BundleAdjustOutput {
        ssrlcv::ptr::value<ssrlcv::Unity<float3>> points;
    };

    void doBundleAdjust(ssrlcv::BundleAdjustInput *in, ssrlcv::BundleAdjustOutput *out);
}

#endif