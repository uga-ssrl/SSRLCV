#ifndef SFM_CUH
#define SFM_CUH

#include "common_includes.hpp"
#include "SIFT_FeatureFactory.cuh"
#include "MatchFactory.cuh"

namespace ssrlcv {
    struct FeatureGenerationInput {
        const std::string seedPath;
        const std::vector<std::string> imagePaths;
        const int numImages;

        FeatureGenerationInput(
            const std::string seedPath,
            const std::vector<std::string> imagePaths,
            const int numImages):
                seedPath(seedPath),
                imagePaths(imagePaths),
                numImages(numImages)
            {}
    };

    struct FeatureGenerationOutput {
        std::shared_ptr<ssrlcv::Unity<ssrlcv::Feature<ssrlcv::SIFT_Descriptor>>> seedFeatures;
        std::vector<std::shared_ptr<ssrlcv::Unity<ssrlcv::Feature<ssrlcv::SIFT_Descriptor>>>> allFeatures;
        std::vector<std::shared_ptr<ssrlcv::Image>> images;
    };

    void doFeatureGeneration(ssrlcv::FeatureGenerationInput *in, ssrlcv::FeatureGenerationOutput *out);

    struct FeatureMatchingInput {
        std::shared_ptr<ssrlcv::Unity<ssrlcv::Feature<ssrlcv::SIFT_Descriptor>>> seedFeatures;
        const std::vector<std::shared_ptr<ssrlcv::Unity<ssrlcv::Feature<ssrlcv::SIFT_Descriptor>>>> allFeatures;
        const std::vector<std::shared_ptr<ssrlcv::Image>> images;

        FeatureMatchingInput(
            std::shared_ptr<ssrlcv::Unity<ssrlcv::Feature<ssrlcv::SIFT_Descriptor>>> seedFeatures,
            const std::vector<std::shared_ptr<ssrlcv::Unity<ssrlcv::Feature<ssrlcv::SIFT_Descriptor>>>> allFeatures,
            const std::vector<std::shared_ptr<ssrlcv::Image>> images):
                seedFeatures(seedFeatures),
                allFeatures(allFeatures),
                images(images)
            {}
    };

    struct FeatureMatchingOutput {
        ssrlcv::MatchSet matchSet;
    };

    void doFeatureMatching(ssrlcv::FeatureMatchingInput *in, ssrlcv::FeatureMatchingOutput *out);
}

#endif