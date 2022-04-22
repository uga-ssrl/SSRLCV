#ifndef SFM_CUH
#define SFM_CUH

#include "common_includes.hpp"
#include "SIFT_FeatureFactory.cuh"

namespace ssrlcv {
    struct FeatureGenerationArgs {
        const std::string seedPath;
        const std::vector<std::string> imagePaths;
        const int numImages;
        
        ssrlcv::Unity<ssrlcv::Feature<ssrlcv::SIFT_Descriptor>> *seedFeatures;
        std::vector<ssrlcv::Unity<ssrlcv::Feature<ssrlcv::SIFT_Descriptor>> *> allFeatures;
        std::vector<ssrlcv::Image *> images;

        FeatureGenerationArgs(
            const std::string seedPath,
            const std::vector<std::string> imagePaths,
            const int numImages):
                seedPath(seedPath),
                imagePaths(imagePaths),
                numImages(numImages)
            {}
    };
}

#endif