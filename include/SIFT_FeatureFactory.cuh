/** \file This file contains methods and kernels for the SIFT_FeatureFactory
*
*/
#ifndef SIFT_FEATUREFACTORY_CUH
#define SIFT_FEATUREFACTORY_CUH

#include "common_includes.h"
#include "FeatureFactory.cuh"

namespace ssrlcv{
  /**
  * \brief this class creates a Feature array with SIFT_Descriptor's
  * \note default configurations = {dense=false, maxOrientations=2,orientationThreshold=0.8,orientationContribWidth=1.5,descriptorContribWidth=6}
  */
  class SIFT_FeatureFactory : public FeatureFactory{

  private:

    Unity<Feature<SIFT_Descriptor>>* createFeatures(uint2 imageSize, float orientationThreshold, unsigned int maxOrientations, float pixelWidth, Unity<int2>* gradients, Unity<float2>* keyPoints);

  public:
    /**
    * \brief set contributer window width for descriptor computation
    */
    void setDescriptorContribWidth(float descriptorContribWidth);

    SIFT_FeatureFactory(float orientationContribWidth = 1.5f, float descriptorContribWidth = 6.0f);


    /**
    * \brief generate an array of Feature's with SIFT_Descriptor's from an Image
    */
    Unity<Feature<SIFT_Descriptor>>* generateFeatures(Image* image, bool dense = false, unsigned int maxOrientations = 2, float orientationThreshold = 0.8);
  };

  __device__ __forceinline__ unsigned long getGlobalIdx_2D_1D();

  __global__ void computeThetas(const unsigned long numKeyPoints, const unsigned int imageWidth,
    const float pixelWidth, const float lambda, const float windowWidth, const float2* __restrict__ keyPointLocations,
    const int2* gradients, int* __restrict__ thetaNumbers, const unsigned int maxOrientations, const float orientationThreshold,
    float* __restrict__ thetas);

  __global__ void fillDescriptors(const unsigned long numFeatures, const unsigned int imageWidth, Feature<SIFT_Descriptor>* features, 
    const float pixelWidth, const float lambda, const float windowWidth, const float* __restrict__ thetas,
    const int* __restrict__ keyPointAddresses, const float2* __restrict__ keyPointLocations, const int2* __restrict__ gradients);


  __global__ void checkKeyPoints(unsigned int numKeyPoints, unsigned int keyPointIndex, uint2 imageSize, float pixelWidth, float lambda, FeatureFactory::ScaleSpace::SSKeyPoint* keyPoints);

  //implement
  __global__ void fillDescriptors(unsigned int numFeatures, unsigned int keyPointIndex, uint2 imageSize, Feature<SIFT_Descriptor>* features,
    float pixelWidth, float lambda, FeatureFactory::ScaleSpace::SSKeyPoint* keyPoints, float2* gradients);


}

#endif /* SIFT_FEATUREFACTORY_CUH */
