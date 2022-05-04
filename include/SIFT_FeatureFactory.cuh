/** \file This file contains methods and kernels for the SIFT_FeatureFactory
*
*/
#pragma once
#ifndef SIFT_FEATUREFACTORY_CUH
#define SIFT_FEATUREFACTORY_CUH

#include "common_includes.hpp"
#include "FeatureFactory.cuh"

namespace ssrlcv{
  /**
  * \brief featureFactory to fill in Feature<SIFT_Descriptor>.
  * \details This class is a child of FeatureFactory and is purposed 
  * for creating and filling in a Feature<SIFT_Descriptor> set.
  * \see FeatureFactory
  * \see Feature
  * \see SIFT_Descriptor
  */
  class SIFT_FeatureFactory : public FeatureFactory{

  private:
    /**
    * \brief Method for generating features with a set of key points and gradients. 
    * \details This method is primarily used for dense Feature<SIFT_Descriptor> generation. 
    */
    ssrlcv::ptr::value<Unity<Feature<SIFT_Descriptor>>> createFeatures(uint2 imageSize, float orientationThreshold, unsigned int maxOrientations, float pixelWidth, ssrlcv::ptr::value<ssrlcv::Unity<float2>> gradients, ssrlcv::ptr::value<ssrlcv::Unity<float2>> keyPoints);

  public:
    /**
    * \brief Set contributer window width for descriptor computation.
    * \details This method sets the contributor window width for features 
    * which is just the radius of influence of neighboring pixel data. 
    */
    void setDescriptorContribWidth(float descriptorContribWidth);


    /**
    * \brief Primary constructor for SIFT_FeatureFactory.
    * \details This constructor is simply sets the contribution 
    * windows for continuous Feature<SIFT_Descriptor> generation. The 
    * member variables being set in this are inherited by the FeatureFactory class. 
    * Defaults for optional parameters come from https://www.robots.ox.ac.uk/~vgg/research/affine/det_eval_files/lowe_ijcv2004.pdf.
    * \param orientationContribWidth - The radius of influence for neighboring pixel
    * data in regards to dominant orientation computation for the float theta value 
    * in the SIFT_Desctiptor. (option, defaults to 1.5)
    * \param descriptorContribWidth - The radius of influence for neighboring pixel 
    * data in regards to HOG (histogram of oriented gradients) computation for the 
    * unsigned char values[128] in SIFT_Descriptor. (optional, defaults to 6.0)
    * \see FeatureFactory
    * \see SIFT_Descriptor 
    */
    SIFT_FeatureFactory(float orientationContribWidth = 1.5f, float descriptorContribWidth = 6.0f);


    /**
    * \brief Generate Feature<SIFT_Descriptor> set for an Image. 
    * \details This method can be used for dense or sparse Feature<SIFT_Descriptor> generation. 
    * Dense simply utilizes all pixels other than those that are too close to the border 
    * for feature generation, while sparse will utilize a FeatureFactory::ScaleSpace 
    * to find extrema in the image. 
    * \param image - the image holding pixel information
    * \param dense - bool signifying dense generation or use of FeatureFcatory::ScaleSpace
    * \param maxOrientations - Maximum allowable features to come from a key point with 
    * multiple dominant orientations. 
    * \param orientationThreshold - Threshold for considering an orientation dominant. (optional, defaults to 0.8 -> magnitude/maxMagnitude)
    * \see Image
    * \see Feature
    * \see SIFT_Descriptor
    * \see FeatureFactory::ScaleSpace
    */
    ssrlcv::ptr::value<Unity<Feature<SIFT_Descriptor>>> generateFeatures(ssrlcv::ptr::value<Image> image, bool dense, unsigned int maxOrientations, float orientationThreshold = 0.8);
  };

  __device__ __forceinline__ unsigned long getGlobalIdx_2D_1D();

  __global__ void computeThetas(const unsigned long numKeyPoints, const unsigned int imageWidth,
    const float pixelWidth, const float lambda, const float windowWidth, const float2* __restrict__ keyPointLocations,
    const float2* gradients, int* __restrict__ thetaNumbers, const unsigned int maxOrientations, const float orientationThreshold,
    float* __restrict__ thetas);

  __global__ void fillDescriptors(const unsigned long numFeatures, const unsigned int imageWidth, Feature<SIFT_Descriptor>* features, 
    const float pixelWidth, const float lambda, const float windowWidth, const float* __restrict__ thetas,
    const int* __restrict__ keyPointAddresses, const float2* __restrict__ keyPointLocations, const float2* __restrict__ gradients);


  __global__ void checkKeyPoints(unsigned int numKeyPoints, unsigned int keyPointIndex, uint2 imageSize, float pixelWidth, float lambda, FeatureFactory::ScaleSpace::SSKeyPoint* keyPoints);

  //implement
  __global__ void fillDescriptors(unsigned int numFeatures, uint2 imageSize, Feature<SIFT_Descriptor>* features,
    float pixelWidth, float lambda, FeatureFactory::ScaleSpace::SSKeyPoint* keyPoints, float2* gradients);


}

#endif /* SIFT_FEATUREFACTORY_CUH */
