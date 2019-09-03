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
    bool dense;
    unsigned int maxOrientations;
    float orientationThreshold;
    float orientationContribWidth;
    float descriptorContribWidth;

    void findKeyPoints(ScaleSpace::Octave* octave);//needs to be implemented
    Unity<Feature<SIFT_Descriptor>>* createFeatures(uint2 imageSize, float pixelWidth, float sigma, Unity<int2>* gradients, Unity<float2>* keyPoints);

    void buildFeatures();//2

  public:

    /**
    * \brief if setting true then all pixels will be represented  as keypoints
    */
    void setDensity(bool dense);
    /**
    * \brief set maximum number Feature's a keypoint can generate
    */
    void setMaxOrientations(unsigned int maxOrientations);
    /**
    * \brief set threshold for a keypoint orientation to make a new Feature
    */
    void setOrientationThreshold(float orientationThreshold);
    /**
    * \brief set contributer window width for orientation computation
    */
    void setOrientationContribWidth(float orientationContribWidth);
    /**
    * \brief set contributer window width for descriptor computation
    */
    void setDescriptorContribWidth(float descriptorContribWidth);

    /**
    * \brief constructor for SIFT_FeatureFactory
    * \detail All parameters are optional
    * \param dense if true consider all pixels keyPoints (default = false)
    * \param maxOrientations maximum number of Feature's a keypoint can generate (default = 2)
    * \param orientationThreshold threshold in orientation histogram for consideration as a Feature  (defualt = 0.8)
    */
    SIFT_FeatureFactory(bool dense = false, unsigned int maxOrientations = 2, float orientationThreshold = 0.8);

    /**
    * \brief generate an array of Feature's with SIFT_Descriptor's from an Image
    */
    Unity<Feature<SIFT_Descriptor>>* generateFeatures(Image* image);
  };


  __device__ __forceinline__ unsigned long getGlobalIdx_2D_1D();
  __device__ __forceinline__ float getMagnitude(const int2 &vector);
  __device__ __forceinline__ float getMagnitude(const float2 &vector);
  __device__ __forceinline__ float getMagnitudeSq(const int2 &vector);
  __device__ __forceinline__ float getMagnitudeSq(const float2 &vector);
  __device__ __forceinline__ float getTheta(const int2 &vector);
  __device__ __forceinline__ float getTheta(const float2 &vector);
  __device__ __forceinline__ float getTheta(const float2 &vector, const float &offset);
  __device__ void trickleSwap(const float2 &compareWValue, float2* arr, const int &index, const int &length);
  __device__ __forceinline__ long4 getOrientationContributers(const long2 &loc, const uint2 &imageSize);
  __device__ __forceinline__ int floatToOrderedInt(float floatVal);
  __device__ __forceinline__ float orderedIntToFloat(int intVal);
  __device__ __forceinline__ float atomicMinFloat (float * addr, float value);
  __device__ __forceinline__ float atomicMaxFloat (float * addr, float value);
  __device__ __forceinline__ float modulus(const float &x, const float &y);
  __device__ __forceinline__ float2 rotateAboutPoint(const int2 &loc, const float &theta, const float2 &origin);


  __global__ void computeThetas(const unsigned long numKeyPoints, const unsigned int imageWidth, const float sigma,
    const float pixelWidth, const float lambda, const float windowWidth, const float2* __restrict__ keyPointLocations,
    const int2* gradients, int* __restrict__ thetaNumbers, const unsigned int maxOrientations, const float orientationThreshold,
    float* __restrict__ thetas);

  __global__ void fillDescriptors(const unsigned long numFeatures, const unsigned int imageWidth, Feature<SIFT_Descriptor>* features,
    const float sigma, const float pixelWidth, const float lambda, const float windowWidth, const float* __restrict__ thetas,
    const int* __restrict__ keyPointAddresses, const float2* __restrict__ keyPointLocations, const int2* __restrict__ gradients);

}

#endif /* SIFT_FEATUREFACTORY_CUH */
