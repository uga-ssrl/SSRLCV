/** \file FeatureFactory.cuh
 * \brief This file contains the base feature class definition.
 * All feature factories should be derivative of this class
 * and should include this file.
*/

#ifndef FEATUREFACTORY_CUH
#define FEATUREFACTORY_CUH

#include "common_includes.h"
#include "cuda_util.cuh"
#include "Image.cuh"
#include "Feature.cuh"
#include "Unity.cuh"
#include <thrust/device_ptr.h>
#include <thrust/copy.h>
#include <thrust/scan.h>

#include "cuda_util.cuh"

#define SIFTBORDER 12

namespace ssrlcv{

  /**
  * \brief Parent factory for generating a Feature array from an Image
  *
  * \detail Contains methods and members that could be useful
  * for any type of feature factory.
  */
  class FeatureFactory{

  public:

    /**
    * \brief this is a struct to house a set of octaves making a scale space
    * \todo implement
    */
    struct ScaleSpace{

      /**
      * \brief this represents an iterative convolutional sample of a ScaleSpace
      * \todo implement
      */
      struct Octave{
        int binRatio;/**\brief 1 == parent | 1> is upsampled by bilinear interpolation*/
        unsigned int numBlurs;
        float* sigmas;/**\brief values used to generate gaussian kernel for each blur*/
        Unity<unsigned char>** blurs;/**\brief array of Unity structs holding pixel values*/
        Octave();
        Octave(unsigned int numBlurs, float* sigmas);
        ~Octave();

      };

      unsigned int numOctaves;
      Octave* octaves;
      unsigned int parentOctave;
      Image_Descriptor parentImageDescriptor;

      ScaleSpace();
      ScaleSpace(unsigned int numOctaves, int startingOctave, unsigned int numBlurs, Image* image);
      ~ScaleSpace();

    };
    /**
    * \brief Empty Constructor
    *
    */
    FeatureFactory();

    /**
    * \breif creates ScaleSpace from an Image
    * \todo implement
    */
    ScaleSpace* generateScaleSpace(Image* image, int numOctaves, int numBlurs, float initialSigma, float sigmaMultiplier);

  };

  /**
  * \brief this class creates a Feature array with SIFT_Descriptor(s)
  * \todo move to a SIFT_FeatureFactory.cuh file
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


  /*
  CUDA variables, methods and kernels
  */

  extern __constant__ float pi;

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


#endif /* FEATUREFACTORY_CUH */
