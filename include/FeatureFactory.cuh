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
      struct SSKeyPoint{
        int octave;
        int blur;
        float2 loc;
        float sigma;
      };

      /**
      * \brief this represents an iterative convolutional sample of a ScaleSpace
      * \todo implement
      */
      struct Octave{
        struct Blur{
          unsigned int colorDepth;
          uint2 size;
          float sigma;
          Unity<float>* pixels;/**\brief vector of Unity structs holding pixel values*/
          Blur();
          Blur(float sigma, int2 kernelSize, Unity<unsigned char>* pixels, uint2 size, unsigned int colorDepth, float pixelWidth);
          ~Blur();
        };
        unsigned int numBlurs;
        Blur** blurs;/**\brief array of blur pointers*/
        Octave();
        //may want to remove kernelSize as it is static in anatomy
        Octave(unsigned int numBlurs, int2 kernelSize, float* sigmas, Unity<unsigned char>* pixels, uint2 depth, unsigned int colorDepth, float pixelWidth);
        ~Octave();

      };

      uint2 depth;//octave,blur
      Octave** octaves;

      ScaleSpace();
      ScaleSpace(Image* image, int startingOctave, uint2 scaleSpaceDim, float initialSigma, float2 sigmaMultiplier, int2 kernelSize);
      ~ScaleSpace();
    };
    typedef ScaleSpace DOG;

    /**
    * \brief Empty Constructor
    *
    */
    FeatureFactory();
    ~FeatureFactory();  

    /**
    * \brief this method generates a difference of gaussians from an acceptable scaleSpace
    */
    DOG* generateDOG(ScaleSpace* scaleSpace);
    Unity<ScaleSpace::SSKeyPoint>* findExtrema(DOG* dog);
    /**
    * \brief this method finds local subpixel extrema in a difference of gaussian scale space
    */
    void refineSubPixel(DOG* dog, Unity<ScaleSpace::SSKeyPoint>* extrema);
    /**
    * \brief this method filters out subpixel keypoints that have an intensity lower than a noise threshold
    */
    void removeNoise(DOG* dog, Unity<ScaleSpace::SSKeyPoint>* extrema, float noiseThreshold);
    /**
    * \brief this method filters out subpixel keypoints that are considered edges using the harris corner detector
    */
    void removeEdges(DOG* dog, Unity<ScaleSpace::SSKeyPoint>* extrema);
    

    /**
    * \brief this method finds keypoints from within a scale space at a pixel or subpixel level
    */
    Unity<float3>* findKeyPoints(Image* image, int startingOctave, uint2 scaleSpaceDim, float initialSigma, float2 sigmaMultiplier, int2 kernelSize,float noiseThreshold, bool subPixel = false);

  };

  /*
  CUDA variables, methods and kernels
  */

  extern __constant__ float pi;

  __device__ __host__ __forceinline__ bool subpixelRefiner(float3& keyPoint, const float3& derivatives, float hessian[3][3]);

  __global__ void subtractImages(unsigned int numPixels, float* pixelsUpper, float* pixelsLower, float* pixelsOut);


  __global__ void convertSSKPToLKP(unsigned int numKeyPoints, float3* localizedKeyPoints, FeatureFactory::ScaleSpace::SSKeyPoint* scaleSpaceKP);


}


#endif /* FEATUREFACTORY_CUH */
