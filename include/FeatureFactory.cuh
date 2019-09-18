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

      /**
      * \brief this represents an iterative convolutional sample of a ScaleSpace
      * \todo implement
      */
      struct Octave{
        struct Blur{
          unsigned int colorDepth;
          uint2 size;
          float sigma;
          Unity<unsigned char>* pixels;/**\brief vector of Unity structs holding pixel values*/
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

      uint2 depth;
      Octave** octaves;
      int parentOctave;

      ScaleSpace();
      ScaleSpace(Image* image, int startingOctave, uint2 scaleSpaceDim, float initialSigma, float2 sigmaMultiplier, int2 kernelSize);
      ~ScaleSpace();

    };
    /**
    * \brief Empty Constructor
    *
    */
    FeatureFactory();
    ~FeatureFactory();  

    /**
    * \brief this method generates a difference of gaussians from an acceptable scaleSpace
    */
    ScaleSpace* generateDOG(ScaleSpace* scaleSpace);
    /**
    * \brief this method finds local extrema in a difference of gaussian scale space
    */
    Unity<int4>* findExtrema(ScaleSpace* dog);
    /**
    * \brief this method finds local subpixel extrema in a difference of gaussian scale space
    */
    Unity<float4>* findSubPixelExtrema(ScaleSpace* dog);
    /**
    * \brief this method filters out keypoints that have an intensity lower than a noise threshold
    */
    Unity<int4>* filterNoise(int4* extrema, ScaleSpace* dog, float threshold);
    /**
    * \brief this method filters out subpixel keypoints that have an intensity lower than a noise threshold
    */
    Unity<float4>* filterNoise(float4* extrema, ScaleSpace* dog, float noiseThreshold);
    /**
    * \brief this method filters out keypoints that are considered edges using the harris corner detector
    */
    Unity<int4>* filterEdges(int4* extrema, ScaleSpace* dog);
    /**
    * \brief this method filters out subpixel keypoints that are considered edges using the harris corner detector
    */
    Unity<float4>* filterEdges(float4* extrema, ScaleSpace* dog);

    /**
    * \brief this method finds keypoints from within a scale space at a pixel or subpixel level
    */
    float3* findKeyPoints(ScaleSpace* scaleSpace, float noiseThreshold, bool subpixel = false);

  };

  /*
  CUDA variables, methods and kernels
  */

  extern __constant__ float pi;


}


#endif /* FEATUREFACTORY_CUH */
