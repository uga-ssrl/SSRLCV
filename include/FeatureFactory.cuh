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
    * \breif creates ScaleSpace from an Image
    * \todo implement
    */
    ScaleSpace* generateScaleSpace(Image* image, int startingOctave, uint2 scaleSpaceDim, float initialSigma, float2 sigmaMultiplier, int2 kernelSize);
    float2* findKeyPoints(ScaleSpace* scaleSpace);//add other variables for filtering keypoints

  };

  /*
  CUDA variables, methods and kernels
  */

  extern __constant__ float pi;


}


#endif /* FEATUREFACTORY_CUH */
