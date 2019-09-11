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
        uint2 size;
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

      ScaleSpace();
      ScaleSpace(unsigned int numOctaves, int startingOctave, unsigned int numBlurs, Image* image);
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
    ScaleSpace* generateScaleSpace(Image* image, int numOctaves, int numBlurs, float initialSigma, float sigmaMultiplier);

  };

  /*
  CUDA variables, methods and kernels
  */

  extern __constant__ float pi;


}


#endif /* FEATUREFACTORY_CUH */
