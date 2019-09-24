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
#include "io_util.h"


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
        float2 abs_loc;
        float intensity;
        float sigma;
        bool discard;
      };

      /**
      * \brief this represents an iterative convolutional sample of a ScaleSpace
      * \todo implement
      */
      struct Octave{
        struct Blur{
          uint2 size;
          float sigma;
          Unity<float>* pixels;/**\brief vector of Unity structs holding pixel values*/
          Blur();
          Blur(float sigma, int2 kernelSize, Unity<float>* blurable, uint2 size, float pixelWidth);
          ~Blur();
        };
        unsigned int numBlurs;
        Blur** blurs;/**\brief array of blur pointers*/
        float pixelWidth;
        Octave();
        //may want to remove kernelSize as it is static in anatomy
        Octave(unsigned int numBlurs, int2 kernelSize, float* sigmas, Unity<unsigned char>* pixels, uint2 depth, float pixelWidth);
        ~Octave();

      };

      uint2 depth;//octave,blur
      Octave** octaves;

      ScaleSpace();
      ScaleSpace(Image* image, int startingOctave, uint2 scaleSpaceDim, float initialSigma, float2 sigmaMultiplier, int2 kernelSize);
      void dumpData(std::string filePath);
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
    void removeEdges(DOG* dog, Unity<ScaleSpace::SSKeyPoint>* extrema, float edgeThreshold);
    

    /**
    * \brief this method finds keypoints from within a scale space at a pixel or subpixel level
    */
    Unity<float3>* findKeyPoints(Image* image, int startingOctave, uint2 scaleSpaceDim, float initialSigma, float2 sigmaMultiplier, 
      int2 kernelSize, float noiseThreshold = 0.015f, float edgeThreshold = 10.0f, bool subPixel = false);

  };

  /*
  CUDA variables, methods and kernels
  */

  extern __constant__ float pi;


  __device__ __forceinline__ float atomicMinFloat (float * addr, float value);
  __device__ __forceinline__ float atomicMaxFloat (float * addr, float value);

  __global__ void subtractImages(unsigned int numPixels, float* pixelsUpper, float* pixelsLower, float* pixelsOut);


  //implement
  __global__ void findMaxima(uint2 imageSize, float* pixelsUpper, float* pixelsMiddle, float* pixelsLower, int* maxima);
  __global__ void fillMaxima(int numKeyPoints, uint2 imageSize,float pixelWidth,int2 ssLoc, int* maximaAddresses, float* pixels, FeatureFactory::ScaleSpace::SSKeyPoint* scaleSpaceKP);
 
  __global__ void refineLocation(unsigned int numKeyPoints, uint2 imageSize, float sigmaMin, float pixelWidthRatio, float pixelWidth, float* pixelsUpper, float* pixelsMiddle, float* pixelsLower, FeatureFactory::ScaleSpace::SSKeyPoint* scaleSpaceKP);
 
  __global__ void flagNoise(uint2 imageSize, FeatureFactory::ScaleSpace::SSKeyPoint* scaleSpaceKP, float threshold);
  __global__ void flagEdges(uint2 imageSize, FeatureFactory::ScaleSpace::SSKeyPoint* scaleSpaceKP, float threshold);



  __global__ void convertSSKPToLKP(unsigned int numKeyPoints, float3* localizedKeyPoints, FeatureFactory::ScaleSpace::SSKeyPoint* scaleSpaceKP);


}


#endif /* FEATUREFACTORY_CUH */
