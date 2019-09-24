/** \file FeatureFactory.cuh
 * \brief This file contains the base feature class definition.
 * All feature factories should be derivative of this class
 * and should include this file.
*/

#ifndef FEATUREFACTORY_CUH
#define FEATUREFACTORY_CUH

#include "common_includes.h"
#include "cuda_util.cuh"
#include "matrix_util.cuh"
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
      struct discard{
        __device__ __host__ bool operator()(const ScaleSpace::SSKeyPoint &kp){
          return kp.discard;
        }
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
        int id;
        unsigned int numBlurs;
        Blur** blurs;/**\brief array of blur pointers*/
        float pixelWidth;
        
        Unity<SSKeyPoint>* extrema;
        int* extremaBlurIndices;



        Octave();
        //may want to remove kernelSize as it is static in anatomy
        Octave(int id, unsigned int numBlurs, int2 kernelSize, float* sigmas, Unity<unsigned char>* pixels, uint2 depth, float pixelWidth);
        
        void searchForExtrema();
        void discardExtrema();
        void refineExtremaLocation();//this is going to have to reorient extrema in scalespace
        void removeNoise(float noiseThreshold);
        void removeEdges(float edgeThreshold);
        
        ~Octave();

      };

      uint2 depth;//octave,blur
      Octave** octaves;

      ScaleSpace();
      ScaleSpace(Image* image, int startingOctave, uint2 scaleSpaceDim, float initialSigma, float2 sigmaMultiplier, int2 kernelSize);
      
      void convertToDOG();      
      void dumpData(std::string filePath);
      void findKeyPoints(float noiseThreshold, float edgeThreshold, bool subpixel = false);
      Unity<SSKeyPoint>* getAllKeyPoints(MemoryState destination = gpu);
      
      ~ScaleSpace();
    };
    typedef ScaleSpace DOG;

    /**
    * \brief Empty Constructor
    *
    */
    FeatureFactory();
    ~FeatureFactory();  

  };

  /*
  CUDA variables, methods and kernels
  */

  extern __constant__ float pi;


  __device__ __forceinline__ float atomicMinFloat (float * addr, float value);
  __device__ __forceinline__ float atomicMaxFloat (float * addr, float value);
  __device__ __forceinline__ float edgeness(const float (&hessian)[2][2]);

  __global__ void subtractImages(unsigned int numPixels, float* pixelsUpper, float* pixelsLower, float* pixelsOut);


  //implement
  __global__ void findExtrema(uint2 imageSize, float* pixelsUpper, float* pixelsMiddle, float* pixelsLower, int* extrema);
  __global__ void fillExtrema(int numKeyPoints, uint2 imageSize,float pixelWidth,int2 ssLoc, int* extremaAddresses, float* pixels, FeatureFactory::ScaleSpace::SSKeyPoint* scaleSpaceKP);
 
  __global__ void refineLocation(unsigned int numKeyPoints, uint2 imageSize, float sigmaMin, float pixelWidthRatio, float pixelWidth, float* pixelsUpper, float* pixelsMiddle, float* pixelsLower, FeatureFactory::ScaleSpace::SSKeyPoint* scaleSpaceKP);
 
  __global__ void flagNoise(unsigned int numKeyPoints, FeatureFactory::ScaleSpace::SSKeyPoint* scaleSpaceKP, float threshold);
  __global__ void flagEdges(unsigned int numKeyPoints, unsigned int startingIndex, uint2 imageSize, FeatureFactory::ScaleSpace::SSKeyPoint* scaleSpaceKP, float* pixels, float threshold);
}


#endif /* FEATUREFACTORY_CUH */
