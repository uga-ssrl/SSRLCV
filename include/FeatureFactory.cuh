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
#include <thrust/sort.h>
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
  protected:
    float orientationContribWidth;
    float descriptorContribWidth;

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
        float theta;
        bool discard;
        __host__ __device__ bool operator<(const SSKeyPoint &kp) const{
          return this->blur < kp.blur;
        }
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
          Unity<float2>* gradients;
          Blur();
          Blur(float sigma, int2 kernelSize, Unity<float>* blurable, uint2 size, float pixelWidth);
          void computeGradients();
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
        void refineExtremaLocation(float minScaleSpacePixelWidth);//this is going to have to reorient extrema in scalespace
        void removeNoise(float noiseThreshold);
        void removeEdges(float edgeThreshold);

        ~Octave();

      };

      uint2 depth;//octave,blur
      Octave** octaves;

      ScaleSpace();
      ScaleSpace(Image* image, int startingOctave, uint2 scaleSpaceDim, float initialSigma, float2 sigmaMultiplier, int2 kernelSize);
      
      /**
      * \brief will convert scalespace to a difference of gaussians
      * numBlurs-- will occur
      */
      void convertToDOG();      
      void dumpData(std::string filePath);
      
      /**
      * \brief will find all key points within a scale space
      * \param noiseThreshold intensity threshold for removing noise
      * \param edgeThreshold edgeness threshold for removing edges
      * \param subpixel if true will compute and refine keypoint location to subpixel
      */
      void findKeyPoints(float noiseThreshold, float edgeThreshold, bool subpixel = false);
      /**
      * \brief will return all key points in octaves of scalespace
      */
      Unity<SSKeyPoint>* getAllKeyPoints(MemoryState destination = gpu);

      /**
      * \brief compute orientations for key points - will generate more features based on orientations above threshold
      * \todo implement
      */
      void computeKeyPointOrientations(float orientationThreshold = 0.8f, unsigned int maxOrientations = 2, float contributerWindowWidth = 1.5f, bool keepGradients = false);
      
      ~ScaleSpace();
    };
    typedef ScaleSpace DOG;

    /**
    * \brief Constructor
    *
    */
    FeatureFactory(float orientationContribWidth = 1.5f, float descriptorContribWidth = 6.0f);
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
    ~FeatureFactory();  

  };

  /*
  CUDA variables, methods and kernels
  */
  __device__ float getMagnitude(const int2 &vector);
  __device__ float getMagnitude(const float2 &vector);
  __device__ float getMagnitudeSq(const int2 &vector);
  __device__ float getMagnitudeSq(const float2 &vector);
  __device__ float getTheta(const int2 &vector);
  __device__ float getTheta(const float2 &vector);
  __device__ float getTheta(const float2 &vector, const float &offset);
  __device__ void trickleSwap(const float2 &compareWValue, float2* arr, const int &index, const int &length);
  __device__ long4 getOrientationContributers(const long2 &loc, const uint2 &imageSize);
  __device__ int floatToOrderedInt(float floatVal);
  __device__ float orderedIntToFloat(int intVal);
  __device__ float atomicMinFloat (float * addr, float value);
  __device__ float atomicMaxFloat (float * addr, float value);
  __device__ float modulus(const float &x, const float &y);
  __device__ float2 rotateAboutPoint(const int2 &loc, const float &theta, const float2 &origin);

  extern __constant__ float pi;


  __device__ float atomicMinFloat (float * addr, float value);
  __device__ float atomicMaxFloat (float * addr, float value);
  __device__ float edgeness(const float (&hessian)[2][2]);

  __global__ void subtractImages(unsigned int numPixels, float* pixelsUpper, float* pixelsLower, float* pixelsOut);

  __global__ void findExtrema(uint2 imageSize, float* pixelsUpper, float* pixelsMiddle, float* pixelsLower, int* extrema);
  __global__ void fillExtrema(int numKeyPoints, uint2 imageSize,float pixelWidth,int2 ssLoc, int* extremaAddresses, float* pixels, FeatureFactory::ScaleSpace::SSKeyPoint* scaleSpaceKP); 
  
  __global__ void refineLocation(unsigned int numKeyPoints, uint2 imageSize, float sigmaMin, float pixelWidthRatio, float pixelWidth, unsigned int numBlurs, float** pixels, FeatureFactory::ScaleSpace::SSKeyPoint* scaleSpaceKP);
  __global__ void flagNoise(unsigned int numKeyPoints, FeatureFactory::ScaleSpace::SSKeyPoint* scaleSpaceKP, float threshold);
  __global__ void flagEdges(unsigned int numKeyPoints, unsigned int startingIndex, uint2 imageSize, FeatureFactory::ScaleSpace::SSKeyPoint* scaleSpaceKP, float* pixels, float threshold);


  __global__ void computeThetas(unsigned long numKeyPoints, unsigned int keyPointIndex, uint2 imageSize, float pixelWidth, 
  float lambda, FeatureFactory::ScaleSpace::SSKeyPoint* keyPoints, float2* gradients, 
  int* thetaNumbers, unsigned int maxOrientations, float orientationThreshold, float* thetas);

  __global__ void expandKeyPoints(unsigned int numKeyPoints, FeatureFactory::ScaleSpace::SSKeyPoint* keyPointsIn, FeatureFactory::ScaleSpace::SSKeyPoint* keyPointsOut, int* thetaAddresses, float* thetas);
}


#endif /* FEATUREFACTORY_CUH */
