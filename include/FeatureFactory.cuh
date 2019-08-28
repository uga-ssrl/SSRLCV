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

  class FeatureFactory{

  public:

    //TODO implement after dense sift works again
    struct ScaleSpace{

      struct Octave{
        int binRatio;//1 == parent | 1> is upsampled by bilinear interpolation
        unsigned int numBlurs;
        float* sigmas;
        Unity<unsigned char>** blurs;
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

    FeatureFactory();

    //TODO implement
    ScaleSpace* generateScaleSpace(Image* image);//needs kernels too

  };

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

    void setOrientationContribWidth(float orientationContribWidth);
    void setDescriptorContribWidth(float descriptorContribWidth);

    SIFT_FeatureFactory(bool dense = false, unsigned int maxOrientations = 2, float orientationThreshold = 0.8);

    Unity<Feature<SIFT_Descriptor>>* generateFeatures(Image* image);
  };

  /*
  TODO implement KAZE, SURF, and other feature detectors here
  */

  /*
  CUDA variables, methods and kernels
  */
  /* CUDA variable, method and kernel defintions */

  extern __constant__ float pi;
  extern __constant__ int2 immediateNeighbors[9];

  __device__ __forceinline__ unsigned long getGlobalIdx_2D_1D();
  __device__ __forceinline__ float getMagnitude(const int2 &vector);
  __device__ __forceinline__ float getTheta(const int2 &vector);
  __device__ __forceinline__ float getTheta(const float2 &vector);
  __device__ __forceinline__ float getTheta(const float2 &vector, const float &offset);
  __device__ void trickleSwap(float2 compareWValue, float2* arr, int index, const int &length);
  __device__ __forceinline__ long4 getOrientationContributers(const long2 &loc, const uint2 &imageSize);
  __device__ __forceinline__ int floatToOrderedInt(float floatVal);
  __device__ __forceinline__ float orderedIntToFloat(int intVal);
  __device__ __forceinline__ float atomicMinFloat (float * addr, float value);
  __device__ __forceinline__ float atomicMaxFloat (float * addr, float value);
  __device__ __forceinline__ float modulus(const float &x, const float &y);
  __device__ __forceinline__ float2 rotateAboutPoint(const int2 &loc, const float &theta, const float2 &origin);


  __global__ void computeThetas(unsigned long numKeyPoints, uint2 imageSize, float sigma, float pixelWidth, int contributerWindowWidth, float2* keyPointLocations, int2* gradients,
    int* thetaNumbers, unsigned int maxOrientations, float orientationThreshold, float* thetas);

  __global__ void fillDescriptors(unsigned long numFeatures, uint2 imageSize, Feature<SIFT_Descriptor>* features, float sigma,
    float pixelWidth, float lambda, int contributerWindowWidth, float* thetas, int* keyPointAddresses, float2* keyPointLocations, int2* gradients);
}


#endif /* FEATUREFACTORY_CUH */
