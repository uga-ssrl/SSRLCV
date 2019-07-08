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

//TODO make sure to implement methods without usage of Quadtree

namespace ssrlcv{
  class FeatureFactory{

  public:
    FeatureFactory();
  };

  class SIFT_FeatureFactory : public FeatureFactory{

  private:
    int numOrientations;
    //the bool dense might need to be changed to some other metric as
    // this could be where scale space is implemented
    Unity<Feature<SIFT_Descriptor>>* generateBlankFeatures(Image* image, bool dense);
    void fillDescriptors(Image* image, Unity<Feature<SIFT_Descriptor>>* features);

  public:
    SIFT_FeatureFactory();
    SIFT_FeatureFactory(int numOrientations);
    void setNumOrientations(int numOrientations);
    Unity<Feature<SIFT_Descriptor>>* generateFeaturesDensly(Image* image);
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
  __device__ void trickleSwap(const float2 &compareWValue, float2* &arr, int index, const int &length);
  __device__ __forceinline__ int4 getOrientationContributers(const int2 &loc, const int2 &imageSize);
  __device__ __forceinline__ int floatToOrderedInt(float floatVal);
  __device__ __forceinline__ float orderedIntToFloat(int intVal);
  __device__ __forceinline__ float atomicMinFloat (float * addr, float value);
  __device__ __forceinline__ float atomicMaxFloat (float * addr, float value);
  __device__ __forceinline__ float modulus(float &x, float &y);
  __device__ __forceinline__ float2 rotateAboutPoint(int2 &loc, float &theta, float2 &origin);

  // __global__ void initFeatureArrayNoZeros(ssrlcv::Image_Descriptor query, ssrlcv::Image_Descriptor target, unsigned int totalFeatures, ssrlcv::Image_Descriptor image, ssrlcv::SIFT_Feature* features, int* numFeatureExtractor, unsigned char* pixels);
  // __global__ void initFeatureArray(ssrlcv::Image_Descriptor query, ssrlcv::Image_Descriptor target, unsigned int totalFeatures, ssrlcv::Image_Descriptor image, ssrlcv::SIFT_Feature* features, int* numFeatureExtractor);
  // __global__ void computeThetas(unsigned int totalFeatures, ssrlcv::Image_Descriptor image, int numOrientations, unsigned char* pixels, ssrlcv::SIFT_Feature* features, ssrlcv::SIFT_Descriptor* descriptors);
  // __global__ void fillDescriptorsDensly(unsigned int totalFeatures, ssrlcv::Image_Descriptor image, int numOrientations, unsigned char* pixels, ssrlcv::SIFT_Feature* features, ssrlcv::SIFT_Descriptor* descriptors);


}

#endif /* FEATUREFACTORY_CUH */
