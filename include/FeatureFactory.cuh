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
    FeatureFactory();
  };

  class SIFT_FeatureFactory : public FeatureFactory{

  private:
    //the bool dense might need to be changed to some other metric as
    // this could be where scale space is implemented
    void fillDescriptors(Image* image, Unity<Feature<SIFT_Descriptor>>* features);

  public:
    SIFT_FeatureFactory();
    Unity<Feature<SIFT_Descriptor>>* generateFeaturesDensly(Image* image, unsigned int binDepth = 0);
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
  __device__ __forceinline__ long4 getOrientationContributers(const long2 &loc, const uint2 &imageSize);
  __device__ __forceinline__ int floatToOrderedInt(float floatVal);
  __device__ __forceinline__ float orderedIntToFloat(int intVal);
  __device__ __forceinline__ float atomicMinFloat (float * addr, float value);
  __device__ __forceinline__ float atomicMaxFloat (float * addr, float value);
  __device__ __forceinline__ float modulus(const float &x, const float &y);
  __device__ __forceinline__ float2 rotateAboutPoint(const int2 &loc, const float &theta, const float2 &origin);

  //this method will fill a feature array where the first index in the sift descriptor is actually the pixel value
  //NOTE THIS MAY BE WASTEFUL ^^^ and should think of better option here.
  //numbers will be filled and then compressed by thrust stream compaction so that addresses are available for looking at real features
  __global__ void findValidFeatures(unsigned int numNodes, unsigned int nodeDepthIndex, Quadtree<unsigned char>::Node* nodes, unsigned int* featureNumbers, unsigned int* featureAddresses);
  __global__ void fillValidFeatures(unsigned int numFeatures, Feature<SIFT_Descriptor>* features, unsigned int* featureAddresses, Quadtree<unsigned char>::Node* nodes);

  __global__ void computeThetas(unsigned long numFeatures, Feature<SIFT_Descriptor>* features, Quadtree<unsigned char>::Node* nodes, unsigned char* pixels);
  __global__ void fillDescriptorsDensly(unsigned long numFeatures, Feature<SIFT_Descriptor>* features, Quadtree<unsigned char>::Node* nodes, unsigned char* pixels);


}

#endif /* FEATUREFACTORY_CUH */
