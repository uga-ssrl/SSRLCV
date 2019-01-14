#ifndef FEATUREFACTORY_CUH
#define FEATUREFACTORY_CUH

#include "common_includes.h"
#include "cuda_util.cuh"
#include "Image.cuh"
#include "Feature.cuh"
#include <thrust/device_ptr.h>
#include <thrust/copy.h>
#include <thrust/scan.h>

#include "cuda_util.cuh"

#define SIFTBORDER 12
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

__global__ void initFeatureArrayNoZeros(Image_Descriptor query, Image_Descriptor target, unsigned int totalFeatures, Image_Descriptor image, SIFT_Feature* features, int* numFeatureExtractor, unsigned char* pixels);
__global__ void initFeatureArray(Image_Descriptor query, Image_Descriptor target, unsigned int totalFeatures, Image_Descriptor image, SIFT_Feature* features, int* numFeatureExtractor);
__global__ void computeThetas(unsigned int totalFeatures, Image_Descriptor image, int numOrientations, unsigned char* pixels, SIFT_Feature* features, SIFT_Descriptor* descriptors);
__global__ void fillDescriptorsDensly(unsigned int totalFeatures, Image_Descriptor image, int numOrientations, unsigned char* pixels, SIFT_Feature* features, SIFT_Descriptor* descriptors);

class FeatureFactory{

protected:

public:
  bool allowZeros;
  Image* image;
  FeatureFactory();
  void setImage(Image* image);
};

class SIFT_FeatureFactory : public FeatureFactory{

private:
  int numOrientations;
  void generateDescriptors(SIFT_Feature* features_device, SIFT_Descriptor* descriptors_device);//NOTE not implemented
  void generateDescriptorsDensly(SIFT_Feature* features_device, SIFT_Descriptor* descriptors_device);

public:
  SIFT_FeatureFactory();
  SIFT_FeatureFactory(bool allowZeros);
  SIFT_FeatureFactory(int numOrientations);
  SIFT_FeatureFactory(bool allowZeros, int numOrientations);
  void setZeroAllowance(bool allowZeros);
  void setNumOrientations(int numOrientations);
  void generateFeatures();//NOTE not implemented
  void generateFeaturesDensly();
};

class SURF_FeatureFactory : public FeatureFactory{

};
class KAZE_FeatureFactory : public FeatureFactory{

};

#endif /* FEATUREFACTORY_CUH */
