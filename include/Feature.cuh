/** \file Feature.cuh
* \brief this file contains all feature/feature descriptor definitions
*/
#ifndef FEATURE_CUH
#define FEATURE_CUH
#include "common_includes.h"

namespace ssrlcv{
  /**
  * \brief The base feature struct that can contain any descriptor type.
  */
  template<typename D>
  struct Feature{
    int parent;
    float2 loc;
    D descriptor;
    __device__ __host__ Feature();
    __device__ __host__ Feature(float2 loc);
    __device__ __host__ Feature(float2 loc, D descriptor);
  };

  template<typename D>
  __device__ __host__ Feature<D>::Feature(){
    this->loc = {-1.0f,-1.0f};
    this->parent = -1;
  }
  template<typename D>
  __device__ __host__ Feature<D>::Feature(float2 loc) : loc(loc){
    this->parent = -1;
  }
  template<typename D>
  __device__ __host__ Feature<D>::Feature(float2 loc, D descriptor) : loc(loc), descriptor(descriptor){
    this->parent = -1;
  }

  /**
  * \brief a descriptor for unsigned char[128] SIFT feature descriptor.
  */
  struct SIFT_Descriptor{
    float sigma;
    float theta;//in radians
    unsigned char values[128];//ordered left to right, top to bottom, 0->360 degrees
    __device__ __host__ SIFT_Descriptor();
    __device__ __host__ SIFT_Descriptor(float theta);
    __device__ __host__ SIFT_Descriptor(float theta, unsigned char values[128]);
  };


}

#endif /* FEATURE_CUH */
