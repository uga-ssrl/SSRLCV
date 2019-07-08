#ifndef FEATURE_CUH
#define FEATURE_CUH
#include "common_includes.h"

namespace ssrlcv{
  template<typename D>
  struct Feature{
    float2 loc;
    D descriptor;
    __device__ __host__ Feature();
    __device__ __host__ Feature(float2 loc);
    __device__ __host__ Feature(float2 loc, D descriptor);
  };

  struct SIFT_Descriptor{
    float theta;//in radians
    unsigned char descriptor[128];//ordered left to right, top to bottom, 0->360 degrees
    __device__ __host__ SIFT_Descriptor();
    __device__ __host__ SIFT_Descriptor(float theta);
    __device__ __host__ SIFT_Descriptor(float theta, unsigned char descriptor[128]);
  };
}

#endif /* FEATURE_CUH */
