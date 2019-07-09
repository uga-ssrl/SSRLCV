#ifndef FEATURE_CUH
#define FEATURE_CUH
#include "common_includes.h"

namespace ssrlcv{
  /*
  BASE FEATURE STRUCT
  */
  template<typename D>
  struct Feature{
    float2 loc;
    D descriptor;
    __device__ __host__ Feature();
    __device__ __host__ Feature(float2 loc);
    __device__ __host__ Feature(float2 loc, D descriptor);
  };

  //included in header to prevent linkage issues
  template<typename D>
  __device__ __host__ Feature<D>::Feature(){
    this->loc = {-1.0f,-1.0f};
  }
  template<typename D>
  __device__ __host__ Feature<D>::Feature(float2 loc){
    this->loc = loc;
  }
  template<typename D>
  __device__ __host__ Feature<D>::Feature(float2 loc, D descriptor){
    this->loc = loc;
    this->descriptor = descriptor;
  }

  /*
  DECLARATIONS OF DESCRIPTORS TO USE WITH FEATURE
  */
  //TODO add KAZE, SURF, ORB, etc
  struct SIFT_Descriptor{
    float theta;//in radians
    unsigned char values[128];//ordered left to right, top to bottom, 0->360 degrees
    __device__ __host__ SIFT_Descriptor();
    __device__ __host__ SIFT_Descriptor(float theta);
    __device__ __host__ SIFT_Descriptor(float theta, unsigned char values[128]);
  };
}

#endif /* FEATURE_CUH */
