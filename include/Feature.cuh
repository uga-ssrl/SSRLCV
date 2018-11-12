#ifndef FEATURE_CUH
#define FEATURE_CUH
#include "common_includes.h"

struct Feature{
  int2 loc;
  bool real;
  int parentImage;
};

struct SIFT_Feature : public Feature{
  float sigma;
  __device__ __host__ SIFT_Feature();
  __device__ __host__ SIFT_Feature(int2 loc, int parentImage);
  __device__ __host__ SIFT_Feature(int2 loc, int parentImage, bool real);
  __device__ __host__ SIFT_Feature(int2 loc, int parentImage, bool real, float sigma);
};

struct Descriptor{
  /*
  ADD COMMONALITIES BETWEEN TYPES OF DESCRIPTORS
  */
};

struct SIFT_Descriptor : public Descriptor{
  float theta;//in radians
  unsigned char descriptor[128];//ordered left to right, top to bottom, 0->360 egrees
  __device__ __host__ SIFT_Descriptor();
  __device__ __host__ SIFT_Descriptor(float theta);
  __device__ __host__ SIFT_Descriptor(float theta, unsigned char descriptor[128]);
};

struct feature_is_inbounds{
  __host__ __device__
  bool operator()(Feature f){
    return f.real;
  }
};



#endif /* FEATURE_CUH */
