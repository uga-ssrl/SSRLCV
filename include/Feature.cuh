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
    //dist here is euclidian distance squared
    __device__ __host__ float distProtocol(const SIFT_Descriptor& b, const float &bestMatch);
  };

  struct Window_3x3{
    unsigned char values[3][3];
    __device__ __host__ Window_3x3();
    __device__ __host__ Window_3x3(unsigned char values[3][3]);
    //distance here is sum of absolute differences
    __device__ __host__ float distProtocol(const Window_3x3& b, const float &bestMatch);
  };
  struct Window_9x9{
    unsigned char values[9][9];
    __device__ __host__ Window_9x9();
    __device__ __host__ Window_9x9(unsigned char values[9][9]);
    //distance here is sum of absolute differences
    __device__ __host__ float distProtocol(const Window_9x9& b, const float &bestMatch);
  };
  struct Window_15x15{
    unsigned char values[15][15];
    __device__ __host__ Window_15x15();
    __device__ __host__ Window_15x15(unsigned char values[15][15]);
    //distance here is sum of absolute differences
    __device__ __host__ float distProtocol(const Window_15x15& b, const float &bestMatch);
  };
  struct Window_25x25{
    unsigned char values[25][25];
    __device__ __host__ Window_25x25();
    __device__ __host__ Window_25x25(unsigned char values[25][25]);
    //distance here is sum of absolute differences
    __device__ __host__ float distProtocol(const Window_25x25& b, const float &bestMatch);
  };
  struct Window_31x31{
    unsigned char values[31][31];
    __device__ __host__ Window_31x31();
    __device__ __host__ Window_31x31(unsigned char values[31][31]);
    //distance here is sum of absolute differences
    __device__ __host__ float distProtocol(const Window_31x31& b, const float &bestMatch);
  };
}


#endif /* FEATURE_CUH */
