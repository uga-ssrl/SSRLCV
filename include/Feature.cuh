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


  /*
  TYPEDEFS FOR EXISTING FEATURES
  */
  typedef Feature<SIFT_Descriptor> SIFT_Feature;

  typedef Feature<unsigned char[3][3]> Window_3x3; 
  typedef Feature<unsigned char[9][9]> Window_9x9; 
  typedef Feature<unsigned char[15][15]> Window_15x15; 
  typedef Feature<unsigned char[25][25]> Window_25x25; 
  typedef Feature<unsigned char[35][35]> Window_35x35; 


}

#endif /* FEATURE_CUH */
