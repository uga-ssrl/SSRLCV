/** 
* \file Feature.cuh
* \brief This file contains all feature/feature descriptor definitions
* \details This file is where all current and future descriptors will 
* be defined. All features need a distProtocol implemented to be 
* compatible with MatchFactory. 
*/
#ifndef FEATURE_CUH
#define FEATURE_CUH
#include <iostream>
//#include <cstdio>
#include <stdio.h>
#include <cuda.h>
#include <cfloat>

//#include "common_includes.hpp"

namespace ssrlcv{
  /**
  * \defgroup features
  * \ingroup feature_detection
  * \{
  */
  /**
  * \brief The base feature struct that can contain any descriptor type.
  * \tparam Type of the descriptor 
  * \note it is advised to implement a D.distProtocol(D) method for 
  * MatchFactory compatibility
  */
  template<typename D>
  struct Feature{
    int parent;///< parent image ID
    float2 loc;///< location on parent image
    D descriptor;///< descriptor of feature
    __device__ __host__ Feature();
    __device__ __host__ Feature(float2 loc);
    __device__ __host__ Feature(float2 loc, D descriptor);
  };
  /** 
  * \}
  */
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
  * \ingroup features
  * \defgroup feature_descriptors
  * \{
  */

  /**
  * \brief Descriptor for unsigned char[128] SIFT feature descriptor.
  * \details This descriptor includes all the information generated during 
  * SIFT feature detection.  
  * \see SIFT_FeatureFactory
  */
  struct SIFT_Descriptor{
    /**
    * \brief location in space space
    * \see FeatureFactory::ScaleSpace 
    */
    float sigma;
    float theta;///< dominant directional vector from gradient analysis in radians
    /**
    * \brief Actual SIFT descriptor values.
    * \details These values outline a flattened HOG (histogram of oriented gradients) 
    * ordered left to right, top to bottom, 0->360 degrees.
    */
    unsigned char values[128];///< actual SIFT descriptor values ordered left to right, top to bottom, 0->360 degrees
    __device__ __host__ SIFT_Descriptor();
    __device__ __host__ SIFT_Descriptor(float theta);
    __device__ __host__ SIFT_Descriptor(float theta, unsigned char values[128]);
    /**
    * \brief The distance protocol used for matching.
    * \details This distance protocol is euclidian distance squared between the value[128] arrays of 2 descriptors
    * \see MatchFactory
    */
    __device__ __host__ float distProtocol(const SIFT_Descriptor& b, const float &bestMatch = FLT_MAX);
    /**
    * \brief Prints sigma, theta and 128D vector of SIFT_Descriptor
    */
    __device__ __host__ void print();//not recommended to execute on the gpu with more than one feature
  };

  struct Window_3x3{
    unsigned char values[3][3];
    __device__ __host__ Window_3x3();
    __device__ __host__ Window_3x3(unsigned char values[3][3]);
    //distance here is sum of absolute differences
    __device__ __host__ float distProtocol(const Window_3x3& b, const float &bestMatch = FLT_MAX);
  };
  struct Window_9x9{
    unsigned char values[9][9];
    __device__ __host__ Window_9x9();
    __device__ __host__ Window_9x9(unsigned char values[9][9]);
    //distance here is sum of absolute differences
    __device__ __host__ float distProtocol(const Window_9x9& b, const float &bestMatch = FLT_MAX);
  };
  struct Window_15x15{
    unsigned char values[15][15];
    __device__ __host__ Window_15x15();
    __device__ __host__ Window_15x15(unsigned char values[15][15]);
    //distance here is sum of absolute differences
    __device__ __host__ float distProtocol(const Window_15x15& b, const float &bestMatch = FLT_MAX);
  };
  struct Window_25x25{
    unsigned char values[25][25];
    __device__ __host__ Window_25x25();
    __device__ __host__ Window_25x25(unsigned char values[25][25]);
    //distance here is sum of absolute differences
    __device__ __host__ float distProtocol(const Window_25x25& b, const float &bestMatch = FLT_MAX);
  };
  struct Window_31x31{
    unsigned char values[31][31];
    __device__ __host__ Window_31x31();
    __device__ __host__ Window_31x31(unsigned char values[31][31]);
    //distance here is sum of absolute differences
    __device__ __host__ float distProtocol(const Window_31x31& b, const float &bestMatch = FLT_MAX);
  };
  /** 
  * \} 
  */
}


#endif /* FEATURE_CUH */
