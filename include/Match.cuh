/** 
* \file Match.cuh
* \brief this file contains the various data structures for matched pairs
*/
#pragma once
#ifndef MATCH_CUH
#define MATCH_CUH

#include "common_includes.hpp"
#include <thrust/device_ptr.h>
#include <thrust/copy.h>
#include <thrust/scan.h>

namespace ssrlcv {
    /**
    * \defgroup matching
    * \{
    */
  
    struct uint2_pair{
      uint2 a;
      uint2 b;
    };
  
    /**
    * \brief simple struct meant to fill out matches
    */
    struct KeyPoint{
      int parentId;
      float2 loc;
    };
  
    /**
    * \brief struct for holding reference to keypoints that make up multiview match
    */
    struct MultiMatch{
      unsigned int numKeyPoints;
      int index;
    };
  
    /**
    * \brief struct to pass around MultiMatches and KeyPoint sets
    */
    struct MatchSet{
      ssrlcv::ptr::value<Unity<KeyPoint>> keyPoints;
      ssrlcv::ptr::value<Unity<MultiMatch>> matches;
    };
  
    /**
    * \brief base Match struct pair of keypoints
    */
    struct Match{
      bool invalid;
      KeyPoint keyPoints[2];
    };
    /**
    * \brief derived Match struct with distance
    * \note distance is squared here to prevent use of sqrtf
    */
    struct DMatch: Match{
      float distance;
    };
    /**
    * \brief derived DMatch struct with descriptors
    */
    template<typename T>
    struct FeatureMatch : DMatch{
      T descriptors[2];
    };
}

#endif 