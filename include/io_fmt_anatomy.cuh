#pragma once
#ifndef IO_FMT_ANATOMY_CUH
#define IO_FMT_ANATOMY_CUH

#include "Unity.cuh" 
#include "Feature.cuh"
#include "MatchFactory.cuh"

#include <istream>
#include <iostream>

/**
 * \brief Import methods for Anatomy of Sift 
 */ 
namespace ssrlcv { 
 namespace io { 
  namespace anatomy { 

    /**
     * \brief Imports Anatomy keypoints (result of bin/sift_cli) from an iostream
     * \return A NEW Unity of SIFT feature descriptors 
     */ 
    Unity<ssrlcv::Feature<ssrlcv::SIFT_Descriptor>> *   readFeatures      (std::istream & stream);

    /**
     * \brief Imports Anatomy matches (result of bin/match_sli) from an iostream
     * \return A NEW Unity of Match objects 
     */
    Unity<ssrlcv::Match> *                              readMatches       (std::istream & stream); 

  }
 }
}
#endif /* IO_FMT_ANATOMY_CUH */
