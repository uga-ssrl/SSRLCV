/**
 * \file CNN.h
 * \brief Houses definitions for system calls involved in CNN usage.
 * \details This will mostly be for cloud segmentation. 
 */
#ifndef CNN_H
#define CNN_H

#include "common_includes.h"

namespace ssrlcv{
  /**
   * \brief
   * \details 
   * \param pathToFile - path to image file 
   * \returns path of hashed file??? maybe of hash
   */
  std::string segmentClouds(std::string pathToFile);
}



#endif /* CNN_H */
