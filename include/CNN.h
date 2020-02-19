/**
 * \file CNN.h
 * \brief Houses definitions for system calls involved in CNN usage.
 * \details This will mostly be for cloud segmentation. 
 */
#ifndef CNN_H
#define CNN_H

#include "common_includes.h"
#include "Image.cuh"
#include "io_util.h"

namespace ssrlcv{
  /**
   * \brief
   * \details 
   * \param pathToFile - path to image file 
   * \returns path of hashed file??? maybe of hash
   */
  std::string segmentClouds(std::string pathToFile);

  void combineRGBImagesToBW(std::string r_folder, std::string g_folder, std::string b_folder, std::string bw_folder);
}



#endif /* CNN_H */
