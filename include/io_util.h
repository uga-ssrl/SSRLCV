/** \file io_util.h
* \brief This file contains image io methods.
*/
#ifndef IO_UTIL_H
#define IO_UTIL_H

#include "common_includes.h"
#include "Unity.cuh"
#include "tinyply.h"


namespace ssrlcv{

  void getImagePaths(std::string dirPath, std::vector<std::string> &imagePaths);
  std::vector<std::string> findFiles(std::string path);

  unsigned char* getPixelArray(unsigned char** &row_pointers, const unsigned int &width, const unsigned int &height, const int numValues);

  unsigned char* readPNG(const char* filePath, unsigned int &height, unsigned int &width, unsigned int& colorDepth);

  void writePNG(const char* filePath, unsigned char* image, const unsigned int &colorDepth, const unsigned int &width, const unsigned int &height);

  void writePLY(const char* filePath, Unity<float3>* points, bool binary = false);

}

#endif /* IO_UTIL_H */
