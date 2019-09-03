/** \file io_util.h
* \brief This file contains image io methods.
*/
#ifndef IO_UTIL_H
#define IO_UTIL_H

#include "common_includes.h"
#include "Unity.cuh"
#include "tinyply.h"


namespace ssrlcv{

  /**
  * \brief get vector of imagePaths from a directory location
  */
  void getImagePaths(std::string dirPath, std::vector<std::string> &imagePaths);
  /**
  * \brief find pngs in a location
  */
  std::vector<std::string> findFiles(std::string path);

  /**
  * \brief get pixel array from a row of pointers generated from ssrlcv::readPNG utilizing png.h
  */
  unsigned char* getPixelArray(unsigned char** &row_pointers, const unsigned int &width, const unsigned int &height, const int numValues);

  /**
  * \brief get pixel values from an image file
  * \returns pixel array flattened row-wise
  */
  unsigned char* readPNG(const char* filePath, unsigned int &height, unsigned int &width, unsigned int& colorDepth);

  /**
  * \brief will write png from pixel array
  */
  void writePNG(const char* filePath, unsigned char* image, const unsigned int &colorDepth, const unsigned int &width, const unsigned int &height);


  /**
  * \brief will write ply from point array
  */
  void writePLY(const char* filePath, Unity<float3>* points, bool binary = false);

}

#endif /* IO_UTIL_H */
