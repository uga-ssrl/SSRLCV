/** \file io_util.h
* \brief This file contains image io methods.
*/
#ifndef IO_UTIL_H
#define IO_UTIL_H

#include "common_includes.h"
#include "Unity.cuh"
#include "tinyply.h"
#include "Octree.cuh"
#include <stdio.h>

namespace ssrlcv{

  /*
  ARG PARSING
  */
  extern std::map<std::string, std::string> cl_args;//filled out in io_util.cpp
  bool fileExists(std::string fileName);
  bool directoryExists(std::string dirPath);
  std::string getFileExtension(std::string path);
  void getImagePaths(std::string dirPath, std::vector<std::string> &imagePaths);
  struct arg{};
  struct img_arg : public arg{
    std::string path;
    img_arg(char* path);
  };
  struct img_dir_arg : public arg{
    std::vector<std::string> paths;
    img_dir_arg(char* path);
  };
  struct flt_arg : public arg{
    float val;
    flt_arg(char* val);
  };
  struct int_arg : public arg{
    int val;
    int_arg(char* val);
  };
  typedef std::pair<std::string, arg*> arg_pair;
  std::vector<std::string> findFiles(std::string path);//going to be deprecated

  std::map<std::string, arg*> parseArgs(int numArgs, char* args[]);


  /*
  IMAGE IO
  */

  /**
  * \brief get pixel array from a row of pointers generated from ssrlcv::readPNG utilizing png.h
  */
  unsigned char* getPixelArray(unsigned char** &row_pointers, const unsigned int &width, const unsigned int &height, const int numValues);

  /**
  * \brief get pixel values from an image file
  * \returns pixel array flattened row-wise
  */
  unsigned char* readPNG(const char* filePath, unsigned int &height, unsigned int &width, unsigned int &colorDepth);

  /**
  * \brief will write png from pixel array
  */
  void writePNG(const char* filePath, unsigned char* image, const unsigned int &colorDepth, const unsigned int &width, const unsigned int &height);


  unsigned char* readTIFF(const char* filePath, unsigned int &height, unsigned int &width, unsigned int &colorDepth);
  void writeTIFF(const char* filePath, unsigned char* image, const unsigned int &colorDepth, const unsigned int &width, const unsigned int &height);

  unsigned char* readJPEG(const char* filePath, unsigned int &height, unsigned int &width, unsigned int &colorDepth);
  void writeJPEG(const char* filePath, unsigned char* image, const unsigned int &colorDepth, const unsigned int &width, const unsigned int &height);

  /*
  FEATURE AND MATCH IO
  */
  //add match writes here

  /*
  PLY IO
  */

  //TODO make readPLY
  /**
  * \brief will write ply from point array
  */
  void writePLY(const char* filePath, Unity<float3>* points, bool binary = false);

  //
  // Binary files - Gitlab #58
  //

  /*
   * Since the Image class and Image_Descriptor struct are both CUDified, I'll leave them that way for now and make this struct here
   *
   * Until further documentation is established, this struct declaration is the official specification of the .bcp format.
   * A .bcp file will be a binary serialization of the structure below in order from top to bottom.
   *
   * This definition is coupled with the reading method in io_util.cpp and the loading method in Image.cuh
   */
  struct bcpFormat {
  	float pos[3];
  	float vec[3];
  	float fov[2];
    float foc;
    float dpix[2];
  };

  bool readImageMeta(std::string imgpath, bcpFormat & out);




}

#endif /* IO_UTIL_H */
