#ifndef IMAGE_IO_H
#define IMAGE_IO_H

#include "common_includes.h"

void getImagePaths(std::string dirPath, std::vector<std::string> &imagePaths);
std::vector<std::string> findFiles(std::string path);

unsigned char* getPixelArray(unsigned char** &row_pointers, const int &width, const int &height, const int numValues);

unsigned char* readPNG(const char* filePath, int &height, int &width, unsigned char& colorDepth);

void writePNG(const char* filePath, const unsigned char* &image, const int &width, const int &height);

#endif /*IMAGE_IO_H*/
