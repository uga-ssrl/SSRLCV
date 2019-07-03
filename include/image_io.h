#ifndef IMAGE_IO_H
#define IMAGE_IO_H

#include "common_includes.h"

void getImagePaths(std::string dirPath, std::vector<std::string> &imagePaths);
std::vector<std::string> findFiles(std::string path);

unsigned char* getPixelArray(unsigned char** &row_pointers, const int &width, const int &height, const int numValues);

unsigned char* readPNG(const char* filePath, int &height, int &width, unsigned char& colorDepth);

void writePNG(const char* filePath, const unsigned char* &image, const int &width, const int &height);

// Meta 

typedef struct { 
  float3 position;
  float3 orientation;
} image_meta;

typedef struct { 
	float fov;		// in radians
	float focal;	// in meters
} camera_meta; 

image_meta readImageMeta(std::string image);
camera_meta readCameraMeta(std::string path);

#endif /*IMAGE_IO_H*/
