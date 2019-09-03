/** \file Image.cuh
* \brief Image related structs, methods and CUDA kernels
*/
#ifndef IMAGE_CUH
#define IMAGE_CUH

#include "common_includes.h"
#include "io_util.h"
#include "Feature.cuh"
#include "cuda_util.cuh"
#include "Quadtree.cuh"
#include "Unity.cuh"

namespace ssrlcv{
  /**
  * \brief this struct is meant to house image and camera parameters.
  */
  struct Image_Descriptor{
    int id;
    uint2 size;
    float3 cam_pos;
    float3 cam_vec;
    float fov;
    float foc;
    float dpix;
    long long int timeStamp;//seconds since Jan 01, 1070
    __device__ __host__ Image_Descriptor();
    __device__ __host__ Image_Descriptor(int id, uint2 size);
    __device__ __host__ Image_Descriptor(int id, uint2 size, float3 cam_pos, float3 camp_dir);
  };

  /**
  * \brief this class holds the information necessary to describe an image
  */
  class Image{

  public:

    Image_Descriptor descriptor;
    std::string filePath;
    Quadtree<unsigned int>* quadtree;//quadtree->data = pixel ref
    unsigned int colorDepth;//consider moving to Image_Descriptor
    Unity<unsigned char>* pixels;

    Image();
    Image(std::string filePath, int id = -1);
    Image(std::string filePath, unsigned int convertColorDepthTo, int id = -1);
    ~Image();

    void generateQuadtree(unsigned int depth = 0);
    void alterSize(int binDepth);
  };


  Unity<unsigned char>* bin(uint2 imageSize, unsigned int colorDepth, Unity<unsigned char>* pixels);

  void convertToBW(Unity<unsigned char>* pixels, unsigned int colorDepth);

  Unity<unsigned int>* applyBorder(Image* image, float2 border);
  Unity<unsigned int>* applyBorder(uint2 imageSize, float2 border);
  Unity<float2>* getLocationsWithinBorder(Image* image, float2 border);
  Unity<float2>* getLocationsWithinBorder(uint2 imageSize, float2 border);

  Unity<int2>* generatePixelGradients(Image* image);
  Unity<int2>* generatePixelGradients(uint2 imageSize, Unity<unsigned char>* pixels);

  //consider returning an Image
  Unity<unsigned char>* convolve(uint2 imageSize, Unity<unsigned char>* pixels, unsigned int colorDepth, unsigned int kernelSize, float* kernel);

  void calcFundamentalMatrix_2View(Image_Descriptor query, Image_Descriptor target, float3 *F);
  void get_cam_params2view(Image_Descriptor &cam1, Image_Descriptor &cam2, std::string infile);


  /* CUDA variable, method and kernel defintions */

  __device__ __forceinline__ unsigned long getGlobalIdx_2D_1D();
  __device__ __forceinline__ unsigned char bwaToBW(const uchar2 &color);
  __device__ __forceinline__ unsigned char rgbToBW(const uchar3 &color);
  __device__ __forceinline__ unsigned char rgbaToBW(const uchar4 &color);
  __global__ void generateBW(int numPixels, unsigned int colorDepth, unsigned char* colorPixels, unsigned char* bwPixels);
  __global__ void binImage(uint2 imageSize, unsigned int colorDepth, unsigned char* pixels, unsigned char* binnedImage);
  __global__ void convolveImage(uint2 imageSize, unsigned char* pixels, unsigned int colorDepth, unsigned int kernelSize, float* kernel, unsigned char* convolvedImage);
  __global__ void applyBorder(uint2 imageSize, unsigned int* featureNumbers, unsigned int* featureAddresses, float2 border);
  __global__ void getPixelCenters(unsigned int numValidPixels, uint2 imageSize, unsigned int* pixelAddresses, float2* pixelCenters);

  __global__ void calculatePixelGradients(uint2 imageSize, unsigned char* pixels, int2* gradients);

}

#endif /* IMAGE_CUH */
