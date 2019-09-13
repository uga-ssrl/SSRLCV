/** \file Image.cuh
* \brief Image related structs, methods and CUDA kernels
*/
#ifndef IMAGE_CUH
#define IMAGE_CUH

#include "common_includes.h"
#include "io_util.h"
#include "Feature.cuh"
#include "cuda_util.cuh"
#include "Unity.cuh"
#include <thrust/sort.h>
#include <thrust/unique.h>
#include <thrust/device_vector.h>
#include <thrust/scan.h>
#include <thrust/device_ptr.h>
#include <thrust/copy.h>

namespace ssrlcv{

  /**
  * \brief this class holds the information necessary to describe an image
  */
  class Image{

  public:
    /**
    * \brief this struct is meant to house image and camera parameters.
    */
    struct Camera{
      float3 cam_pos;/**\brief position of camera*/
      float3 cam_vec;/**\brief pointing vector of camera*/
      float fov;/**\brief feild of fiew of camera*/
      float foc;/**\brief focal length of camera*/
      float2 dpix;/**\brief real world size of each pixel*/
      long long int timeStamp;/**\brief seconds since Jan 01, 1070*/
      uint2 size; /**identical to the image size param, but used in GPU camera modification methods */
      __device__ __host__ Camera();
      __device__ __host__ Camera(uint2 size);
      __device__ __host__ Camera(uint2 size, float3 cam_pos, float3 camp_dir);
    };

    std::string filePath;/**\brief path to image file*/
    int id;/**\brief parent image id*/
    uint2 size;/**\brief size of image*/
    unsigned int colorDepth;/**\brief colorDepth of image*/
    Camera camera;/**\brief Camera struct holding all camera parameters*/
    Unity<unsigned char>* pixels;/**\brief pixels of image flattened row-wise*/

    Image();
    Image(std::string filePath, int id = -1);
    Image(std::string filePath, unsigned int convertColorDepthTo, int id = -1);
    ~Image();

    void convertColorDepthTo(unsigned int colorDepth);
    Unity<int2>* getPixelGradients();
    /**
    *\breif This method will either bin or upsample an image based on the scaling factor
    *\param scalingFactor if >= 1 binning will occur if <= -1 upsampling will occur
    */
    void alterSize(int scalingFactor);

    // Binary camera params [Gitlab #58]
    void bcp_in(bcpFormat data) {
      this->camera.cam_pos.x  = data.pos[0];
      this->camera.cam_pos.y  = data.pos[1];
      this->camera.cam_pos.z  = data.pos[2];

      this->camera.cam_vec.x  = data.vec[0];
      this->camera.cam_vec.y  = data.vec[1];
      this->camera.cam_vec.z  = data.vec[2];

      this->camera.fov        = data.fov;
      this->camera.foc        = data.foc;

      this->camera.dpix.x     = data.dpix[0];
      this->camera.dpix.y     = data.dpix[1];
    }

  };

  /**
  *\brief generate int2 gradients for each pixel
  */
  Unity<int2>* generatePixelGradients(uint2 imageSize, Unity<unsigned char>* pixels);
  /**
  *\brief bins an image by a factor of 2 in the x and y direction
  */
  Unity<unsigned char>* bin(uint2 imageSize, unsigned int colorDepth, Unity<unsigned char>* pixels);
  /**
  *\brief upsamples and image by a factor of 2 in the x and y directions
  */
  Unity<unsigned char>* upsample(uint2 imageSize, unsigned int colorDepth, Unity<unsigned char>* pixels);
  /**
  *\brief same as bin and upsample without constraining scaling to factor of two
  */
  Unity<unsigned char>* scaleImage(uint2 imageSize, unsigned int colorDepth, Unity<unsigned char>* pixels, float outputPixelWidth);
  Unity<unsigned char>* convolve(uint2 imageSize, Unity<unsigned char>* pixels, unsigned int colorDepth, int2 kernelSize, float* kernel);

  void convertToBW(Unity<unsigned char>* pixels, unsigned int colorDepth);
  void convertToRGB(Unity<unsigned char>* pixels, unsigned int colorDepth);

  void calcFundamentalMatrix_2View(Image* query, Image* target, float3 *F);
  void get_cam_params2view(Image* cam1, Image* cam2, std::string infile);


  /* CUDA variable, method and kernel defintions */

  __device__ __forceinline__ float atomicMinFloat (float * addr, float value);
  __device__ __forceinline__ float atomicMaxFloat (float * addr, float value);

  __device__ __forceinline__ unsigned long getGlobalIdx_2D_1D();

  __device__ __forceinline__ unsigned char bwaToBW(const uchar2 &color);
  __device__ __forceinline__ unsigned char rgbToBW(const uchar3 &color);
  __device__ __forceinline__ unsigned char rgbaToBW(const uchar4 &color);

  /**
  *\note upsampling color is not an exact science
  */
  __device__ __forceinline__ uchar3 bwToRGB(const unsigned char &color);
  __device__ __forceinline__ uchar3 bwaToRGB(const uchar2 &color);
  __device__ __forceinline__ uchar3 rgbaToRGB(const uchar4 &color);

  __global__ void generateBW(int numPixels, unsigned int colorDepth, unsigned char* colorPixels, unsigned char* pixels);
  __global__ void generateRGB(int numPixels, unsigned int colorDepth, unsigned char* colorPixels, unsigned char* pixels);


  __global__ void binImage(uint2 imageSize, unsigned int colorDepth, unsigned char* pixels, unsigned char* binnedImage);
  __global__ void upsampleImage(uint2 imageSize, unsigned int colorDepth, unsigned char* pixels, unsigned char* upsampledImage);
  __global__ void bilinearInterpolation(uint2 imageSize, unsigned int colorDepth, unsigned char* pixels, unsigned char* outputPixels, float outputPixelWidth);
  __global__ void convolveImage(uint2 imageSize, unsigned char* pixels, unsigned int colorDepth, int2 kernelSize, float* kernel, float* convolvedImage, float* min, float* max);
  __global__ void convertToCharImage(unsigned int numPixels, unsigned char* pixels, float* fltPixels, float* min, float* max);
  __global__ void applyBorder(uint2 imageSize, unsigned int* featureNumbers, unsigned int* featureAddresses, float2 border);
  __global__ void getPixelCenters(unsigned int numValidPixels, uint2 imageSize, unsigned int* pixelAddresses, float2* pixelCenters);

  __global__ void calculatePixelGradients(uint2 imageSize, unsigned char* pixels, int2* gradients);

}

#endif /* IMAGE_CUH */