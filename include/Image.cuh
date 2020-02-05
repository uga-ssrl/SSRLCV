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
      /**\brief position of camera*/
      float3 cam_pos;
      /**\brief the x, y, z rotations of the camera*/
      float3 cam_rot;
      /**\brief feild of fiew of camera*/
      float2 fov;
      /**\brief focal length of camera*/
      float foc;
      /**\brief real world size of each pixel*/
      float2 dpix;
      /**\brief seconds since Jan 01, 1070*/
      long long int timeStamp;
      /**identical to the image size param, but used in GPU camera modification methods */
      uint2 size;
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
    Image(uint2 size, unsigned int colorDepth, Unity<unsigned char>* pixels);
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

      this->camera.cam_rot.x  = data.vec[0];
      this->camera.cam_rot.y  = data.vec[1];
      this->camera.cam_rot.z  = data.vec[2];

      this->camera.fov.x      = data.fov[0];
      this->camera.fov.y      = data.fov[1];
      this->camera.foc        = data.foc;

      this->camera.dpix.x     = data.dpix[0];
      this->camera.dpix.y     = data.dpix[1];
    }

  };

  Unity<unsigned char>* addBufferBorder(uint2 size, ssrlcv::Unity<unsigned char>* pixels, int2 border);
  Unity<float>* addBufferBorder(uint2 size, ssrlcv::Unity<float>* pixels, int2 border);

  Unity<unsigned char>* convertImageToChar(Unity<float>* pixels);
  Unity<float>* convertImageToFlt(Unity<unsigned char>* pixels);

  void normalizeImage(Unity<float>* pixels);
  void normalizeImage(Unity<float>* pixels, float2 minMax);

  void convertToBW(Unity<unsigned char>* pixels, unsigned int colorDepth);
  void convertToRGB(Unity<unsigned char>* pixels, unsigned int colorDepth);

  //TODO implement
  void calcFundamentalMatrix_2View(float cam0[3][3], float cam1[3][3], float (&F)[3][3]);

  void calcFundamentalMatrix_2View(Image* query, Image* target, float3 (&F)[3]);
  void get_cam_params2view(Image* cam1, Image* cam2, std::string infile);

  /**
  *\brief generate int2 gradients for each pixel with borders being symmetrized with an offset inward
  * this symmetrization is based on finite difference and gradient approx
  */
  Unity<int2>* generatePixelGradients(uint2 imageSize, Unity<unsigned char>* pixels);
  /**
  *\brief generate float2 gradients for each pixel with borders being symmetrized with an offset inward
  * this symmetrization is based on finite difference and gradient approx
  */
  Unity<float2>* generatePixelGradients(uint2 imageSize, Unity<float>* pixels);

  void makeBinnable(uint2 &size, Unity<unsigned char>* pixels, int plannedDepth);
  void makeBinnable(uint2 &size, Unity<float>* pixels, int plannedDepth);

  /**
  *\brief bins an image by a factor of 2 in the x and y direction
  */
  Unity<unsigned char>* bin(uint2 imageSize, Unity<unsigned char>* pixels);
  Unity<float>* bin(uint2 imageSize, Unity<float>* pixels);

  /**
  *\brief upsamples and image by a factor of 2 in the x and y directions
  */
  Unity<unsigned char>* upsample(uint2 imageSize, Unity<unsigned char>* pixels);
  Unity<float>* upsample(uint2 imageSize, Unity<float>* pixels);

  /**
  *\brief same as bin and upsample without constraining scaling to factor of two
  *\todo think about adding referenced imageSize for new image size
  */
  Unity<unsigned char>* scaleImage(uint2 imageSize, Unity<unsigned char>* pixels, float outputPixelWidth);
  Unity<float>* scaleImage(uint2 imageSize, Unity<float>* pixels, float outputPixelWidth);



  Unity<float>* convolve(uint2 imageSize, Unity<unsigned char>* pixels, int2 kernelSize, float* kernel, bool symmetric = true);
  Unity<float>* convolve(uint2 imageSize, Unity<float>* pixels, int2 kernelSize, float* kernel, bool symmetric = true);


  /* CUDA variable, method and kernel defintions */

  /**
  * \brief symmetrizes a coordinate based
  * \todo determine if this causes the image to act spherical (circular with respect to x and y)
  */
  __device__ __host__ __forceinline__ int getSymmetrizedCoord(int i, unsigned int l);

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

  __global__ void binImage(uint2 imageSize, unsigned int colorDepth, float* pixels, float* binnedImage);
  __global__ void upsampleImage(uint2 imageSize, unsigned int colorDepth, float* pixels, float* upsampledImage);
  __global__ void bilinearInterpolation(uint2 imageSize, unsigned int colorDepth, float* pixels, float* outputPixels, float outputPixelWidth);


  //border condition 0
  __global__ void convolveImage(uint2 imageSize, unsigned char* pixels, unsigned int colorDepth, int2 kernelSize, float* kernel, float* convolvedImage);
  __global__ void convolveImage(uint2 imageSize, float* pixels, unsigned int colorDepth, int2 kernelSize, float* kernel, float* convolvedImage);
  //border condition non0
  __global__ void convolveImage_symmetric(uint2 imageSize, unsigned char* pixels, unsigned int colorDepth, int2 kernelSize, float* kernel, float* convolvedImage);
  __global__ void convolveImage_symmetric(uint2 imageSize, float* pixels, unsigned int colorDepth, int2 kernelSize, float* kernel, float* convolvedImage);

  __global__ void convertToCharImage(unsigned int numPixels, unsigned char* pixels, float* fltPixels);
  __global__ void convertToFltImage(unsigned int numPixels, unsigned char* pixels, float* fltPixels);
  __global__ void normalize(unsigned long numPixels, float* pixels, float2 minMax);

  __global__ void applyBorder(uint2 imageSize, unsigned int* featureNumbers, unsigned int* featureAddresses, float2 border);
  __global__ void getPixelCenters(unsigned int numValidPixels, uint2 imageSize, unsigned int* pixelAddresses, float2* pixelCenters);

  __global__ void calculatePixelGradients(uint2 imageSize, float* pixels, float2* gradients);
  __global__ void calculatePixelGradients(uint2 imageSize, unsigned char* pixels, int2* gradients);

}
#endif /* IMAGE_CUH */
