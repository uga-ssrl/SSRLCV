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

  void calcFundamentalMatrix_2View(Image* query, Image* target, float3 *F);
  void get_cam_params2view(Image* cam1, Image* cam2, std::string infile);


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
