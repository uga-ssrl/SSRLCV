/**
* \file Image.cuh
* \brief Image related structs, methods and CUDA kernels
* \todo Convert all methods and struct to allow any arithmetic type for pixel values.
*/
#pragma once
#ifndef IMAGE_CUH
#define IMAGE_CUH

#include "common_includes.hpp"
#include "io_util.hpp"
#include "Feature.cuh"
#include "fix_thrust_warning.h"
#include <thrust/sort.h>
#include <thrust/unique.h>
#include <thrust/device_vector.h>
#include <thrust/scan.h>
#include <thrust/device_ptr.h>
#include <thrust/copy.h>
#include <sys/stat.h>

namespace ssrlcv{
  /**
  * \defgroup image_manipulation
  * \{
  */

  /**
  * \brief This class holds the information necessary to describe an image.
  * \details This class hold camera paramters for an image as well as the pixels
  * themselves. All utility methods assume that pixels are flattened row-wise.
  * \todo template this to hold any arithmetic type in pixels
  * \todo consolidate Camera and Image variable (current has redundant information)
  */
  class Image{

  public:
    /**
    * \brief This struct is meant to house image and camera parameters.
    */
    struct Camera{
      float3 cam_pos;///<\brief position of camera
      float3 cam_rot;///<\brief the x, y, z rotations of the camera
      float2 fov;///<\brief feild of fiew of camera
      float foc;///<\brief focal length of camera
      float2 dpix;///<\brief real world size of each pixel
      long long int timeStamp;///<\brief seconds since Jan 01, 1070
      /**
       * \brief identical to the image size param, but used in GPU camera modification method
       */
      float3 ecef_offset; //<\position offset (add this to convert to ECEF)
      uint2 size;
      __device__ __host__ Camera();
      __device__ __host__ Camera(uint2 size);
      __device__ __host__ Camera(uint2 size, float3 cam_pos, float3 cam_rot);
    };

    /**
     * \brief this struct is meant to house pushbrum camera paramters
     */
    struct PushbroomCamera{
      float3 start_pos; ///<\brief the starting coordinate of a pushbroom scan
      float3 end_pos;  ///<\brief the ending coordinate or a pushbroom scan
      float2 projection_center; ///<\brief lattitude and longitude coords representing
      float axis_radius; // <\brief the radius from the center of the body being orbited to the projection center
      float roll; //<\brief the craft roll in the plane purpendicular to the flight vector
      float altitude;  ///<\brief Assumes circular orbit, the altitude
      float foc; ///<\brief focal length of camera
      float fov; ///<\brief feild of fiew of camera
      float gsd; ///<\brief the ideal GSD
      float2 dpix; ///<\brief real world size of each pixel
      /**
       * \brief identical to the image size param, but used in GPU camera modification method
       */
      uint2 size;
    };

    std::string filePath;///< \brief path to image file
    int id;///<\brief parent image id
    uint2 size;///<\brief size of image
    unsigned int colorDepth;///<\brief colorDepth of image
    Camera camera; ///<\brief Camera struct holding all camera parameters
    PushbroomCamera pushbroom; ///<\brief Camera struct holding all camera parameters
    bool isPushbroom; ///<\brief variable is true if the image was generated with a pushbroom camera
    ssrlcv::ptr::value<ssrlcv::Unity<unsigned char>> pixels;///<\brief pixels of image flattened row-wise

    // =============================================================================================================
    //
    // Constructors and Destructors
    //
    // =============================================================================================================

    Image();///< \brief default constructor
    /**
    * \brief Constructor utilizing already read images.
    * \details This constructor exists to allow users
    * to read in images in their own way and instantiate an Image.
    * It is always assumed in utility methods and other methods within
    * the ssrlcv namespace that the pixels are flattened row-wise.
    * \param size - {width, height} of image in pixels
    * \param colorDepth - number of values per pixel
    * \param pixels - pixel values flattened row-wise within a Unity structure
    * \see Unity
    */

    Image(uint2 size, unsigned int colorDepth, ssrlcv::ptr::value<ssrlcv::Unity<unsigned char>> pixels);
    /**
    * \brief Primary constructor utilizing ssrlcv image io.
    * \details This constructor uses a file path to a jpg/jpeg, png or tif/tiff
    * to fill in the pixel array, or a checkpointed .cpimg file.
    * \param filePath - path to image
    * \param id - id of image for referencing with multiple images (optional, defaults to -1)
    */
    Image(std::string filePath, int id = -1);

    /**
    * \brief Constructor utilizing ssrlcv image io and allowing immediate color conversion.
    * \details This constructor uses a file path to a jpg/jpeg, png or tif/tiff
    * to fill in the pixel array and then allows the user to specify the target colorDepth
    * after reading in the image.
    * \param filePath - path to image
    * \param convertColorDepthTo - target colorDepth of image after reading
    * \param id - id of image for referencing with multiple images (optional, defaults to -1)
    * \see Image::convertColorDepthTo
    * \warning it is only recommened to convert color down (rgb -> grayscale) and not the other way
    */
    Image(std::string filePath, unsigned int convertColorDepthTo, int id = -1);

    void checkpoint(std::string dirPath = "./");

    // =============================================================================================================
    //
    // Image Host Methods
    //
    // =============================================================================================================

    /**
    * \brief Convert this->pixels to a specified colorDepth.
    * \details This method is used to change the number of values
    * associated with one pixel. This currently only support "to grayscale" and "to rgb" conversion.
    * \see convertToBW
    * \see convertToRGB
    * \warning it is only recommened to convert color down (rgb -> grayscale) and not the other way
    */
    void convertColorDepthTo(unsigned int colorDepth);
    /**
    * \brief Generate pixel gradients from this->pixels.
    * \details This method calls generatePixelGradients() to generate int2s
    * that signify {x,y} gradients.
    * \returns ssrlcv::ptr::value<ssrlcv::Unity<int2>> gradients in the form of {x,y}
    * \see generatePixelGradients
    * \see Unity
    */
    ssrlcv::ptr::value<ssrlcv::Unity<int2>> getPixelGradients();
    /**
    * \brief Scale an image by factors of 2.
    * \details This method will scale an image by a set factor of 2. Scaling will occur
    * abs(scalingFactor) times with sign of scalingFactor determining upsampling (0 >) or downsampling (< 0).
    * \param scalingFactor - abs(scalingFactor) = scaling degree, > 0 = downsample, < 0 = upsample
    * \see upsample
    * \see downsample
    */
    void alterSize(int scalingFactor);

    /**
    * retuns the camera paramters as a float vector where all data types are cast to floats
    * removes the unix time stamp
    * @returns array of floats representing the camera parameters in the order X pos, Y pos, Z pos, X rot, Y rot, Z rot, fov X, fov Y, foc, dpix x, dpix y
    */
    ssrlcv::ptr::value<ssrlcv::Unity<float>> getFloatVector();

    /**
    * retuns the camera paramters as a float vector of length len, where all data types are cast to floats
    * removes the unix time stamp
    * @param len the desired lentgth of the vector, has the effect of extracting less parameters
    * @returns array of floats representing the camera parameters in the order X pos, Y pos, Z pos, X rot, Y rot, Z rot, fov X, fov Y, foc, dpix x, dpix y
    */
    ssrlcv::ptr::value<ssrlcv::Unity<float>> getFloatVector(int len);

    /**
    * updates the camera parameters from a float vector representing camera parameters
    * if there are less than 11 params the camera will still be updated, retaining values for params not included
    * @param array of floats which should update the current parameters in the order X pos, Y pos, Z pos, X rot, Y rot, Z rot, fov X, fov Y, foc, dpix x, dpix y
    */
    void setFloatVector(ssrlcv::ptr::value<ssrlcv::Unity<float>> params);

    /**
    * calculates a vector represnting the difference between the input extinsic camera parameters and the local extinsic camera paramters
    * caclulated for the cam_pos and cam_rot vectors as local camera - input camera
    * @param other the other camera's extrinsic parameters
    * @returns difference the difference between the extinsic camera params
    */
    ssrlcv::ptr::value<ssrlcv::Unity<float>> getExtrinsicDifference(Camera other);

    /*
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
    }*/

  };

  // =============================================================================================================
  //
  // Other Host Methods
  //
  // =============================================================================================================

  /**
  * \brief Generate a new image with a border.
  * \details This method takes in a Unity<unsigned char> pixel array and will add
  * a border to it. If the border is positive, it will return a larger image
  * with 0'd pixels added as the border. If the border is negative, it will
  * remove pixels from the image and return a smaller image.
  * \param size - size of image {width,height}
  * \param pixels - pixels flattened row-wise
  * \param border - border to apply to pixels {x,y}
  * \returns ssrlcv::ptr::value<ssrlcv::Unity<unsigned char>> of pixels with border applied flattened row-wise
  * \see Unity
  */
  ssrlcv::ptr::value<ssrlcv::Unity<unsigned char>> addBufferBorder(uint2 size, ssrlcv::ptr::value<ssrlcv::Unity<unsigned char>> pixels, int2 border);
  /**
  * \brief Generate a new image with a border.
  * \details This method takes in a Unity<unsigned char> pixel array and will add
  * a border to it. If the border is positive, it will return a larger image
  * with 0'd pixels added as the border. If the border is negative, it will
  * remove pixels from the image and return a smaller image.
  * \param size - size of image {width,height}
  * \param pixels - pixels flattened row-wise
  * \param border - border to apply to pixels {x,y}
  * \returns ssrlcv::ptr::value<ssrlcv::Unity<unsigned char>> of pixels with border applied flattened row-wise
  * \see Unity
  */
  ssrlcv::ptr::value<ssrlcv::Unity<float>> addBufferBorder(uint2 size, ssrlcv::ptr::value<ssrlcv::Unity<float>> pixels, int2 border);
  /**
  * \brief Convert ssrlcv::ptr::value<ssrlcv::Unity<float>> to ssrlcv::ptr::value<ssrlcv::Unity<unsigned char>>
  * \details This method will determine min and max pixel values
  * and use those to convert the float set to unsigned char values
  * between 0-255, where 0 is min and 255 is max.
  * \param pixels - pixels in float representation
  * \returns copied pixels scaled between 0-255 in unsigned char representation
  * \see Unity
  * \see convertToCharImage
  */
  ssrlcv::ptr::value<ssrlcv::Unity<unsigned char>> convertImageToChar(ssrlcv::ptr::value<ssrlcv::Unity<float>> pixels);
  /**
  * \brief Convert ssrlcv::ptr::value<ssrlcv::Unity<unsigned char>> to ssrlcv::ptr::value<ssrlcv::Unity<float>>
  * \details This method will simply convert unsigned char values to
  * floats without changing the information at all.
  * \param pixels - pixels in unsigned char representation
  * \returns copied pixels in float representation
  * \see Unity
  * \see convertToFltImage
  */
  ssrlcv::ptr::value<ssrlcv::Unity<float>> convertImageToFlt(ssrlcv::ptr::value<ssrlcv::Unity<unsigned char>> pixels);
  /**
  * \brief Normalize float values in Unity from 0-1.
  * \details This method will determine min and max for the values
  * in the ssrlcv::ptr::value<ssrlcv::Unity<float>> and then normalize between 0 and 1.
  * \param pixels - float values
  * \see Unity
  * \see normalize(unsigned long, float*, float2)
  * \todo add option to normalize between two numbers
  */
  void normalizeImage(ssrlcv::ptr::value<ssrlcv::Unity<float>> pixels);
  /**
  * \brief Normalize float values in Unity from 0-1.
  * \details This method will use the provided min and max for the values
  * in the ssrlcv::ptr::value<ssrlcv::Unity<float>> and then normalize between 0 and 1.
  * \param pixels - float values
  * \param minMax - minimum and maximum float values {min,max}
  * \see Unity
  * \see normalize(unsigned long, float*, float2)
  * \todo add option to normalize between two numbers
  * \warning if the minMax values here are incorrect then normalization will
  * be incorrect
  */
  void normalizeImage(ssrlcv::ptr::value<ssrlcv::Unity<float>> pixels, float2 minMax);
  /**
  * \brief Convert pixel values to grayscale.
  * \details This method will take pixels of a higher colorDepth and
  * convert them to 1, making each pixel value just one unsigned char.
  * \param pixels - The pixel values flattened row-wise to be converted.
  * \param colorDepth - The original colorDepth of the pixels.
  * \see Unity
  * \see generateBW
  */
  void convertToBW(ssrlcv::ptr::value<ssrlcv::Unity<unsigned char>> pixels, unsigned int colorDepth);
  /**
  * \brief Convert pixel values to RGB.
  * \details This method will take pixels and
  * convert their colorDepth to 3, making each pixel value three unsigned char.
  * \param pixels - The pixel values flattened row-wise to be converted.
  * \param colorDepth - The original colorDepth of the pixels.
  * \see Unity
  * \see generateRGB
  * \note going from a colorDepth of 1 or 2 is permitted but will not be perfect.
  * \todo improve colorDepth upsample procedur
  */
  void convertToRGB(ssrlcv::ptr::value<ssrlcv::Unity<unsigned char>> pixels, unsigned int colorDepth);

  /**
  * \brief Generate 3x3 fundamental matrix from 2 camera matrices.
  * \details This method will generate a fundamental matrix
  * for epipolar line calculations utilizing camera matrices
  * from 2 images.
  * \param cam0 - camera matrix for first camera
  * \param cam1 - camera matrix for second camera
  * \param F - passed-by-reference matrix to fill in
  * \todo implement!!!!!
  */
  void calcFundamentalMatrix_2View(float cam0[3][3], float cam1[3][3], float (&F)[3][3]);
  /**
  * \brief Generate float3[3] fundamental matrix from 2 images.
  * \details This method takes in two images and uses their camera variables to
  * calculate a fundamental matrix for epipolar line calculations.
  * \param query - primary image in this calculation
  * \param target - image that is being related to primary image
  * \param F - passed-by-reference matrix to fill in
  * \see Image
  * \see Image::Camera
  */
  void calcFundamentalMatrix_2View(Image* query, Image* target, float3 (&F)[3]);
  /**
  * \brief
  * \details
  * \todo fill in doxy for this method
  */
  void get_cam_params2view(Image* cam1, Image* cam2, std::string infile);

  /**
  * \brief Generate gradients for an unsigned char image.
  * \details This method generates int2 gradients {x,y} for every pixel with borders
  * being symmetrized with an offset inward. The symmetrization is based on finite
  * difference and gradient approximation methods for images.
  * \param imageSize - {width,height} of image
  * \param pixels - pixels of image flattened row-wise
  * \returns ssrlcv::ptr::value<ssrlcv::Unity<int2>> that has gradients {x,y} stored in same order as pixel
  * \see Unity
  * \see calculatePixelGradients(uint2,unsigned char*,int2*)
  * \note No need to pass in colorDepth as that can be determined by
  * looking at numElements of pixels and size of image.
  */
  ssrlcv::ptr::value<ssrlcv::Unity<int2>> generatePixelGradients(uint2 imageSize, ssrlcv::ptr::value<ssrlcv::Unity<unsigned char>> pixels);
  /**
  * \brief Generate gradients for float image.
  * \details This method generates float2 gradients {x,y} for every pixel with borders
  * being symmetrized with an offset inward. The symmetrization is based on finite
  * difference and gradient approximation methods for images.
  * \param imageSize - {width,height} of image
  * \param pixels - pixels of image flattened row-wise.
  * \returns ssrlcv::ptr::value<ssrlcv::Unity<float2>> that has gradients {x,y} stored in same order as pixel
  * \see Unity
  * \see calculatePixelGradients(uint2,float*,int2*)
  * \note No need to pass in colorDepth as that can be determined by
  * looking at numElements of pixels and size of image.
  */
  ssrlcv::ptr::value<ssrlcv::Unity<float2>> generatePixelGradients(uint2 imageSize, ssrlcv::ptr::value<ssrlcv::Unity<float>> pixels);

  /**
  * \brief Ensure that an unsigned char image can be binned to a certain depth.
  * \details This method is used to ensure that the an image can be
  * binned to a certain depth. Due to odd valued side lengths of original image
  * or a later binning stage, this is necessary to avoid referencing issues and memory
  * errors associated with floating point dimensions of images. The basic concept is
  * to just add a border to allow for later binning.
  * \param size - passed-by-reference {width,height} of image to be changed
  * \param pixels - pixels of image flattened row-wise
  * \param plannedDepth - the number of times the image will be binned
  * \see Unity
  * \see addBufferBorder(uint2,ssrlcv::ptr::value<ssrlcv::Unity<unsigned char>>,int2)
  * \note No need to pass in colorDepth as that can be determined by
  * looking at numElements of pixels and size of image.
  */
  void makeBinnable(uint2 &size, ssrlcv::ptr::value<ssrlcv::Unity<unsigned char>> pixels, int plannedDepth);
  /**
  * \brief Ensure that a float image can be binned to a certain depth.
  * \details This method is used to ensure that the an image can be
  * binned to a certain depth. Due to odd valued side lengths of original image
  * or a later binning stage, this is necessary to avoid referencing issues and memory
  * errors associated with floating point dimensions of images. The basic concept is
  * to just add a border to allow for later binning.
  * \param size - passed-by-reference {width,height} of image to be changed
  * \param pixels - pixels of image flattened row-wise
  * \param plannedDepth - the number of times the image will be binned
  * \see Unity
  * \see addBufferBorder(uint2,ssrlcv::ptr::value<ssrlcv::Unity<float>>,int2)
  * \note No need to pass in colorDepth as that can be determined by
  * looking at numElements of pixels and size of image.
  */
  void makeBinnable(uint2 &size, ssrlcv::ptr::value<ssrlcv::Unity<float>> pixels, int plannedDepth);

  /**
  * \brief Downsample an unsigned char image by a factor of 2.
  * \details This method will generate an image from a provided image that
  * is half the width and half height of the original image.
  * \param imageSize - {width,height} of image
  * \param pixels - pixels of image flattened row-wise
  * \returns - ssrlcv::ptr::value<ssrlcv::Unity<unsigned char>> holding the binned version of the provided image
  * \see Unity
  * \see binImage(uint2,unsigned int,unsigned char*,unsigned char*)
  * \note No need to pass in colorDepth as that can be determined by
  * looking at numElements of pixels and size of image.
  */
  ssrlcv::ptr::value<ssrlcv::Unity<unsigned char>> bin(uint2 imageSize, ssrlcv::ptr::value<ssrlcv::Unity<unsigned char>> pixels);
  /**
  * \brief Downsample an float image by a factor of 2.
  * \details This method will generate an image from a provided image that
  * is half the width and half height of the original image.
  * \param imageSize - {width,height} of image
  * \param pixels - pixels of image flattened row-wise
  * \returns - ssrlcv::ptr::value<ssrlcv::Unity<float>> holding the binned version of the provided image
  * \see Unity
  * \see binImage(uint2,unsigned int,float*,float*)
  * \note No need to pass in colorDepth as that can be determined by
  * looking at numElements of pixels and size of image.
  */
  ssrlcv::ptr::value<ssrlcv::Unity<float>> bin(uint2 imageSize, ssrlcv::ptr::value<ssrlcv::Unity<float>> pixels);

  /**
  * \brief Upsample an unsigned char image by a factor of 2.
  * \details This method will generate an image from a provided image that
  * is double the width and half height of the original image.
  * \param imageSize - {width,height} of image
  * \param pixels - pixels of image flattened row-wise
  * \returns - ssrlcv::ptr::value<ssrlcv::Unity<unsigned char>> holding the upsampled version of the provided image
  * \see Unity
  * \see upsampleImage(uint2,unsigned int,unsigned char*,unsigned char*)
  * \note No need to pass in colorDepth as that can be determined by
  * looking at numElements of pixels and size of image.
  */
  ssrlcv::ptr::value<ssrlcv::Unity<unsigned char>> upsample(uint2 imageSize, ssrlcv::ptr::value<ssrlcv::Unity<unsigned char>> pixels);  /**
  * \brief Upsample an float image by a factor of 2.
  * \details This method will generate an image from a provided image that
  * is double the width and half height of the original image.
  * \param imageSize - {width,height} of image
  * \param pixels - pixels of image flattened row-wise
  * \returns - ssrlcv::ptr::value<ssrlcv::Unity<float>> holding the upsampled version of the provided image
  * \see Unity
  * \see upsampleImage(uint2,unsigned int,float*,float*)
  * \note No need to pass in colorDepth as that can be determined by
  * looking at numElements of pixels and size of image.
  */
  ssrlcv::ptr::value<ssrlcv::Unity<float>> upsample(uint2 imageSize, ssrlcv::ptr::value<ssrlcv::Unity<float>> pixels);

  /**
  * \brief Scale an unsigned char image by a specified factor.
  * \details This method will generate an image from a provided image that
  * is scaled to a specified pixel width in relation to the original image where pixel
  * width is 1.
  * \param imageSize - {width,height} of image
  * \param pixels - pixels of image flattened row-wise
  * \param outputPixelWidth - the desired size of a pixel in the returned Unity
  * \returns - ssrlcv::ptr::value<ssrlcv::Unity<unsigned char>> holding the scaled version of the provided image
  * \see Unity
  * \see bilinearInterpolation(uint2,unsigned int,unsigned char*,unsigned char*,float)
  * \note No need to pass in colorDepth as that can be determined by
  * looking at numElements of pixels and size of image.
  */
  ssrlcv::ptr::value<ssrlcv::Unity<unsigned char>> scaleImage(uint2 imageSize, ssrlcv::ptr::value<ssrlcv::Unity<unsigned char>> pixels, float outputPixelWidth);
  /**
  * \brief Scale an float image by a specified factor.
  * \details This method will generate an image from a provided image that
  * is scaled to a specified pixel width in relation to the original image where pixel
  * width is 1.
  * \param imageSize - {width,height} of image
  * \param pixels - pixels of image flattened row-wise
  * \param outputPixelWidth - the desired size of a pixel in the returned Unity
  * \returns - ssrlcv::ptr::value<ssrlcv::Unity<float>> holding the scaled version of the provided image
  * \see Unity
  * \see bilinearInterpolation(uint2,unsigned int,float*,float*,float)
  * \note No need to pass in colorDepth as that can be determined by
  * looking at numElements of pixels and size of image.
  */
  ssrlcv::ptr::value<ssrlcv::Unity<float>> scaleImage(uint2 imageSize, ssrlcv::ptr::value<ssrlcv::Unity<float>> pixels, float outputPixelWidth);


  /**
  * \brief Convolve an unsigned char image with a specified kernel.
  * \details This method will convolve and image with a provided
  * kernel and return a ssrlcv::ptr::value<ssrlcv::Unity<float>> containing the convolved image. There
  * is an optional argument to specify that the convolution should or should not be
  * symmetric. It is assumed that it should be, meaning that border pixels will
  * still be convolved by symmetrizing coordinate references based on getSymmetrizedCoord.
  * \param imageSize - {width,height} of image
  * \param pixels - pixels of image flattened row-wise
  * \param kernelSize - {width,height} of kernel (must have odd dimensions)
  * \param kernel - kernel with floating point values to be convolved on every pixel
  * \param symmetric - bool that specifies if convolution should be symmetric (optiona, defaults to true)
  * \returns convolved image of same size
  * \see convolveImage(uint2,unsigned char*,unsigned int,int2,float*,float*)
  * \see convolveImage_symmetric(uint2,unsigned char*,unsigned int,int2,float*,float*)
  * \see getSymmetrizedCoord
  * \see Unity
  */
  ssrlcv::ptr::value<ssrlcv::Unity<float>> convolve(uint2 imageSize, ssrlcv::ptr::value<ssrlcv::Unity<unsigned char>> pixels, int2 kernelSize, float* kernel, bool symmetric = true);
  /**
  * \brief Convolve an float image with a specified kernel.
  * \details This method will convolve and image with a provided
  * kernel and return a ssrlcv::ptr::value<ssrlcv::Unity<float>> containing the convolved image. There
  * is an optional argument to specify that the convolution should or should not be
  * symmetric. It is assumed that it should be, meaning that border pixels will
  * still be convolved by symmetrizing coordinate references based on getSymmetrizedCoord.
  * \param imageSize - {width,height} of image
  * \param pixels - pixels of image flattened row-wise
  * \param kernelSize - {width,height} of kernel (must have odd dimensions)
  * \param kernel - kernel with floating point values to be convolved on every pixel
  * \param symmetric - bool that specifies if convolution should be symmetric (optiona, defaults to true)
  * \returns convolved image of same size
  * \see convolveImage(uint2,float*,unsigned int,int2,float*,float*)
  * \see convolveImage_symmetric(uint2,float*,unsigned int,int2,float*,float*)
  * \see getSymmetrizedCoord
  * \see Unity
  */
  ssrlcv::ptr::value<ssrlcv::Unity<float>> convolve(uint2 imageSize, ssrlcv::ptr::value<ssrlcv::Unity<float>> pixels, int2 kernelSize, float* kernel, bool symmetric = true);

  // =============================================================================================================
  //
  // Device Kernels
  //
  // =============================================================================================================

  /* CUDA variable, method and kernel defintions */

  /**
  * \ingroup cuda_util
  * \{
  */

  /**
  * \brief symmetrizes a coordinate based
  * \todo determine if this causes the image to act spherical (circular with respect to x and y)
  */
  __device__ __host__ __forceinline__ int getSymmetrizedCoord(int i, unsigned int l);

  __device__ __host__ unsigned char bwaToBW(const uchar2 &color);
  __device__ __host__ unsigned char rgbToBW(const uchar3 &color);
  __device__ __host__ unsigned char rgbaToBW(const uchar4 &color);

  __device__ __host__ uchar3 bwToRGB(const unsigned char &color);
  __device__ __host__ uchar3 bwaToRGB(const uchar2 &color);
  __device__ __host__ uchar3 rgbaToRGB(const uchar4 &color);

  /**
   * Sets P to a projection matrix that maps a point in homogenous R3 (ECEF) to its projection
   * onto this image's camera (as a point in homogenous R2)
   * 
   * @param P pass-by-value matrix that is set to the projection matrix
   */
  __device__ __host__ void getProjectionMatrix(float4 (&P)[3], ssrlcv::Image::Camera *camera);

  /**
  * \}
  * \ingroup cuda_kernels
  * \defgroup image_manipulation_kernels
  * \{
  */

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
  /**\}*/
  /**\}*/
  /**\}*/
}
#endif /* IMAGE_CUH */
