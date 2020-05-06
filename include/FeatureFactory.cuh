/** \file FeatureFactory.cuh
* \brief This file contains the base feature class definition.
* \details All feature factories should be derivative of this class
* and should include this file.
*/

#ifndef FEATUREFACTORY_CUH
#define FEATUREFACTORY_CUH

#include "common_includes.hpp"
#include "Image.cuh"
#include "Feature.cuh"
#include <thrust/device_ptr.h>
#include <thrust/copy.h>
#include <thrust/scan.h>
#include <thrust/sort.h>
#include <thrust/remove.h>
#include "io_util.hpp"


#define SIFTBORDER 12

namespace ssrlcv{
  /**
  * \defgroup feature_detection
  * \{
  */

  /**
  * \brief Parent factory for generating a Feature array from an Image
  * \details Contains methods and members that could be useful
  * for any type of feature factory.
  */
  class FeatureFactory{
  protected:
    float orientationContribWidth;
    float descriptorContribWidth;

  public:

    /**
    * \ingroup feature_detection
    * \defgroup scalespace
    * \{
    */
    /**
    * \brief A scale space holding a heirarchy of octaves and blurs.
    * \details 
    * \todo Allow for other kernels (not just gaussians) to be passed in for convolution. 
    */
    struct ScaleSpace{
    private:
      bool isDOG;
      /**
      * \brief Convert scalespace to a difference of gaussians
      * \details 
      */
      void convertToDOG();  
    public:
      /**
      * \ingroup scalespace
      */
      struct SSKeyPoint{
        int octave;
        int blur;
        float2 loc;
        float intensity;
        float sigma;
        float theta;
        bool discard;
        __host__ __device__ bool operator<(const SSKeyPoint &kp) const{
          return this->blur < kp.blur;
        }
      };
      
      struct discard{
        __device__ __host__ bool operator()(const ScaleSpace::SSKeyPoint &kp){
          return kp.discard;
        }
      };

      /**
      * \brief this represents an iterative convolutional sample of a ScaleSpace
      * \todo implement
      * \ingroup scalespace
      */
      struct Octave{
        /**
        * \brief 
        * \ingroup scalespace
        */
        struct Blur{
          uint2 size;
          float sigma;
          Unity<float>* pixels;/**\brief vector of Unity structs holding pixel values*/
          Unity<float2>* gradients;
          Blur();
          Blur(float sigma, int2 kernelSize, Unity<float>* pixels, uint2 size, float pixelWidth);
          ~Blur();
          void computeGradients();
        };


        int id;
        unsigned int numBlurs;
        Blur** blurs;/**\brief array of blur pointers*/
        float pixelWidth;
        Unity<SSKeyPoint>* extrema;
        int* extremaBlurIndices;

        Octave();
        Octave(int id, unsigned int numBlurs, int2 kernelSize, float* sigmas, Unity<float>* pixels, uint2 size, float pixelWidth, int keepPixelsAfterBlur);      
        void searchForExtrema();
        void discardExtrema();
        void refineExtremaLocation();
        void removeNoise(float noiseThreshold);
        void removeEdges(float edgeThreshold);
        void removeBorder(float2 border);
        void normalize();

        ~Octave();

      };

      uint2 depth;///< depth of ScaleSpace {numOctaves, numBlurs}
      Octave** octaves;

      ScaleSpace();
      ScaleSpace(Image* image, int startingOctave, uint2 scaleSpaceDim, float initialSigma, float2 sigmaMultiplier, int2 kernelSize, bool makeDOG = false);
      
      bool checkIfDOG();
    
      void dumpData(std::string filePath);
      
      /**
      * \brief will find all key points within a scale space
      * \param noiseThreshold intensity threshold for removing noise
      * \param edgeThreshold edgeness threshold for removing edges
      * \param subpixel if true will compute and refine keypoint location to subpixel
      */
      void findKeyPoints(float noiseThreshold, float edgeThreshold, bool subpixel = false);
      /**
      * \brief will return all key points in octaves of scalespace
      */
      Unity<SSKeyPoint>* getAllKeyPoints(MemoryState destination = gpu);

      /**
      * \brief compute orientations for key points - will generate more features based on orientations above threshold
      * \todo implement
      */
      void computeKeyPointOrientations(float orientationThreshold = 0.8f, unsigned int maxOrientations = 1, float contributerWindowWidth = 1.5f, bool keepGradients = false);
      
      ~ScaleSpace();
    };
    typedef ScaleSpace DOG;
    /**
    * \}
    */

    /**
    * \brief Constructor
    *
    */
    FeatureFactory(float orientationContribWidth = 1.5f, float descriptorContribWidth = 6.0f);
     /**
    * \brief set maximum number Feature's a keypoint can generate
    * \details 
    */
    void setMaxOrientations(unsigned int maxOrientations);
    /**
    * \brief set threshold for a keypoint orientation to make a new Feature
    * \details 
    */
    void setOrientationThreshold(float orientationThreshold);
    /**
    * \brief set contributer window width for orientation computation
    * \details 
    */
    void setOrientationContribWidth(float orientationContribWidth);

    /**
    * \brief feature generators for 3x3 pixel window
    * \details 
    */
    Unity<Feature<Window_3x3>>* generate3x3Windows(Image* image);
    /**
    * \brief feature generators for 9x9 pixel window
    * \details 
    */
    Unity<Feature<Window_9x9>>* generate9x9Windows(Image* image);
    /**
    * \brief feature generators for 15x15 pixel window
    * \details 
    */
    Unity<Feature<Window_15x15>>* generate15x15Windows(Image* image);
    /**
    * \brief feature generators for 25x25 pixel window
    * \details 
    */
    Unity<Feature<Window_25x25>>* generate25x25Windows(Image* image);
    /**
    * \brief feature generators for 31x31 pixel window
    * \details 
    */
    Unity<Feature<Window_31x31>>* generate31x31Windows(Image* image);

    ~FeatureFactory();  

  };

  /**
  * \}
  */

  /*
  CUDA variables, methods and kernels
  */
  /**
  * \ingroup cuda_util
  * \{
  */
  extern __constant__ float pi;
  __device__ __forceinline__ float getMagnitude(const int2 &vector);
  __device__ __forceinline__ float getMagnitude(const float2 &vector);
  __device__ void trickleSwap(const float2 &compareWValue, float2* arr, const int &index, const int &length);
  __device__ __forceinline__ float atomicMinFloat (float * addr, float value);
  __device__ __forceinline__ float atomicMaxFloat (float * addr, float value);
  __device__ __forceinline__ float edgeness(const float (&hessian)[2][2]);
  /**
  * \}
  */

  /**
  * \ingroup feature_detection
  * \ingroup cuda_kernels
  * \defgroup feature_detection_kernels
  * \{
  */

  __global__ void fillWindows(uint2 size, int parent, unsigned char* pixels, Feature<Window_3x3>* windows);
  __global__ void fillWindows(uint2 size, int parent, unsigned char* pixels, Feature<Window_9x9>* windows);
  __global__ void fillWindows(uint2 size, int parent, unsigned char* pixels, Feature<Window_15x15>* windows);
  __global__ void fillWindows(uint2 size, int parent, unsigned char* pixels, Feature<Window_25x25>* windows);
  __global__ void fillWindows(uint2 size, int parent, unsigned char* pixels, Feature<Window_31x31>* windows);


  __global__ void subtractImages(unsigned int numPixels, float* pixelsUpper, float* pixelsLower, float* pixelsOut);

  __global__ void findExtrema(uint2 imageSize, float* pixelsUpper, float* pixelsMiddle, float* pixelsLower, int* extrema);
  __global__ void fillExtrema(int numKeyPoints, uint2 imageSize,float pixelWidth,int2 ssLoc, float sigma, int* extremaAddresses, float* pixels, FeatureFactory::ScaleSpace::SSKeyPoint* scaleSpaceKP); 
  
  __global__ void refineLocation(unsigned int numKeyPoints, uint2 imageSize, float sigmaMin, float blurSigmaMultiplier, unsigned int numBlurs, float** pixels, FeatureFactory::ScaleSpace::SSKeyPoint* scaleSpaceKP);
  __global__ void flagNoise(unsigned int numKeyPoints, FeatureFactory::ScaleSpace::SSKeyPoint* scaleSpaceKP, float threshold);
  __global__ void flagEdges(unsigned int numKeyPoints, unsigned int startingIndex, uint2 imageSize, FeatureFactory::ScaleSpace::SSKeyPoint* scaleSpaceKP, float* pixels, float threshold);

  __global__ void flagBorder(unsigned int numKeyPoints, uint2 imageSize, FeatureFactory::ScaleSpace::SSKeyPoint* scaleSpaceKP, float2 border);

  __global__ void computeThetas(unsigned long numKeyPoints, unsigned int keyPointIndex, uint2 imageSize, float pixelWidth, 
  float lambda, FeatureFactory::ScaleSpace::SSKeyPoint* keyPoints, float2* gradients, 
  int* thetaNumbers, unsigned int maxOrientations, float orientationThreshold, float* thetas);

  __global__ void expandKeyPoints(unsigned int numKeyPoints, FeatureFactory::ScaleSpace::SSKeyPoint* keyPointsIn, FeatureFactory::ScaleSpace::SSKeyPoint* keyPointsOut, int* thetaAddresses, float* thetas);

  /** 
  * \}
  */
}


#endif /* FEATUREFACTORY_CUH */
