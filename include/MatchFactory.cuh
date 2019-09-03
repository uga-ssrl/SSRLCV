/** \file MatchFactory.cuh
* \brief this file contains all feature matching methods
*/
#ifndef MATCHFACTORY_CUH
#define MATCHFACTORY_CUH

#include "common_includes.h"
#include "Image.cuh"
#include "Feature.cuh"
#include "Unity.cuh"
#include <thrust/device_ptr.h>
#include <thrust/copy.h>
#include <thrust/scan.h>

namespace ssrlcv{

  /**
  * \brief method that allows Feature's with SIFT_Descriptor's to be matching in this class
  */
  __device__ __forceinline__ float calcElucidSq(const Feature<SIFT_Descriptor>& a, const Feature<SIFT_Descriptor>& b);
  /**
  * \brief method that allows Feature's with SIFT_Descriptor's to be matching in this class
  */
  __device__ __forceinline__ float calcElucidSq(const Feature<SIFT_Descriptor>& a, const Feature<SIFT_Descriptor>& b, const float &bestMatch);



  /**
  * \brief represents pair of features
  */
  template<typename T>
  struct Match{
    int parentId[2];
    Feature<T> features[2];
    float distance;
  };

  namespace{
    struct Spline{
      float coeff[6][6][4][4];
    };
    typedef struct Spline Spline;

    struct SubpixelM7x7{
      float M1[9][9];
      float M2[9][9];
    };
    typedef struct SubpixelM7x7 SubpixelM7x7;

    template<typename T>
    struct match_above_cutoff{
      __host__ __device__
      bool operator()(Match<T> m){
        return m.distance > 0.0f;
      }
    };
  }

  /**
  * \brief Factory for generating matches for accepted features
  * \note if attempting to add new Feature support implement calcElucidSq
  * as modeled by calcElucidSq(const Feature<SIFT_Descriptor>& a,
  * const Feature<SIFT_Descriptor>& b);calcElucidSq(const Feature<SIFT_Descriptor>& a, const Feature<SIFT_Descriptor>& b)
  * and calcElucidSq(const Feature<SIFT_Descriptor>& a, const Feature<SIFT_Descriptor>& b),
  * then add template declaration at the top MatchFactory.cu like template class ssrlcv::MatchFactory<ssrlcv::SIFT_Descriptor>;
  */
  template<typename T>
  class MatchFactory{

  public:

    MatchFactory();

    //NOTE nothing for nview is implemented
    //TODO consider making it so features are computed if they arent instead of throwing errors with image parameters

    void refineMatches(Unity<Match<T>>* matches, float cutoffRatio);

    /**
    * \brief Generates matches between sift features
    */
    Unity<Match<T>>* generateMatchesBruteForce(Image* query, Unity<Feature<T>>* queryFeatures, Image* target, Unity<Feature<T>>* targetFeatures);
    /**
    * \brief Generates matches between sift features constrained by epipolar line
    */
    Unity<Match<T>>* generateMatchesConstrained(Image* query, Unity<Feature<T>>* queryFeatures, Image* target, Unity<Feature<T>>* targetFeatures, float epsilon);
    /**
    * \brief Generates subpixel matches between sift features
    */
    Unity<Match<T>>* generateSubPixelMatchesBruteForce(Image* query, Unity<Feature<T>>* queryFeatures, Image* target, Unity<Feature<T>>* targetFeatures);
    /**
    * \brief Generates subpixel matches between sift features constrained by the epipolar line
    */
    Unity<Match<T>>* generateSubPixelMatchesConstrained(Image* query, Unity<Feature<T>>* queryFeatures, Image* target, Unity<Feature<T>>* targetFeatures, float epsilon);


  };
  /* CUDA variable, method and kernel defintions */

  extern __constant__ float matchTreshold;
  extern __constant__ int splineHelper[4][4];
  extern __constant__ int splineHelperInv[4][4];

  __device__ __host__ __forceinline__ float sum(const float3 &a);
  __device__ __forceinline__ float square(const float &a);
  __device__ __forceinline__ float atomicMinFloat (float * addr, float value);
  __device__ __forceinline__ float findSubPixelContributer(const float2 &loc, const int &width);




  /*
  Pairwise stuff
  */
  template<typename T>
  __global__ void matchFeaturesBruteForce(unsigned int queryImageID, unsigned long numFeaturesQuery,
    Feature<T>* featuresQuery, unsigned int targetImageID, unsigned long numFeaturesTarget,
    Feature<T>* featuresTarget, Match<T>* matches);

  template<typename T>
  __global__ void matchFeaturesConstrained(unsigned int queryImageID, unsigned long numFeaturesQuery,
    Feature<T>* featuresQuery, unsigned int targetImageID, unsigned long numFeaturesTarget,
    Feature<T>* featuresTarget, Match<T>* matches, float epsilon, float3 fundamental[3]);

  template<typename T>
  __global__ void initializeSubPixels(unsigned long numMatches, Match<T>* matches, SubpixelM7x7* subPixelDescriptors,
    uint2 querySize, unsigned long numFeaturesQuery, Feature<T>* featuresQuery,
    uint2 targetSize, unsigned long numFeaturesTarget, Feature<T>* featuresTarget);

  __global__ void fillSplines(unsigned long numMatches, SubpixelM7x7* subPixelDescriptors, Spline* splines);
  template<typename T>
  __global__ void determineSubPixelLocationsBruteForce(float increment, unsigned long numMatches, Match<T>* matches, Spline* splines);

  template<typename T>
  __global__ void refineWCutoffRatio(unsigned long numMatches, Match<T>* matches, int* matchCounter, float2 minMax, float cutoffRatio);
  template<typename T>
  __global__ void copyMatches(unsigned long numMatches, int* matchCounter, Match<T>* minimizedMatches, Match<T>* matches);

}


#endif /* MATCHFACTORY_CUH */
