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
  * \brief method that allows SIFT_Descriptor's to be matching in this class
  */
  __device__ __forceinline__ float calcElucidSq(const SIFT_Descriptor& a, const SIFT_Descriptor& b);
  /**
  * \brief method that allows SIFT_Descriptor's to be matching in this class
  */
  __device__ __forceinline__ float calcElucidSq(const SIFT_Descriptor& a, const SIFT_Descriptor& b, const float &bestMatch);




  /**
  * \brief represents pair of features
  */
  struct PointPair2D{
    int parentIds[2];
    float2 locations[2];
  };
  struct Match : PointPair2D{
    float distance;
  };
  template<typename T>
  struct FeatureMatch : Match{
    T descriptors[2];
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

    struct match_above_cutoff{
      __host__ __device__
      bool operator()(Match m){
        return m.distance > 0.0f;
      }
    };

    struct distance_thresholder{
      float threshold;
      distance_thresholder(float threshold) : threshold(threshold){};
      __host__ __device__
      bool operator()(Match m){
        return (m.distance > threshold);
      }
    };
    struct match_comparator{
      __host__ __device__
      bool operator()(const Match& a, const Match& b){
        return a.distance < b.distance;
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

    void refineMatches(Unity<Match>* matches, float cutoffRatio);
    void refineMatches(Unity<FeatureMatch<T>>* matches, float cutoffRatio);

    /**
    * \brief sorts all matches by mismatch distance
    * \note this is a cpu version
    */
    void sortMatches(Unity<Match>* matches);
    void sortMatches(Unity<FeatureMatch<T>>* matches);
    Unity<Match>* getRawMatches(Unity<FeatureMatch<T>>* matches);

    /**
    * \brief Generates matches between sift features
    */
    Unity<FeatureMatch<T>>* generateMatchesBruteForce(Image* query, Unity<Feature<T>>* queryFeatures, Image* target, Unity<Feature<T>>* targetFeatures);
    /**
    * \brief Generates matches between sift features constrained by epipolar line
    */
    Unity<FeatureMatch<T>>* generateMatchesConstrained(Image* query, Unity<Feature<T>>* queryFeatures, Image* target, Unity<Feature<T>>* targetFeatures, float epsilon);
    /**
    * \brief Generates subpixel matches between sift features
    */
    Unity<FeatureMatch<T>>* generateSubPixelMatchesBruteForce(Image* query, Unity<Feature<T>>* queryFeatures, Image* target, Unity<Feature<T>>* targetFeatures);
    /**
    * \brief Generates subpixel matches between sift features constrained by the epipolar line
    */
    Unity<FeatureMatch<T>>* generateSubPixelMatchesConstrained(Image* query, Unity<Feature<T>>* queryFeatures, Image* target, Unity<Feature<T>>* targetFeatures, float epsilon);


  };
  /* CUDA variable, method and kernel defintions */

  extern __constant__ float matchTreshold;
  extern __constant__ int splineHelper[4][4];
  extern __constant__ int splineHelperInv[4][4];

  __device__ __host__ __forceinline__ float sum(const float3 &a);
  __device__ __forceinline__ float square(const float &a);
  __device__ __forceinline__ float atomicMinFloat (float * addr, float value);
  __device__ __forceinline__ float findSubPixelContributer(const float2 &loc, const int &width);



  //base matching kernels
  template<typename T>
  __global__ void matchFeaturesBruteForce(unsigned int queryImageID, unsigned long numFeaturesQuery,
    Feature<T>* featuresQuery, unsigned int targetImageID, unsigned long numFeaturesTarget,
    Feature<T>* featuresTarget, Match* matches);
  template<typename T>
  __global__ void matchFeaturesBruteForce(unsigned int queryImageID, unsigned long numFeaturesQuery,
    Feature<T>* featuresQuery, unsigned int targetImageID, unsigned long numFeaturesTarget,
    Feature<T>* featuresTarget, FeatureMatch<T>* matches);
  template<typename T>
  __global__ void matchFeaturesConstrained(unsigned int queryImageID, unsigned long numFeaturesQuery,
    Feature<T>* featuresQuery, unsigned int targetImageID, unsigned long numFeaturesTarget,
    Feature<T>* featuresTarget, Match* matches, float epsilon, float3 fundamental[3]);

  template<typename T>
  __global__ void matchFeaturesConstrained(unsigned int queryImageID, unsigned long numFeaturesQuery,
    Feature<T>* featuresQuery, unsigned int targetImageID, unsigned long numFeaturesTarget,
    Feature<T>* featuresTarget, FeatureMatch<T>* matches, float epsilon, float3 fundamental[3]);



  //subpixel kernels
  template<typename T>
  __global__ void initializeSubPixels(unsigned long numMatches, FeatureMatch<T>* matches, SubpixelM7x7* subPixelDescriptors,
    uint2 querySize, unsigned long numFeaturesQuery, Feature<T>* featuresQuery,
    uint2 targetSize, unsigned long numFeaturesTarget, Feature<T>* featuresTarget);

  __global__ void fillSplines(unsigned long numMatches, SubpixelM7x7* subPixelDescriptors, Spline* splines);
  template<typename T>
  __global__ void determineSubPixelLocationsBruteForce(float increment, unsigned long numMatches, FeatureMatch<T>* matches, Spline* splines);

}


#endif /* MATCHFACTORY_CUH */
