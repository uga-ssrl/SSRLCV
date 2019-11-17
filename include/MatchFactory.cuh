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
  __device__ __forceinline__ float dist(const Feature<SIFT_Descriptor>& a, const Feature<SIFT_Descriptor>& b);
  /**
  * \brief method that allows Feature's with SIFT_Descriptor's to be matching in this class
  */
  __device__ __forceinline__ float dist(const Feature<SIFT_Descriptor>& a, const Feature<SIFT_Descriptor>& b, const float &bestMatch);
  /**
  * \brief method that allows SIFT_Descriptor's to be matching in this class
  */
  __device__ __forceinline__ float dist(const SIFT_Descriptor& a, const SIFT_Descriptor& b);
  /**
  * \brief method that allows SIFT_Descriptor's to be matching in this class
  */
  __device__ __forceinline__ float dist(const SIFT_Descriptor& a, const SIFT_Descriptor& b, const float &bestMatch);

  __device__ __forceinline__ int dist(const Window_3x3& a, const Window_3x3& b);
  __device__ __forceinline__ int dist(const Window_9x9& a, const Window_9x9& b);
  __device__ __forceinline__ int dist(const Window_15x15& a, const Window_15x15& b);
  __device__ __forceinline__ int dist(const Window_25x25& a, const Window_25x25& b);
  __device__ __forceinline__ int dist(const Window_35x35& a, const Window_35x35& b);
  __device__ __forceinline__ int dist(const Window_3x3& a, const Window_3x3& b, const int &bestMatch);
  __device__ __forceinline__ int dist(const Window_9x9& a, const Window_9x9& b, const int &bestMatch);
  __device__ __forceinline__ int dist(const Window_15x15& a, const Window_15x15& b, const int &bestMatch);
  __device__ __forceinline__ int dist(const Window_25x25& a, const Window_25x25& b, const int &bestMatch);
  __device__ __forceinline__ int dist(const Window_35x35& a, const Window_35x35& b, const int &bestMatch);

  /**
  * \brief simple struct meant to fill out matches
  */
  struct KeyPoint{
    int parentId;
    float2 loc;
  };

  /**
  * \brief struct for holding reference to keypoints that make up multiview match
  */
  struct MultiMatch{
    unsigned int numKeyPoints;
    int index;
  };

  /**
  * \brief struct to pass around MultiMatches and KeyPoint sets
  */
  struct MatchSet{
    Unity<KeyPoint>* keyPoints;
    Unity<MultiMatch>* matches;
  };

  /**
  * \brief base Match struct pair of keypoints
  */
  struct Match{
    bool invalid;
    KeyPoint keyPoints[2];
  };
  struct validate{
    __device__ __host__ bool operator()(const Match &m){
      return m.invalid;
    }
  };
  /**
  * \brief derived Match struct with distance
  */
  struct DMatch: Match{
    float distance;
  };
  /**
  * \brief derived DMatch struct with descriptors
  */
  template<typename T>
  struct FeatureMatch : DMatch{
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
      bool operator()(DMatch m){
        return m.distance > 0.0f;
      }
    };

    struct match_dist_thresholder{
      float threshold;
      match_dist_thresholder(float threshold) : threshold(threshold){};
      __host__ __device__
      bool operator()(DMatch m){
        return (m.distance > threshold);
      }
    };
    struct match_dist_comparator{
      __host__ __device__
      bool operator()(const DMatch& a, const DMatch& b){
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
  private:
    Unity<Feature<T>>* seedFeatures;
  public:
    float absoluteThreshold;
    float relativeThreshold;
    MatchFactory();
    MatchFactory(float relativeThreshold, float absoluteThreshold);
    void setSeedFeatures(Unity<Feature<T>>* seedFeatures);//implement

    //NOTE nothing for nview is implemented
    //TODO consider making it so features are computed if they arent instead of throwing errors with image parameters

    void validateMatches(Unity<Match>* matches);
    void validateMatches(Unity<DMatch>* matches);
    void validateMatches(Unity<FeatureMatch<T>>* matches);

    void refineMatches(Unity<DMatch>* matches, float threshold);
    void refineMatches(Unity<FeatureMatch<T>>* matches, float threshold);

    /**
    * \brief sorts all matches by mismatch distance
    * \note this is a cpu version
    */
    void sortMatches(Unity<DMatch>* matches);
    void sortMatches(Unity<FeatureMatch<T>>* matches);
    Unity<Match>* getRawMatches(Unity<DMatch>* matches);
    Unity<Match>* getRawMatches(Unity<FeatureMatch<T>>* matches);

    Unity<float>* getSeedDistances(Unity<Feature<T>>* features);

    /**
    * \brief Generates matches between sift features
    */
    Unity<Match>* generateMatches(Image* query, Unity<Feature<T>>* queryFeatures, Image* target, Unity<Feature<T>>* targetFeatures, Unity<float>* seedDistances = nullptr);
    /**
    * \brief Generates matches between sift features constrained by epipolar line
    * \warning This method requires Images to have filled out Camera variables
    */
    Unity<Match>* generateMatchesConstrained(Image* query, Unity<Feature<T>>* queryFeatures, Image* target, Unity<Feature<T>>* targetFeatures, float epsilon, Unity<float>* seedDistances = nullptr);
    
    /**
    * \brief Generates matches between sift features
    */
    Unity<DMatch>* generateDistanceMatches(Image* query, Unity<Feature<T>>* queryFeatures, Image* target, Unity<Feature<T>>* targetFeatures, Unity<float>* seedDistances = nullptr);
    /**
    * \brief Generates matches between sift features constrained by epipolar line
    * \warning This method requires Images to have filled out Camera variables
    */
    Unity<DMatch>* generateDistanceMatchesConstrained(Image* query, Unity<Feature<T>>* queryFeatures, Image* target, Unity<Feature<T>>* targetFeatures, float epsilon, Unity<float>* seedDistances = nullptr);
    
    /**
    * \brief Generates matches between sift features
    */
    Unity<FeatureMatch<T>>* generateFeatureMatches(Image* query, Unity<Feature<T>>* queryFeatures, Image* target, Unity<Feature<T>>* targetFeatures, Unity<float>* seedDistances = nullptr);
    /**
    * \brief Generates matches between sift features constrained by epipolar line
    * \warning This method requires Images to have filled out Camera variables
    */
    Unity<FeatureMatch<T>>* generateFeatureMatchesConstrained(Image* query, Unity<Feature<T>>* queryFeatures, Image* target, Unity<Feature<T>>* targetFeatures, float epsilon, Unity<float>* seedDistances = nullptr);

    /**
    * \brief interpolates Matches between multiple images
    * \todo implement
    */
    MatchSet* getMultiViewMatches(std::vector<Image*> images, Unity<Match>* matches);
    /**
    * \brief interpolates Matches between multiple images
    * \todo implement
    */
    MatchSet* getMultiViewMatches(std::vector<Image*> images, Unity<DMatch>* matches);
    /**
    * \brief interpolates Matches between multiple images
    * \todo implement
    */
    MatchSet* getMultiViewMatches(std::vector<Image*> images, Unity<FeatureMatch<T>>* matches);


    /*
    METHODS IN MATCHFACTORY BELOW THIS ONLY WORK FOR DENSE FEATURES THAT HAVE NOT BEEN FILTERED
    */
    /**
    * \brief Generates subpixel matches between sift features
    * \warning This only works for dense features
    */
    Unity<FeatureMatch<T>>* generateSubPixelMatches(Image* query, Unity<Feature<T>>* queryFeatures, Image* target, Unity<Feature<T>>* targetFeatures);
    /**
    * \brief Generates subpixel matches between sift features constrained by the epipolar line
    * \warning This only works for dense features
    * \warning This method requires Images to have filled out Camera variable
    */
    Unity<FeatureMatch<T>>* generateSubPixelMatchesConstrained(Image* query, Unity<Feature<T>>* queryFeatures, Image* target, Unity<Feature<T>>* targetFeatures, float epsilon);


  };

  void writeMatchFile(Unity<Match>* matches, std::string pathToFile);

  /* CUDA variable, method and kernel defintions */

  extern __constant__ float matchTreshold;
  extern __constant__ int splineHelper[4][4];
  extern __constant__ int splineHelperInv[4][4];

  __device__ __host__ __forceinline__ float sum(const float3 &a);
  __device__ __forceinline__ float square(const float &a);
  __device__ __forceinline__ float atomicMinFloat (float * addr, float value);
  __device__ __forceinline__ float findSubPixelContributer(const float2 &loc, const int &width);



  template<typename T>
  __global__ void getSeedMatchDistances(unsigned long numFeaturesQuery, Feature<T>* featuresQuery, unsigned long numSeedFeatures,
    Feature<T>* seedFeatures, float* matchDistances);

  //base matching kernels
  template<typename T>
  __global__ void matchFeaturesBruteForce(unsigned int queryImageID, unsigned long numFeaturesQuery,
    Feature<T>* featuresQuery, unsigned int targetImageID, unsigned long numFeaturesTarget,
    Feature<T>* featuresTarget, Match* matches, float absoluteThreshold);
  template<typename T>
  __global__ void matchFeaturesConstrained(unsigned int queryImageID, unsigned long numFeaturesQuery,
    Feature<T>* featuresQuery, unsigned int targetImageID, unsigned long numFeaturesTarget,
    Feature<T>* featuresTarget, Match* matches, float epsilon, float3 fundamental[3], float absoluteThreshold);
  template<typename T>
  __global__ void matchFeaturesBruteForce(unsigned int queryImageID, unsigned long numFeaturesQuery,
    Feature<T>* featuresQuery, unsigned int targetImageID, unsigned long numFeaturesTarget,
    Feature<T>* featuresTarget, Match* matches, float* seedDistances ,float relativeThreshold, float absoluteThreshold);
  template<typename T>
  __global__ void matchFeaturesConstrained(unsigned int queryImageID, unsigned long numFeaturesQuery,
    Feature<T>* featuresQuery, unsigned int targetImageID, unsigned long numFeaturesTarget,
    Feature<T>* featuresTarget, Match* matches, float epsilon, float3 fundamental[3], float* seedDistances ,float relativeThreshold, float absoluteThreshold);


  template<typename T>
  __global__ void matchFeaturesBruteForce(unsigned int queryImageID, unsigned long numFeaturesQuery,
    Feature<T>* featuresQuery, unsigned int targetImageID, unsigned long numFeaturesTarget,
    Feature<T>* featuresTarget, DMatch* matches, float absoluteThreshold);
  template<typename T>
  __global__ void matchFeaturesConstrained(unsigned int queryImageID, unsigned long numFeaturesQuery,
    Feature<T>* featuresQuery, unsigned int targetImageID, unsigned long numFeaturesTarget,
    Feature<T>* featuresTarget, DMatch* matches, float epsilon, float3 fundamental[3], float absoluteThreshold);
  template<typename T>
  __global__ void matchFeaturesBruteForce(unsigned int queryImageID, unsigned long numFeaturesQuery,
    Feature<T>* featuresQuery, unsigned int targetImageID, unsigned long numFeaturesTarget,
    Feature<T>* featuresTarget, DMatch* matches, float* seedDistances, float relativeThreshold, float absoluteThreshold);
  template<typename T>
  __global__ void matchFeaturesConstrained(unsigned int queryImageID, unsigned long numFeaturesQuery,
    Feature<T>* featuresQuery, unsigned int targetImageID, unsigned long numFeaturesTarget,
    Feature<T>* featuresTarget, DMatch* matches, float epsilon, float3 fundamental[3], float* seedDistances,
    float relativeThreshold, float absoluteThreshold);
  
  template<typename T>
  __global__ void matchFeaturesBruteForce(unsigned int queryImageID, unsigned long numFeaturesQuery,
    Feature<T>* featuresQuery, unsigned int targetImageID, unsigned long numFeaturesTarget,
    Feature<T>* featuresTarget, FeatureMatch<T>* matches, float absoluteThreshold);
  template<typename T>
  __global__ void matchFeaturesConstrained(unsigned int queryImageID, unsigned long numFeaturesQuery,
    Feature<T>* featuresQuery, unsigned int targetImageID, unsigned long numFeaturesTarget,
    Feature<T>* featuresTarget, FeatureMatch<T>* matches, float epsilon, float3 fundamental[3], float absoluteThreshold);
  template<typename T>
  __global__ void matchFeaturesBruteForce(unsigned int queryImageID, unsigned long numFeaturesQuery,
    Feature<T>* featuresQuery, unsigned int targetImageID, unsigned long numFeaturesTarget,
    Feature<T>* featuresTarget, FeatureMatch<T>* matches, float* seedDistances, float relativeThreshold, float absoluteThreshold);
  template<typename T>
  __global__ void matchFeaturesConstrained(unsigned int queryImageID, unsigned long numFeaturesQuery,
    Feature<T>* featuresQuery, unsigned int targetImageID, unsigned long numFeaturesTarget,
    Feature<T>* featuresTarget, FeatureMatch<T>* matches, float epsilon, float3 fundamental[3], float* seedDistances, 
    float relativeThreshold, float absoluteThreshold);



  //subpixel kernels
  template<typename T>
  __global__ void initializeSubPixels(unsigned long numMatches, FeatureMatch<T>* matches, SubpixelM7x7* subPixelDescriptors,
    uint2 querySize, unsigned long numFeaturesQuery, Feature<T>* featuresQuery,
    uint2 targetSize, unsigned long numFeaturesTarget, Feature<T>* featuresTarget);

  __global__ void fillSplines(unsigned long numMatches, SubpixelM7x7* subPixelDescriptors, Spline* splines);
  template<typename T>
  __global__ void determineSubPixelLocationsBruteForce(float increment, unsigned long numMatches, FeatureMatch<T>* matches, Spline* splines);

  //utility kernels
  __global__ void convertMatchToRaw(unsigned long numMatches, ssrlcv::Match* rawMatches, ssrlcv::DMatch* matches);
  template<typename T>
  __global__ void convertMatchToRaw(unsigned long numMatches, ssrlcv::Match* rawMatches, ssrlcv::FeatureMatch<T>* matches);

}


#endif /* MATCHFACTORY_CUH */
