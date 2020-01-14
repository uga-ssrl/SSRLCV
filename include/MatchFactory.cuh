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

  //TODO differentiate distance methods and pass function pointers to matching kernels

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
    __host__ __device__ bool operator()(const Match &m){
      return m.invalid;
    }
  };
  /**
  * \brief derived Match struct with distance
  * \note distance is squared here to prevent use of sqrtf
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
    float absoluteThreshold;//squared distance
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
    Unity<Match>* generateMatchesConstrained(Image* query, Unity<Feature<T>>* queryFeatures, Image* target, Unity<Feature<T>>* targetFeatures, float epsilon, float fundamental[3][3], Unity<float>* seedDistances = nullptr);

    /**
    * \brief Generates matches between sift features
    */
    Unity<DMatch>* generateDistanceMatches(Image* query, Unity<Feature<T>>* queryFeatures, Image* target, Unity<Feature<T>>* targetFeatures, Unity<float>* seedDistances = nullptr);
    /**
    * \brief Generates matches between sift features constrained by epipolar line
    * \warning This method requires Images to have filled out Camera variables
    */
    Unity<DMatch>* generateDistanceMatchesConstrained(Image* query, Unity<Feature<T>>* queryFeatures, Image* target, Unity<Feature<T>>* targetFeatures, float epsilon, float fundamental[3][3], Unity<float>* seedDistances = nullptr);
    
    /**
    * \brief Generates matches between sift features
    */
    Unity<FeatureMatch<T>>* generateFeatureMatches(Image* query, Unity<Feature<T>>* queryFeatures, Image* target, Unity<Feature<T>>* targetFeatures, Unity<float>* seedDistances = nullptr);
    /**
    * \brief Generates matches between sift features constrained by epipolar line
    * \warning This method requires Images to have filled out Camera variables
    */
    Unity<FeatureMatch<T>>* generateFeatureMatchesConstrained(Image* query, Unity<Feature<T>>* queryFeatures, Image* target, Unity<Feature<T>>* targetFeatures, float epsilon, float fundamental[3][3], Unity<float>* seedDistances = nullptr);

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


    
  };
  Unity<Match>* generateDiparityMatches(uint2 querySize, Unity<unsigned char>* queryPixels, uint2 targetSize, Unity<unsigned char>* targetPixels, 
    float fundamental[3][3], unsigned int maxDisparity, unsigned int windowSize = 3, Direction direction = undefined);

  void writeMatchFile(Unity<Match>* matches, std::string pathToFile, bool binary = false);
  Unity<Match>* readMatchFile(std::string pathToFile);

  /* CUDA variable, method and kernel defintions */

  extern __constant__ float matchTreshold;
  extern __constant__ int splineHelper[4][4];
  extern __constant__ int splineHelperInv[4][4];

  __host__ __device__ __forceinline__ float sum(const float3 &a);
  __host__ __device__ __forceinline__ float square(const float &a);
  __device__ __forceinline__ float atomicMinFloat (float * addr, float value);
  __host__ __device__ __forceinline__ float findSubPixelContributer(const float2 &loc, const int &width);



  template<typename T>
  __global__ void getSeedMatchDistances(unsigned long numFeaturesQuery, Feature<T>* featuresQuery, unsigned long numSeedFeatures,
    Feature<T>* seedFeatures, float* matchDistances);

  //base matching kernels
  __global__ void disparityMatching(uint2 querySize, unsigned char* pixelsQuery, uint2 targetSize, unsigned char* pixelsTarget, float* fundamental, Match* matches, unsigned int maxDisparity, Direction direction);
  __global__ void disparityScanMatching(uint2 querySize, unsigned char* pixelsQuery, uint2 targetSize, unsigned char* pixelsTarget, Match* matches, unsigned int maxDisparity, Direction direction);

  template<typename T>
  __global__ void matchFeaturesBruteForce(unsigned int queryImageID, unsigned long numFeaturesQuery,
    Feature<T>* featuresQuery, unsigned int targetImageID, unsigned long numFeaturesTarget,
    Feature<T>* featuresTarget, Match* matches, float absoluteThreshold);
  template<typename T>
  __global__ void matchFeaturesConstrained(unsigned int queryImageID, unsigned long numFeaturesQuery,
    Feature<T>* featuresQuery, unsigned int targetImageID, unsigned long numFeaturesTarget,
    Feature<T>* featuresTarget, Match* matches, float epsilon, float* fundamental, float absoluteThreshold);
  template<typename T>
  __global__ void matchFeaturesBruteForce(unsigned int queryImageID, unsigned long numFeaturesQuery,
    Feature<T>* featuresQuery, unsigned int targetImageID, unsigned long numFeaturesTarget,
    Feature<T>* featuresTarget, Match* matches, float* seedDistances ,float relativeThreshold, float absoluteThreshold);
  template<typename T>
  __global__ void matchFeaturesConstrained(unsigned int queryImageID, unsigned long numFeaturesQuery,
    Feature<T>* featuresQuery, unsigned int targetImageID, unsigned long numFeaturesTarget,
    Feature<T>* featuresTarget, Match* matches, float epsilon, float* fundamental, float* seedDistances ,float relativeThreshold, float absoluteThreshold);


  template<typename T>
  __global__ void matchFeaturesBruteForce(unsigned int queryImageID, unsigned long numFeaturesQuery,
    Feature<T>* featuresQuery, unsigned int targetImageID, unsigned long numFeaturesTarget,
    Feature<T>* featuresTarget, DMatch* matches, float absoluteThreshold);
  template<typename T>
  __global__ void matchFeaturesConstrained(unsigned int queryImageID, unsigned long numFeaturesQuery,
    Feature<T>* featuresQuery, unsigned int targetImageID, unsigned long numFeaturesTarget,
    Feature<T>* featuresTarget, DMatch* matches, float epsilon, float* fundamental, float absoluteThreshold);
  template<typename T>
  __global__ void matchFeaturesBruteForce(unsigned int queryImageID, unsigned long numFeaturesQuery,
    Feature<T>* featuresQuery, unsigned int targetImageID, unsigned long numFeaturesTarget,
    Feature<T>* featuresTarget, DMatch* matches, float* seedDistances, float relativeThreshold, float absoluteThreshold);
  template<typename T>
  __global__ void matchFeaturesConstrained(unsigned int queryImageID, unsigned long numFeaturesQuery,
    Feature<T>* featuresQuery, unsigned int targetImageID, unsigned long numFeaturesTarget,
    Feature<T>* featuresTarget, DMatch* matches, float epsilon, float* fundamental, float* seedDistances,
    float relativeThreshold, float absoluteThreshold);
  
  template<typename T>
  __global__ void matchFeaturesBruteForce(unsigned int queryImageID, unsigned long numFeaturesQuery,
    Feature<T>* featuresQuery, unsigned int targetImageID, unsigned long numFeaturesTarget,
    Feature<T>* featuresTarget, FeatureMatch<T>* matches, float absoluteThreshold);
  template<typename T>
  __global__ void matchFeaturesConstrained(unsigned int queryImageID, unsigned long numFeaturesQuery,
    Feature<T>* featuresQuery, unsigned int targetImageID, unsigned long numFeaturesTarget,
    Feature<T>* featuresTarget, FeatureMatch<T>* matches, float epsilon, float* fundamental, float absoluteThreshold);
  template<typename T>
  __global__ void matchFeaturesBruteForce(unsigned int queryImageID, unsigned long numFeaturesQuery,
    Feature<T>* featuresQuery, unsigned int targetImageID, unsigned long numFeaturesTarget,
    Feature<T>* featuresTarget, FeatureMatch<T>* matches, float* seedDistances, float relativeThreshold, float absoluteThreshold);
  template<typename T>
  __global__ void matchFeaturesConstrained(unsigned int queryImageID, unsigned long numFeaturesQuery,
    Feature<T>* featuresQuery, unsigned int targetImageID, unsigned long numFeaturesTarget,
    Feature<T>* featuresTarget, FeatureMatch<T>* matches, float epsilon, float* fundamental, float* seedDistances, 
    float relativeThreshold, float absoluteThreshold);
 
  //utility kernels
  __global__ void convertMatchToRaw(unsigned long numMatches, ssrlcv::Match* rawMatches, ssrlcv::DMatch* matches);
  template<typename T>
  __global__ void convertMatchToRaw(unsigned long numMatches, ssrlcv::Match* rawMatches, ssrlcv::FeatureMatch<T>* matches);

}


#endif /* MATCHFACTORY_CUH */
