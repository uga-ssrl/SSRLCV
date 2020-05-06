/** 
* \file MatchFactory.cuh
* \brief this file contains all feature matching methods
*/
#pragma once
#ifndef MATCHFACTORY_CUH
#define MATCHFACTORY_CUH

#include "common_includes.hpp"
#include "Image.cuh"
#include "Feature.cuh"
#include <thrust/device_ptr.h>
#include <thrust/copy.h>
#include <thrust/scan.h>

namespace ssrlcv{
  /**
  * \defgroup matching
  * \{
  */

  struct uint2_pair{
    uint2 a;
    uint2 b;
  };

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
    /**
     * Structs used with thrust::remove_if on GPU arrays
     */
    struct validate{
      __host__ __device__ bool operator()(const Match &m){
        return m.invalid;
      }
      __host__ __device__ bool operator()(const uint2_pair &m){
        return m.a.x == m.b.x && m.a.y == m.b.y;
      }
    };
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

    /**
     * Struct for comparison to be used with thrust::sort on GPU arrays
     */
    struct match_dist_comparator{
      __host__ __device__
      bool operator()(const DMatch& a, const DMatch& b){
        return a.distance < b.distance;
      }
    };
  }

  /**
   * \brief Factory for generating matches for accepted features
   * \details This factory has a series of methods for matching 
   * Fetures with filled descriptors. The descriptors implemented must 
   * have the distProtocol method implemented (see Feature.cuh). For 
   * better matching it is recommended to utilize a seed image where  
   * images in question will have no overlap. Additionally, before 
   * this becomes a header only implementation it is necessary 
   * to forward declare the new descriptor at the top of MatchFactory.cu like 
   * "template class ssrlcv::MatchFactory<Descriptor>;"".
   * \see Feature 
   * \see SIFT_Descriptor
   * \see Window3x3
   * \see Window9x9
   * \see Window15x15
   * \see Window25x25
   * \see Window31x31
   * \todo allow user to pass in their own distProtocol methods for features
   */
  template<typename T>
  class MatchFactory{
  private:
    Unity<Feature<T>>* seedFeatures;///<\brief Features generated from false seed image
  public:
    float absoluteThreshold;///<\brief Absolute distance threshold relative to the return of Descrpitor type T's distProtocol()
    float relativeThreshold;///<\brief Relative threshold based on closest descriptor neighbor/closest seed descriptor (fraction)
    /**
     * \brief Default constructor
     * \details This constructor sets absoluteThreshold to FLT_MAX and 
     * and relativeThreshold to 1.0. Keeping these values is not recommended. 
     */
    MatchFactory();
    /**
     * \brief Primary constructor
     * \param relativeThreshold seed distance threshold
     * \param absoluteThreshold distProtocol returm threshold
     */
    MatchFactory(float relativeThreshold, float absoluteThreshold);
    /**
     * \brief Sets seed features of MatchFactory to enable relative thresholding. 
     * \details These feature should come from and image that will have no overlap 
     * with iamges that are attempting to be matched. 
     * \param seedFeatures Unity of Features from a nonoverlapping image. 
     * \see Unity
     */
    void setSeedFeatures(Unity<Feature<T>>* seedFeatures);

    /**
     * \brief Discards matches flagged as invalid
     * \details This will discard any Match with the 
     * invalid parameter set to true.
     * \param matches Unity with matches
     * \see Unity
     */
    void validateMatches(Unity<Match>* matches);
    /**
     * \brief Discards matches flagged as invalid
     * \details This will discard any DMatch with the 
     * invalid parameter set to true.
     * \param matches Unity with matches
     * \see Unity
     */
    void validateMatches(Unity<DMatch>* matches);
    /**
     * \brief Discards matches flagged as invalid
     * \details This will discard any FeatureMatch<T> with the 
     * invalid parameter set to true.
     * \param matches Unity with matches
     * \see Unity
     */
    void validateMatches(Unity<FeatureMatch<T>>* matches);
    /**
     * \brief Discards matches flagged as invalid
     * \details This will discard any uint2_pair with the 
     * invalid parameter set to true.
     * \param matches Unity with matches
     * \see Unity
     */
    void validateMatches(Unity<uint2_pair>* matches);

    /**
     * \brief Discards matches with distance over an absolute threshold.
     * \details Depending on the descriptor that the DMatch came from,
     * the distProtocol method fills in the distance parameter. If 
     * the distance is less than the treshold it is kept. 
     * \param matches Unity with matches
     * \param threshold absolute threshold for checking distance
     * \see Unity
     */
    void refineMatches(Unity<DMatch>* matches, float threshold);
    /**
     * \brief Discards matches with distance over an absolute threshold.
     * \details Depending on the descriptor that the FeatureMatch<T> came from,
     * the distProtocol method fills in the distance parameter. If 
     * the distance is less than the treshold it is kept. 
     * \param matches Unity with matches
     * \param threshold absolute threshold for checking distance
     * \see Unity
     */
    void refineMatches(Unity<FeatureMatch<T>>* matches, float threshold);

    /**
     * \brief sorts all DMatches by mismatch distance
     * \param matches Unity of DMatches to be sorted by mismatch distance
     * \see Unity 
     */
    void sortMatches(Unity<DMatch>* matches);
    /**
     * \brief sorts all FeatureMatch<T> by mismatch distance
     * \param matches Unity of FeatureMatches to be sorted by mismatch distance
     * \see Unity
     */
    void sortMatches(Unity<FeatureMatch<T>>* matches);
    /**
     * \brief Returns Unity of simplified Match structs.
     * \details This simply converts every element in the Unity 
     * to the base Match struct, removing distance as a parameter.
     * \param matches Unity of DMatches to be converted to Matches
     * \returns Unity<Match>* an array of simplified matches
     * \see Unity
     */
    Unity<Match>* getRawMatches(Unity<DMatch>* matches);
    /**
     * \brief Returns Unity of simplified Match structs.
     * \details This simply converts every element in the Unity 
     * to the base Match struct, removing distance and the matched descriptors.
     * \param matches Unity of FeatureMatches to be converted to Matches
     * \returns Unity<Match>* an array of simplified matches
     * \see Unity
     */
    Unity<Match>* getRawMatches(Unity<FeatureMatch<T>>* matches);

    /**
     * \brief Generates distances between a set of features and the closest seedFeatures.
     * \details This method matches this->seedFeatures and the passed in Unity of Features 
     * and returns the distance of the closest seedFeature based on the distProtocol method 
     * of the descriptor. 
     * \param features features to be matches against this->seedFeatures
     * \returns Unity<float>* an array same length as features with distances associated 
     */
    Unity<float>* getSeedDistances(Unity<Feature<T>>* features);

    /**
     * \brief Generates Matches between Feature<Descriptor> when Descriptor::distProtocol() is implemented
     */
    Unity<Match>* generateMatches(Image* query, Unity<Feature<T>>* queryFeatures, Image* target, Unity<Feature<T>>* targetFeatures, Unity<float>* seedDistances = nullptr);
    /**
     * \brief Generates Matches between Feature<Descriptor> when Descriptor::distProtocol() is implemented with epipolar constaints
     * \param fundamental fundamental matrix generated from the query and target image camera parameters
     * \warning If bad fundamental matrix between two images, matches will be very bad. 
     */
    Unity<Match>* generateMatchesConstrained(Image* query, Unity<Feature<T>>* queryFeatures, Image* target, Unity<Feature<T>>* targetFeatures, float epsilon, float fundamental[3][3], Unity<float>* seedDistances = nullptr);

    /**
     * \brief Generates DMatches between Feature<Descriptor> when Descriptor::distProtocol() is implemented
     */
    Unity<DMatch>* generateDistanceMatches(Image* query, Unity<Feature<T>>* queryFeatures, Image* target, Unity<Feature<T>>* targetFeatures, Unity<float>* seedDistances = nullptr);
    /**
     * \brief Generates DMatches between Feature<Descriptor> when Descriptor::distProtocol() is implemented with epipolar constaints
     * \warning If bad fundamental matrix between two images, matches will be very bad. 
     */
    Unity<DMatch>* generateDistanceMatchesConstrained(Image* query, Unity<Feature<T>>* queryFeatures, Image* target, Unity<Feature<T>>* targetFeatures, float epsilon, float fundamental[3][3], Unity<float>* seedDistances = nullptr);
    
    /**
     * \brief Generates FeatureMatch<Descriptor>s between Feature<Descriptor> when Descriptor::distProtocol() is implemented
     */
    Unity<FeatureMatch<T>>* generateFeatureMatches(Image* query, Unity<Feature<T>>* queryFeatures, Image* target, Unity<Feature<T>>* targetFeatures, Unity<float>* seedDistances = nullptr);
    /**
     * \brief Generates FeatureMatch<Descriptor>s between Feature<Descriptor> when Descriptor::distProtocol() is implemented with epipolar constraints
     * \warning This method requires Images to have filled out Camera variables
     */
    Unity<FeatureMatch<T>>* generateFeatureMatchesConstrained(Image* query, Unity<Feature<T>>* queryFeatures, Image* target, Unity<Feature<T>>* targetFeatures, float epsilon, float fundamental[3][3], Unity<float>* seedDistances = nullptr);


    Unity<uint2_pair>* generateMatchesIndexOnly(Image* query, Unity<Feature<T>>* queryFeatures, Image* target, Unity<Feature<T>>* targetFeatures, Unity<float>* seedDistances = nullptr);
    Unity<uint2_pair>* generateMatchesConstrainedIndexOnly(Image* query, Unity<Feature<T>>* queryFeatures, Image* target, Unity<Feature<T>>* targetFeatures, float epsilon, float fundamental[3][3], Unity<float>* seedDistances = nullptr);

    /**
    * \brief Match a set of images 
    * \param images images that features were generated from
    * \param features features generated from images
    * \param ordered is the image set ordered or not
    * \param estimatedOverlap fraction of overlap from one image to the next
    */
    MatchSet generateMatchesExaustive(std::vector<Image*> images, std::vector<Unity<Feature<T>>*> features, bool ordered = true, float estimatedOverlap = 0.0f);
  
  };

  Unity<Match>* generateDiparityMatches(uint2 querySize, Unity<unsigned char>* queryPixels, uint2 targetSize, Unity<unsigned char>* targetPixels, 
    float fundamental[3][3], unsigned int maxDisparity, unsigned int windowSize = 3, Direction direction = undefined);

  /**
  * \defgroup match_io
  * \{
  */
  void writeMatchFile(Unity<Match>* matches, std::string pathToFile, bool binary = false);
  void writeMatchFile(MatchSet multiview_matches, std::string pathToFile, bool binary = false);
  Unity<Match>* readMatchFile(std::string pathToFile);
  /** \} */

  /* CUDA variable, method and kernel defintions */


  /**
  * \ingroup cuda_util 
  * \{
  */
  __host__ __device__ __forceinline__ float sum(const float3 &a);
  __host__ __device__ __forceinline__ float square(const float &a);
  __device__ __forceinline__ float atomicMinFloat (float * addr, float value);
  /** \} */

  /**
  * \ingroup cuda_kernels
  * \defgroup matching_kernels
  * \{
  */

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


  template<typename T>
  __global__ void matchFeaturesBruteForce(unsigned int queryImageID, unsigned long numFeaturesQuery,
    Feature<T>* featuresQuery, unsigned int targetImageID, unsigned long numFeaturesTarget,
    Feature<T>* featuresTarget, uint2_pair* matches, float absoluteThreshold);
  template<typename T>
  __global__ void matchFeaturesConstrained(unsigned int queryImageID, unsigned long numFeaturesQuery,
    Feature<T>* featuresQuery, unsigned int targetImageID, unsigned long numFeaturesTarget,
    Feature<T>* featuresTarget, uint2_pair* matches, float epsilon, float* fundamental, float absoluteThreshold);
  template<typename T>
  __global__ void matchFeaturesBruteForce(unsigned int queryImageID, unsigned long numFeaturesQuery,
    Feature<T>* featuresQuery, unsigned int targetImageID, unsigned long numFeaturesTarget,
    Feature<T>* featuresTarget, uint2_pair* matches, float* seedDistances, float relativeThreshold, float absoluteThreshold);
  template<typename T>
  __global__ void matchFeaturesConstrained(unsigned int queryImageID, unsigned long numFeaturesQuery,
    Feature<T>* featuresQuery, unsigned int targetImageID, unsigned long numFeaturesTarget,
    Feature<T>* featuresTarget, uint2_pair* matches, float epsilon, float* fundamental, float* seedDistances, 
    float relativeThreshold, float absoluteThreshold);

  //utility kernels

  __global__ void checkOverlap();

  __global__ void convertMatchToRaw(unsigned long numMatches, ssrlcv::Match* rawMatches, ssrlcv::DMatch* matches);
  template<typename T>
  __global__ void convertMatchToRaw(unsigned long numMatches, ssrlcv::Match* rawMatches, ssrlcv::FeatureMatch<T>* matches);
  
  /** /} */
  /** /} */

}


#endif /* MATCHFACTORY_CUH */
