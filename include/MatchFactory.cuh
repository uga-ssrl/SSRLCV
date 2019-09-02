/** @file MatchFactory.cuh
* @brief this file contains all feature matching methods
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
  struct Spline{
    float coeff[6][6][4][4];
  };
  typedef struct Spline Spline;

  struct SubpixelM7x7{
    float M1[9][9];
    float M2[9][9];
  };
  typedef struct SubpixelM7x7 SubpixelM7x7;

  struct Match{
    Feature<unsigned int> features[2];//descriptor == parentImage id
    float distance;
  };
  typedef struct Match Match;

  namespace{
    struct match_above_cutoff{
      __host__ __device__
      bool operator()(Match m){
        return m.distance > 0.0f;
      }
    };
  }

  class MatchFactory{

  public:

    MatchFactory();

    //NOTE nothing for nview is implemented
    //TODO consider making it so features are computed if they arent instead of throwing errors with image parameters

    void refineMatches(Unity<Match>* matches, float cutoffRatio);

    Unity<Match>* generateMatchesBruteForce(Image* query, Unity<Feature<SIFT_Descriptor>>* queryFeatures, Image* target, Unity<Feature<SIFT_Descriptor>>* targetFeatures);
    Unity<Match>* generateMatchesConstrained(Image* query, Unity<Feature<SIFT_Descriptor>>* queryFeatures, Image* target, Unity<Feature<SIFT_Descriptor>>* targetFeatures, float epsilon);
    //NOTE currently brute force
    Unity<Match>* generateSubPixelMatchesBruteForce(Image* query, Unity<Feature<SIFT_Descriptor>>* queryFeatures, Image* target, Unity<Feature<SIFT_Descriptor>>* targetFeatures);
    Unity<Match>* generateSubPixelMatchesConstrained(Image* query, Unity<Feature<SIFT_Descriptor>>* queryFeatures, Image* target, Unity<Feature<SIFT_Descriptor>>* targetFeatures, float epsilon);


  };
  /* CUDA variable, method and kernel defintions */

  extern __constant__ float matchTreshold;
  extern __constant__ int splineHelper[4][4];
  extern __constant__ int splineHelperInv[4][4];

  __device__ __host__ __forceinline__ float sum(const float3 &a);
  __device__ __forceinline__ float square(const float &a);
  __device__ __forceinline__ float calcElucid(const int2 &a, const int2 &b);
  __device__ __forceinline__ float calcElucid(const unsigned char a[128], const unsigned char b[128]);
  __device__ __forceinline__ float atomicMinFloat (float * addr, float value);
  __device__ __forceinline__ float atomicMaxFloat (float * addr, float value);
  __device__ __forceinline__ float findSubPixelContributer(const float2 &loc, const int &width);
  /*
  Pairwise stuff
  */
  __global__ void matchFeaturesBruteForce(unsigned int queryImageID, unsigned long numFeaturesQuery,
    Feature<SIFT_Descriptor>* featuresQuery, unsigned int targetImageID, unsigned long numFeaturesTarget,
    Feature<SIFT_Descriptor>* featuresTarget, Match* matches);

  __global__ void matchFeaturesConstrained(unsigned int queryImageID, unsigned long numFeaturesQuery,
    Feature<SIFT_Descriptor>* featuresQuery, unsigned int targetImageID, unsigned long numFeaturesTarget,
    Feature<SIFT_Descriptor>* featuresTarget, Match* matches, float epsilon, float3 fundamental[3]);

  //pairwise subPixelLocations //TODO fix
  __global__ void initializeSubPixels(unsigned long numMatches, Match* matches, SubpixelM7x7* subPixelDescriptors,
    Image_Descriptor query, unsigned long numFeaturesQuery, Feature<SIFT_Descriptor>* featuresQuery,
    Image_Descriptor target, unsigned long numFeaturesTarget, Feature<SIFT_Descriptor>* featuresTarget);

  __global__ void fillSplines(unsigned long numMatches, SubpixelM7x7* subPixelDescriptors, Spline* splines);
  __global__ void determineSubPixelLocationsBruteForce(float increment, unsigned long numMatches, Match* matches, Spline* splines);

  __global__ void refineWCutoffRatio(unsigned long numMatches, Match* matches, int* matchCounter, float2 minMax, float cutoffRatio);
  __global__ void copyMatches(unsigned long numMatches, int* matchCounter, Match* minimizedMatches, Match* matches);

}


#endif /* MATCHFACTORY_CUH */
