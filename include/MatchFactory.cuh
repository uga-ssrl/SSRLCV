#ifndef MATCHFACTORY_CUH
#define MATCHFACTORY_CUH

#include "common_includes.h"
#include "Image.cuh"
#include "Feature.cuh"
#include <thrust/device_ptr.h>
#include <thrust/copy.h>
#include <thrust/scan.h>


extern __constant__ float matchTreshold;
extern __constant__ int splineHelper[4][4];
extern __constant__ int splineHelperInv[4][4];

//TODO make it so that this can be feature classes not just sift

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
  SIFT_Feature features[2];
  float distance[2];
};
typedef struct Match Match;

struct match_above_cutoff{
  __host__ __device__
  bool operator()(Match m){
    return m.distance[0] > 0.0f;
  }
};

struct SubPixelMatch : public Match{
  float2 subLocations[2];
};
typedef struct SubPixelMatch SubPixelMatch;


//WARNING MATCHSET CANNOT HAVE MemoryState of both RIGHT NOW

struct MatchSet{
  Match* matches;
  int numMatches;
  MemoryState memoryState;
};
typedef struct MatchSet MatchSet;

struct SubPixelMatchSet{
  SubPixelMatch* matches;
  int numMatches;
  MemoryState memoryState;
};
typedef struct SubPixelMatchSet SubPixelMatchSet;


__device__ __host__ __forceinline__ float sum(const float3 &a);
__device__ __forceinline__ float square(const float &a);
__device__ __forceinline__ float calcElucid(const int2 &a, const int2 &b);
__device__ __forceinline__ float calcElucid_SIFTDescriptor(const unsigned char a[128], const unsigned char b[128]);
__device__ __forceinline__ float atomicMinFloat (float * addr, float value);
__device__ __forceinline__ float atomicMaxFloat (float * addr, float value);
/*
Pairwise stuff
*/
__global__ void matchFeaturesPairwiseBruteForce(int numFeaturesQuery, int numOrientationsQuery,
int numFeaturesTarget, int numOrientationsTarget, SIFT_Descriptor* descriptorsQuery, SIFT_Feature* featuresQuery,
SIFT_Descriptor* descriptorsTarget, SIFT_Feature* featuresTarget, Match* matches);

__global__ void matchFeaturesPairwiseConstrained(int numFeaturesQuery, int numOrientationsQuery,
int numFeaturesTarget, int numOrientationsTarget, SIFT_Descriptor* descriptorsQuery, SIFT_Feature* featuresQuery,
SIFT_Descriptor* descriptorsTarget, SIFT_Feature* featuresTarget, Match* matches, float epsilon, float3 fundamental[3]);

  //pairwise subPixelLocations //TODO fix
  __global__ void initializeSubPixels(Image_Descriptor query, Image_Descriptor target, unsigned long numMatches, Match* matches, SubPixelMatch* subPixelMatches, SubpixelM7x7* subPixelDescriptors,
  SIFT_Descriptor* queryDescriptors, int numFeaturesQuery, int numDescriptorsPerFeatureQuery, SIFT_Descriptor* targetDescriptors, int numFeaturesTarget, int numDescriptorsPerFeatureTarget);
  __global__ void fillSplines(unsigned long numMatches, SubpixelM7x7* subPixelDescriptors, Spline* splines);
  __global__ void determineSubPixelLocationsBruteForce(float increment, unsigned long numMatches, SubPixelMatch* subPixelMatches, Spline* splines);


__global__ void refineWCutoffRatio(int numMatches, Match* matches, int* matchCounter, float2 minMax, float cutoffRatio);


/*
Funundamental matrix stuff
*/
float3 multiply3x3x1(const float3 A[3], const float3 &B);
void multiply3x3(const float3 A[3], const float3 B[3], float3 *C);
void transpose3x3(const float3 M[3], float3 (&M_T)[3]);
void inverse3x3(float3 M[3], float3 (&Minv)[3]);
void calcFundamentalMatrix_2View(Image_Descriptor query, Image_Descriptor target, float3 *F);

class MatchFactory{

protected:
  unsigned long totalFeatures;

public:

  float cutoffRatio;
  int numImages;

  MatchFactory();

  //NOTE nothing for nview is implemented
  //TODO consider making it so features are computed if they arent instead of throwing errors with image parameters

  void setCutOffRatio(float cutoffRatio);

  void refineMatches(MatchSet* &matchSet);


  void generateMatchesPairwiseBruteForce(Image* query, Image* target, MatchSet* &matchSet, MemoryState return_state);
  void generateMatchesPairwiseConstrained(Image* query, Image* target, float epsilon, MatchSet* &matchSet, MemoryState return_state);
  //NOTE currently brute force
  void generateSubPixelMatchesPairwiseBruteForce(Image* query, Image* target, SubPixelMatchSet* &matchSet, MemoryState return_state);
  void generateSubPixelMatchesPairwiseConstrained(Image* query, Image* target, float epsilon, SubPixelMatchSet* &matchSet, MemoryState return_state);


};

#endif /* MATCHFACTORY_CUH */
