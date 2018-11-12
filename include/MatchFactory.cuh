#ifndef MATCHFACTORY_CUH
#define MATCHFACTORY_CUH

#include "common_includes.h"
#include "Image.cuh"
#include "Feature.cuh"


extern __constant__ float matchTreshold;
extern __constant__ int splineHelper[4][4];
extern __constant__ int splineHelperInv[4][4];

//TODO make it so that this can be feature classes not just sift

struct Spline{
  float coeff[6][6][4][4];
};

struct SubpixelM7x7{
  float M1[9][9];
  float M2[9][9];
};

struct Match{
  SIFT_Feature features[2];
  float distance[2];
};
struct SubPixelMatch : public Match{
  float2 subLocations[2];
};


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
  __global__ void initializeSubPixels(unsigned long numMatches, Match* matches, SubPixelMatch* subPixelMatches, SubpixelM7x7* subPixelDescriptors,
  SIFT_Descriptor* queryDescriptors, int numFeaturesQuery, int numDescriptorsPerFeatureQuery, SIFT_Descriptor* targetDescriptors, int numFeaturesTarget, int numDescriptorsPerFeatureTarget);

  __global__ void fillSplines(unsigned long numMatches, SubpixelM7x7* subPixelDescriptors, Spline* splines);
  __global__ void determineSubPixelLocationsBruteForce(float increment, unsigned long numMatches, SubPixelMatch* subPixelMatches, Spline* splines);

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

  int numImages;

  MatchFactory();

  //NOTE nothing for nview is implemented
  //TODO consider making it so features are computed if they arent instead of throwing errors with image parameters

  Match* generateMatchesPairwiseBruteForce(Image* query, Image* target, MemoryState return_state);
  Match* generateMatchesPairwiseConstrained(Image* query, Image* target, float epsilon, MemoryState return_state);
  //NOTE currently brute force
  SubPixelMatch* generateSubPixelMatchesPairwiseBruteForce(Image* query, Image* target, MemoryState return_state);
  SubPixelMatch* generateSubPixelMatchesPairwiseConstrained(Image* query, Image* target, float epsilon, MemoryState return_state);


};

#endif /* MATCHFACTORY_CUH */
