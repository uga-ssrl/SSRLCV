/** \file PointCloudFactory.cuh
* \brief this file contains methods for deriving point clouds
*/
#ifndef POINTCLOUDFACTORY_CUH
#define POINTCLOUDFACTORY_CUH

#include "common_includes.h"
#include "Image.cuh"
#include "MatchFactory.cuh"
#include "Unity.cuh"
#include "io_util.h"


namespace ssrlcv{

   /**
    * \brief A structure to define a line my a vector and a point in R3
    */
  struct Bundle{
    /**
     * \brief A line in R3 point vector format
     */
    struct Line{
      float3 vec;
      float3 pnt;
    };
    // The number of lines
    unsigned int numLines;
    // the index of a single line
    int index;
  };

  struct BundleSet{
    Unity<Bundle::Line>* lines;
    Unity<Bundle>* bundles;
  };



  /**
  * \brief This class contains methods to generate point clouds from a set of Match structs.
  * \param Array of Matches
  */
  class PointCloudFactory {

  public:
  	PointCloudFactory();

    // this is not a good name anymore
    //Unity<float3>* reproject(Unity<Match>* matches, Image* target, Image* query);

    BundleSet generateBundles(MatchSet* matchSet, std::vector<ssrlcv::Image*> images);

    // stereo with auto cacluated scalar from camera params
    ssrlcv::Unity<float3>* stereo_disparity(Unity<Match>* matches, Image::Camera* cameras);

    // setero with scalar pass thru
    ssrlcv::Unity<float3>* stereo_disparity(Unity<Match>* matches, float scale);

    ssrlcv::Unity<float3>* stereo_disparity(Unity<Match>* matches, float foc, float baseline, float doffset);


    ssrlcv::Unity<float3>* twoViewTriangulate(BundleSet bunlesSet);

  };

  uchar3 heatMap(float value);

  void writeDisparityImage(Unity<float3>* points, unsigned int interpolationRadius, std::string pathToFile);

  __global__ void generateBundle(unsigned int numBundles, Bundle* bundles, Bundle::Line* lines, MultiMatch* matches, KeyPoint* keyPoints, Image::Camera* cameras);

  __global__ void computeStereo(unsigned int numMatches, Match* matches, float3* points, float foc, float baseLine, float doffset);

  __global__ void computeStereo(unsigned int numMatches, Match* matches, float3* points, float scale);

  __global__ void interpolateDepth(uint2 disparityMapSize, int influenceRadius, float* disparities, float* interpolated);

  __global__ void computeTwoViewTriangulate();

  __global__ void two_view_reproject(int numMatches, float4* matches, float cam1C[3],
  	float cam1V[3],float cam2C[3], float cam2V[3], float K_inv[9],
  	float rotationTranspose1[9], float rotationTranspose2[9], float3* points);

}




#endif /* POINTCLOUDFACTORY_CUH */



















































// yeet
