/** \file PointCloudFactory.cuh
* \brief this file contains methods for deriving point clouds
*/
#ifndef POINTCLOUDFACTORY_CUH
#define POINTCLOUDFACTORY_CUH

#include "common_includes.h"
#include "Image.cuh"
#include "MatchFactory.cuh"
#include "Unity.cuh"


namespace ssrlcv{

   /**
    * \brief A structure to define a line my a vector and a point in R3
    */
  // struct Line {
  //   float3 vec; // vector in R3
  //   float3 pnt; // point in R3
  // };
  // /*
  //  * \brief A structure that contains lines, the collection of lines are considered a bindle and are generated from the same match set
  //  */
  // struct Bundle {
  //   Line* line;
  //   short n;
  // };

  struct Bundle{
    struct Line{
      float3 vec;
      float3 pnt;
    };
    unsigned int numLines;
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

    ssrlcv::Unity<float3>* stereo_disparity(Unity<Match>* matches, float scale);

  };



  __global__ void generateBundle(unsigned int numBundles, Bundle* bundles, Bundle::Line* lines, MultiMatch* matches, KeyPoint* keyPoints, Image::Camera* cameras);

  __global__ void computeStereo(unsigned int numMatches, Match* matches, float3* points, float scale);

  __global__ void two_view_reproject(int numMatches, float4* matches, float cam1C[3],
  	float cam1V[3],float cam2C[3], float cam2V[3], float K_inv[9],
  	float rotationTranspose1[9], float rotationTranspose2[9], float3* points);

}




#endif /* POINTCLOUDFACTORY_CUH */
