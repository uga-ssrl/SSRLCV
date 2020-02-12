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

    /**
    * The CPU method that sets up the GPU enabled two view tringulation.
    * @param bundleSet a set of lines and bundles that should be triangulated
    * @param linearError is the total linear error of the triangulation, it is an analog for reprojection error
    */
    ssrlcv::Unity<float3>* twoViewTriangulate(BundleSet bundleSet, float* linearError);

    /**
    * The CPU method that sets up the GPU enabled two view tringulation.
    * @param bundleSet a set of lines and bundles that should be triangulated
    * @param the individual linear errors (for use in debugging and histogram)
    * @param linearError is the total linear error of the triangulation, it is an analog for reprojection error
    */
    ssrlcv::Unity<float3>* twoViewTriangulate(BundleSet bundleSet, Unity<float>* errors, float* linearError);

    /**
    * The CPU method that sets up the GPU enabled two view tringulation.
    * @param bundleSet a set of lines and bundles that should be triangulated
    * @param the individual linear errors (for use in debugging and histogram)
    * @param linearError is the total linear error of the triangulation, it is an analog for reprojection error
    * @param linearErrorCutoff is a value that all linear errors should be less than. points with larger errors are discarded.
    */
    ssrlcv::Unity<float3>* twoViewTriangulate(BundleSet bundleSet, Unity<float>* errors, float* linearError, float* linearErrorCutoff);

    /**
     * Same method as two view triangulation, but all that is desired fro this method is a calculation of the linearError
     * @param bundleSet a set of lines and bundles that should be triangulated
     * @param the individual linear errors (for use in debugging and histogram)
     * @param linearError is the total linear error of the triangulation, it is an analog for reprojection error
     * @param linearErrorCutoff is a value that all linear errors should be less than. points with larger errors are discarded.
     */
    void voidTwoViewTriangulate(BundleSet bundleSet, float* linearError, float* linearErrorCutof);

    /**
     * A Naive bundle adjustment based on a two-view triangulation and a first order descrete gradient decent
     * @param matchSet a group of matches
     * @param a group of images, used only for their stored camera parameters
     * @return a bundle adjusted point cloud
     */
    ssrlcv::Unity<float3>* BundleAdjustTwoView(MatchSet* matchSet, std::vector<ssrlcv::Image*> images);

  };

  uchar3 heatMap(float value);

  void writeDisparityImage(Unity<float3>* points, unsigned int interpolationRadius, std::string pathToFile);

  __global__ void generateBundle(unsigned int numBundles, Bundle* bundles, Bundle::Line* lines, MultiMatch* matches, KeyPoint* keyPoints, Image::Camera* cameras);

  __global__ void computeStereo(unsigned int numMatches, Match* matches, float3* points, float foc, float baseLine, float doffset);

  __global__ void computeStereo(unsigned int numMatches, Match* matches, float3* points, float scale);

  __global__ void interpolateDepth(uint2 disparityMapSize, int influenceRadius, float* disparities, float* interpolated);

  __global__ void computeTwoViewTriangulate(float* linearError, unsigned long pointnum, Bundle::Line* lines, Bundle* bundles, float3* pointcloud);

  __global__ void computeTwoViewTriangulate(float* linearError, float* errors, unsigned long pointnum, Bundle::Line* lines, Bundle* bundles, float3* pointcloud);

  __global__ void computeTwoViewTriangulate(float* linearError, float* linearErrorCutoff, float* errors, unsigned long pointnum, Bundle::Line* lines, Bundle* bundles, float3* pointcloud);

  __global__ void voidComputeTwoViewTriangulate(float* linearError, float* linearErrorCutoff, unsigned long pointnum, Bundle::Line* lines, Bundle* bundles);

  __global__ void two_view_reproject(int numMatches, float4* matches, float cam1C[3],
  	float cam1V[3],float cam2C[3], float cam2V[3], float K_inv[9],
  	float rotationTranspose1[9], float rotationTranspose2[9], float3* points);

}




#endif /* POINTCLOUDFACTORY_CUH */



















































// yeet
