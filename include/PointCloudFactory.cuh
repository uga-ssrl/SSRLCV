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
  * \defgroup pointcloud
  * \{
  */

  /**
   * \brief A structure to aid in indexing lines and vectors in R3, additionally contains a bit for help removing bad indexes
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
    // if the bundle is invald, then it should be removed in a bundle adjustment
    bool invalid;
  };


  /**
   * \brief a set of lines in point vector format, and indexes (stored as bundles) that represent bundled lines for reprojection
   */
  struct BundleSet{
    Unity<Bundle::Line>* lines;
    Unity<Bundle>* bundles;
  };

  /**
   * \brief basically the same as the line but for camera adjustments in a view view bundle adjustment
   */
   struct CamAdjust2{
     float3 cam_pos0;
     float3 cam_rot0;
     float3 cam_pos1;
     float3 cam_rot1;
   };

  /**
  * \brief This class contains methods to generate point clouds from a set of Match structs.
  * \param Array of Matches
  */
  class PointCloudFactory {

  public:
  	PointCloudFactory();

    // =============================================================================================================
    //
    // Stereo Disparity Methods
    //
    // =============================================================================================================

    // stereo with auto cacluated scalar from camera params
    ssrlcv::Unity<float3>* stereo_disparity(Unity<Match>* matches, Image::Camera* cameras);

    // setero with scalar pass thru
    ssrlcv::Unity<float3>* stereo_disparity(Unity<Match>* matches, float scale);

    ssrlcv::Unity<float3>* stereo_disparity(Unity<Match>* matches, float foc, float baseline, float doffset);

    // =============================================================================================================
    //
    // 2 View Methods
    //
    // =============================================================================================================

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
    * The CPU method that sets up the GPU enabled two view tringulation.
    * This method uses the extra bit in the float3 data structure as a "filter" bit which can be used to remove bad points
    * @param bundleSet a set of lines and bundles that should be triangulated
    * @param the individual linear errors (for use in debugging and histogram)
    * @param linearError is the total linear error of the triangulation, it is an analog for reprojection error
    * @param linearErrorCutoff is a value that all linear errors should be less than. points with larger errors are discarded.
    */
    ssrlcv::Unity<float3_b>* twoViewTriangulate_b(BundleSet bundleSet, Unity<float>* errors, float* linearError, float* linearErrorCutoff);

    /**
     * Same method as two view triangulation, but all that is desired fro this method is a calculation of the linearError
     * @param bundleSet a set of lines and bundles that should be triangulated
     * @param the individual linear errors (for use in debugging and histogram)
     * @param linearError is the total linear error of the triangulation, it is an analog for reprojection error
     * @param linearErrorCutoff is a value that all linear errors should be less than. points with larger errors are discarded.
     */
    void voidTwoViewTriangulate(BundleSet bundleSet, float* linearError, float* linearErrorCutof);

    // =============================================================================================================
    //
    // N View Methods
    //
    // =============================================================================================================

    /**
     * The CPU method that sets up the GPU enabled n view triangulation.
     * @param bundleSet a set of lines and bundles to be triangulated
     */
    ssrlcv::Unity<float3>* nViewTriangulate(BundleSet bundleSet);

    /**
     * The CPU method that sets up the GPU enabled n view triangulation.
     * @param bundleSet a set of lines and bundles to be triangulated
     * @param angularError the total diff between vectors
     */
    ssrlcv::Unity<float3>* nViewTriangulate(BundleSet bundleSet, float* angularError);

    /**
     * The CPU method that sets up the GPU enabled n view triangulation.
     * @param bundleSet a set of lines and bundles to be triangulated
     * @param errors the individual angular errors per point
     * @param angularError the total diff between vectors
     */
    ssrlcv::Unity<float3>* nViewTriangulate(BundleSet bundleSet, Unity<float>* errors, float* angularError);

    /**
     * The CPU method that sets up the GPU enabled n view triangulation.
     * @param bundleSet a set of lines and bundles to be triangulated
     * @param errors the individual angular errors per point
     * @param angularError the total diff between vectors
     * @param angularErrorCutoff generated points that have an error over this cutoff are marked invalid
     */
    ssrlcv::Unity<float3>* nViewTriangulate(BundleSet bundleSet, Unity<float>* errors, float* angularError, float* angularErrorCutoff);

    // =============================================================================================================
    //
    // Bundle Adjustment Methods
    //
    // =============================================================================================================

    /**
    * The CPU method that sets up the GPU enabled line generation, which stores lines
    * and sets of lines as bundles
    * @param matchSet a group of maches
    * @param a group of images, used only for their stored camera parameters
    */
    BundleSet generateBundles(MatchSet* matchSet, std::vector<ssrlcv::Image*> images);


    /**
     * A Naive bundle adjustment based on a two-view triangulation and a first order descrete gradient decent
     * @param matchSet a group of matches
     * @param a group of images, used only for their stored camera parameters
     * @return a bundle adjusted point cloud
     */
    ssrlcv::Unity<float3>* BundleAdjustTwoView(MatchSet* matchSet, std::vector<ssrlcv::Image*> images);

    // =============================================================================================================
    //
    // Debug Methods
    //
    // =============================================================================================================

    /**
     * Saves a point cloud as a PLY while also saving cameras and projected points of those cameras
     * all as points in R3. Each is color coded RED for the cameras, GREEN for the point cloud, and
     * BLUE for the reprojected points.
     * @param pointCloud a Unity float3 that represents the point cloud itself
     * @param bundleSet is a BundleSet that contains lines and points to be drawn in front of the cameras
     * @param images a vector of images that contain value camera information
     */
    void saveDebugCloud(Unity<float3>* pointCloud, BundleSet bundleSet, std::vector<ssrlcv::Image*> images);

    /**
     * Saves a point cloud as a PLY while also saving cameras and projected points of those cameras
     * all as points in R3. Each is color coded RED for the cameras, GREEN for the point cloud, and
     * BLUE for the reprojected points.
     * @param pointCloud a Unity float3 that represents the point cloud itself
     * @param bundleSet is a BundleSet that contains lines and points to be drawn in front of the cameras
     * @param images a vector of images that contain value camera information
     * @param fineName a filename for the debug cloud
     */
    void saveDebugCloud(Unity<float3>* pointCloud, BundleSet bundleSet, std::vector<ssrlcv::Image*> images, std::string filename);

    /**
     * Saves a point cloud as a PLY while also saving cameras and projected points of those cameras
     * all as points in R3. Each is color coded RED for the cameras, GREEN for the point cloud, and
     * BLUE for the reprojected points.
     * @param pointCloud a Unity float3 that represents the point cloud itself
     * @param bundleSet is a BundleSet that contains lines and points to be drawn in front of the cameras
     * @param images a vector of images that contain value camera information
     * @param fineName a filename for the debug cloud
     */
    void saveDebugCloud(Unity<float3>* pointCloud, BundleSet bundleSet, std::vector<ssrlcv::Image*> images, const char* filename);

    /**
     * Saves a colored point cloud where the colors correspond do the linear errors from within the cloud.
     * @param matchSet a group of matches
     * @param images a group of images, used only for their stored camera parameters
     * @param filename the name of the file that should be saved
     */
    void saveDebugLinearErrorCloud(ssrlcv::MatchSet* matchSet, std::vector<ssrlcv::Image*> images, const char* filename);

    /**
     * Saves a colored point cloud where the colors correspond to the number of images matched in each color
     * @param matchSet a group of matches
     * @param images a group of images, used only for their stored camera parameters
     * @param filename the name of the file that should be saved
     */
    void saveViewNumberCloud(ssrlcv::MatchSet* matchSet, std::vector<ssrlcv::Image*> images, const char* filename);

    /**
     * Saves several CSV's which have (x,y) coordinates representing step the step from an intial condution and
     * the output error for that condition, this should be graphed
     * @param matchSet a group of matches
     * @param images a group of images, used only for their stored camera parameters
     * @param filename the name of the file that should be saved
     */
    void generateSensitivityFunctions(ssrlcv::MatchSet* matchSet, std::vector<ssrlcv::Image*> images, std::string filename);

    // =============================================================================================================
    //
    // Filtering Methods
    //
    // =============================================================================================================

    /**
     * Deterministically filters, with the assumption that the data is guassian, statistical outliers of the pointcloud
     * set and returns a matchSet without such outliers. The method is deterministic by taking a uniformly spaced sample of points
     * within the matcheSet.
     * @param matchSet a group of matches
     * @param iamges a group of images, used only for their stored camera parameters
     * @param sigma is the variance to cutoff from
     * @param sampleSize represents a percentage and should be between 0.0 and 1.0
     */
    void deterministicStatisticalFilter(ssrlcv::MatchSet* matchSet, std::vector<ssrlcv::Image*> images, float sigma, float sampleSize);

    /**
     * NonDeterministically filters, with the assumption that the data is guassian, statistical outliers of the pointcloud
     * set and returns a matchSet without such outliers. It is the same as the deterministicStatisticalFilter only samples
     * are chosen randomly rather than equally spaced.
     * @param matchSet a group of matches
     * @param images a group of images, used only for their stored camera parameters
     * @param sigma is the variance to cutoff from
     * @param sampleSize represents a percentage and should be between 0.0 and 1.0
     */
    void nonDeterministicStatisticalFilter(ssrlcv::MatchSet* matchSet, std::vector<ssrlcv::Image*> images, float sigma, float sampleSize);

    /**
     * A filter that removes all points with a linear error greater than the cutoff. Modifies the matchSet that is pass thru
     * @param matchSet a group of matches
     * @param images a group of images, used only for their stored camera parameters
     * @param cutoff the float that no linear errors should be greater than
     */
    void linearCutoffFilter(ssrlcv::MatchSet* matchSet, std::vector<ssrlcv::Image*> images, float cutoff);

  };

  // =======
  // MISC.
  // =======

  uchar3 heatMap(float value);

  void writeDisparityImage(Unity<float3>* points, unsigned int interpolationRadius, std::string pathToFile);

  /**
  * \ingroup pointcloud
  * \ingroup cuda_kernels
  * \defgroup pointcloud_kernels
  * \{
  */

  // =============================================================================================================
  //
  // Bundle Adjustment Kernels
  //
  // =============================================================================================================

  __global__ void generateBundle(unsigned int numBundles, Bundle* bundles, Bundle::Line* lines, MultiMatch* matches, KeyPoint* keyPoints, Image::Camera* cameras);

  // =============================================================================================================
  //
  // Stereo Kernels
  //
  // =============================================================================================================
  __global__ void computeStereo(unsigned int numMatches, Match* matches, float3* points, float foc, float baseLine, float doffset);

  __global__ void computeStereo(unsigned int numMatches, Match* matches, float3* points, float scale);

  __global__ void interpolateDepth(uint2 disparityMapSize, int influenceRadius, float* disparities, float* interpolated);

  // =============================================================================================================
  //
  // 2 View Kernels
  //
  // =============================================================================================================

  __global__ void computeTwoViewTriangulate(float* linearError, unsigned long pointnum, Bundle::Line* lines, Bundle* bundles, float3* pointcloud);

  __global__ void computeTwoViewTriangulate(float* linearError, float* errors, unsigned long pointnum, Bundle::Line* lines, Bundle* bundles, float3* pointcloud);

  __global__ void computeTwoViewTriangulate(float* linearError, float* linearErrorCutoff, float* errors, unsigned long pointnum, Bundle::Line* lines, Bundle* bundles, float3* pointcloud);

  __global__ void computeTwoViewTriangulate_b(float* linearError, float* linearErrorCutoff, float* errors, unsigned long pointnum, Bundle::Line* lines, Bundle* bundles, float3_b* pointcloud);

  __global__ void voidComputeTwoViewTriangulate(float* linearError, float* linearErrorCutoff, unsigned long pointnum, Bundle::Line* lines, Bundle* bundles);

  __global__ void two_view_reproject(int numMatches, float4* matches, float cam1C[3],
  	float cam1V[3],float cam2C[3], float cam2V[3], float K_inv[9],
  	float rotationTranspose1[9], float rotationTranspose2[9], float3* points);

  // =============================================================================================================
  //
  // N View Kernels
  //
  // =============================================================================================================

    /**
    * the CUDA kernel for Nview triangulate
    */
  __global__ void computeNViewTriangulate(unsigned long pointnum, Bundle::Line* lines, Bundle* bundles, float3* pointcloud);

  /**
  * the CUDA kernel for Nview triangulation with angular error
  */
  __global__ void computeNViewTriangulate(float* angularError, unsigned long pointnum, Bundle::Line* lines, Bundle* bundles, float3* pointcloud);

  /**
  * the CUDA kernel for Nview triangulation with angular error
  */
  __global__ void computeNViewTriangulate(float* angularError, float* errors, unsigned long pointnum, Bundle::Line* lines, Bundle* bundles, float3* pointcloud);

  /**
  * the CUDA kernel for Nview triangulation with angular error
  */
  __global__ void computeNViewTriangulate(float* angularError, float* angularErrorCutoff, float* errors, unsigned long pointnum, Bundle::Line* lines, Bundle* bundles, float3* pointcloud);


}




#endif /* POINTCLOUDFACTORY_CUH */



















































// yeet
