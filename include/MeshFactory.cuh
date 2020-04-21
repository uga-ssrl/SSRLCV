/** \file MeshFactory.cuh
* \brief this file contains all mesh generation methods
*/
#ifndef MESHFACTORY_CUH
#define MESHFACTORY_CUH

#include "common_includes.h"
#include "Image.cuh"
#include "Octree.cuh"
#include <thrust/scan.h>
#include <thrust/device_ptr.h>

namespace ssrlcv{

  /**
  * \brief defines a line in point vector format
  */
  struct Line{
    float3 vec;
    float3 pnt;
  };

  /**
  * \defgroup meshing
  * \{
  */
  /**
  * \brief Factory for generating meshes from PointClouds.
  */
  class MeshFactory{

  public:
    // Octree is used for reconstructon, minimal meshes need not use this
    Octree* octree;

    float3* surfaceVertices;
    int numSurfaceVertices;
    int3* surfaceTriangles;
    int numSurfaceTriangles;
    float* vertexImplicitDevice;

    /**
     * the face encoding is used to defined if the mesh is encoded with triangles or
     * encoded with quadrilaterals.
     * this value is 3 for triangles and 4 for quadrilaterals
     */
    int faceEncoding;
    // faces stored where ever (N = faceEncoding) are grouped
    Unity<int>* faces;

    // The points within the mesh
    Unity<float3>* points;

    // RBG colors for the point cloud
    Unity<uchar3>* colors;

    bool pointsSet = false;
    bool octreeSet = false;

    // =============================================================================================================
    //
    // Constructors and Destructors
    //
    // =============================================================================================================

    // default constructor
    MeshFactory();

    // constructor given existing points and faces
    MeshFactory(Unity<float3>* in_points, Unity<int>* in_faces, int in_faceEncoding);

    // default destructor
    ~MeshFactory();

    /**
     * An octree based constructor
     * @param octree an SSRLCV octree data structure storing a point cloud
     */
    MeshFactory(Octree* octree);

    // =============================================================================================================
    //
    // Mesh Setters, Getter, Loading, and Saving Methods
    //
    // =============================================================================================================

    /**
     * Loads in a point cloud into the mesh, this will override any existing point data
     * and should be used sparingly
     * @param pointcloud a unity of float3 that represents a point cloud to be set to internal points
     */
    void setPoints(Unity<float3>* pointcloud);

    /**
     * Loads faces into the mesh, this will override any existing face data
     * and should be used sparingly
     * @param faces a unity of int that represents the indexes of points which make faces
     * @param faceEncoding the face encoding scheme 3 or 4
     */
    void setFaces(Unity<int>* faces, int faceEncoding);

    /**
     * loads a mesh from a file into
     * currently only ASCII encoded PLY files are supported
     * @param filePath the filepath, relative to the install location
     */
    void loadMesh(const char* filePath);

    /**
    * loads points from an ASCII encoded PLY file into the mesh
    * overloads existing points
    * @param filePath the filepath, relative to the install location
    */
    void loadPoints(const char* filePath);

    /**
     * saves a PLY encoded Mesh as a given filename to the out directory
     * @param filename the filename
     */
    void saveMesh(const char* filename);

    /**
     * saves only the points as a PLY
     * @param filename the filename
     */
    void savePoints(const char* filename);

    // =============================================================================================================
    //
    // Comparison and Error methods
    //
    // =============================================================================================================

    /**
     * Assuming that a point cloud and the mesh are alligned in the same plane, this method takes each point of the
     * input pointcloud and calculates the distance purpendicular to the plane they are both in. That discance can be
     * thought of as the "error" between that point and the mesh. This method caclculates the average error between
     * a mesh and a point cloud
     * @param pointCloud the input point cloud to compare to the mesh
     * @param planeNormal a float3 representing a vector normal to the shared plane of the point cloud and mesh
     * @return averageError this is number is a float that is always positive or 0.0f, it is -1.0f if an error has occured
     */
    float calculateAverageDifference(Unity<float3>* pointCloud, float3 planeNormal);

    /**
     * Assuming that a point cloud and the mesh are alligned in the same plane, this method takes each point of the
     * input pointcloud and calculates the distance purpendicular to the plane they are both in. That discance can be
     * thought of as the "error" between that point and the mesh. This method caclculates the error between
     * a mesh and a point cloud for each point and returns it
     * @param pointCloud the input point cloud to compare to the mesh
     * @param planeNormal a float3 representing a vector normal to the shared plane of the point cloud and mesh
     * @return errorList a unity array of floats that contain errors
     */
    ssrlcv::Unity<float>* calculatePerPointDifference(Unity<float3>* pointCloud, float3 planeNormal);

    // =============================================================================================================
    //
    // Filtering Methods
    //
    // =============================================================================================================

    /**
     * caclualtes the average distance to N neightbors for each points
     * @param n the number of neignbors to calculate an average distance to
     * @return float a unity of floats representing the average distance to N neighbors
     */
    ssrlcv::Unity<float>* calculateAverageDistancesToOctreeNeighbors(int n);

    /**
     * caclualtes the average distance to N neightbors for each point on average
     * @param n the number of neignbors to calculate an average distance to
     * @return float which is the average distance to n neighbors
     */
    float calculateAverageDistanceToOctreeNeighbors(int n);

    /**
     * filters points from the mesh by caclulating their average distances to their neighbors
     * and then calculating the variance of the data, and removing points past sigma
     * @param sigma the statistical value to remove points after
     */
    void filterByOctreeNeighborDistance(float sigma);

    /**
     * caclualtes the average distance to N neightbors for each points
     * @param n the number of neignbors to calculate an average distance to
     * @return float a unity of floats representing the average distance to N neighbors
     */
    ssrlcv::Unity<float>* calculateAverageDistancesToNeighbors(int n);

    /**
     * caclualtes the average distance to N neightbors for each point on average
     * @param n the number of neignbors to calculate an average distance to
     * @return float which is the average distance to n neighbors
     */
    float calculateAverageDistanceToNeighbors(int n);

    /**
     * filters points from the mesh by caclulating their average distances to their neighbors
     * and then calculating the variance of the data, and removing points past sigma
     * @param sigma the statistical value to remove points after
     */
    void filterByNeighborDistance(float sigma);

    // =============================================================================================================
    //
    // Other MeshFactory Methods
    //
    // =============================================================================================================

    void computeVertexImplicitJAX(int focusDepth);
    void adaptiveMarchingCubes();
    void marchingCubes();

    // =============================================================================================================
    //
    // Mesh Generation Methods
    //
    // =============================================================================================================

    void jaxMeshing();
    void generateMesh(bool binary);
    void generateMesh();
    void generateMeshWithFinestEdges();

  };
  /**
  * \}
  */
  /* CUDA variable, method and kernel defintions */

  // =============================================================================================================
  //
  // Device Kernels
  //
  // =============================================================================================================

  namespace{
    struct is_not_neg_int{
      __host__ __device__
      bool operator()(const int x)
      {
        return (x >= 0);
      }
    };
    struct is_not_zero_float{
      __host__ __device__
      bool operator()(const float x)
      {
        return (x != 0.0f);
      }
    };
  }


  extern __constant__ int cubeCategoryTrianglesFromEdges[256][15];
  extern __constant__ int cubeCategoryEdgeIdentity[256];
  extern __constant__ int numTrianglesInCubeCategory[256];



  __device__ __host__ float3 blenderPrime(const float3 &a, const float3 &b, const float &bw);
  __device__ __host__ float3 blenderPrimePrime(const float3 &a, const float3 &b, const float &bw);

  /**
  * \ingroup meshing
  * \ingroup cuda_kernels
  * \defgroup meshing_kernels
  * \{
  */

  /**
   * this measures the distance between each point in a point cloud and where they "collide"
   * with the mesh along a single given vector fro all points. This is returned as a sum
   */
  __global__ void sumCollisionDistance(float* averageDistance, int* misses, unsigned long pointnum, float3* pointcloud, float3* vector, float3* vertices, unsigned long facenum, int* faces, int* faceEncoding);

  /**
   * Measures individual collision distances between each point in the point cloud and the mesh
   * and returns those distances in the errors unity
   */
  __global__ void generateCollisionDistances(float* errors, int* misses, unsigned long pointnum, float3* pointcloud, float3* vector, float3* vertices, unsigned long facenum, int* faces, int* faceEncoding);

  /**
   * exaustivley caclulates the average distance to N nearest neightbors
   */
  __global__ void averageDistToNeighbors(int * d_num, unsigned long pointnum, float3* points, float* averages);

  __global__ void vertexImplicitFromNormals(int numVertices, Octree::Vertex* vertexArray, Octree::Node* nodeArray, float3* normals, float3* points, float* vertexImplicit);
  __global__ void calcVertexNumbers(int numEdges, int depthIndex, Octree::Edge* edgeArray, float* vertexImplicit, int* vertexNumbers);

  //adaptive marching cubes
  __global__ void categorizeCubesRecursively_child(int parent, int parentCategory, Octree::Edge* edgeArray, Octree::Node* nodeArray, int* vertexNumbers, int* cubeCategory, int* triangleNumbers);
  __global__ void categorizeCubesRecursively(int firstChildrenIndex, Octree::Edge* edgeArray, Octree::Node* nodeArray, int* vertexNumbers, int* cubeCategory, int* triangleNumbers);
  __global__ void minimizeVertices(int numEdges, Octree::Edge* edgeArray, Octree::Node* nodeArray, int* cubeCategory, int* vertexNumbers);

  //marching cubes
  __global__ void determineCubeCategories(int numNodes, int nodeIndex, int edgeIndex, Octree::Node* nodeArray, int* vertexNumbers, int* cubeCategory, int* triangleNumbers);
  __global__ void generateSurfaceVertices(int numEdges, int depthIndex, Octree::Edge* edgeArray, Octree::Vertex* vertexArray, int* vertexNumbers, int* vertexAddresses, float3* surfaceVertices);
  __global__ void generateSurfaceTriangles(int numNodes, int nodeIndex, int edgeIndex, Octree::Node* nodeArray, int* vertexAddresses, int* triangleAddresses, int* cubeCategory, int3* surfaceTriangles);
  /**
  * \}
  */
}


#endif /* MESHFACTORY_CUH */
