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

    // =============================================================================================================
    //
    // Constructors and Destructors
    //
    // =============================================================================================================

    // default constructor
    MeshFactory();

    // default destructor
    ~MeshFactory();

    /**
     * An octree based constructor
     * @param octree an SSRLCV octree data structure storing a point cloud
     */
    MeshFactory(Octree* octree);

    // =============================================================================================================
    //
    // Mesh Loading & Saving Methods
    //
    // =============================================================================================================

    /**
     * loads a mesh from a file into
     * currently only ASCII encoded PLY files are supported
     * @param filePath the filepath, relative to the install location
     */
    void loadMesh(const char* filePath);

    /**
     * saves a PLY encoded Mesh as a given filename to the out directory
     * @param filename the filename
     */
    void saveMesh(const char* filename);

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
   * with the mesh along a single given vector fro all points. This is returned as an average
   */
  __global__ void averageCollisionDistance(float* averageDistance, unsigned long pointnum, float3* pointcloud, float3* vector, float3* vertices, int* faces, int* faceEncoding);

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
