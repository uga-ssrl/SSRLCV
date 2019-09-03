/** \file Quadtree.cuh
 * \brief File contains all things related to CUDA Quadtree.
*/
#ifndef QUADTREE_CUH
#define QUADTREE_CUH

#include "common_includes.h"
#include "Unity.cuh"
#include <thrust/sort.h>
#include <thrust/pair.h>
#include <thrust/unique.h>
#include <thrust/gather.h>
#include <thrust/iterator/counting_iterator.h>
#include <thrust/device_vector.h>
#include <thrust/scan.h>
#include <thrust/device_ptr.h>
#include <thrust/copy.h>

namespace ssrlcv{

  /**
   * \brief simple struct for holding float2 as location and
   * data of anytype.
   * \todo find better place for this struct
  */
  template<typename D>
  struct LocalizedData{
    float2 loc;
    D data;
  };

  //TODO make Quadtree exceptions and add to a namespace
  //TODO consider allowing quadtrees that are not square
  //TODO evaluate use of unsigned int or long for index holders


  /**
  * \brief Class that can hold the following classes:
  * float2, LocalizedData<T>, unsigned int, unsigned char
  */
  template<typename T>
  class Quadtree{
    unsigned int colorDepth;

    void generateLeafNodes();
    void generateParentNodes();
    void fillNeighborhoods();


  public:
    /**
    * \brief Quadtree<T>::Node class for GPU and CPU utilization
    */
    struct Node{
      int key;
      int dataIndex;
      int numElements;
      float2 center;
      int depth;//maybe remove as its derivable or change to char | can be used to calc width
      int parent;
      int children[4];
      int neighbors[9];
      int edges[4];
      int vertices[4];
      bool flag;

      __device__ __host__ Node();
    };
    /**
    * \brief Quadtree<T>::Vertex class for GPU and CPU utilization
    */
    struct Vertex{
      float2 loc;
      int nodes[4];
      int depth;
      __device__ __host__ Vertex();
    };
    /**
    * \brief Quadtree<T>::Edge class for GPU and CPU utilization
    */
    struct Edge{
      int2 vertices;
      int nodes[2];
      int depth;
      __device__ __host__ Edge();
    };

    uint2 size;
    int2 border;
    unsigned int depth;


    ssrlcv::Unity<T>* data;
    ssrlcv::Unity<Node>* nodes;
    ssrlcv::Unity<Vertex>* vertices;
    ssrlcv::Unity<Edge>* edges;

    ssrlcv::Unity<unsigned int>* dataNodeIndex;

    //TODO change these to strict host variables so no need for unity
    ssrlcv::Unity<unsigned int>* nodeDepthIndex;
    ssrlcv::Unity<unsigned int>* vertexDepthIndex;
    ssrlcv::Unity<unsigned int>* edgeDepthIndex;

    Quadtree();

    /**
    * \brief Constructor for full quadtrees holding indices to each pixel
    * \param size size of image {x,y}
    * \param depth depth of quadtree
    * \param border size of border (optional parameter)
    */
    Quadtree(uint2 size, unsigned int depth, int2 border = {0,0});
    /**
    * \brief Constructor for full quadtrees holding actual data
    * \param size size of image {x,y}
    * \param depth depth of quadtree
    * \param data of type T
    * \param border size of border (optional parameter)
    */
    Quadtree(uint2 size, unsigned int depth, ssrlcv::Unity<T>* data, unsigned int colorDepth = 0, int2 border = {0,0});
    //generally not necessary and takes up a lot of memory - useful for testing small scale
    /**
    * \brief Method to fill Quadtree Vertex array.
    */
    void generateVertices();
    /**
    * \brief Method to fill Quadtree Edge array.
    */
    void generateEdges();
    /**
    * \brief Method to fill Quadtree Edge and Vertex arrays.
    */
    void generateVerticesAndEdges();

    /**
    * \brief Method to set Quadtree::Node flags based on a hashmap.
    * \param hashMap Unity struct with bool array representing hashmap
    * \param requireFullNeighbors if true then border pixels have flag set to false
    * \param depthRange used to specify depths of quadtree evaluated in method
    */
    void setNodeFlags(Unity<bool>* hashMap, bool requireFullNeighbors = false, uint2 depthRange = {0,0});
    /**
    * \brief Method to set Quadtree::Node flags based on a border.
    * \param flagBorder border where flags will be set to false
    * \param requireFullNeighbors if true then border pixels have flag set to false
    * \param depthRange used to specify depths of quadtree evaluated in method
    */
    void setNodeFlags(float2 flagBorder, bool requireFullNeighbors = false, uint2 depthRange = {0,0});

    /**
    * \brief Method to print ply of image with Quadtree<unsigned int>
    * \param pixels pixels to be shown
    */
    void writePLY(Unity<unsigned char>* pixels);
    void writePLY();
    void writePLY(Node* nodes_device, unsigned long numNodes);
    void writePLY(float2* points_device, unsigned long numPoints);

    ~Quadtree();


  };

  namespace{
    struct is_flagged{
      template<typename T>
      __host__ __device__
      bool operator()(const typename Quadtree<T>::Node& n){
        return (n.flag);
      }
    };
  }


  /*
  CUDA KERNEL DEFINITIONS
  */

  __global__ void getKeys(int* keys, float2* nodeCenters, uint2 size, int2 border, unsigned int depth);
  __global__ void getKeys(int* keys, float2* nodeCenters, uint2 size, int2 border, unsigned int depth, unsigned int colorDepth);
  __global__ void getKeys(unsigned int numPoints, float2* points, int* keys, float2* nodeCenters, uint2 size, unsigned int depth);
  __global__ void getKeys(unsigned int numLocalizedPointers, ssrlcv::LocalizedData<unsigned int>* localizedPointers, int* keys, float2* nodeCenters, uint2 size, unsigned int depth);

  template<typename T>
  __global__ void fillLeafNodes(unsigned long numDataElements, unsigned long numLeafNodes, typename Quadtree<T>::Node* leafNodes,int* keys, float2* nodeCenters, unsigned int* nodeDataIndex, unsigned int depth);

  template<typename T>
  __global__ void findAllNodes(unsigned long numUniqueNodes, unsigned int* nodeNumbers, typename Quadtree<T>::Node* uniqueNodes);

  template<typename T>
  __global__ void fillNodesAtDepth(unsigned long numUniqueNodes, unsigned int* nodeNumbers, unsigned int* nodeAddresses, typename Quadtree<T>::Node* existingNodes,
    typename Quadtree<T>::Node* allNodes, unsigned int currentDepth, unsigned int totalDepth);

  template<typename T>
  __global__ void buildParentalNodes(unsigned long numChildNodes, unsigned long childDepthIndex, typename Quadtree<T>::Node* childNodes, typename Quadtree<T>::Node* parentNodes, uint2 width);

  template<typename T>
  __global__ void fillParentIndex(unsigned int numNodesAtDepth, unsigned int depthStartingIndex, typename Quadtree<T>::Node* nodes);

  template<typename T>
  __global__ void fillDataNodeIndex(unsigned long numLeafNodes, typename Quadtree<T>::Node* nodes, unsigned int* dataNodeIndex);


  template<typename T>
  __global__ void computeNeighboringNodes(unsigned int numNodesAtDepth, unsigned int currentDepthIndex, unsigned int* parentLUT, unsigned int* childLUT, typename Quadtree<T>::Node* nodes);

  template<typename T>
  __global__ void findVertexOwners(unsigned int numNodesAtDepth, unsigned int depthIndex, typename Quadtree<T>::Node* nodes, int* numVertices, int* ownerInidices, int* vertexPlacement);

  template<typename T>
  __global__ void fillUniqueVertexArray(unsigned int depthIndex, typename Quadtree<T>::Node* nodes, unsigned long numVertices, int vertexIndex,
  typename Quadtree<T>::Vertex* vertices, int depth, int* ownerInidices, int* vertexPlacement, uint2 size);

  template<typename T>
  __global__ void findEdgeOwners(unsigned int numNodesAtDepth, unsigned int depthIndex, typename Quadtree<T>::Node* nodes, int* numEdges, int* ownerInidices, int* edgePlacement);

  template<typename T>
  __global__ void fillUniqueEdgeArray(unsigned int depthIndex, typename Quadtree<T>::Node* nodes, unsigned long numEdges, int edgeIndex,
  typename Quadtree<T>::Edge* edges, int depth, int* ownerInidices, int* edgePlacement);

  template<typename T>
  __global__ void applyNodeFlags(unsigned int numNodes, unsigned int depthIndex, typename Quadtree<T>::Node* nodes, bool* hashMap, bool requireFullNeighbors);
  template<typename T>
  __global__ void applyNodeFlags(unsigned int numNodes, unsigned int depthIndex, typename Quadtree<T>::Node* nodes, float4 flagBounds, bool requireFullNeighbors);

}


#endif /* QUADTREE_CUH */
