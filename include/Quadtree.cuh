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

  //consider putting in util
  template<typename D>
  struct LocalizedData{
    float2 loc;
    D data;
  };

  //TODO make Quadtree exceptions and add to a namespace
  //TODO consider allowing quadtrees that are not square


  /*
  BASE QUADTREE CLASS
  */
  template<typename T>
  class Quadtree{

    void generateLeafNodes();

    void generateParentNodes();



  public:
    struct Node{
      int key;
      int dataIndex;
      int numElements;
      float2 center;
      int depth;//maybe remove as its derivable or change to char
      int parent;
      int children[4];
      int neighbors[9];
      int edges[4];
      int vertices[4];

      __device__ __host__ Node();
    };
    struct Vertex{
      float2 loc;
      int nodes[4];
      int depth;
      __device__ __host__ Vertex();
    };
    struct Edge{
      int2 vertices;
      int nodes[2];
      int depth;
      __device__ __host__ Edge();
    };

    uint2 size;
    uint2 depth;//{min,max}

    ssrlcv::Unity<T>* data;
    ssrlcv::Unity<Node>* nodes;
    ssrlcv::Unity<Vertex>* vertices;
    ssrlcv::Unity<Edge>* edges;

    ssrlcv::Unity<unsigned int>* nodeDepthIndex;
    ssrlcv::Unity<unsigned int>* vertexDepthIndex;
    ssrlcv::Unity<unsigned int>* edgeDepthIndex;

    Quadtree();

    //for full quadtrees only holding data indices
    //can only be used with Quadtree<unsigned int>()
    Quadtree(uint2 size);

    Quadtree(uint2 size, ssrlcv::Unity<T>* data);


    ~Quadtree();


  };

  /*
  CUDA KERNEL DEFINITIONS
  */

  __global__ void getKeys(int* keys, float2* nodeCenters, uint2 size, int depth);

  template<typename T>
  __global__ void fillLeafNodes(unsigned long numLeafNodes, typename Quadtree<T>::Node* leafNodes,int* keys, float2* nodeCenters, unsigned int* nodeDataIndex);

}


#endif /* QUADTREE_CUH */
