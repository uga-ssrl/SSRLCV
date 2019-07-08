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
  //look for better solution here
  template<typename D>
  struct LocalizedData{
    float2 loc;
    D data;
  };

  //TODO make Quadtree exceptions and add to a namespace
  //TODO make depth variable

  /*
  BASE QUADTREE CLASS
  */
  template<typename T>
  class Quadtree{

    void generateLeafNodes(int depth = -1);//Will generate dense tree

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

    uint2 imageSize;
    int width;
    unsigned int depth;

    ssrlcv::Unity<T>* data;
    ssrlcv::Unity<Node>* nodes;
    ssrlcv::Unity<Vertex>* vertices;
    ssrlcv::Unity<Edge>* edges;

    ssrlcv::Unity<unsigned int>* nodeDepthIndex;
    ssrlcv::Unity<unsigned int>* vertexDepthIndex;
    ssrlcv::Unity<unsigned int>* edgeDepthIndex;

    Quadtree();

    Quadtree(uint2 imageSize, ssrlcv::Unity<T>* data);
    Quadtree(uint2 imageSize, ssrlcv::Unity<bool>* hashMap, ssrlcv::Unity<T>* data);
    Quadtree(ssrlcv::Unity<int2>* data);
    Quadtree(ssrlcv::Unity<float2>* data);
    Quadtree(ssrlcv::Unity<LocalizedData<T>>* data);

    ~Quadtree();

    void setHashMap(ssrlcv::Unity<bool>* hashMap);

  };

  /*
  CUDA KERNEL DEFINITIONS
  */
  template<typename T>
  __global__ void fillLeafNodesDensly(typename Quadtree<T>::Node* leafNodes, unsigned int width, uint2 imageSize, int depth);

}


#endif /* QUADTREE_CUH */
