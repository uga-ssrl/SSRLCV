#ifndef __IO_FILE__
#define __IO_FILE__

#include "common_includes.h"
#include "Unity.cuh"
#include "Octree.cuh"
#include "io_util.h"

#include <fstream> 

namespace ssrlcv {
  namespace io { 
    namespace ply { 
 
      


  class Writer
  {
  public:
    Writer(const char * path);
    void Write(Unity<float3> * points); 
    void Write(Octree * octree, bool verts = true, bool edges = true);

  private:
    const char * path;
    bool binary = true;
    std::ofstream file; 

    void header_intro();
    void header_vertex(int count); 
    void header_edge(int count); 
    void header_end();
  };


  void write_points(const char * path, Unity<float3> * points); 
  void write_octree(const char * path, Octree * octree, bool verts = true, bool edges = true);




    }
  }
}

#endif