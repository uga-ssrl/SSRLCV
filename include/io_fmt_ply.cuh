#pragma once
#ifndef IO_FMT_PLY_CUH
#define IO_FMT_PLY_CUH

#include "common_includes.hpp"
#include "Octree.cuh"
#include "io_util.hpp"
#include <fstream> 

namespace ssrlcv {
  namespace io { 
    namespace ply { 
 
      


  class Writer
  {
  public:
    Writer(const char * path);
    void Write(std::shared_ptr<ssrlcv::Unity<float3>> points); 
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


  void write_points(const char * path, std::shared_ptr<ssrlcv::Unity<float3>> points); 
  void write_octree(const char * path, Octree * octree, bool verts = true, bool edges = true);




    }
  }
}

#endif /* IO_FMT_PLY_CUH */
