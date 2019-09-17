#ifndef __IO_FILE__
#define __IO_FILE__

#include "common_includes.h"
#include "Unity.cuh"
#include "Octree.cuh"
#include "io_util.h"


namespace ssrlcv{


  class Writer3d {

  public:
    // These must be made explicit for the abstraction to work, because C++
    Writer3d();
    ~Writer3d();

    Unity<float3> * points;

    Octree * octree;
    bool octreeVertices = true, octreeEdges = true;
    bool octreeCenters = false, octreeNormals = false;

    virtual void write() = 0;
  };

  class WriterPLY : public Writer3d
  {
  private:
    const char * path;
    bool binary = true;
  public:
    WriterPLY(const char * path, bool binary = true);
    void write();
  };



}




#endif
