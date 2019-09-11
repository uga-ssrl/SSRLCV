#include "io_file.cuh"
#include <fstream>
#include <arpa/inet.h>

namespace ssrlcv{

  Writer3d::Writer3d() { }
  Writer3d::~Writer3d() { }

  WriterPLY::WriterPLY(const char * path, bool binary) {
    this->path = path;
    this->binary = binary;
  }

  void WriterPLY::Write()
  {
    //TODO: err
    std::ofstream file;
    file.exceptions(std::ios::failbit);
    file.open(path, std::ofstream::binary /* TODO: Non binary? */);

    if(binary) {
      if(htonl(69) == 69) file << "ply\nformat binary_big_endian 1.0\n";
      else                file << "ply\nformat binary_little_endian 1.0\n";
    } else                file << "ply\nformat ascii 1.0\n";

    //
    // The order of header declarations here must match the order where they're written below
    //

    if(this->points) {
      file << "element vertex " << std::to_string(this->points->numElements);
      file << "\nproperty float x\nproperty float y\nproperty float z\n";
    }

    else if(this->octree) {
      if(this->octreeCenters){
        file << "element vertex " << std::to_string(this->octree->vertices->numElements);
        file << "\nproperty float x\nproperty float y\nproperty float z\n";
      }
      if(this->octreeEdges) {
        file << "element edge " << std::to_string(this->octree->edges->numElements);
        file << "\nproperty int v1\nproperty int v2\n";
      }
    }

    file << "end_header\n";

    //
    // Data
    //

    #define FWRITE(x) file.write(reinterpret_cast<char *>(&x), sizeof(x))

    if(this->points) {
      if(this->points->state != cpu) this->points->transferMemoryTo(cpu);
      for(int i = 0; i < this->points->numElements; i++) {
        FWRITE(this->points->host[i].x);
        FWRITE(this->points->host[i].y);
        FWRITE(this->points->host[i].z);
      }
    }

    else if(this->octree)
    {
      if(this->octreeVertices) {
        Unity<Octree::Vertex> * vertices = this->octree->vertices;
        if(vertices->state != cpu) vertices->transferMemoryTo(cpu);
        for(int i =0; i < vertices->numElements; i++) {
          FWRITE(vertices->host[i].coord.x);
          FWRITE(vertices->host[i].coord.y);
          FWRITE(vertices->host[i].coord.z);
        }
      }

      if(this->octreeEdges) {
        Unity<Octree::Edge> * edges = this->octree->edges;
        if(edges->state != cpu) edges->transferMemoryTo(cpu);
        for(int i =0; i < edges->numElements; i++) {
          FWRITE(edges->host[i].v1);
          FWRITE(edges->host[i].v2);
        }
      }
    }

    #undef FWRITE
    file.close();
  }

}
