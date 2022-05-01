#include "io_fmt_ply.cuh"
#include <arpa/inet.h>

namespace ssrlcv { 
  namespace io { 
    namespace ply { 
      


  Writer::Writer(const char * path) {
    this->path = path;
    this->binary = true; // for now it's only binary cause that was the task I was given 

    this->file = std::ofstream();
    this->file.exceptions(std::ios::failbit);
    this->file.open(path, std::ofstream::binary /* TODO: Non binary? */);
  }

  //
  // Header helper functions
  //

  // This outputs the beginning of the header - it's always the first output called
  void Writer::header_intro() { 
    if(binary) {
      if(htonl(69) == 69) this->file << "ply\nformat binary_big_endian 1.0\n";
      else                this->file << "ply\nformat binary_little_endian 1.0\n";
    } 
    else                  this->file << "ply\nformat ascii 1.0\n";
  }

  // This declares the vertex element
  void Writer::header_vertex(int count) { 
    this->file << "element vertex " << std::to_string(count);
    this->file << "\nproperty float x\nproperty float y\nproperty float z\n";
  }

  // This declares the edge element
  void Writer::header_edge(int count) { 
    this->file << "element edge " << std::to_string(count);
    this->file << "\nproperty int v1\nproperty int v2\n";
  }

  void Writer::header_end() { 
    this->file << "end_header\n";
  }


  //
  // Main thingz 
  //

  #define FWRITE(x) this->file.write(reinterpret_cast<char *>(&x), sizeof(x))

  void Writer::Write(std::shared_ptr<ssrlcv::Unity<float3>> points) { 
    this->header_intro();
    this->header_vertex(points->size());
    this->header_end(); 


    if(points->getMemoryState() != cpu) points->transferMemoryTo(cpu);
    for(int i = 0; i < points->size(); i++) {
      FWRITE(points->host.get()[i].x);
      FWRITE(points->host.get()[i].y);
      FWRITE(points->host.get()[i].z);
    }


    this->file.close();
  }



  void Writer::Write(Octree * octree, bool vertices, bool edges )
  {
    this->header_intro();
    if(vertices)  this->header_vertex(octree->vertices->size());
    if(edges)     this->header_edge(octree->edges->size());
    this->header_end(); 

    if(vertices) {
      std::shared_ptr<ssrlcv::Unity<Octree::Vertex>> vertices = octree->vertices;
      if(vertices->getMemoryState() != cpu) vertices->transferMemoryTo(cpu);
      for(int i =0; i < vertices->size(); i++) {
        FWRITE(vertices->host.get()[i].coord.x);
        FWRITE(vertices->host.get()[i].coord.y);
        FWRITE(vertices->host.get()[i].coord.z);
      }
    }

    if(edges) { 
      std::shared_ptr<ssrlcv::Unity<Octree::Edge>> edges = octree->edges;
      if(edges->getMemoryState() != cpu) edges->transferMemoryTo(cpu);
      for(int i =0; i < edges->size(); i++) {
        FWRITE(edges->host.get()[i].v1);
        FWRITE(edges->host.get()[i].v2);
      }
    }

    this->file.close(); 
  }

  #undef FWRITE


  //
  // Interface methods
  // 

  void write_points(const char * path, std::shared_ptr<ssrlcv::Unity<float3>> points){ 
    Writer(path).Write(points); 
  }

  void write_octree(const char * path, Octree * octree, bool verts, bool edges) {
    Writer(path).Write(octree, verts, edges) ;
  }



    }
  }
}