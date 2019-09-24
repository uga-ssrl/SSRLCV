#include "io_3d.cuh"
#include "tinyply.h"

#include "unit-testing.h"
#include <iostream>
#include <vector>

using namespace ssrlcv;
using namespace std;
using namespace tinyply;

int main(int argc, char ** argv) {

  const char * testfile = "tmp/ply.ply";

  vector<float3> points;
  points.push_back({ 1.0, 2.0, 3.0 });
  points.push_back({ 420.0, 69.0, 666.0 });

  Unity<float3> data(nullptr, points.size(), cpu);
  memcpy(data.host, points.data(), sizeof(float3) * points.size());


  WriterPLY writer(testfile);
  writer.points = &data;
  writer.write();

  // Read it back - currently testing with tinyply
  // We could switch these to static test cases if we wanted to ditch tinyply altogether

  ifstream plyinput(testfile, std::ios::binary);
  if(plyinput.fail()) { cerr << "Failed to re-read file" << endl; return 2; }

  PlyFile plyread;

  try {
    plyread.parse_header(plyinput);


    std::shared_ptr<PlyData> rVertices = plyread.request_properties_from_element("vertex", { "x", "y", "z" });
    plyread.read(plyinput);

    TEST(rVertices->count == points.size(), "size mismatch");
    TEST(rVertices->t == Type::FLOAT32, "size mismatch");

    std::vector<float3> vertices(rVertices->count);
    std::memcpy(vertices.data(), rVertices->buffer.get(), rVertices->buffer.size_bytes());

    for(int i = 0; i < vertices.size(); i++) {
      float3 in = points[i], out=vertices[i];
      TEST(in.x == out.x, "x mismatch");
      TEST(in.y == out.y, "y mismatch");
      TEST(in.z == out.z, "z mismatch");
    }
  }
  catch (const std::exception & e) { cerr << "Exception re-reading PLY: " << e.what() << endl; return 1; }

  return 0;
}
