#ifndef IMAGE_CUH
#define IMAGE_CUH

#include "common_includes.h"
#include "image_io.h"
#include "Feature.cuh"
#include "cuda_util.cuh"

__device__ __forceinline__ unsigned long getGlobalIdx_2D_1D();
__device__ __forceinline__ unsigned char bwaToBW(const uchar2 &color);
__device__ __forceinline__ unsigned char rgbToBW(const uchar3 &color);
__device__ __forceinline__ unsigned char rgbaToBW(const uchar4 &color);
__global__ void generateBW(int numPixels, unsigned char colorDepth, unsigned char* colorPixels, unsigned char* bwPixels);

struct Image_Descriptor{
  int id;
  int2 size;
  float3 cam_pos;
  float3 cam_vec;
  float fov;
  float foc;
  float dpix;
  int2 segment_size;
  int2 segment_num;
  __device__ __host__ Image_Descriptor();
  __device__ __host__ Image_Descriptor(int id, int2 size);
  __device__ __host__ Image_Descriptor(int id, int2 size, float3 cam_pos, float3 camp_dir);
};

struct Segment_Helper{
  bool is_segment;
  int segment_number;
  int2 size;
  int pix_filled;
  // unsigned char* pixels; not needed
};

void get_cam_params2view(Image_Descriptor &cam1, Image_Descriptor &cam2, std::string infile);

class Image{

public:

  Image_Descriptor descriptor;
  std::string filePath;

  MemoryState arrayStates[3];//pix,features,featureDescritors

  // this is for image segmentation
  Image* segments;
  Segment_Helper segment_helper;

  //TODO find way to allow for feature to be of any type

  int numFeatures;
  size_t feature_size;
  SIFT_Feature* features;
  SIFT_Feature* features_device;
  int numDescriptorsPerFeature;
  size_t featureDescriptor_size;
  SIFT_Descriptor* featureDescriptors;
  SIFT_Descriptor* featureDescriptors_device;

  //numPixels can be derived from descriptor.size
  unsigned char colorDepth;
  int totalPixels;
  unsigned char* pixels;
  unsigned char* pixels_device;

  Image();
  Image(std::string filePath);
  Image(std::string filePath, int id);
  Image(std::string filePath, int id, MemoryState arrayStates[3]);
  ~Image();

  void setPixelState(MemoryState pixelState);
  unsigned char* readColorPixels();
  void convertToBW();
  void segment(int x_num, int y_num, int x_size, int y_size);

private:
  bool isInSegment(int x_run, int y_run);
  int getSegNum(int x_run, int y_run);
};


#endif /* IMAGE_CUH */
