#ifndef __IMAGE_H__
#define __IMAGE_H__

// Gitlab #58 - Moved Image_Descriptor to a non-CUDA header so it can be populated from io_util 

namespace ssrlcv{
  struct Image_Descriptor{
    int id;
    uint2 size;
    float3 cam_pos;
    float3 cam_vec;
    float fov;
    float foc;
    float dpix;
    long long int timeStamp;//seconds since Jan 01, 1070
    __device__ __host__ Image_Descriptor();
    __device__ __host__ Image_Descriptor(int id, uint2 size);
    __device__ __host__ Image_Descriptor(int id, uint2 size, float3 cam_pos, float3 camp_dir);
  };
}

#endif 