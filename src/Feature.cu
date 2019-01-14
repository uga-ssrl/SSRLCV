#include "Feature.cuh"

/*
STRUCTURE METHODS
*/
__device__ __host__ SIFT_Feature::SIFT_Feature(){
  this->loc = {-1,-1};
  this->parentImage = -1;
  this->real = true;
  this->sigma = 0.0f;
  this->descriptorIndex = -1;
}
__device__ __host__ SIFT_Feature::SIFT_Feature(int2 loc, int parentImage){
  this->loc = loc;
  this->parentImage = parentImage;
  this->real = true;
  this->sigma = 0.0f;
  this->descriptorIndex = -1;
}
__device__ __host__ SIFT_Feature::SIFT_Feature(int2 loc, int parentImage, bool real){
  this->loc = loc;
  this->parentImage = parentImage;
  this->real = real;
  this->sigma = 0.0f;
  this->descriptorIndex = -1;
}
__device__ __host__ SIFT_Feature::SIFT_Feature(int2 loc, int parentImage, bool real, float sigma){
  this->loc = loc;
  this->parentImage = parentImage;
  this->real = real;
  this->sigma = sigma;
  this->descriptorIndex = -1;
}
__device__ __host__ SIFT_Descriptor::SIFT_Descriptor(){
  this->theta = 0.0f;
}
__device__ __host__ SIFT_Descriptor::SIFT_Descriptor(float theta){
  this->theta = theta;
}
__device__ __host__ SIFT_Descriptor::SIFT_Descriptor(float theta, unsigned char descriptor[128]){
  this->theta = theta;
  for(int i = 0; i < 128; ++i){
    this->descriptor[i] = descriptor[i];
  }
}
