#include "Feature.cuh"

/*
STRUCTURE METHODS
*/

__device__ __host__ ssrlcv::SIFT_Descriptor::SIFT_Descriptor(){
  this->theta = 0.0f;
}
__device__ __host__ ssrlcv::SIFT_Descriptor::SIFT_Descriptor(float theta){
  this->theta = theta;
}
__device__ __host__ ssrlcv::SIFT_Descriptor::SIFT_Descriptor(float theta, unsigned char values[128]){
  this->theta = theta;
  for(int i = 0; i < 128; ++i){
    this->values[i] = values[i];
  }
}
