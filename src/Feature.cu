#include "Feature.cuh"

/*
STRUCTURE METHODS
*/
template<typename D>
__device__ __host__ ssrlcv::Feature<D>::Feature(){
  this->loc = {-1.0f,-1.0f};
}
template<typename D>
__device__ __host__ ssrlcv::Feature<D>::Feature(float2 loc){
  this->loc = loc;
}
template<typename D>
__device__ __host__ ssrlcv::Feature<D>::Feature(float2 loc, D descriptor){
  this->loc = loc;
  this->descriptor = descriptor;
}
__device__ __host__ ssrlcv::SIFT_Descriptor::SIFT_Descriptor(){
  this->theta = 0.0f;
}
__device__ __host__ ssrlcv::SIFT_Descriptor::SIFT_Descriptor(float theta){
  this->theta = theta;
}
__device__ __host__ ssrlcv::SIFT_Descriptor::SIFT_Descriptor(float theta, unsigned char descriptor[128]){
  this->theta = theta;
  for(int i = 0; i < 128; ++i){
    this->descriptor[i] = descriptor[i];
  }
}
