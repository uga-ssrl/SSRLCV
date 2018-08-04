#include "cuda_util.cuh"

__device__ __host__ float3 operator+(const float3 &a, const float3 &b) {
  return {a.x+b.x, a.y+b.y, a.z+b.z};
}
__device__ __host__ float3 operator-(const float3 &a, const float3 &b) {
  return {a.x-b.x, a.y-b.y, a.z-b.z};
}
__device__ __host__ float3 operator/(const float3 &a, const float3 &b) {
  return {a.x/b.x, a.y/b.y, a.z/b.z};
}
__device__ __host__ float3 operator*(const float3 &a, const float3 &b) {
  return {a.x*b.x, a.y*b.y, a.z*b.z};
}
__device__ __host__ float dotProduct(const float3 &a, const float3 &b){
  return (a.x*b.x) + (a.y*b.y) + (a.z*b.z);
}
__device__ __host__ float3 operator+(const float3 &a, const float &b){
  return {a.x+b, a.y+b, a.z+b};
}
__device__ __host__ float3 operator-(const float3 &a, const float &b){
  return {a.x-b, a.y-b, a.z-b};
}
__device__ __host__ float3 operator/(const float3 &a, const float &b){
  return {a.x/b, a.y/b, a.z/b};
}
__device__ __host__ float3 operator*(const float3 &a, const float &b){
  return {a.x*b, a.y*b, a.z*b};
}
__device__ __host__ float3 operator+(const float &a, const float3 &b) {
  return {a+b.x, a+b.y, a+b.z};
}
__device__ __host__ float3 operator-(const float &a, const float3 &b) {
  return {a-b.x, a-b.y, a-b.z};
}
__device__ __host__ float3 operator/(const float &a, const float3 &b) {
  return {a/b.x, a/b.y, a/b.z};
}
__device__ __host__ float3 operator*(const float &a, const float3 &b) {
  return {a*b.x, a*b.y, a*b.z};
}
__device__ __host__ bool operator==(const float3 &a, const float3 &b){
  return (a.x==b.x)&&(a.y==b.y)&&(a.z==b.z);
}

__host__ void cusolverCheckError(cusolverStatus_t cusolver_status){
  switch (cusolver_status){
      case CUSOLVER_STATUS_SUCCESS:
          std::cout<<"CUSOLVER_SUCCESS"<<std::endl;
          break;

      case CUSOLVER_STATUS_NOT_INITIALIZED:
          std::cout<<"CUSOLVER_STATUS_NOT_INITIALIZED"<<std::endl;
          exit(-1);

      case CUSOLVER_STATUS_ALLOC_FAILED:
          std::cout<<"CUSOLVER_STATUS_ALLOC_FAILED"<<std::endl;
          exit(-1);

      case CUSOLVER_STATUS_INVALID_VALUE:
          std::cout<<"CUSOLVER_STATUS_INVALID_VALUE"<<std::endl;
          exit(-1);

      case CUSOLVER_STATUS_ARCH_MISMATCH:
          std::cout<<"CUSOLVER_STATUS_ARCH_MISMATCH"<<std::endl;
          exit(-1);

      case CUSOLVER_STATUS_EXECUTION_FAILED:
          std::cout<<"CUSOLVER_STATUS_EXECUTION_FAILED"<<std::endl;
          exit(-1);

      case CUSOLVER_STATUS_INTERNAL_ERROR:
          std::cout<<"CUSOLVER_STATUS_INTERNAL_ERROR"<<std::endl;
          exit(-1);

      case CUSOLVER_STATUS_MATRIX_TYPE_NOT_SUPPORTED:
          std::cout<<"CUSOLVER_STATUS_MATRIX_TYPE_NOT_SUPPORTED"<<std::endl;
          exit(-1);
  }
}
