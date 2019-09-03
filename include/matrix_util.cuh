/** \file matrix_util.cuh
 * \brief file for housing matrix utility functions
*/
#ifndef MATRIXUTIL_CUH
#define MATRIXUTIL_CUH

#include "common_includes.h"
#include "cuda_util.cuh"

namespace ssrlcv{
  __device__ __host__ float sum(const float3 &a);
  __device__ int getIndex_gpu(int x, int y);
  __device__ void multiply3x3x1_gpu(float A[9], float B[3], float (&C)[3]);
  __device__ void multiply3x3x1_gpu(float A[3][3], float B[3], float (&C)[3]);
  __device__ float dot_product_gpu(float a[3], float b[3]);
  __device__ float magnitude_gpu(float v[3]);
  __device__ void normalize_gpu(float (&v)[3]);
  __device__ int getGlobalIdx_1D_1D();
  __device__ void inverse3x3_gpu(float M[3][3], float (&Minv)[3][3]);

  void transpose_cpu(float M[3][3], float (&M_t)[3][3]);
  void inverse3x3_cpu(float M[3][3], float (&Minv)[3][3]);
  void multiply3x3_cpu(float A[3][3], float B[3][3], float (&C)[3][3]);
  void multiply3x3x1_cpu(float A[3][3], float B[3], float (&C)[3]);

  /*
  Funundamental matrix stuff
  */
  float3 multiply3x3x1(const float3 A[3], const float3 &B);
  void multiply3x3(const float3 A[3], const float3 B[3], float3 *C);
  void transpose3x3(const float3 M[3], float3 (&M_T)[3]);
  void inverse3x3(float3 M[3], float3 (&Minv)[3]);

}

#endif /* MATRIXUTIL_CUH */
