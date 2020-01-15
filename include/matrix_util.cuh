/** \file matrix_util.cuh
 * \brief file for housing matrix utility functions
*/
#ifndef MATRIXUTIL_CUH
#define MATRIXUTIL_CUH

#include "common_includes.h"
#include "cuda_util.cuh"

namespace ssrlcv{
  __device__ __host__ float sum(const float3 &a);
  __device__ __host__ void multiply(const float (&A)[9], const float (&B)[3][3], float (&C)[3][3]);
  __device__ __host__ void multiply(const float3 (&A)[3], const float3 (&B)[3], float3 (&C)[3]);
  __device__ __host__ void multiply(const float (&A)[3][3], const float (&B)[3][3], float (&C)[3][3]);
  __device__ __host__ void multiply(const float (&A)[9], const float (&B)[3], float (&C)[3]);
  __device__ __host__ void multiply(const float3 (&A)[3], const float3 &B, float3 &C);
  __device__ __host__ void multiply(const float (&A)[3][3], const float (&B)[3], float (&C)[3]);
  __device__ __host__ void multiply(const float (&A)[3], const float (&B)[3][3], float (&C)[3]);
  __device__ __host__ void multiply(const float (&A)[2][2], const float (&B)[2][2], float (&C)[2][2]);
  __device__ __host__ float dotProduct(const float (&A)[3], const float (&B)[3]);
  __device__ __host__ bool inverse(const float (&M)[3][3], float (&M_out)[3][3]);
  __device__ __host__ bool inverse(const float3 (&M)[3], float3 (&M_out)[3]);
  __device__ __host__ void transpose(const float3 (&M)[3], float3 (&M_out)[3]);
  __device__ __host__ void transpose(const float (&M)[3][3], float (&M_out)[3][3]);
  __device__ __host__ void transpose(const float (&M)[2][2], float (&M_out)[2][2]);

  __device__ __host__ float determinant(const float (&M)[2][2]);
  __device__ __host__ float trace(const float(&M)[2][2]);
  __device__ __host__ float trace(const float(&M)[3][3]);

  __device__ __host__ void normalize(float (&v)[3]);
  __device__ __host__ void normalize(float3 &v);
  __device__ __host__ float magnitude(const float (&v)[3]);
  __device__ __host__ float magnitude(const float3 &v);

  __device__ float3 matrixMulVector(float3 x, float A[3][3]);
  __device__ float3 getVectorAngles(float3 v);
  __device__ float3 rotatePoint(float3 point, float3 angles);
  __device__ float3 rotatePointKP(float3 point, float3 goal, float axangle);

}

#endif /* MATRIXUTIL_CUH */
