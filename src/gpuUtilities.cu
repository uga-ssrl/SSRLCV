#ifndef GPU_UTILS
#define GPU_UTILS
/*
#include <cuda.h>
#include <cuda_runtime.h>

__device__ int getIndex_gpu(int x, int y)
{
	return (3*x +y);
}

__device__ void multiply3x3x1_gpu(float A[9], float B[3], float (&C)[3])
{
  for (int r = 0; r < 3; ++r)
  {
    float val = 0;
    for (int c = 0; c < 3; ++c)
    {
      val += A[getIndex_gpu(r,c)] * B[c];
    }
    C[r] = val;
  }
}

__device__ void multiply3x3x1_gpu(float A[3][3], float B[3], float (&C)[3])
{
  for (int r = 0; r < 3; ++r)
  {
    float val = 0;
    for (int c = 0; c < 3; ++c)
    {
      val += A[r][c] * B[c];
    }
    C[r] = val;
  }
}

__device__ float dot_product_gpu(float a[3], float b[3])
{
  return a[0]*b[0] + a[1]*b[1] + a[2]*b[2];
}

__device__ float magnitude_gpu(float v[3])
{
  return sqrt(dot_product_gpu(v, v));
}

__device__ void normalize_gpu(float (&v)[3])
{
  float mag = magnitude_gpu(v);
  if(mag > 0)
	{
    v[0] = v[0]/mag;
    v[1] = v[1]/mag;
    v[2] = v[2]/mag;
  }
}

__device__ int getGlobalIdx_1D_1D()
{
	return blockIdx.x *blockDim.x + threadIdx.x;
}

__device__ void inverse3x3_gpu(float M[3][3], float (&Minv)[3][3])
{
  float d1 = M[1][1] * M[2][2] - M[2][1] * M[1][2];
  float d2 = M[1][0] * M[2][2] - M[1][2] * M[2][0];
  float d3 = M[1][0] * M[2][1] - M[1][1] * M[2][0];
  float det = M[0][0]*d1 - M[0][1]*d2 + M[0][2]*d3;
  if(det == 0)
	{
    // return pinv(M);
  }
  float invdet = 1/det;
  Minv[0][0] = d1*invdet;
  Minv[0][1] = (M[0][2]*M[2][1] - M[0][1]*M[2][2]) * invdet;
  Minv[0][2] = (M[0][1]*M[1][2] - M[0][2]*M[1][1]) * invdet;
  Minv[1][0] = -1 * d2 * invdet;
  Minv[1][1] = (M[0][0]*M[2][2] - M[0][2]*M[2][0]) * invdet;
  Minv[1][2] = (M[1][0]*M[0][2] - M[0][0]*M[1][2]) * invdet;
  Minv[2][0] = d3 * invdet;
  Minv[2][1] = (M[2][0]*M[0][1] - M[0][0]*M[2][1]) * invdet;
  Minv[2][2] = (M[0][0]*M[1][1] - M[1][0]*M[0][1]) * invdet;
}*/
#endif
