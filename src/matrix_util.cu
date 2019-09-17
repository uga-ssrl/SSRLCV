#include "matrix_util.cuh"

//
//
// GPU Methods
//
//

__device__ __host__ float ssrlcv::sum(const float3 &a){
  return a.x + a.y + a.z;
}
__device__ int ssrlcv::getIndex_gpu(int x, int y){
        return (3*x +y);
}
__device__ void ssrlcv::multiply3x3x1_gpu(float A[9], float B[3], float (&C)[3]){
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
__device__ void ssrlcv::multiply3x3x1_gpu(float A[3][3], float B[3], float (&C)[3]){
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
__device__ float ssrlcv::dot_product_gpu(float a[3], float b[3]){
  return a[0]*b[0] + a[1]*b[1] + a[2]*b[2];
}
__device__ float ssrlcv::magnitude_gpu(float v[3]){
  return sqrt(dot_product_gpu(v, v));
}
__device__ void ssrlcv::normalize_gpu(float (&v)[3]){
  float mag = magnitude_gpu(v);
  if(mag > 0)
        {
    v[0] = v[0]/mag;
    v[1] = v[1]/mag;
    v[2] = v[2]/mag;
  }
}
__device__ int ssrlcv::getGlobalIdx_1D_1D(){
        return blockIdx.x *blockDim.x + threadIdx.x;
}
__device__ void ssrlcv::inverse3x3_gpu(float M[3][3], float (&Minv)[3][3]){
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
}

__device__ float ssrlcv::dotProduct3(float3 a, float3 b){
  float product = 0.0;
  product += a.x * b.x;
  product += a.y * b.y;
  product += a.z * b.z;
  return product;
}

__device__ float3 ssrlcv::normalizeVector(float3 v){
  float mag = sqrt(dotProduct(v,v));
  v.x /= mag;
  v.y /= mag;
  v.z /= mag;
  return v;
}


__device__ float3 ssrlcv::matrixMulVector(float3 x, float A[3][3]){
  float temp[3] = {x.x, x.y, x.z};
  float b[3];
  for (int r = 0; r < 3; ++r)
  {
    float val = 0;
    for (int c = 0; c < 3; ++c)
    {
      val += A[r][c] * temp[c];
    }
    b[r] = val;
  }
  return {b[0], b[1], b[2]};
}

__device__ float3 ssrlcv::getVectorAngles(float3 v){
  float3 angles;
  float3 x_n = {1.0f, 0.0f, 0.0f};
  float3 y_n = {0.0f, 1.0f, 0.0f};
  float3 z_n = {0.0f, 0.0f, 1.0f};
  // x angle
  float a = dotProduct3(v,x_n);
  float b = dotProduct3(v,v);
  float c = (a)/(sqrtf(b));
  angles.x = acosf(c);
  // y angle
  a = dotProduct3(v,y_n);
  b = dotProduct3(v,v);
  c = (a)/(sqrtf(b));
  angles.y = acosf(c);
  // z angle
  a = dotProduct3(v,z_n);
  b = dotProduct3(v,v);
  c = (a)/(sqrtf(b));
  angles.z = acosf(c);
  return angles;
}

__device__ float3 ssrlcv::rotatePoint(float3 point, float3 angle) {
  // this is just a 3D rotation matrix
  // contains all R3 rotations multiplied together
  float rotationMatrix[3][3];
  rotationMatrix[0][0] = cosf(angle.y) * cosf(angle.z);
  rotationMatrix[0][1] = -1 * cosf(angle.y) * sinf(angle.z);
  rotationMatrix[0][2] = sinf(angle.y);
  rotationMatrix[1][0] = sinf(angle.x) * sinf(angle.y) * cosf(angle.z) + cosf(angle.x) * sinf(angle.z);
  rotationMatrix[1][1] = cosf(angle.x) * cosf(angle.z) - sinf(angle.x) * sinf(angle.y) * sinf(angle.z);
  rotationMatrix[1][2] = -1 * sinf(angle.x) * cosf(angle.y);
  rotationMatrix[2][0] = sinf(angle.x) * sinf(angle.z) - cosf(angle.x) * sinf(angle.y) * cosf(angle.z);
  rotationMatrix[2][1] = cosf(angle.x) * sinf(angle.y) * sinf(angle.z) + sinf(angle.x) * cosf(angle.z);
  rotationMatrix[2][2] = cosf(angle.x) * cosf(angle.y);
  point = matrixMulVector(point, rotationMatrix);
  return point;
}

//
//
// CPU methods
//
//

void ssrlcv::transpose_cpu(float M[3][3], float (&M_t)[3][3]){
  for(int r = 0; r < 3; ++r)
  {
    for(int c = 0; c < 3; ++c)
    {
      M_t[r][c] = M[c][r];
    }
  }
}
void ssrlcv::inverse3x3_cpu(float M[3][3], float (&Minv)[3][3]){
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
}
void ssrlcv::multiply3x3_cpu(float A[3][3], float B[3][3], float (&C)[3][3]){
  for(int r = 0; r < 3; ++r)
  {
    for(int c = 0; c < 3; ++c)
    {
      float entry = 0;
      for(int z = 0; z < 3; ++z)
      {
        entry += A[r][z]*B[z][c];
      }
      C[r][c] = entry;
    }
  }
}
void ssrlcv::multiply3x3x1_cpu(float A[3][3], float B[3], float (&C)[3]){
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

/*
fundamental matrix stuff
*/
float3 ssrlcv::multiply3x3x1(const float3 A[3], const float3 &B) {
  return {sum(A[0]*B),sum(A[1]*B),sum(A[2]*B)};
}
void ssrlcv::multiply3x3(const float3 A[3], const float3 B[3], float3 *C) {
  float3 bX = {B[0].x,B[1].x,B[2].x};
  float3 bY = {B[0].y,B[1].y,B[2].y};
  float3 bZ = {B[0].z,B[1].z,B[2].z};
  C[0] = {sum(A[0]*bX),sum(A[0]*bY),sum(A[0]*bZ)};
  C[1] = {sum(A[1]*bX),sum(A[1]*bY),sum(A[1]*bZ)};
  C[2] = {sum(A[2]*bX),sum(A[2]*bY),sum(A[2]*bZ)};
}
void ssrlcv::transpose3x3(const float3 M[3], float3 (&M_T)[3]) {
  M_T[0] = {M[0].x,M[1].x,M[2].x};
  M_T[1] = {M[0].y,M[1].y,M[2].y};
  M_T[2] = {M[0].z,M[1].z,M[2].z};
}
void ssrlcv::inverse3x3(float3 M[3], float3 (&Minv)[3]) {
  float d1 = M[1].y * M[2].z - M[2].y * M[1].z;
  float d2 = M[1].x * M[2].z - M[1].z * M[2].x;
  float d3 = M[1].x * M[2].y - M[1].y * M[2].x;
  float det = M[0].x*d1 - M[0].y*d2 + M[0].z*d3;
  if(det == 0) {
    // return pinv(M);
  }
  float invdet = 1/det;
  Minv[0].x = d1*invdet;
  Minv[0].y = (M[0].z*M[2].y - M[0].y*M[2].z) * invdet;
  Minv[0].z = (M[0].y*M[1].z - M[0].z*M[1].y) * invdet;
  Minv[1].x = -1 * d2 * invdet;
  Minv[1].y = (M[0].x*M[2].z - M[0].z*M[2].x) * invdet;
  Minv[1].z = (M[1].x*M[0].z - M[0].x*M[1].z) * invdet;
  Minv[2].x = d3 * invdet;
  Minv[2].y = (M[2].x*M[0].y - M[0].x*M[2].y) * invdet;
  Minv[2].z = (M[0].x*M[1].y - M[1].x*M[0].y) * invdet;
}
