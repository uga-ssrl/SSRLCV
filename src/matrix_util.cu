#include "matrix_util.cuh"

__device__ __host__ float ssrlcv::sum(const float3 &a){
  return a.x + a.y + a.z;
}
__device__ __host__ void ssrlcv::multiply(const float (&A)[9], const float (&B)[3][3], float (&C)[3][3]){
  for(int r = 0; r < 3; ++r){
    for(int c = 0; c < 3; ++c){
      float entry = 0;
      for(int z = 0; z < 3; ++z){
        entry += A[r*3 + z]*B[z][c];
      }
      C[r][c] = entry;
    }
  }
}
__device__ __host__ void ssrlcv::multiply(const float3 (&A)[3], const float3 (&B)[3], float3 (&C)[3]){
  for(int r = 0; r < 3; ++r){
    C[r].x = (A[r].x*B[0].x) + (A[r].y*B[1].x) + (A[r].z*B[2].x);
    C[r].y = (A[r].x*B[0].y) + (A[r].y*B[1].y) + (A[r].z*B[2].y);
    C[r].z = (A[r].x*B[0].z) + (A[r].y*B[1].z) + (A[r].z*B[2].z);
  }
}
__device__ __host__ void ssrlcv::multiply(const float (&A)[3][3], const float (&B)[3][3], float (&C)[3][3]){
  for(int r = 0; r < 3; ++r){
    for(int c = 0; c < 3; ++c){
      float entry = 0;
      for(int z = 0; z < 3; ++z){
        entry += A[r][z]*B[z][c];
      }
      C[r][c] = entry;
    }
  }
}
__device__ __host__ void ssrlcv::multiply(const float3 (&A)[3], const float4 (&B)[3], float4 (&C)[3]){
  for(int r = 0; r < 3; ++r){
    C[r].x = (A[r].x*B[0].x) + (A[r].y*B[1].x) + (A[r].z*B[2].x);
    C[r].y = (A[r].x*B[0].y) + (A[r].y*B[1].y) + (A[r].z*B[2].y);
    C[r].z = (A[r].x*B[0].z) + (A[r].y*B[1].z) + (A[r].z*B[2].z);
    C[r].w = (A[r].x*B[0].w) + (A[r].y*B[1].w) + (A[r].z*B[2].w);
  }
}
__device__ __host__ void ssrlcv::multiply(const float (&A)[9], const float (&B)[3], float (&C)[3]){
   for (int r = 0; r < 3; ++r){
    float val = 0;
    for (int c = 0; c < 3; ++c){
      val += A[r*3 + c] * B[c];
    }
    C[r] = val;
  }
}
__device__ __host__ void ssrlcv::multiply(const float3 (&A)[3], const float3 &B, float3 &C){
  C.x = (A[0].x * B.x) + (A[0].y * B.y) + (A[0].z * B.z);
  C.y = (A[1].x * B.x) + (A[1].y * B.y) + (A[1].z * B.z);
  C.z = (A[2].x * B.x) + (A[2].y * B.y) + (A[2].z * B.z);
}

__device__ __host__ void ssrlcv::multiply(const float4 (&A)[3], const float4 &B, float3 &C){
  C.x = (A[0].x * B.x) + (A[0].y * B.y) + (A[0].z * B.z) + (A[0].w * B.w);
  C.y = (A[1].x * B.x) + (A[1].y * B.y) + (A[1].z * B.z) + (A[1].w * B.w);
  C.z = (A[2].x * B.x) + (A[2].y * B.y) + (A[2].z * B.z) + (A[2].w * B.w);
}

__device__ __host__ void ssrlcv::multiply(const float (&A)[3][3], const float (&B)[3], float (&C)[3]){
  for (int r = 0; r < 3; ++r){
    float val = 0;
    for (int c = 0; c < 3; ++c){
      val += A[r][c] * B[c];
    }
    C[r] = val;
  }
}
__device__ __host__ void ssrlcv::multiply(const float (&A)[3], const float (&B)[3][3], float (&C)[3]){
  for (int c = 0; c < 3; ++c){
    float val = 0;
    for (int r = 0; r < 3; ++r){
      val += B[r][c] * A[r];
    }
    C[c] = val;
  }
}
__device__ __host__ void ssrlcv::multiply(const float (&A)[2][2], const float (&B)[2][2], float (&C)[2][2]){
   for(int r = 0; r < 2; ++r){
    for(int c = 0; c < 2; ++c){
      float entry = 0;
      for(int z = 0; z < 3; ++z){
        entry += A[r][z]*B[z][c];
      }
      C[r][c] = entry;
    }
  }
}

__device__ __host__ float ssrlcv::dotProduct(const float (&A)[3], const float (&B)[3]){
  return (A[0]*B[0]) + (A[1]*B[1]) + (A[2]*B[2]);
}

__device__ __host__ float ssrlcv::dotProduct(const float4 A, const float4 B){
  return (A.x*B.x) + (A.y*B.y) + (A.z*B.z) + (A.w*B.w);
}

__device__ __host__ float3 ssrlcv::crossProduct(const float3 A, const float3 B){
  return {(A.y * B.z - A.z * B.y),(A.z * B.x - A.x * B.z),(A.x * B.y - A.y * B.x)};
}

__device__ __host__ bool ssrlcv::inverse(const float (&M)[3][3], float (&M_out)[3][3]){
  float d1 = M[1][1] * M[2][2] - M[2][1] * M[1][2];
  float d2 = M[1][0] * M[2][2] - M[1][2] * M[2][0];
  float d3 = M[1][0] * M[2][1] - M[1][1] * M[2][0];
  float det = M[0][0]*d1 - M[0][1]*d2 + M[0][2]*d3;
  if(det == 0){
    return false;
  }
  float invdet = 1/det;
  M_out[0][0] = d1*invdet;
  M_out[0][1] = (M[0][2]*M[2][1] - M[0][1]*M[2][2]) * invdet;
  M_out[0][2] = (M[0][1]*M[1][2] - M[0][2]*M[1][1]) * invdet;
  M_out[1][0] = -1 * d2 * invdet;
  M_out[1][1] = (M[0][0]*M[2][2] - M[0][2]*M[2][0]) * invdet;
  M_out[1][2] = (M[1][0]*M[0][2] - M[0][0]*M[1][2]) * invdet;
  M_out[2][0] = d3 * invdet;
  M_out[2][1] = (M[2][0]*M[0][1] - M[0][0]*M[2][1]) * invdet;
  M_out[2][2] = (M[0][0]*M[1][1] - M[1][0]*M[0][1]) * invdet;
  return true;
}
__device__ __host__ bool ssrlcv::inverse(const float3 (&M)[3], float3 (&M_out)[3]){
  float d1 = M[1].y * M[2].z - M[2].y * M[1].z;
  float d2 = M[1].x * M[2].z - M[1].z * M[2].x;
  float d3 = M[1].x * M[2].y - M[1].y * M[2].x;
  float det = M[0].x*d1 - M[0].y*d2 + M[0].z*d3;
  if(det == 0){
    return false;
  }
  float invdet = 1/det;
  M_out[0].x = d1*invdet;
  M_out[0].y = (M[0].z*M[2].y - M[0].y*M[2].z) * invdet;
  M_out[0].z = (M[0].y*M[1].z - M[0].z*M[1].y) * invdet;
  M_out[1].x = -1 * d2 * invdet;
  M_out[1].y = (M[0].x*M[2].z - M[0].z*M[2].x) * invdet;
  M_out[1].z = (M[1].x*M[0].z - M[0].x*M[1].z) * invdet;
  M_out[2].x = d3 * invdet;
  M_out[2].y = (M[2].x*M[0].y - M[0].x*M[2].y) * invdet;
  M_out[2].z = (M[0].x*M[1].y - M[1].x*M[0].y) * invdet;
  return true;
}
__device__ __host__ void ssrlcv::transpose(const float (&M)[3][3], float (&M_out)[3][3]){
  for(int r = 0; r < 3; ++r){
    for(int c = 0; c < 3; ++c){
      M_out[r][c] = M[c][r];
    }
  }
}
__device__ __host__ void ssrlcv::transpose(const float3 (&M)[3], float3 (&M_out)[3]){
  M_out[0].x = M[0].x;
  M_out[0].y = M[1].x;
  M_out[0].z = M[2].x;
  M_out[1].x = M[0].y;
  M_out[1].y = M[1].y;
  M_out[1].z = M[2].y;
  M_out[2].x = M[0].z;
  M_out[2].y = M[1].z;
  M_out[2].z = M[2].z;
}

__device__ __host__ void ssrlcv::transpose(const float (&M)[2][2], float (&M_out)[2][2]){
  for(int r = 0; r < 2; ++r){
    for(int c = 0; c < 2; ++c){
      M_out[c][r] = M[r][c];
    }
  }
}

/**
 * \brief Multiplies a 3x1 vector by its transpose producing a 3x3 matrix.
 * \details Multiplies a 3x1 vector by its transpose producing a 3x3 matrix.
 * \param M vector
 * \param M_out the resulting matrix product
 */
__device__ __host__ void ssrlcv::matrixProduct(const float3 (&M), float3(&M_out)[3]){
  M_out[0].x = M.x * M.x;
  M_out[0].y = M.x * M.y;
  M_out[0].z = M.x * M.z;
  M_out[1].x = M.y * M.x;
  M_out[1].y = M.y * M.y;
  M_out[1].z = M.y * M.z;
  M_out[2].x = M.z * M.x;
  M_out[2].y = M.z * M.y;
  M_out[2].z = M.z * M.z;
}

__device__ __host__ float ssrlcv::determinant(const float (&M)[2][2]){
  return (M[0][0]*M[1][1]) - (M[0][1]*M[1][0]);
}
__device__ __host__ float ssrlcv::trace(const float(&M)[2][2]){
  return M[0][0] + M[1][1];
}
__device__ __host__ float ssrlcv::trace(const float(&M)[3][3]){
  return M[0][0] + M[1][1] + M[2][2];
}


__device__ __host__ void ssrlcv::normalize(float (&v)[3]){
  float mag = magnitude(v);
  if(mag > 0){
    v[0] = v[0]/mag;
    v[1] = v[1]/mag;
    v[2] = v[2]/mag;
  }
}
__device__ __host__ void ssrlcv::normalize(float3 &v){
  float mag = magnitude(v);
  if(mag > 0){
    v.x = v.x/mag;
    v.y = v.y/mag;
    v.z = v.z/mag;
  }
}
__device__ __host__ float ssrlcv::magnitude(const float (&v)[3]){
  return sqrtf(dotProduct({v[0],v[1],v[2]}, {v[0],v[1],v[2]}));
}
__device__ __host__ float ssrlcv::magnitude(const float3 &v){
  return sqrtf(dotProduct(v, v));
}

/**
 * computes the L2 norm of vector A
 */
__device__ __host__ float ssrlcv::L2norm(const float3 A){
  return sqrtf(A.x*A.x + A.y*A.y + A.z*A.z);
}

/**
 * \brief returns the euclidean distance of two points
 */
__device__ __host__ float ssrlcv::euclideanDistance(const float3 A, const float3 B){
  return sqrtf((A.x - B.x)*(A.x - B.x) + (A.y - B.y)*(A.y - B.y) + (A.z - B.z)*(A.z - B.z));
}

/**
 * \brief calcualtes x y and z rotations from an input rotation matrix
 * @param R a 3x3 rotation matrix
 * @return rotations a float3 of the x,y,z axis rotations
 */
__device__ __host__ float3 ssrlcv::getAxisRotations(const float(&R)[3][3]){
  float x = atan2f(R[2][1] , R[2][2]);
  float y = atan2f(-R[2][0] , (R[2][2]/cosf(x)));
  float z = atan2f(R[1][0] , R[0][0]);
  
  return {x, y, z};
}

/**
 * \brief calcualtes a rotation matrix from x, y, and z rotations
 * @param angle a float3 of the x,y,z axis rotations  
 * @param R output 3x3 rotation matrix
 */
__device__ __host__ void ssrlcv::getRotationMatrix(const float3 &angle, float(&R)[3][3]){
  R[0][0] = cosf(angle.z) * cosf(angle.y);
  R[0][1] = cosf(angle.z) * sinf(angle.y) * sinf(angle.x) - sinf(angle.z) * cosf(angle.x);
  R[0][2] = cosf(angle.z) * sinf(angle.y) * cosf(angle.x) + sinf(angle.z) * sinf(angle.x);
  R[1][0] = sinf(angle.z) * cosf(angle.y);
  R[1][1] = sinf(angle.z) * sinf(angle.y) * sinf(angle.x) + cosf(angle.z) * cosf(angle.x);
  R[1][2] = sinf(angle.z) * sinf(angle.y) * cosf(angle.x) - cosf(angle.z) * sinf(angle.x);
  R[2][0] = -1 * sinf(angle.y);
  R[2][1] = cosf(angle.y) * sinf(angle.x);
  R[2][2] = cosf(angle.y) * cosf(angle.x);
}

__device__ __host__ float3 ssrlcv::matrixMulVector(float3 x, float A[3][3]){
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
  float a = dotProduct(v,x_n);
  float b = dotProduct(v,v);
  float c = (a)/(sqrtf(b));
  angles.x = acosf(c);
  // y angle
  a = dotProduct(v,y_n);
  b = dotProduct(v,v);
  c = (a)/(sqrtf(b));
  angles.y = acosf(c);
  // z angle
  a = dotProduct(v,z_n);
  b = dotProduct(v,v);
  c = (a)/(sqrtf(b));
  angles.z = acosf(c);
  return angles;
}

/**
 * \brief Rotates a point around the x, y, and z.
 * \details Rotates a point around the x, y, and z axis by the given angles.
 * \param point point to rotate
 * \param angles angle to rotate
 * \return point after rotation
 */
__host__ __device__ float3 ssrlcv::rotatePoint(float3 point, float3 angle) {
  float rotationMatrix[3][3];
  rotationMatrix[0][0] = cosf(angle.z) * cosf(angle.y);
  rotationMatrix[0][1] = cosf(angle.z) * sinf(angle.y) * sinf(angle.x) - sinf(angle.z) * cosf(angle.x);
  rotationMatrix[0][2] = cosf(angle.z) * sinf(angle.y) * cosf(angle.x) + sinf(angle.z) * sinf(angle.x);
  rotationMatrix[1][0] = sinf(angle.z) * cosf(angle.y);
  rotationMatrix[1][1] = sinf(angle.z) * sinf(angle.y) * sinf(angle.x) + cosf(angle.z) * cosf(angle.x);
  rotationMatrix[1][2] = sinf(angle.z) * sinf(angle.y) * cosf(angle.x) - cosf(angle.z) * sinf(angle.x);
  rotationMatrix[2][0] = -1 * sinf(angle.y);
  rotationMatrix[2][1] = cosf(angle.y) * sinf(angle.x);
  rotationMatrix[2][2] = cosf(angle.y) * cosf(angle.x);
  point = matrixMulVector(point, rotationMatrix);
  return point;
}

/**
 * \brief Rotates a point around a given axis.
 * \details Rotates a point around a given axis given by a unit vector through the origin
 * by a given angle. The resulting point can be found by first rotating the axis
 * to the z-axis, performing the rotation, and rotating the axis back to the orginal orientation.
 * \param point point to rotate
 * \param axis axis to rotate
 * \param angle angle to rotate
 * \return point after the rotation
 */
__host__ __device__ float3 ssrlcv::rotatePointArbitrary(float3 point, float3 axis, float angle) {
  float rotationMatrix[3][3];
  float k = (1- cosf(angle));
  normalize(axis);
  rotationMatrix[0][0] = axis.x * axis.x * k + cosf(angle);
  rotationMatrix[0][1] = axis.x * axis.y * k - axis.z * sinf(angle);
  rotationMatrix[0][2] = axis.x * axis.z * k + axis.y * sinf(angle);
  rotationMatrix[1][0] = axis.x * axis.y * k + axis.z * sinf(angle);
  rotationMatrix[1][1] = axis.y * axis.y * k + cosf(angle);
  rotationMatrix[1][2] = axis.y * axis.z * k - axis.x * sinf(angle);
  rotationMatrix[2][0] = axis.x * axis.z * k - axis.y * sinf(angle);
  rotationMatrix[2][1] = axis.y * axis.z * k + axis.x * sinf(angle);
  rotationMatrix[2][2] = axis.z * axis.z * k + cosf(angle);
  point = matrixMulVector(point, rotationMatrix);
  return point;
}















































// yeet
