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

// this assumes we are ALWAYS starting with a rotation
// this should only be used for reprojection
// TODO this could be combined into one matrix, and thus one matrix multiplication
// we should move to that. I (caleb) only did it this way at first because it was much much easier to test
__device__ float3 ssrlcv::rotatePointKP(float3 point, float3 goal, float axangle) {

    // old
  // // we rotate in the z axis around <0 0 1> first, so we make a z rotation matrix
  // float zRotationMatrix[3][3];
  // zRotationMatrix[0][0] = cosf(axangle);
  // zRotationMatrix[0][1] = -1 * sinf(axangle);
  // zRotationMatrix[0][2] = 0;
  // zRotationMatrix[1][0] = sinf(axangle);
  // zRotationMatrix[1][1] = cosf(axangle);
  // zRotationMatrix[1][2] = 0;
  // zRotationMatrix[2][0] = 0;
  // zRotationMatrix[2][1] = 0;
  // zRotationMatrix[2][2] = 1;
  // point = matrixMulVector(point, zRotationMatrix);



  // now calculate the needed rotation in the x axis
  // this isn't explicitly given, so we have to calculate it
  // from our current heading <0 0 1>
  float3 tempgoal = {0.0, goal.y, goal.z};
  normalize(tempgoal);
  float xrot = acosf( dotProduct({1.0, 0.0, 0.0}, tempgoal));
  printf("xRotation Estimate: %f \n", xrot);
  float xRotationMatrix[3][3];
  xRotationMatrix[0][0] = 1;
  xRotationMatrix[0][1] = 0;
  xRotationMatrix[0][2] = 0;
  xRotationMatrix[1][0] = 0;
  xRotationMatrix[1][1] = cosf(xrot);
  xRotationMatrix[1][2] = -1 * sinf(xrot);
  xRotationMatrix[2][0] = 0;
  xRotationMatrix[2][1] = sinf(xrot);
  xRotationMatrix[2][2] = cosf(xrot);
  point = matrixMulVector(point, xRotationMatrix);
  // tempgoal =
  // Last step is to calculate the rotation in the y axis
  // using the same method and assumtion as above.
  float3 tempgoal2 = {goal.x, 0.0, goal.z};
  normalize(tempgoal2);
  float yrot = acosf( dotProduct({0.0, 1.0, 0.0}, tempgoal2));
  printf("yRotation Estimate: %f \n", yrot);
  float yRotationMatrix[3][3];
  yRotationMatrix[0][0] = cosf(yrot);
  yRotationMatrix[0][1] = 0;
  yRotationMatrix[0][2] = sinf(yrot);
  yRotationMatrix[1][0] = 0;
  yRotationMatrix[1][1] = 1;
  yRotationMatrix[1][2] = 0;
  yRotationMatrix[2][0] = -1 * sinf(yrot);
  yRotationMatrix[2][1] = 0;
  yRotationMatrix[2][2] = cosf(yrot);
  point = matrixMulVector(point, yRotationMatrix);
  // now calc the change in the z axis that is needed
  float3 tempgoal3 = {goal.x, goal.y, 0.0};
  normalize(tempgoal3);
  float zrot = acosf( dotProduct({0.0, 0.0, 1.0}, tempgoal3));
  printf("zRotation Estimate: %f \n", zrot);
  float zRotationMatrix[3][3];
  zRotationMatrix[0][0] = cosf(zrot);
  zRotationMatrix[0][1] = -1 * sinf(zrot);
  zRotationMatrix[0][2] = 0;
  zRotationMatrix[1][0] = sinf(zrot);
  zRotationMatrix[1][1] = cosf(zrot);
  zRotationMatrix[1][2] = 0;
  zRotationMatrix[2][0] = 0;
  zRotationMatrix[2][1] = 0;
  zRotationMatrix[2][2] = 1;
  point = matrixMulVector(point, zRotationMatrix);
  // then we're done!
  // then we're done!
  return point;
}


















































// yeet
