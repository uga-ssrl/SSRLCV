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
  __device__ __host__ float3 crossProduct(const float3 A, const float3 B);
  __device__ __host__ bool inverse(const float (&M)[3][3], float (&M_out)[3][3]);
  __device__ __host__ bool inverse(const float3 (&M)[3], float3 (&M_out)[3]);
  __device__ __host__ void transpose(const float3 (&M)[3], float3 (&M_out)[3]);
  __device__ __host__ void transpose(const float (&M)[3][3], float (&M_out)[3][3]);
  __device__ __host__ void transpose(const float (&M)[2][2], float (&M_out)[2][2]);

  /**
   * \brief Multiplies a 3x1 vector by its transpose producing a 3x3 matrix.
   * \details Multiplies a 3x1 vector by its transpose producing a 3x3 matrix.
   * \param M vector
   * \param M_out the resulting matrix product
   */
  __device__ __host__ void matrixProduct(const float3 (&M), float3(&M_out)[3]);

  __device__ __host__ float determinant(const float (&M)[2][2]);
  __device__ __host__ float trace(const float(&M)[2][2]);
  __device__ __host__ float trace(const float(&M)[3][3]);

  __device__ __host__ void normalize(float (&v)[3]);
  __device__ __host__ void normalize(float3 &v);
  __device__ __host__ float magnitude(const float (&v)[3]);
  __device__ __host__ float magnitude(const float3 &v);

  /**
   * computes the L2 norm of vector A
   */
  __device__ __host__ float L2norm(const float3 A);

  /**
   * \brief returns the euclidean distance of two points
   */
  __device__ __host__ float euclideanDistance(const float3 A, const float3 B);

  /**
   * \brief calcualtes x y and z rotations from an input rotation matrix
   * @param R a 3x3 rotation matrix
   * @return rotations a float3 of the x,y,z axis rotations
   */
  __device__ __host__ float3 getAxisRotations(const float(&R)[3][3]);

  __device__ float3 matrixMulVector(float3 x, float A[3][3]);
  __device__ float3 getVectorAngles(float3 v);

  /**
   * \brief Rotates a point around the x, y, and z.
   * \details Rotates a point around the x, y, and z axis by the given angles.
   * \param point point to rotate
   * \param angles angle to rotate
   * \return point after rotation
   */
  __device__ float3 rotatePoint(float3 point, float3 angles);

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
  __device__ float3 rotatePointArbitrary(float3 point, float3 axis, float angle);

}

#endif /* MATRIXUTIL_CUH */
