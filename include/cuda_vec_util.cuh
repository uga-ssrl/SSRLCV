/**
* \file cuda_vec_util.cuh
* \brief This file contains operator overloads and utility functions for cuda built in vectors.
*/
#pragma once
#ifndef CUDA_VEC_UTIL_CUH
#define CUDA_VEC_UTIL_CUH

#include "cuda_util.cuh"

__device__ void orderInt3(int3 &toOrder);

//TODO do this for all vector types and related type operations
/*
u&char1,2,3,4
u&short1,2,3,4
u&int1,2,3,4
u&long1,2,3,4
u&longlong1,2,3,4
float1,2,3,4
double1,2,3,4
*/

// =============================================================================================================
//
// Struct Definitions
//
// =============================================================================================================

struct colorPoint : float3 {
  u_char r;
  u_char b;
  u_char g;
};

/**
* \ingroup cuda_util
* \defgroup cuda_validation_vectors
* \{
*/

struct float2_b : float2{
  bool invalid;
};
struct float3_b : float3{
  bool invalid;
};
struct float4_b : float4{
  bool invalid;
};
struct double2_b : double2{
  bool invalid;
};
struct double3_b : double3{
  bool invalid;
};
struct double4_b : double4{
  bool invalid;
};
struct char2_b : char2{
  bool invalid;
};
struct char3_b : char3{
  bool invalid;
};
struct char4_b : char4{
  bool invalid;
};
struct short2_b : short2{
  bool invalid;
};
struct short3_b : short3{
  bool invalid;
};
struct short4_b : short4{
  bool invalid;
};
struct int2_b : int2{
  bool invalid;
};
struct int3_b : int3{
  bool invalid;
};
struct int4_b : int4{
  bool invalid;
};
struct long2_b : long2{
  bool invalid;
};
struct long3_b : long3{
  bool invalid;
};
struct long4_b : long4{
  bool invalid;
};
struct longlong2_b : longlong2{
  bool invalid;
};
struct longlong3_b : longlong3{
  bool invalid;
};
struct longlong4_b : longlong4{
  bool invalid;
};
struct uchar2_b : uchar2{
  bool invalid;
};
struct uchar3_b : uchar3{
  bool invalid;
};
struct uchar4_b : uchar4{
  bool invalid;
};
struct ushort2_b : ushort2{
  bool invalid;
};
struct ushort3_b : ushort3{
  bool invalid;
};
struct ushort4_b : ushort4{
  bool invalid;
};
struct uint2_b : uint2{
  bool invalid;
};
struct uint3_b : uint3{
  bool invalid;
};
struct uint4_b : uint4{
  bool invalid;
};
struct ulong2_b : ulong2{
  bool invalid;
};
struct ulong3_b : ulong3{
  bool invalid;
};
struct ulong4_b : ulong4{
  bool invalid;
};
struct ulonglong2_b : ulonglong2{
  bool invalid;
};
struct ulonglong3_b : ulonglong3{
  bool invalid;
};
struct ulonglong4_b : ulonglong4{
  bool invalid;
};

/**
* \}
*/

// =============================================================================================================
//
// Vector Removes
//
// =============================================================================================================

/**
* \ingroup cuda_util
* \defgroup cuda_vector_removes
* \see Unity
* \{
*/
void remove(ssrlcv::ptr::value<ssrlcv::Unity<float2_b>> array);
void remove(ssrlcv::ptr::value<ssrlcv::Unity<float3_b>> array);
void remove(ssrlcv::ptr::value<ssrlcv::Unity<float4_b>> array);
void remove(ssrlcv::ptr::value<ssrlcv::Unity<double2_b>> array);
void remove(ssrlcv::ptr::value<ssrlcv::Unity<double3_b>> array);
void remove(ssrlcv::ptr::value<ssrlcv::Unity<double4_b>> array);
void remove(ssrlcv::ptr::value<ssrlcv::Unity<char2_b>> array);
void remove(ssrlcv::ptr::value<ssrlcv::Unity<char3_b>> array);
void remove(ssrlcv::ptr::value<ssrlcv::Unity<char4_b>> array);
void remove(ssrlcv::ptr::value<ssrlcv::Unity<uchar2_b>> array);
void remove(ssrlcv::ptr::value<ssrlcv::Unity<uchar3_b>> array);
void remove(ssrlcv::ptr::value<ssrlcv::Unity<uchar4_b>> array);
void remove(ssrlcv::ptr::value<ssrlcv::Unity<short2_b>> array);
void remove(ssrlcv::ptr::value<ssrlcv::Unity<short3_b>> array);
void remove(ssrlcv::ptr::value<ssrlcv::Unity<short4_b>> array);
void remove(ssrlcv::ptr::value<ssrlcv::Unity<ushort2_b>> array);
void remove(ssrlcv::ptr::value<ssrlcv::Unity<ushort3_b>> array);
void remove(ssrlcv::ptr::value<ssrlcv::Unity<ushort4_b>> array);
void remove(ssrlcv::ptr::value<ssrlcv::Unity<int2_b>> array);
void remove(ssrlcv::ptr::value<ssrlcv::Unity<int3_b>> array);
void remove(ssrlcv::ptr::value<ssrlcv::Unity<int4_b>> array);
void remove(ssrlcv::ptr::value<ssrlcv::Unity<uint2_b>> array);
void remove(ssrlcv::ptr::value<ssrlcv::Unity<uint3_b>> array);
void remove(ssrlcv::ptr::value<ssrlcv::Unity<uint4_b>> array);
void remove(ssrlcv::ptr::value<ssrlcv::Unity<long2_b>> array);
void remove(ssrlcv::ptr::value<ssrlcv::Unity<long3_b>> array);
void remove(ssrlcv::ptr::value<ssrlcv::Unity<long4_b>> array);
void remove(ssrlcv::ptr::value<ssrlcv::Unity<ulong2_b>> array);
void remove(ssrlcv::ptr::value<ssrlcv::Unity<ulong3_b>> array);
void remove(ssrlcv::ptr::value<ssrlcv::Unity<ulong4_b>> array);
void remove(ssrlcv::ptr::value<ssrlcv::Unity<longlong2_b>> array);
void remove(ssrlcv::ptr::value<ssrlcv::Unity<longlong3_b>> array);
void remove(ssrlcv::ptr::value<ssrlcv::Unity<longlong4_b>> array);
void remove(ssrlcv::ptr::value<ssrlcv::Unity<ulonglong2_b>> array);
void remove(ssrlcv::ptr::value<ssrlcv::Unity<ulonglong3_b>> array);
void remove(ssrlcv::ptr::value<ssrlcv::Unity<ulonglong4_b>> array);
/**
* \}
* \ingroup cuda_util
* \defgroup cuda_vector_operators
*/

// =============================================================================================================
//
// Comparison Operators
//
// =============================================================================================================

/**
* \ingroup cuda_vector_operators
* \defgroup comparison
* \{
*/
__device__ __host__ bool operator==(const float2 &a, const float2 &b);
__device__ __host__ bool operator==(const float3 &a, const float3 &b);
__device__ __host__ bool operator==(const float4 &a, const float4 &b);
__device__ __host__ bool operator==(const double2 &a, const double2 &b);
__device__ __host__ bool operator==(const double3 &a, const double3 &b);
__device__ __host__ bool operator==(const double4 &a, const double4 &b);
__device__ __host__ bool operator==(const char2 &a, const char2 &b);
__device__ __host__ bool operator==(const char3 &a, const char3 &b);
__device__ __host__ bool operator==(const char4 &a, const char4 &b);
__device__ __host__ bool operator==(const uchar2 &a, const uchar2 &b);
__device__ __host__ bool operator==(const uchar3 &a, const uchar3 &b);
__device__ __host__ bool operator==(const uchar4 &a, const uchar4 &b);
__device__ __host__ bool operator==(const short2 &a, const short2 &b);
__device__ __host__ bool operator==(const short3 &a, const short3 &b);
__device__ __host__ bool operator==(const short4 &a, const short4 &b);
__device__ __host__ bool operator==(const ushort2 &a, const ushort2 &b);
__device__ __host__ bool operator==(const ushort3 &a, const ushort3 &b);
__device__ __host__ bool operator==(const ushort4 &a, const ushort4 &b);
__device__ __host__ bool operator==(const int2 &a, const int2 &b);
__device__ __host__ bool operator==(const int3 &a, const int3 &b);
__device__ __host__ bool operator==(const int4 &a, const int4 &b);
__device__ __host__ bool operator==(const uint2 &a, const uint2 &b);
__device__ __host__ bool operator==(const uint3 &a, const uint3 &b);
__device__ __host__ bool operator==(const uint4 &a, const uint4 &b);
__device__ __host__ bool operator==(const long2 &a, const long2 &b);
__device__ __host__ bool operator==(const long3 &a, const long3 &b);
__device__ __host__ bool operator==(const long4 &a, const long4 &b);
__device__ __host__ bool operator==(const ulong2 &a, const ulong2 &b);
__device__ __host__ bool operator==(const ulong3 &a, const ulong3 &b);
__device__ __host__ bool operator==(const ulong4 &a, const ulong4 &b);
__device__ __host__ bool operator==(const longlong2 &a, const longlong2 &b);
__device__ __host__ bool operator==(const longlong3 &a, const longlong3 &b);
__device__ __host__ bool operator==(const longlong4 &a, const longlong4 &b);
__device__ __host__ bool operator==(const ulonglong2 &a, const ulonglong2 &b);
__device__ __host__ bool operator==(const ulonglong3 &a, const ulonglong3 &b);
__device__ __host__ bool operator==(const ulonglong4 &a, const ulonglong4 &b);

__device__ __host__ bool operator!=(const float2 &a, const float2 &b);
__device__ __host__ bool operator!=(const float3 &a, const float3 &b);
__device__ __host__ bool operator!=(const float4 &a, const float4 &b);
__device__ __host__ bool operator!=(const double2 &a, const double2 &b);
__device__ __host__ bool operator!=(const double3 &a, const double3 &b);
__device__ __host__ bool operator!=(const double4 &a, const double4 &b);
__device__ __host__ bool operator!=(const char2 &a, const char2 &b);
__device__ __host__ bool operator!=(const char3 &a, const char3 &b);
__device__ __host__ bool operator!=(const char4 &a, const char4 &b);
__device__ __host__ bool operator!=(const uchar2 &a, const uchar2 &b);
__device__ __host__ bool operator!=(const uchar3 &a, const uchar3 &b);
__device__ __host__ bool operator!=(const uchar4 &a, const uchar4 &b);
__device__ __host__ bool operator!=(const short2 &a, const short2 &b);
__device__ __host__ bool operator!=(const short3 &a, const short3 &b);
__device__ __host__ bool operator!=(const short4 &a, const short4 &b);
__device__ __host__ bool operator!=(const ushort2 &a, const ushort2 &b);
__device__ __host__ bool operator!=(const ushort3 &a, const ushort3 &b);
__device__ __host__ bool operator!=(const ushort4 &a, const ushort4 &b);
__device__ __host__ bool operator!=(const int2 &a, const int2 &b);
__device__ __host__ bool operator!=(const int3 &a, const int3 &b);
__device__ __host__ bool operator!=(const int4 &a, const int4 &b);
__device__ __host__ bool operator!=(const uint2 &a, const uint2 &b);
__device__ __host__ bool operator!=(const uint3 &a, const uint3 &b);
__device__ __host__ bool operator!=(const uint4 &a, const uint4 &b);
__device__ __host__ bool operator!=(const long2 &a, const long2 &b);
__device__ __host__ bool operator!=(const long3 &a, const long3 &b);
__device__ __host__ bool operator!=(const long4 &a, const long4 &b);
__device__ __host__ bool operator!=(const ulong2 &a, const ulong2 &b);
__device__ __host__ bool operator!=(const ulong3 &a, const ulong3 &b);
__device__ __host__ bool operator!=(const ulong4 &a, const ulong4 &b);
__device__ __host__ bool operator!=(const longlong2 &a, const longlong2 &b);
__device__ __host__ bool operator!=(const longlong3 &a, const longlong3 &b);
__device__ __host__ bool operator!=(const longlong4 &a, const longlong4 &b);
__device__ __host__ bool operator!=(const ulonglong2 &a, const ulonglong2 &b);
__device__ __host__ bool operator!=(const ulonglong3 &a, const ulonglong3 &b);
__device__ __host__ bool operator!=(const ulonglong4 &a, const ulonglong4 &b);

__device__ __host__ bool operator<(const float2 &a, const float2 &b);
__device__ __host__ bool operator<(const float3 &a, const float3 &b);
__device__ __host__ bool operator<(const float4 &a, const float4 &b);
__device__ __host__ bool operator>(const float2 &a, const float2 &b);
__device__ __host__ bool operator>(const float3 &a, const float3 &b);
__device__ __host__ bool operator>(const float4 &a, const float4 &b);
__device__ __host__ bool operator<(const double2 &a, const double2 &b);
__device__ __host__ bool operator<(const double3 &a, const double3 &b);
__device__ __host__ bool operator<(const double4 &a, const double4 &b);
__device__ __host__ bool operator>(const double2 &a, const double2 &b);
__device__ __host__ bool operator>(const double3 &a, const double3 &b);
__device__ __host__ bool operator>(const double4 &a, const double4 &b);
__device__ __host__ bool operator<(const char2 &a, const char2 &b);
__device__ __host__ bool operator<(const char3 &a, const char3 &b);
__device__ __host__ bool operator<(const char4 &a, const char4 &b);
__device__ __host__ bool operator>(const char2 &a, const char2 &b);
__device__ __host__ bool operator>(const char3 &a, const char3 &b);
__device__ __host__ bool operator>(const char4 &a, const char4 &b);
__device__ __host__ bool operator<(const uchar2 &a, const uchar2 &b);
__device__ __host__ bool operator<(const uchar3 &a, const uchar3 &b);
__device__ __host__ bool operator<(const uchar4 &a, const uchar4 &b);
__device__ __host__ bool operator>(const uchar2 &a, const uchar2 &b);
__device__ __host__ bool operator>(const uchar3 &a, const uchar3 &b);
__device__ __host__ bool operator>(const uchar4 &a, const uchar4 &b);
__device__ __host__ bool operator<(const short2 &a, const short2 &b);
__device__ __host__ bool operator<(const short3 &a, const short3 &b);
__device__ __host__ bool operator<(const short4 &a, const short4 &b);
__device__ __host__ bool operator>(const short2 &a, const short2 &b);
__device__ __host__ bool operator>(const short3 &a, const short3 &b);
__device__ __host__ bool operator>(const short4 &a, const short4 &b);
__device__ __host__ bool operator<(const ushort2 &a, const ushort2 &b);
__device__ __host__ bool operator<(const ushort3 &a, const ushort3 &b);
__device__ __host__ bool operator<(const ushort4 &a, const ushort4 &b);
__device__ __host__ bool operator>(const ushort2 &a, const ushort2 &b);
__device__ __host__ bool operator>(const ushort3 &a, const ushort3 &b);
__device__ __host__ bool operator>(const ushort4 &a, const ushort4 &b);
__device__ __host__ bool operator<(const int2 &a, const int2 &b);
__device__ __host__ bool operator<(const int3 &a, const int3 &b);
__device__ __host__ bool operator<(const int4 &a, const int4 &b);
__device__ __host__ bool operator>(const int2 &a, const int2 &b);
__device__ __host__ bool operator>(const int3 &a, const int3 &b);
__device__ __host__ bool operator>(const int4 &a, const int4 &b);
__device__ __host__ bool operator<(const uint2 &a, const uint2 &b);
__device__ __host__ bool operator<(const uint3 &a, const uint3 &b);
__device__ __host__ bool operator<(const uint4 &a, const uint4 &b);
__device__ __host__ bool operator>(const uint2 &a, const uint2 &b);
__device__ __host__ bool operator>(const uint3 &a, const uint3 &b);
__device__ __host__ bool operator>(const uint4 &a, const uint4 &b);
__device__ __host__ bool operator<(const long2 &a, const long2 &b);
__device__ __host__ bool operator<(const long3 &a, const long3 &b);
__device__ __host__ bool operator<(const long4 &a, const long4 &b);
__device__ __host__ bool operator>(const long2 &a, const long2 &b);
__device__ __host__ bool operator>(const long3 &a, const long3 &b);
__device__ __host__ bool operator>(const long4 &a, const long4 &b);
__device__ __host__ bool operator<(const ulong2 &a, const ulong2 &b);
__device__ __host__ bool operator<(const ulong3 &a, const ulong3 &b);
__device__ __host__ bool operator<(const ulong4 &a, const ulong4 &b);
__device__ __host__ bool operator>(const ulong2 &a, const ulong2 &b);
__device__ __host__ bool operator>(const ulong3 &a, const ulong3 &b);
__device__ __host__ bool operator>(const ulong4 &a, const ulong4 &b);
__device__ __host__ bool operator<(const longlong2 &a, const longlong2 &b);
__device__ __host__ bool operator<(const longlong3 &a, const longlong3 &b);
__device__ __host__ bool operator<(const longlong4 &a, const longlong4 &b);
__device__ __host__ bool operator>(const longlong2 &a, const longlong2 &b);
__device__ __host__ bool operator>(const longlong3 &a, const longlong3 &b);
__device__ __host__ bool operator>(const longlong4 &a, const longlong4 &b);
__device__ __host__ bool operator<(const ulonglong2 &a, const ulonglong2 &b);
__device__ __host__ bool operator<(const ulonglong3 &a, const ulonglong3 &b);
__device__ __host__ bool operator<(const ulonglong4 &a, const ulonglong4 &b);
__device__ __host__ bool operator>(const ulonglong2 &a, const ulonglong2 &b);
__device__ __host__ bool operator>(const ulonglong3 &a, const ulonglong3 &b);
__device__ __host__ bool operator>(const ulonglong4 &a, const ulonglong4 &b);


__device__ __host__ bool operator<=(const float2 &a, const float2 &b);
__device__ __host__ bool operator<=(const float3 &a, const float3 &b);
__device__ __host__ bool operator<=(const float4 &a, const float4 &b);
__device__ __host__ bool operator>=(const float2 &a, const float2 &b);
__device__ __host__ bool operator>=(const float3 &a, const float3 &b);
__device__ __host__ bool operator>=(const float4 &a, const float4 &b);
__device__ __host__ bool operator<=(const double2 &a, const double2 &b);
__device__ __host__ bool operator<=(const double3 &a, const double3 &b);
__device__ __host__ bool operator<=(const double4 &a, const double4 &b);
__device__ __host__ bool operator>=(const double2 &a, const double2 &b);
__device__ __host__ bool operator>=(const double3 &a, const double3 &b);
__device__ __host__ bool operator>=(const double4 &a, const double4 &b);
__device__ __host__ bool operator<=(const char2 &a, const char2 &b);
__device__ __host__ bool operator<=(const char3 &a, const char3 &b);
__device__ __host__ bool operator<=(const char4 &a, const char4 &b);
__device__ __host__ bool operator>=(const char2 &a, const char2 &b);
__device__ __host__ bool operator>=(const char3 &a, const char3 &b);
__device__ __host__ bool operator>=(const char4 &a, const char4 &b);
__device__ __host__ bool operator<=(const uchar2 &a, const uchar2 &b);
__device__ __host__ bool operator<=(const uchar3 &a, const uchar3 &b);
__device__ __host__ bool operator<=(const uchar4 &a, const uchar4 &b);
__device__ __host__ bool operator>=(const uchar2 &a, const uchar2 &b);
__device__ __host__ bool operator>=(const uchar3 &a, const uchar3 &b);
__device__ __host__ bool operator>=(const uchar4 &a, const uchar4 &b);
__device__ __host__ bool operator<=(const short2 &a, const short2 &b);
__device__ __host__ bool operator<=(const short3 &a, const short3 &b);
__device__ __host__ bool operator<=(const short4 &a, const short4 &b);
__device__ __host__ bool operator>=(const short2 &a, const short2 &b);
__device__ __host__ bool operator>=(const short3 &a, const short3 &b);
__device__ __host__ bool operator>=(const short4 &a, const short4 &b);
__device__ __host__ bool operator<=(const ushort2 &a, const ushort2 &b);
__device__ __host__ bool operator<=(const ushort3 &a, const ushort3 &b);
__device__ __host__ bool operator<=(const ushort4 &a, const ushort4 &b);
__device__ __host__ bool operator>=(const ushort2 &a, const ushort2 &b);
__device__ __host__ bool operator>=(const ushort3 &a, const ushort3 &b);
__device__ __host__ bool operator>=(const ushort4 &a, const ushort4 &b);
__device__ __host__ bool operator<=(const int2 &a, const int2 &b);
__device__ __host__ bool operator<=(const int3 &a, const int3 &b);
__device__ __host__ bool operator<=(const int4 &a, const int4 &b);
__device__ __host__ bool operator>=(const int2 &a, const int2 &b);
__device__ __host__ bool operator>=(const int3 &a, const int3 &b);
__device__ __host__ bool operator>=(const int4 &a, const int4 &b);
__device__ __host__ bool operator<=(const uint2 &a, const uint2 &b);
__device__ __host__ bool operator<=(const uint3 &a, const uint3 &b);
__device__ __host__ bool operator<=(const uint4 &a, const uint4 &b);
__device__ __host__ bool operator>=(const uint2 &a, const uint2 &b);
__device__ __host__ bool operator>=(const uint3 &a, const uint3 &b);
__device__ __host__ bool operator>=(const uint4 &a, const uint4 &b);
__device__ __host__ bool operator<=(const long2 &a, const long2 &b);
__device__ __host__ bool operator<=(const long3 &a, const long3 &b);
__device__ __host__ bool operator<=(const long4 &a, const long4 &b);
__device__ __host__ bool operator>=(const long2 &a, const long2 &b);
__device__ __host__ bool operator>=(const long3 &a, const long3 &b);
__device__ __host__ bool operator>=(const long4 &a, const long4 &b);
__device__ __host__ bool operator<=(const ulong2 &a, const ulong2 &b);
__device__ __host__ bool operator<=(const ulong3 &a, const ulong3 &b);
__device__ __host__ bool operator<=(const ulong4 &a, const ulong4 &b);
__device__ __host__ bool operator>=(const ulong2 &a, const ulong2 &b);
__device__ __host__ bool operator>=(const ulong3 &a, const ulong3 &b);
__device__ __host__ bool operator>=(const ulong4 &a, const ulong4 &b);
__device__ __host__ bool operator<=(const longlong2 &a, const longlong2 &b);
__device__ __host__ bool operator<=(const longlong3 &a, const longlong3 &b);
__device__ __host__ bool operator<=(const longlong4 &a, const longlong4 &b);
__device__ __host__ bool operator>=(const longlong2 &a, const longlong2 &b);
__device__ __host__ bool operator>=(const longlong3 &a, const longlong3 &b);
__device__ __host__ bool operator>=(const longlong4 &a, const longlong4 &b);
__device__ __host__ bool operator<=(const ulonglong2 &a, const ulonglong2 &b);
__device__ __host__ bool operator<=(const ulonglong3 &a, const ulonglong3 &b);
__device__ __host__ bool operator<=(const ulonglong4 &a, const ulonglong4 &b);
__device__ __host__ bool operator>=(const ulonglong2 &a, const ulonglong2 &b);
__device__ __host__ bool operator>=(const ulonglong3 &a, const ulonglong3 &b);
__device__ __host__ bool operator>=(const ulonglong4 &a, const ulonglong4 &b);
/**
* \}
*/

// =============================================================================================================
//
// Vector Dot Products
//
// =============================================================================================================

__device__ __host__ float dotProduct(const float2 &a, const float2 &b);
__device__ __host__ float dotProduct(const float3 &a, const float3 &b);
__device__ __host__ int dotProduct(const int2 &a, const int2 &b);

// =============================================================================================================
//
// Arithmetic Operators
//
// =============================================================================================================

/**
* \ingroup cuda_vector_operators
* \defgroup arithmetic
* \{
*/
__device__ __host__ float3 operator+(const float3 &a, const float3 &b);
__device__ __host__ float3 operator-(const float3 &a, const float3 &b);
__device__ __host__ float3 operator/(const float3 &a, const float3 &b);
__device__ __host__ float3 operator*(const float3 &a, const float3 &b);
__device__ __host__ float2 operator+(const float2 &a, const float2 &b);
__device__ __host__ float2 operator-(const float2 &a, const float2 &b);
__device__ __host__ float2 operator/(const float2 &a, const float2 &b);
__device__ __host__ float2 operator*(const float2 &a, const float2 &b);
__device__ __host__ int2 operator+(const int2 &a, const int2 &b);
__device__ __host__ int2 operator-(const int2 &a, const int2 &b);
__device__ __host__ int2 operator/(const int2 &a, const int2 &b);
__device__ __host__ int2 operator*(const int2 &a, const int2 &b);
__device__ __host__ uint2 operator+(const uint2 &a, const uint2 &b);
__device__ __host__ uint2 operator-(const uint2 &a, const uint2 &b);
__device__ __host__ uint2 operator*(const uint2 &a, const uint2 &b);
__device__ __host__ uint2 operator/(const uint2 &a, const uint2 &b);
/**
* \}
*/

// =============================================================================================================
//
// Compound Assignment Operators
//
// =============================================================================================================

/**
* \ingroup cuda_vector_operators
* \defgroup compound_assignment_operators
* \{
*/

__device__ __host__ float4& operator+=(float4 &a, const float4 &b);
__device__ __host__ float3& operator+=(float3 &a, const float3 &b);
__device__ __host__ float2& operator+=(float2 &a, const float2 &b);
__device__ __host__ float4& operator+=(float4 &a, const int4 &b);
__device__ __host__ float3& operator+=(float3 &a, const int3 &b);
__device__ __host__ float2& operator+=(float2 &a, const int2 &b);
__device__ __host__ int4& operator+=(int4 &a, const float4 &b);
__device__ __host__ int3& operator+=(int3 &a, const float3 &b);
__device__ __host__ int2& operator+=(int2 &a, const float2 &b);
__device__ __host__ int4& operator+=(int4 &a, const int4 &b);
__device__ __host__ int3& operator+=(int3 &a, const int3 &b);
__device__ __host__ int2& operator+=(int2 &a, const int2 &b);

__device__ __host__ float4& operator-=(float4 &a, const float4 &b);
__device__ __host__ float3& operator-=(float3 &a, const float3 &b);
__device__ __host__ float2& operator-=(float2 &a, const float2 &b);
__device__ __host__ float4& operator-=(float4 &a, const int4 &b);
__device__ __host__ float3& operator-=(float3 &a, const int3 &b);
__device__ __host__ float2& operator-=(float2 &a, const int2 &b);
__device__ __host__ int4& operator-=(int4 &a, const float4 &b);
__device__ __host__ int3& operator-=(int3 &a, const float3 &b);
__device__ __host__ int2& operator-=(int2 &a, const float2 &b);
__device__ __host__ int4& operator-=(int4 &a, const int4 &b);
__device__ __host__ int3& operator-=(int3 &a, const int3 &b);
__device__ __host__ int2& operator-=(int2 &a, const int2 &b);

__device__ __host__ float4& operator*=(float4 &a, const float &b);
__device__ __host__ float3& operator*=(float3 &a, const float &b);
__device__ __host__ float2& operator*=(float2 &a, const float &b);
__device__ __host__ float4& operator*=(float4 &a, const int &b);
__device__ __host__ float3& operator*=(float3 &a, const int &b);
__device__ __host__ float2& operator*=(float2 &a, const int &b);
__device__ __host__ int4& operator*=(int4 &a, const float &b);
__device__ __host__ int3& operator*=(int3 &a, const float &b);
__device__ __host__ int2& operator*=(int2 &a, const float &b);
__device__ __host__ int4& operator*=(int4 &a, const int &b);
__device__ __host__ int3& operator*=(int3 &a, const int &b);
__device__ __host__ int2& operator*=(int2 &a, const int &b);

__device__ __host__ float4& operator/=(float4 &a, const float &b);
__device__ __host__ float3& operator/=(float3 &a, const float &b);
__device__ __host__ float2& operator/=(float2 &a, const float &b);
__device__ __host__ float4& operator/=(float4 &a, const int &b);
__device__ __host__ float3& operator/=(float3 &a, const int &b);
__device__ __host__ float2& operator/=(float2 &a, const int &b);
__device__ __host__ int4& operator/=(int4 &a, const float &b);
__device__ __host__ int3& operator/=(int3 &a, const float &b);
__device__ __host__ int2& operator/=(int2 &a, const float &b);
__device__ __host__ int4& operator/=(int4 &a, const int &b);
__device__ __host__ int3& operator/=(int3 &a, const int &b);
__device__ __host__ int2& operator/=(int2 &a, const int &b);


/**
* \}
*/

// =============================================================================================================
//
// Missmatched Arithmetic Operators
//
// =============================================================================================================

/**
* \ingroup arithmetic
* \addtogroup mismatch
* \{
*/
__device__ __host__ float3 operator+(const float3 &a, const float &b);
__device__ __host__ float3 operator-(const float3 &a, const float &b);
__device__ __host__ float3 operator/(const float3 &a, const float &b);
__device__ __host__ float3 operator*(const float3 &a, const float &b);
__device__ __host__ float3 operator+(const float &a, const float3 &b);
__device__ __host__ float3 operator-(const float &a, const float3 &b);
__device__ __host__ float3 operator/(const float &a, const float3 &b);
__device__ __host__ float3 operator*(const float &a, const float3 &b);


__device__ __host__ float2 operator+(const float2 &a, const float &b);
__device__ __host__ float2 operator-(const float2 &a, const float &b);
__device__ __host__ float2 operator/(const float2 &a, const float &b);
__device__ __host__ float2 operator*(const float2 &a, const float &b);
__device__ __host__ float2 operator+(const float &a, const float2 &b);
__device__ __host__ float2 operator-(const float &a, const float2 &b);
__device__ __host__ float2 operator/(const float &a, const float2 &b);
__device__ __host__ float2 operator*(const float &a, const float2 &b);

__device__ __host__ float2 operator+(const float2 &a, const int2 &b);
__device__ __host__ float2 operator-(const float2 &a, const int2 &b);
__device__ __host__ float2 operator/(const float2 &a, const int2 &b);
__device__ __host__ float2 operator*(const float2 &a, const int2 &b);
__device__ __host__ float2 operator+(const int2 &a, const float2 &b);
__device__ __host__ float2 operator-(const int2 &a, const float2 &b);
__device__ __host__ float2 operator/(const int2 &a, const float2 &b);
__device__ __host__ float2 operator*(const int2 &a, const float2 &b);


__device__ __host__ float2 operator/(const float2 &a, const int2 &b);
__device__ __host__ float2 operator/(const int2 &a, const float &b);


__device__ __host__ uint2 operator+(const uint2 &a, const int &b);
__device__ __host__ uint2 operator-(const uint2 &a, const int &b);
__device__ __host__ uint2 operator*(const uint2 &a, const int &b);
__device__ __host__ uint2 operator/(const uint2 &a, const int &b);
__device__ __host__ int2 operator+(const int2 &a, const uint2 &b);
__device__ __host__ int2 operator-(const int2 &a, const uint2 &b);
__device__ __host__ int2 operator*(const int2 &a, const uint2 &b);
__device__ __host__ int2 operator/(const int2 &a, const uint2 &b);
__device__ __host__ int2 operator+(const uint2 &a, const int2 &b);
__device__ __host__ int2 operator-(const uint2 &a, const int2 &b);
__device__ __host__ int2 operator*(const uint2 &a, const int2 &b);
__device__ __host__ int2 operator/(const uint2 &a, const int2 &b);

__device__ __host__ float2 operator+(const int2 &a, const float &b);
__device__ __host__ float2 operator-(const int2 &a, const float &b);
__device__ __host__ float2 operator/(const float2 &a, const float &b);
__device__ __host__ float2 operator*(const int2 &a, const float &b);
__device__ __host__ float2 operator/(const int2 &a, const float &b);

__device__ __host__ int2 operator+(const int2 &a, const int &b);
__device__ __host__ int2 operator-(const int2 &a, const int &b);

__device__ __host__ ulong2 operator+(const ulong2 &a, const int2 &b);
__device__ __host__ ulong2 operator-(const ulong2 &a, const int2 &b);
__device__ __host__ ulong2 operator/(const ulong2 &a, const int2 &b);
__device__ __host__ ulong2 operator*(const ulong2 &a, const int2 &b);
__device__ __host__ ulong2 operator/(const ulong2 &a, const int2 &b);

__device__ __host__ ulong2 operator+(const int2 &a, const ulong2 &b);
__device__ __host__ ulong2 operator-(const int2 &a, const ulong2 &b);
__device__ __host__ ulong2 operator/(const int2 &a, const ulong2 &b);
__device__ __host__ ulong2 operator*(const int2 &a, const ulong2 &b);
__device__ __host__ ulong2 operator/(const int2 &a, const ulong2 &b);

__device__ __host__ bool operator>(const float2 &a, const float &b);
__device__ __host__ bool operator<(const float2 &a, const float &b);
__device__ __host__ bool operator>(const float2 &a, const int2 &b);
__device__ __host__ bool operator<(const float2 &a, const int2 &b);
/**
* \}
*/



#endif /* CUDA_VEC_UTIL_CUH */
