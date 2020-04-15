#include "cuda_vec_util.cuh"



//xorswap
__device__ void orderInt3(int3 &toOrder){
  if(toOrder.x > toOrder.y){
    toOrder.x ^= toOrder.y;
    toOrder.y ^= toOrder.x;
    toOrder.x ^= toOrder.y;
  }
  if(toOrder.x > toOrder.z){
    toOrder.x ^= toOrder.z;
    toOrder.z ^= toOrder.x;
    toOrder.x ^= toOrder.z;
  }
  if(toOrder.y > toOrder.z){
    toOrder.y ^= toOrder.z;
    toOrder.z ^= toOrder.y;
    toOrder.y ^= toOrder.z;
  }
}

// =============================================================================================================
//
// Vector Removes
//
// =============================================================================================================

void remove(ssrlcv::Unity<float2_b>* array){remove(array,check_if_valid<float2_b>);}
void remove(ssrlcv::Unity<float3_b>* array){remove(array,check_if_valid<float3_b>);}
void remove(ssrlcv::Unity<float4_b>* array){remove(array,check_if_valid<float4_b>);}
void remove(ssrlcv::Unity<double2_b>* array){remove(array,check_if_valid<double2_b>);}
void remove(ssrlcv::Unity<double3_b>* array){remove(array,check_if_valid<double3_b>);}
void remove(ssrlcv::Unity<double4_b>* array){remove(array,check_if_valid<double4_b>);}
void remove(ssrlcv::Unity<char2_b>* array){remove(array,check_if_valid<char2_b>);}
void remove(ssrlcv::Unity<char3_b>* array){remove(array,check_if_valid<char3_b>);}
void remove(ssrlcv::Unity<char4_b>* array){remove(array,check_if_valid<char4_b>);}
void remove(ssrlcv::Unity<uchar2_b>* array){remove(array,check_if_valid<uchar2_b>);}
void remove(ssrlcv::Unity<uchar3_b>* array){remove(array,check_if_valid<uchar3_b>);}
void remove(ssrlcv::Unity<uchar4_b>* array){remove(array,check_if_valid<uchar4_b>);}
void remove(ssrlcv::Unity<short2_b>* array){remove(array,check_if_valid<short2_b>);}
void remove(ssrlcv::Unity<short3_b>* array){remove(array,check_if_valid<short3_b>);}
void remove(ssrlcv::Unity<short4_b>* array){remove(array,check_if_valid<short4_b>);}
void remove(ssrlcv::Unity<ushort2_b>* array){remove(array,check_if_valid<ushort2_b>);}
void remove(ssrlcv::Unity<ushort3_b>* array){remove(array,check_if_valid<ushort3_b>);}
void remove(ssrlcv::Unity<ushort4_b>* array){remove(array,check_if_valid<ushort4_b>);}
void remove(ssrlcv::Unity<int2_b>* array){remove(array,check_if_valid<int2_b>);}
void remove(ssrlcv::Unity<int3_b>* array){remove(array,check_if_valid<int3_b>);}
void remove(ssrlcv::Unity<int4_b>* array){remove(array,check_if_valid<int4_b>);}
void remove(ssrlcv::Unity<uint2_b>* array){remove(array,check_if_valid<uint2_b>);}
void remove(ssrlcv::Unity<uint3_b>* array){remove(array,check_if_valid<uint3_b>);}
void remove(ssrlcv::Unity<uint4_b>* array){remove(array,check_if_valid<uint4_b>);}
void remove(ssrlcv::Unity<long2_b>* array){remove(array,check_if_valid<long2_b>);}
void remove(ssrlcv::Unity<long3_b>* array){remove(array,check_if_valid<long3_b>);}
void remove(ssrlcv::Unity<long4_b>* array){remove(array,check_if_valid<long4_b>);}
void remove(ssrlcv::Unity<ulong2_b>* array){remove(array,check_if_valid<ulong2_b>);}
void remove(ssrlcv::Unity<ulong3_b>* array){remove(array,check_if_valid<ulong3_b>);}
void remove(ssrlcv::Unity<ulong4_b>* array){remove(array,check_if_valid<ulong4_b>);}
void remove(ssrlcv::Unity<longlong2_b>* array){remove(array,check_if_valid<longlong2_b>);}
void remove(ssrlcv::Unity<longlong3_b>* array){remove(array,check_if_valid<longlong3_b>);}
void remove(ssrlcv::Unity<longlong4_b>* array){remove(array,check_if_valid<longlong4_b>);}
void remove(ssrlcv::Unity<ulonglong2_b>* array){remove(array,check_if_valid<ulonglong2_b>);}
void remove(ssrlcv::Unity<ulonglong3_b>* array){remove(array,check_if_valid<ulonglong3_b>);}
void remove(ssrlcv::Unity<ulonglong4_b>* array){remove(array,check_if_valid<ulonglong4_b>);}

// =============================================================================================================
//
// Comparison Operators
//
// =============================================================================================================

__device__ __host__ bool operator==(const float2 &a, const float2 &b){
  return a.x == b.x && a.y == b.y;
}
__device__ __host__ bool operator==(const float3 &a, const float3 &b){
  return a.x == b.x && a.y == b.y && a.z == b.z;
}
__device__ __host__ bool operator==(const float4 &a, const float4 &b){
  return a.x == b.x && a.y == b.y && a.z == b.z && a.w == b.w;
}
__device__ __host__ bool operator==(const double2 &a, const double2 &b){
  return a.x == b.x && a.y == b.y;
}
__device__ __host__ bool operator==(const double3 &a, const double3 &b){
  return a.x == b.x && a.y == b.y && a.z == b.z;
}
__device__ __host__ bool operator==(const double4 &a, const double4 &b){
  return a.x == b.x && a.y == b.y && a.z == b.z && a.w == b.w;
}
__device__ __host__ bool operator==(const char2 &a, const char2 &b){
  return a.x == b.x && a.y == b.y;
}
__device__ __host__ bool operator==(const char3 &a, const char3 &b){
  return a.x == b.x && a.y == b.y && a.z == b.z;
}
__device__ __host__ bool operator==(const char4 &a, const char4 &b){
  return a.x == b.x && a.y == b.y && a.z == b.z && a.w == b.w;
}
__device__ __host__ bool operator==(const uchar2 &a, const uchar2 &b){
  return a.x == b.x && a.y == b.y;
}
__device__ __host__ bool operator==(const uchar3 &a, const uchar3 &b){
  return a.x == b.x && a.y == b.y && a.z == b.z;
}
__device__ __host__ bool operator==(const uchar4 &a, const uchar4 &b){
  return a.x == b.x && a.y == b.y && a.z == b.z && a.w == b.w;
}
__device__ __host__ bool operator==(const short2 &a, const short2 &b){
  return a.x == b.x && a.y == b.y;
}
__device__ __host__ bool operator==(const short3 &a, const short3 &b){
  return a.x == b.x && a.y == b.y && a.z == b.z;
}
__device__ __host__ bool operator==(const short4 &a, const short4 &b){
  return a.x == b.x && a.y == b.y && a.z == b.z && a.w == b.w;
}
__device__ __host__ bool operator==(const ushort2 &a, const ushort2 &b){
  return a.x == b.x && a.y == b.y;
}
__device__ __host__ bool operator==(const ushort3 &a, const ushort3 &b){
  return a.x == b.x && a.y == b.y && a.z == b.z;
}
__device__ __host__ bool operator==(const ushort4 &a, const ushort4 &b){
  return a.x == b.x && a.y == b.y && a.z == b.z && a.w == b.w;
}
__device__ __host__ bool operator==(const int2 &a, const int2 &b){
  return a.x == b.x && a.y == b.y;
}
__device__ __host__ bool operator==(const int3 &a, const int3 &b){
  return a.x == b.x && a.y == b.y && a.z == b.z;
}
__device__ __host__ bool operator==(const int4 &a, const int4 &b){
  return a.x == b.x && a.y == b.y && a.z == b.z && a.w == b.w;
}
__device__ __host__ bool operator==(const uint2 &a, const uint2 &b){
  return a.x == b.x && a.y == b.y;
}
__device__ __host__ bool operator==(const uint3 &a, const uint3 &b){
  return a.x == b.x && a.y == b.y && a.z == b.z;
}
__device__ __host__ bool operator==(const uint4 &a, const uint4 &b){
  return a.x == b.x && a.y == b.y && a.z == b.z && a.w == b.w;
}
__device__ __host__ bool operator==(const long2 &a, const long2 &b){
  return a.x == b.x && a.y == b.y;
}
__device__ __host__ bool operator==(const long3 &a, const long3 &b){
  return a.x == b.x && a.y == b.y && a.z == b.z;
}
__device__ __host__ bool operator==(const long4 &a, const long4 &b){
  return a.x == b.x && a.y == b.y && a.z == b.z && a.w == b.w;
}
__device__ __host__ bool operator==(const ulong2 &a, const ulong2 &b){
  return a.x == b.x && a.y == b.y;
}
__device__ __host__ bool operator==(const ulong3 &a, const ulong3 &b){
  return a.x == b.x && a.y == b.y && a.z == b.z;
}
__device__ __host__ bool operator==(const ulong4 &a, const ulong4 &b){
  return a.x == b.x && a.y == b.y && a.z == b.z && a.w == b.w;
}
__device__ __host__ bool operator==(const longlong2 &a, const longlong2 &b){
  return a.x == b.x && a.y == b.y;
}
__device__ __host__ bool operator==(const longlong3 &a, const longlong3 &b){
  return a.x == b.x && a.y == b.y && a.z == b.z;
}
__device__ __host__ bool operator==(const longlong4 &a, const longlong4 &b){
  return a.x == b.x && a.y == b.y && a.z == b.z && a.w == b.w;
}
__device__ __host__ bool operator==(const ulonglong2 &a, const ulonglong2 &b){
  return a.x == b.x && a.y == b.y;
}
__device__ __host__ bool operator==(const ulonglong3 &a, const ulonglong3 &b){
  return a.x == b.x && a.y == b.y && a.z == b.z;
}
__device__ __host__ bool operator==(const ulonglong4 &a, const ulonglong4 &b){
  return a.x == b.x && a.y == b.y && a.z == b.z && a.w == b.w;
}


__device__ __host__ bool operator!=(const float2 &a, const float2 &b){
  return !(a==b);
}
__device__ __host__ bool operator!=(const float3 &a, const float3 &b){
  return !(a==b);
}
__device__ __host__ bool operator!=(const float4 &a, const float4 &b){
  return !(a==b);
}
__device__ __host__ bool operator!=(const double2 &a, const double2 &b){
  return !(a==b);
}
__device__ __host__ bool operator!=(const double3 &a, const double3 &b){
  return !(a==b);
}
__device__ __host__ bool operator!=(const double4 &a, const double4 &b){
  return !(a==b);
}
__device__ __host__ bool operator!=(const char2 &a, const char2 &b){
  return !(a==b);
}
__device__ __host__ bool operator!=(const char3 &a, const char3 &b){
  return !(a==b);
}
__device__ __host__ bool operator!=(const char4 &a, const char4 &b){
  return !(a==b);
}
__device__ __host__ bool operator!=(const uchar2 &a, const uchar2 &b){
  return !(a==b);
}
__device__ __host__ bool operator!=(const uchar3 &a, const uchar3 &b){
  return !(a==b);
}
__device__ __host__ bool operator!=(const uchar4 &a, const uchar4 &b){
  return !(a==b);
}
__device__ __host__ bool operator!=(const short2 &a, const short2 &b){
  return !(a==b);
}
__device__ __host__ bool operator!=(const short3 &a, const short3 &b){
  return !(a==b);
}
__device__ __host__ bool operator!=(const short4 &a, const short4 &b){
  return !(a==b);
}
__device__ __host__ bool operator!=(const ushort2 &a, const ushort2 &b){
  return !(a==b);
}
__device__ __host__ bool operator!=(const ushort3 &a, const ushort3 &b){
  return !(a==b);
}
__device__ __host__ bool operator!=(const ushort4 &a, const ushort4 &b){
  return !(a==b);
}
__device__ __host__ bool operator!=(const int2 &a, const int2 &b){
  return !(a==b);
}
__device__ __host__ bool operator!=(const int3 &a, const int3 &b){
  return !(a==b);
}
__device__ __host__ bool operator!=(const int4 &a, const int4 &b){
  return !(a==b);
}
__device__ __host__ bool operator!=(const uint2 &a, const uint2 &b){
  return !(a==b);
}
__device__ __host__ bool operator!=(const uint3 &a, const uint3 &b){
  return !(a==b);
}
__device__ __host__ bool operator!=(const uint4 &a, const uint4 &b){
  return !(a==b);
}
__device__ __host__ bool operator!=(const long2 &a, const long2 &b){
  return !(a==b);
}
__device__ __host__ bool operator!=(const long3 &a, const long3 &b){
  return !(a==b);
}
__device__ __host__ bool operator!=(const long4 &a, const long4 &b){
  return !(a==b);
}
__device__ __host__ bool operator!=(const ulong2 &a, const ulong2 &b){
  return !(a==b);
}
__device__ __host__ bool operator!=(const ulong3 &a, const ulong3 &b){
  return !(a==b);
}
__device__ __host__ bool operator!=(const ulong4 &a, const ulong4 &b){
  return !(a==b);
}
__device__ __host__ bool operator!=(const longlong2 &a, const longlong2 &b){
  return !(a==b);
}
__device__ __host__ bool operator!=(const longlong3 &a, const longlong3 &b){
  return !(a==b);
}
__device__ __host__ bool operator!=(const longlong4 &a, const longlong4 &b){
  return !(a==b);
}
__device__ __host__ bool operator!=(const ulonglong2 &a, const ulonglong2 &b){
  return !(a==b);
}
__device__ __host__ bool operator!=(const ulonglong3 &a, const ulonglong3 &b){
  return !(a==b);
}
__device__ __host__ bool operator!=(const ulonglong4 &a, const ulonglong4 &b){
  return !(a==b);
}


__device__ __host__ bool operator<(const float2 &a, const float2 &b){
  if(a == b) return false;
  else if(a.x == b.x) return a.y < b.y;
  else return a.x < b.x;
}
__device__ __host__ bool operator<(const float3 &a, const float3 &b){
  if(a == b) return false;
  else if(a.x == b.x){
    if(a.y == b.y){
      return a.z < b.z;
    }
    else return a.y < b.y;
  }
  else return a.x < b.x;
}
__device__ __host__ bool operator<(const float4 &a, const float4 &b){
  if(a == b) return false;
  else if(a.x == b.x){
    if(a.y == b.y){
      if(a.z == b.z){
        return a.w < b.w;
      }
      else return a.z < b.z;
    }
    else return a.y < b.y;
  }
  else return a.x < b.x;
}
__device__ __host__ bool operator>(const float2 &a, const float2 &b){
  return b < a;
}
__device__ __host__ bool operator>(const float3 &a, const float3 &b){
  return b < a;
}
__device__ __host__ bool operator>(const float4 &a, const float4 &b){
  return b < a;

}
__device__ __host__ bool operator<(const double2 &a, const double2 &b){
  if(a == b) return false;
  else if(a.x == b.x) return a.y < b.y;
  else return a.x < b.x;
}
__device__ __host__ bool operator<(const double3 &a, const double3 &b){
  if(a == b) return false;
  else if(a.x == b.x){
    if(a.y == b.y){
      return a.z < b.z;
    }
    else return a.y < b.y;
  }
  else return a.x < b.x;
}
__device__ __host__ bool operator<(const double4 &a, const double4 &b){
  if(a == b) return false;
  else if(a.x == b.x){
    if(a.y == b.y){
      if(a.z == b.z){
        return a.w < b.w;
      }
      else return a.z < b.z;
    }
    else return a.y < b.y;
  }
  else return a.x < b.x;
}
__device__ __host__ bool operator>(const double2 &a, const double2 &b){
  return b < a;
}
__device__ __host__ bool operator>(const double3 &a, const double3 &b){
  return b < a;
}
__device__ __host__ bool operator>(const double4 &a, const double4 &b){
  return b < a;

}
__device__ __host__ bool operator<(const char2 &a, const char2 &b){
  if(a == b) return false;
  else if(a.x == b.x) return a.y < b.y;
  else return a.x < b.x;
}
__device__ __host__ bool operator<(const char3 &a, const char3 &b){
  if(a == b) return false;
  else if(a.x == b.x){
    if(a.y == b.y){
      return a.z < b.z;
    }
    else return a.y < b.y;
  }
  else return a.x < b.x;
}
__device__ __host__ bool operator<(const char4 &a, const char4 &b){
  if(a == b) return false;
  else if(a.x == b.x){
    if(a.y == b.y){
      if(a.z == b.z){
        return a.w < b.w;
      }
      else return a.z < b.z;
    }
    else return a.y < b.y;
  }
  else return a.x < b.x;
}
__device__ __host__ bool operator>(const char2 &a, const char2 &b){
  return b < a;
}
__device__ __host__ bool operator>(const char3 &a, const char3 &b){
  return b < a;
}
__device__ __host__ bool operator>(const char4 &a, const char4 &b){
  return b < a;

}
__device__ __host__ bool operator<(const uchar2 &a, const uchar2 &b){
  if(a == b) return false;
  else if(a.x == b.x) return a.y < b.y;
  else return a.x < b.x;
}
__device__ __host__ bool operator<(const uchar3 &a, const uchar3 &b){
  if(a == b) return false;
  else if(a.x == b.x){
    if(a.y == b.y){
      return a.z < b.z;
    }
    else return a.y < b.y;
  }
  else return a.x < b.x;
}
__device__ __host__ bool operator<(const uchar4 &a, const uchar4 &b){
  if(a == b) return false;
  else if(a.x == b.x){
    if(a.y == b.y){
      if(a.z == b.z){
        return a.w < b.w;
      }
      else return a.z < b.z;
    }
    else return a.y < b.y;
  }
  else return a.x < b.x;
}
__device__ __host__ bool operator>(const uchar2 &a, const uchar2 &b){
  return b < a;
}
__device__ __host__ bool operator>(const uchar3 &a, const uchar3 &b){
  return b < a;
}
__device__ __host__ bool operator>(const uchar4 &a, const uchar4 &b){
  return b < a;

}
__device__ __host__ bool operator<(const short2 &a, const short2 &b){
  if(a == b) return false;
  else if(a.x == b.x) return a.y < b.y;
  else return a.x < b.x;
}
__device__ __host__ bool operator<(const short3 &a, const short3 &b){
  if(a == b) return false;
  else if(a.x == b.x){
    if(a.y == b.y){
      return a.z < b.z;
    }
    else return a.y < b.y;
  }
  else return a.x < b.x;
}
__device__ __host__ bool operator<(const short4 &a, const short4 &b){
  if(a == b) return false;
  else if(a.x == b.x){
    if(a.y == b.y){
      if(a.z == b.z){
        return a.w < b.w;
      }
      else return a.z < b.z;
    }
    else return a.y < b.y;
  }
  else return a.x < b.x;
}
__device__ __host__ bool operator>(const short2 &a, const short2 &b){
  return b < a;
}
__device__ __host__ bool operator>(const short3 &a, const short3 &b){
  return b < a;
}
__device__ __host__ bool operator>(const short4 &a, const short4 &b){
  return b < a;

}
__device__ __host__ bool operator<(const ushort2 &a, const ushort2 &b){
  if(a == b) return false;
  else if(a.x == b.x) return a.y < b.y;
  else return a.x < b.x;
}
__device__ __host__ bool operator<(const ushort3 &a, const ushort3 &b){
  if(a == b) return false;
  else if(a.x == b.x){
    if(a.y == b.y){
      return a.z < b.z;
    }
    else return a.y < b.y;
  }
  else return a.x < b.x;
}
__device__ __host__ bool operator<(const ushort4 &a, const ushort4 &b){
  if(a == b) return false;
  else if(a.x == b.x){
    if(a.y == b.y){
      if(a.z == b.z){
        return a.w < b.w;
      }
      else return a.z < b.z;
    }
    else return a.y < b.y;
  }
  else return a.x < b.x;
}
__device__ __host__ bool operator>(const ushort2 &a, const ushort2 &b){
  return b < a;
}
__device__ __host__ bool operator>(const ushort3 &a, const ushort3 &b){
  return b < a;
}
__device__ __host__ bool operator>(const ushort4 &a, const ushort4 &b){
  return b < a;

}
__device__ __host__ bool operator<(const int2 &a, const int2 &b){
  if(a == b) return false;
  else if(a.x == b.x) return a.y < b.y;
  else return a.x < b.x;
}
__device__ __host__ bool operator<(const int3 &a, const int3 &b){
  if(a == b) return false;
  else if(a.x == b.x){
    if(a.y == b.y){
      return a.z < b.z;
    }
    else return a.y < b.y;
  }
  else return a.x < b.x;
}
__device__ __host__ bool operator<(const int4 &a, const int4 &b){
  if(a == b) return false;
  else if(a.x == b.x){
    if(a.y == b.y){
      if(a.z == b.z){
        return a.w < b.w;
      }
      else return a.z < b.z;
    }
    else return a.y < b.y;
  }
  else return a.x < b.x;
}
__device__ __host__ bool operator>(const int2 &a, const int2 &b){
  return b < a;
}
__device__ __host__ bool operator>(const int3 &a, const int3 &b){
  return b < a;
}
__device__ __host__ bool operator>(const int4 &a, const int4 &b){
  return b < a;

}
__device__ __host__ bool operator<(const uint2 &a, const uint2 &b){
  if(a == b) return false;
  else if(a.x == b.x) return a.y < b.y;
  else return a.x < b.x;
}
__device__ __host__ bool operator<(const uint3 &a, const uint3 &b){
  if(a == b) return false;
  else if(a.x == b.x){
    if(a.y == b.y){
      return a.z < b.z;
    }
    else return a.y < b.y;
  }
  else return a.x < b.x;
}
__device__ __host__ bool operator<(const uint4 &a, const uint4 &b){
  if(a == b) return false;
  else if(a.x == b.x){
    if(a.y == b.y){
      if(a.z == b.z){
        return a.w < b.w;
      }
      else return a.z < b.z;
    }
    else return a.y < b.y;
  }
  else return a.x < b.x;
}
__device__ __host__ bool operator>(const uint2 &a, const uint2 &b){
  return b < a;
}
__device__ __host__ bool operator>(const uint3 &a, const uint3 &b){
  return b < a;
}
__device__ __host__ bool operator>(const uint4 &a, const uint4 &b){
  return b < a;

}
__device__ __host__ bool operator<(const long2 &a, const long2 &b){
  if(a == b) return false;
  else if(a.x == b.x) return a.y < b.y;
  else return a.x < b.x;
}
__device__ __host__ bool operator<(const long3 &a, const long3 &b){
  if(a == b) return false;
  else if(a.x == b.x){
    if(a.y == b.y){
      return a.z < b.z;
    }
    else return a.y < b.y;
  }
  else return a.x < b.x;
}
__device__ __host__ bool operator<(const long4 &a, const long4 &b){
  if(a == b) return false;
  else if(a.x == b.x){
    if(a.y == b.y){
      if(a.z == b.z){
        return a.w < b.w;
      }
      else return a.z < b.z;
    }
    else return a.y < b.y;
  }
  else return a.x < b.x;
}
__device__ __host__ bool operator>(const long2 &a, const long2 &b){
  return b < a;
}
__device__ __host__ bool operator>(const long3 &a, const long3 &b){
  return b < a;
}
__device__ __host__ bool operator>(const long4 &a, const long4 &b){
  return b < a;

}
__device__ __host__ bool operator<(const ulong2 &a, const ulong2 &b){
  if(a == b) return false;
  else if(a.x == b.x) return a.y < b.y;
  else return a.x < b.x;
}
__device__ __host__ bool operator<(const ulong3 &a, const ulong3 &b){
  if(a == b) return false;
  else if(a.x == b.x){
    if(a.y == b.y){
      return a.z < b.z;
    }
    else return a.y < b.y;
  }
  else return a.x < b.x;
}
__device__ __host__ bool operator<(const ulong4 &a, const ulong4 &b){
  if(a == b) return false;
  else if(a.x == b.x){
    if(a.y == b.y){
      if(a.z == b.z){
        return a.w < b.w;
      }
      else return a.z < b.z;
    }
    else return a.y < b.y;
  }
  else return a.x < b.x;
}
__device__ __host__ bool operator>(const ulong2 &a, const ulong2 &b){
  return b < a;
}
__device__ __host__ bool operator>(const ulong3 &a, const ulong3 &b){
  return b < a;
}
__device__ __host__ bool operator>(const ulong4 &a, const ulong4 &b){
  return b < a;

}
__device__ __host__ bool operator<(const longlong2 &a, const longlong2 &b){
  if(a == b) return false;
  else if(a.x == b.x) return a.y < b.y;
  else return a.x < b.x;
}
__device__ __host__ bool operator<(const longlong3 &a, const longlong3 &b){
  if(a == b) return false;
  else if(a.x == b.x){
    if(a.y == b.y){
      return a.z < b.z;
    }
    else return a.y < b.y;
  }
  else return a.x < b.x;
}
__device__ __host__ bool operator<(const longlong4 &a, const longlong4 &b){
  if(a == b) return false;
  else if(a.x == b.x){
    if(a.y == b.y){
      if(a.z == b.z){
        return a.w < b.w;
      }
      else return a.z < b.z;
    }
    else return a.y < b.y;
  }
  else return a.x < b.x;
}
__device__ __host__ bool operator>(const longlong2 &a, const longlong2 &b){
  return b < a;
}
__device__ __host__ bool operator>(const longlong3 &a, const longlong3 &b){
  return b < a;
}
__device__ __host__ bool operator>(const longlong4 &a, const longlong4 &b){
  return b < a;

}
__device__ __host__ bool operator<(const ulonglong2 &a, const ulonglong2 &b){
  if(a == b) return false;
  else if(a.x == b.x) return a.y < b.y;
  else return a.x < b.x;
}
__device__ __host__ bool operator<(const ulonglong3 &a, const ulonglong3 &b){
  if(a == b) return false;
  else if(a.x == b.x){
    if(a.y == b.y){
      return a.z < b.z;
    }
    else return a.y < b.y;
  }
  else return a.x < b.x;
}
__device__ __host__ bool operator<(const ulonglong4 &a, const ulonglong4 &b){
  if(a == b) return false;
  else if(a.x == b.x){
    if(a.y == b.y){
      if(a.z == b.z){
        return a.w < b.w;
      }
      else return a.z < b.z;
    }
    else return a.y < b.y;
  }
  else return a.x < b.x;
}
__device__ __host__ bool operator>(const ulonglong2 &a, const ulonglong2 &b){
  return b < a;
}
__device__ __host__ bool operator>(const ulonglong3 &a, const ulonglong3 &b){
  return b < a;
}
__device__ __host__ bool operator>(const ulonglong4 &a, const ulonglong4 &b){
  return b < a;

}


__device__ __host__ bool operator<=(const float2 &a, const float2 &b){
  if(a == b) return true;
  else if(a.x == b.x) return a.y < b.y;
  else return a.x < b.x;
}
__device__ __host__ bool operator<=(const float3 &a, const float3 &b){
  if(a == b) return true;
  else if(a.x == b.x){
    if(a.y == b.y){
      return a.z < b.z;
    }
    else return a.y < b.y;
  }
  else return a.x < b.x;
}
__device__ __host__ bool operator<=(const float4 &a, const float4 &b){
  if(a == b) return true;
  else if(a.x == b.x){
    if(a.y == b.y){
      if(a.z == b.z){
        return a.w < b.w;
      }
      else return a.z < b.z;
    }
    else return a.y < b.y;
  }
  else return a.x < b.x;
}
__device__ __host__ bool operator>=(const float2 &a, const float2 &b){
  return b <= a;
}
__device__ __host__ bool operator>=(const float3 &a, const float3 &b){
  return b <= a;
}
__device__ __host__ bool operator>=(const float4 &a, const float4 &b){
  return b <= a;

}
__device__ __host__ bool operator<=(const double2 &a, const double2 &b){
  if(a == b) return true;
  else if(a.x == b.x) return a.y < b.y;
  else return a.x < b.x;
}
__device__ __host__ bool operator<=(const double3 &a, const double3 &b){
  if(a == b) return true;
  else if(a.x == b.x){
    if(a.y == b.y){
      return a.z < b.z;
    }
    else return a.y < b.y;
  }
  else return a.x < b.x;
}
__device__ __host__ bool operator<=(const double4 &a, const double4 &b){
  if(a == b) return true;
  else if(a.x == b.x){
    if(a.y == b.y){
      if(a.z == b.z){
        return a.w < b.w;
      }
      else return a.z < b.z;
    }
    else return a.y < b.y;
  }
  else return a.x < b.x;
}
__device__ __host__ bool operator>=(const double2 &a, const double2 &b){
  return b <= a;
}
__device__ __host__ bool operator>=(const double3 &a, const double3 &b){
  return b <= a;
}
__device__ __host__ bool operator>=(const double4 &a, const double4 &b){
  return b <= a;

}
__device__ __host__ bool operator<=(const char2 &a, const char2 &b){
  if(a == b) return true;
  else if(a.x == b.x) return a.y < b.y;
  else return a.x < b.x;
}
__device__ __host__ bool operator<=(const char3 &a, const char3 &b){
  if(a == b) return true;
  else if(a.x == b.x){
    if(a.y == b.y){
      return a.z < b.z;
    }
    else return a.y < b.y;
  }
  else return a.x < b.x;
}
__device__ __host__ bool operator<=(const char4 &a, const char4 &b){
  if(a == b) return true;
  else if(a.x == b.x){
    if(a.y == b.y){
      if(a.z == b.z){
        return a.w < b.w;
      }
      else return a.z < b.z;
    }
    else return a.y < b.y;
  }
  else return a.x < b.x;
}
__device__ __host__ bool operator>=(const char2 &a, const char2 &b){
  return b <= a;
}
__device__ __host__ bool operator>=(const char3 &a, const char3 &b){
  return b <= a;
}
__device__ __host__ bool operator>=(const char4 &a, const char4 &b){
  return b <= a;

}
__device__ __host__ bool operator<=(const uchar2 &a, const uchar2 &b){
  if(a == b) return true;
  else if(a.x == b.x) return a.y < b.y;
  else return a.x < b.x;
}
__device__ __host__ bool operator<=(const uchar3 &a, const uchar3 &b){
  if(a == b) return true;
  else if(a.x == b.x){
    if(a.y == b.y){
      return a.z < b.z;
    }
    else return a.y < b.y;
  }
  else return a.x < b.x;
}
__device__ __host__ bool operator<=(const uchar4 &a, const uchar4 &b){
  if(a == b) return true;
  else if(a.x == b.x){
    if(a.y == b.y){
      if(a.z == b.z){
        return a.w < b.w;
      }
      else return a.z < b.z;
    }
    else return a.y < b.y;
  }
  else return a.x < b.x;
}
__device__ __host__ bool operator>=(const uchar2 &a, const uchar2 &b){
  return b <= a;
}
__device__ __host__ bool operator>=(const uchar3 &a, const uchar3 &b){
  return b <= a;
}
__device__ __host__ bool operator>=(const uchar4 &a, const uchar4 &b){
  return b <= a;
}
__device__ __host__ bool operator<=(const short2 &a, const short2 &b){
  if(a == b) return true;
  else if(a.x == b.x) return a.y < b.y;
  else return a.x < b.x;
}
__device__ __host__ bool operator<=(const short3 &a, const short3 &b){
  if(a == b) return true;
  else if(a.x == b.x){
    if(a.y == b.y){
      return a.z < b.z;
    }
    else return a.y < b.y;
  }
  else return a.x < b.x;
}
__device__ __host__ bool operator<=(const short4 &a, const short4 &b){
  if(a == b) return true;
  else if(a.x == b.x){
    if(a.y == b.y){
      if(a.z == b.z){
        return a.w < b.w;
      }
      else return a.z < b.z;
    }
    else return a.y < b.y;
  }
  else return a.x < b.x;
}
__device__ __host__ bool operator>=(const short2 &a, const short2 &b){
  return b <= a;
}
__device__ __host__ bool operator>=(const short3 &a, const short3 &b){
  return b <= a;
}
__device__ __host__ bool operator>=(const short4 &a, const short4 &b){
  return b <= a;

}
__device__ __host__ bool operator<=(const ushort2 &a, const ushort2 &b){
  if(a == b) return true;
  else if(a.x == b.x) return a.y < b.y;
  else return a.x < b.x;
}
__device__ __host__ bool operator<=(const ushort3 &a, const ushort3 &b){
  if(a == b) return true;
  else if(a.x == b.x){
    if(a.y == b.y){
      return a.z < b.z;
    }
    else return a.y < b.y;
  }
  else return a.x < b.x;
}
__device__ __host__ bool operator<=(const ushort4 &a, const ushort4 &b){
  if(a == b) return true;
  else if(a.x == b.x){
    if(a.y == b.y){
      if(a.z == b.z){
        return a.w < b.w;
      }
      else return a.z < b.z;
    }
    else return a.y < b.y;
  }
  else return a.x < b.x;
}
__device__ __host__ bool operator>=(const ushort2 &a, const ushort2 &b){
  return b <= a;
}
__device__ __host__ bool operator>=(const ushort3 &a, const ushort3 &b){
  return b <= a;
}
__device__ __host__ bool operator>=(const ushort4 &a, const ushort4 &b){
  return b <= a;
}
__device__ __host__ bool operator<=(const int2 &a, const int2 &b){
  if(a == b) return true;
  else if(a.x == b.x) return a.y < b.y;
  else return a.x < b.x;
}
__device__ __host__ bool operator<=(const int3 &a, const int3 &b){
  if(a == b) return true;
  else if(a.x == b.x){
    if(a.y == b.y){
      return a.z < b.z;
    }
    else return a.y < b.y;
  }
  else return a.x < b.x;
}
__device__ __host__ bool operator<=(const int4 &a, const int4 &b){
  if(a == b) return true;
  else if(a.x == b.x){
    if(a.y == b.y){
      if(a.z == b.z){
        return a.w < b.w;
      }
      else return a.z < b.z;
    }
    else return a.y < b.y;
  }
  else return a.x < b.x;
}
__device__ __host__ bool operator>=(const int2 &a, const int2 &b){
  return b <= a;
}
__device__ __host__ bool operator>=(const int3 &a, const int3 &b){
  return b <= a;
}
__device__ __host__ bool operator>=(const int4 &a, const int4 &b){
  return b <= a;

}
__device__ __host__ bool operator<=(const uint2 &a, const uint2 &b){
  if(a == b) return true;
  else if(a.x == b.x) return a.y < b.y;
  else return a.x < b.x;
}
__device__ __host__ bool operator<=(const uint3 &a, const uint3 &b){
  if(a == b) return true;
  else if(a.x == b.x){
    if(a.y == b.y){
      return a.z < b.z;
    }
    else return a.y < b.y;
  }
  else return a.x < b.x;
}
__device__ __host__ bool operator<=(const uint4 &a, const uint4 &b){
  if(a == b) return true;
  else if(a.x == b.x){
    if(a.y == b.y){
      if(a.z == b.z){
        return a.w < b.w;
      }
      else return a.z < b.z;
    }
    else return a.y < b.y;
  }
  else return a.x < b.x;
}
__device__ __host__ bool operator>=(const uint2 &a, const uint2 &b){
  return b <= a;
}
__device__ __host__ bool operator>=(const uint3 &a, const uint3 &b){
  return b <= a;
}
__device__ __host__ bool operator>=(const uint4 &a, const uint4 &b){
  return b <= a;
}
__device__ __host__ bool operator<=(const long2 &a, const long2 &b){
  if(a == b) return true;
  else if(a.x == b.x) return a.y < b.y;
  else return a.x < b.x;
}
__device__ __host__ bool operator<=(const long3 &a, const long3 &b){
  if(a == b) return true;
  else if(a.x == b.x){
    if(a.y == b.y){
      return a.z < b.z;
    }
    else return a.y < b.y;
  }
  else return a.x < b.x;
}
__device__ __host__ bool operator<=(const long4 &a, const long4 &b){
  if(a == b) return true;
  else if(a.x == b.x){
    if(a.y == b.y){
      if(a.z == b.z){
        return a.w < b.w;
      }
      else return a.z < b.z;
    }
    else return a.y < b.y;
  }
  else return a.x < b.x;
}
__device__ __host__ bool operator>=(const long2 &a, const long2 &b){
  return b <= a;
}
__device__ __host__ bool operator>=(const long3 &a, const long3 &b){
  return b <= a;
}
__device__ __host__ bool operator>=(const long4 &a, const long4 &b){
  return b <= a;

}
__device__ __host__ bool operator<=(const ulong2 &a, const ulong2 &b){
  if(a == b) return true;
  else if(a.x == b.x) return a.y < b.y;
  else return a.x < b.x;
}
__device__ __host__ bool operator<=(const ulong3 &a, const ulong3 &b){
  if(a == b) return true;
  else if(a.x == b.x){
    if(a.y == b.y){
      return a.z < b.z;
    }
    else return a.y < b.y;
  }
  else return a.x < b.x;
}
__device__ __host__ bool operator<=(const ulong4 &a, const ulong4 &b){
  if(a == b) return true;
  else if(a.x == b.x){
    if(a.y == b.y){
      if(a.z == b.z){
        return a.w < b.w;
      }
      else return a.z < b.z;
    }
    else return a.y < b.y;
  }
  else return a.x < b.x;
}
__device__ __host__ bool operator>=(const ulong2 &a, const ulong2 &b){
  return b <= a;
}
__device__ __host__ bool operator>=(const ulong3 &a, const ulong3 &b){
  return b <= a;
}
__device__ __host__ bool operator>=(const ulong4 &a, const ulong4 &b){
  return b <= a;

}
__device__ __host__ bool operator<=(const longlong2 &a, const longlong2 &b){
  if(a == b) return true;
  else if(a.x == b.x) return a.y < b.y;
  else return a.x < b.x;
}
__device__ __host__ bool operator<=(const longlong3 &a, const longlong3 &b){
  if(a == b) return true;
  else if(a.x == b.x){
    if(a.y == b.y){
      return a.z < b.z;
    }
    else return a.y < b.y;
  }
  else return a.x < b.x;
}
__device__ __host__ bool operator<=(const longlong4 &a, const longlong4 &b){
  if(a == b) return true;
  else if(a.x == b.x){
    if(a.y == b.y){
      if(a.z == b.z){
        return a.w < b.w;
      }
      else return a.z < b.z;
    }
    else return a.y < b.y;
  }
  else return a.x < b.x;
}
__device__ __host__ bool operator>=(const longlong2 &a, const longlong2 &b){
  return b <= a;
}
__device__ __host__ bool operator>=(const longlong3 &a, const longlong3 &b){
  return b <= a;
}
__device__ __host__ bool operator>=(const longlong4 &a, const longlong4 &b){
  return b <= a;

}
__device__ __host__ bool operator<=(const ulonglong2 &a, const ulonglong2 &b){
  if(a == b) return true;
  else if(a.x == b.x) return a.y < b.y;
  else return a.x < b.x;
}
__device__ __host__ bool operator<=(const ulonglong3 &a, const ulonglong3 &b){
  if(a == b) return true;
  else if(a.x == b.x){
    if(a.y == b.y){
      return a.z < b.z;
    }
    else return a.y < b.y;
  }
  else return a.x < b.x;
}
__device__ __host__ bool operator<=(const ulonglong4 &a, const ulonglong4 &b){
  if(a == b) return true;
  else if(a.x == b.x){
    if(a.y == b.y){
      if(a.z == b.z){
        return a.w < b.w;
      }
      else return a.z < b.z;
    }
    else return a.y < b.y;
  }
  else return a.x < b.x;
}
__device__ __host__ bool operator>=(const ulonglong2 &a, const ulonglong2 &b){
  return b <= a;
}
__device__ __host__ bool operator>=(const ulonglong3 &a, const ulonglong3 &b){
  return b <= a;
}
__device__ __host__ bool operator>=(const ulonglong4 &a, const ulonglong4 &b){
  return b <= a;

}


// =============================================================================================================
//
// Vector Dot Products
//
// =============================================================================================================


__device__ __host__ float dotProduct(const float2 &a, const float2 &b){
  return (a.x*b.x) + (a.y*b.y);
}
__device__ __host__ float dotProduct(const float3 &a, const float3 &b){
  return (a.x*b.x) + (a.y*b.y) + (a.z*b.z);
}
__device__ __host__ int dotProduct(const int2 &a, const int2 &b){
  return (a.x*b.x) + (a.y*b.y);
}

// =============================================================================================================
//
// Arithmetic Operators
//
// =============================================================================================================

__device__ __host__ float2 operator+(const float2 &a, const float2 &b){
  return {a.x + b.x, a.y + b.y};
}
__device__ __host__ float2 operator-(const float2 &a, const float2 &b){
  return {a.x - b.x, a.y - b.y};
}
__device__ __host__ float2 operator/(const float2 &a, const float2 &b){
  return {a.x / b.x, a.y / b.y};
}
__device__ __host__ float2 operator*(const float2 &a, const float2 &b){
  return {a.x * b.x, a.y * b.y};
}
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
__device__ __host__ int2 operator+(const int2 &a, const int2 &b){
  return {a.x + b.x, a.y + b.y};
}
__device__ __host__ int2 operator-(const int2 &a, const int2 &b){
  return {a.x - b.x, a.y - b.y};
}
__device__ __host__ int2 operator/(const int2 &a, const int2 &b){
  return {a.x/b.x,a.y/b.y};
}
__device__ __host__ int2 operator*(const int2 &a, const int2 &b){
  return {a.x * b.x, a.y * b.y};
}
__device__ __host__ uint2 operator+(const uint2 &a, const uint2 &b){
  return {a.x + b.x, a.y + b.y};
}
__device__ __host__ uint2 operator-(const uint2 &a, const uint2 &b){
  return {a.x - b.x, a.y - b.y};
}
__device__ __host__ uint2 operator*(const uint2 &a, const uint2 &b){
  return {a.x * b.x, a.y * b.y};
}
__device__ __host__ uint2 operator/(const uint2 &a, const uint2 &b){
  return {a.x / b.x, a.y / b.y};
}


// =============================================================================================================
//
// Compound Assignment Operators
//
// =============================================================================================================

__device__ __host__ float4 operator+=(const float4 &a, const float4 &b){
  return {a.x + b.x, a.y + b.y, a.z + b.z, a.w + b.w};
}
__device__ __host__ float3 operator+=(const float3 &a, const float3 &b){
  return {a.x + b.x, a.y + b.y, a.z + b.z};
}
__device__ __host__ float2 operator+=(const float2 &a, const float2 &b){
  return {a.x + b.x, a.y + b.y};
}
__device__ __host__ float4 operator+=(const float4 &a, const int4 &b){
  return {a.x + ((float) b.x), a.y + ((float) b.y), a.z + ((float) b.z), a.w + ((float) b.w)};
}
__device__ __host__ float3 operator+=(const float3 &a, const int3 &b){
  return {a.x + ((float) b.x), a.y + ((float) b.y), a.z + ((float) b.z)};
}
__device__ __host__ float2 operator+=(const float2 &a, const int2 &b){
  return {a.x + ((float) b.x), a.y + ((float) b.y)};
}
__device__ __host__ int4 operator+=(const int4 &a, const float4 &b){
  return {a.x + ((int) b.x), a.y + ((int) b.y), a.z + ((int) b.z), a.w + ((int) b.w)};
}
__device__ __host__ int3 operator+=(const int3 &a, const float3 &b){
  return {a.x + ((int) b.x), a.y + ((int) b.y), a.z + ((int) b.z)};
}
__device__ __host__ int2 operator+=(const int2 &a, const float2 &b){
  return {a.x + ((int) b.x), a.y + ((int) b.y)};
}
__device__ __host__ int4 operator+=(const int4 &a, const int4 &b){
  return {a.x + b.x, a.y + b.y, a.z + b.z, a.w + b.w};
}
__device__ __host__ int3 operator+=(const int3 &a, const int3 &b){
  return {a.x + b.x, a.y + b.y, a.z + b.z};
}
__device__ __host__ int2 operator+=(const int2 &a, const int2 &b){
  return {a.x + b.x, a.y + b.y};
}

__device__ __host__ float4 operator-=(const float4 &a, const float4 &b){
  return {a.x - b.x, a.y - b.y, a.z - b.z, a.w - b.w};
}
__device__ __host__ float3 operator-=(const float3 &a, const float3 &b){
  return {a.x - b.x, a.y - b.y, a.z - b.z};
}
__device__ __host__ float2 operator-=(const float2 &a, const float2 &b){
  return {a.x - b.x, a.y - b.y};
}
__device__ __host__ float4 operator-=(const float4 &a, const int4 &b){
  return {a.x - ((float) b.x), a.y - ((float) b.y), a.z - ((float) b.z), a.w - ((float) b.w)};
}
__device__ __host__ float3 operator-=(const float3 &a, const int3 &b){
  return {a.x - ((float) b.x), a.y - ((float) b.y), a.z - ((float) b.z)};
}
__device__ __host__ float2 operator-=(const float2 &a, const int2 &b){
  return {a.x - ((float) b.x), a.y - ((float) b.y)};
}
__device__ __host__ int4 operator-=(const int4 &a, const float4 &b){
  return {a.x - ((int) b.x), a.y - ((int) b.y), a.z - ((int) b.z), a.w - ((int) b.w)};
}
__device__ __host__ int3 operator-=(const int3 &a, const float3 &b){
  return {a.x - ((int) b.x), a.y - ((int) b.y), a.z - ((int) b.z)};
}
__device__ __host__ int2 operator-=(const int2 &a, const float2 &b){
  return {a.x - ((int) b.x), a.y - ((int) b.y)};
}
__device__ __host__ int4 operator-=(const int4 &a, const int4 &b){
  return {a.x - b.x, a.y - b.y, a.z - b.z, a.w - b.w};
}
__device__ __host__ int3 operator-=(const int3 &a, const int3 &b){
  return {a.x - b.x, a.y - b.y, a.z - b.z};
}
__device__ __host__ int2 operator-=(const int2 &a, const int2 &b){
  return {a.x - b.x, a.y - b.y};
}

__device__ __host__ float4 operator*=(const float4 &a, const float &b){
  return {a.x * b, a.y * b, a.z * b, a.w * b};
}
__device__ __host__ float3 operator*=(const float3 &a, const float &b){
  return {a.x * b, a.y * b, a.z * b};
}
__device__ __host__ float2 operator*=(const float2 &a, const float &b){
  return {a.x * b, a.y * b};
}
__device__ __host__ float4 operator*=(const float4 &a, const int &b){
  return {a.x * ((float) b), a.y * ((float) b), a.z * ((float) b), a.w * ((float) b)};
}
__device__ __host__ float3 operator*=(const float3 &a, const int &b){
  return {a.x * ((float) b), a.y * ((float) b), a.z * ((float) b)};
}
__device__ __host__ float2 operator*=(const float2 &a, const int &b){
  return {a.x * ((float) b), a.y * ((float) b)};
}
__device__ __host__ int4 operator*=(const int4 &a, const float &b){
  return {a.x * ((int) b), a.y * ((int) b), a.z * ((int) b), a.w * ((int) b)};
}
__device__ __host__ int3 operator*=(const int3 &a, const float &b){
  return {a.x * ((int) b), a.y * ((int) b), a.z * ((int) b)};
}
__device__ __host__ int2 operator*=(const int2 &a, const float &b){
  return {a.x * ((int) b), a.y * ((int) b)};
}
__device__ __host__ int4 operator*=(const int4 &a, const int &b){
  return {a.x * b, a.y * b, a.z * b, a.w * b};
}
__device__ __host__ int3 operator*=(const int3 &a, const int &b){
  return {a.x * b, a.y * b, a.z * b};
}
__device__ __host__ int2 operator*=(const int2 &a, const int &b){
  return {a.x * b, a.y * b};
}

__device__ __host__ float4 operator/=(const float4 &a, const float &b){
  return {a.x / b, a.y / b, a.z / b, a.w / b};
}
__device__ __host__ float3 operator/=(const float3 &a, const float &b){
  return {a.x / b, a.y / b, a.z / b};
}
__device__ __host__ float2 operator/=(const float2 &a, const float &b){
  return {a.x / b, a.y / b};
}
__device__ __host__ float4 operator/=(const float4 &a, const int &b){
  return {a.x / ((float) b), a.y / ((float) b), a.z / ((float) b), a.w / ((float) b)};
}
__device__ __host__ float3 operator/=(const float3 &a, const int &b){
  return {a.x / ((float) b), a.y / ((float) b), a.z / ((float) b))};
}
__device__ __host__ float2 operator/=(const float2 &a, const int &b){
  return {a.x / ((float) b), a.y / ((float) b))};
}
__device__ __host__ int4 operator/=(const int4 &a, const float &b){
  return {a.x / ((int) b), a.y / ((int) b), a.z / ((int) b), a.w / ((int) b)};
}
__device__ __host__ int3 operator/=(const int3 &a, const float &b){
  return {a.x / ((int) b), a.y / ((int) b), a.z / ((int) b)};
}
__device__ __host__ int2 operator/=(const int2 &a, const float &b){
  return {a.x / ((int) b), a.y / ((int) b)};
}
__device__ __host__ int4 operator/=(const int4 &a, const int &b){
  return {a.x / b, a.y / b, a.z / b, a.w / b};
}
__device__ __host__ int3 operator/=(const int3 &a, const int &b){
  return {a.x / b, a.y / b, a.z / b};
}
__device__ __host__ int2 operator/=(const int2 &a, const int &b){
  return {a.x / b, a.y / b};
}


// =============================================================================================================
//
// Missmatched Arithmetic Operators
//
// =============================================================================================================

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
__device__ __host__ float2 operator+(const float2 &a, const float &b){
  return {a.x + b, a.y + b};
}
__device__ __host__ float2 operator-(const float2 &a, const float &b){
  return {a.x - b, a.y - b};
}
__device__ __host__ float2 operator/(const float2 &a, const float &b){
  return {a.x / b, a.y / b};
}
__device__ __host__ float2 operator*(const float2 &a, const float &b){
  return {a.x * b, a.y * b};
}
__device__ __host__ float2 operator+(const float &a, const float2 &b){
  return {a + b.x, a + b.y};
}
__device__ __host__ float2 operator-(const float &a, const float2 &b){
  return {a - b.x, a - b.y};
}
__device__ __host__ float2 operator/(const float &a, const float2 &b){
  return {a / b.x, a / b.y};
}
__device__ __host__ float2 operator*(const float &a, const float2 &b){
  return {a * b.x, a * b.y};
}
__device__ __host__ float2 operator+(const float2 &a, const int2 &b){
  return {a.x + ((float) b.x), a.y + ((float) b.y)};
}
__device__ __host__ float2 operator-(const float2 &a, const int2 &b){
  return {a.x - ((float) b.x), a.y - ((float) b.y)};
}
__device__ __host__ float2 operator/(const float2 &a, const int2 &b){
  return {a.x / ((float) b.x), a.y / ((float) b.y)};
}
__device__ __host__ float2 operator*(const float2 &a, const int2 &b){
  return {a.x * ((float) b.x), a.y * ((float) b.y)};
}
__device__ __host__ float2 operator+(const int2 &a, const float2 &b){
  return {((float) a.x) + b.x, ((float) a.y) + b.y};
}
__device__ __host__ float2 operator-(const int2 &a, const float2 &b){
  return {((float) a.x) - b.x, ((float) a.y) - b.y};
}
__device__ __host__ float2 operator/(const int2 &a, const float2 &b){
  return {((float) a.x) / b.x, ((float) a.y) / b.y};
}
__device__ __host__ float2 operator*(const int2 &a, const float2 &b){
  return {(float) a.x * b.x, ((float) a.y) * b.y};
}
__device__ __host__ float2 operator/(const int2 &a, const float &b){
  return {((float)a.x)/b, ((float)a.y)/b};
}
__device__ __host__ float2 operator+(const int2 &a, const float &b){
  return {((float)a.x) + b, ((float)a.y) + b};
}
__device__ __host__ float2 operator-(const int2 &a, const float &b){
  return {((float)a.x) - b, ((float)a.y) - b};
}
__device__ __host__ int2 operator+(const int2 &a, const int &b){
  return {a.x + b,a.y + b};
}
__device__ __host__ int2 operator-(const int2 &a, const int &b){
  return {a.x - b,a.y - b};
}
__device__ __host__ uint2 operator+(const uint2 &a, const int &b){
  return {a.x + b, a.y + b};
}
__device__ __host__ uint2 operator-(const uint2 &a, const int &b){
  return {a.x - b, a.y - b};
}
__device__ __host__ uint2 operator*(const uint2 &a, const int &b){
  return {a.x * b, a.y * b};
}
__device__ __host__ uint2 operator/(const uint2 &a, const int &b){
  return {a.x / b, a.y / b};
}
__device__ __host__ int2 operator+(const int2 &a, const uint2 &b){
  return {a.x + (int)b.x, a.y + (int)b.y};
}
__device__ __host__ int2 operator-(const int2 &a, const uint2 &b){
  return {a.x - (int)b.x, a.y - (int)b.y};
}
__device__ __host__ int2 operator*(const int2 &a, const uint2 &b){
  return {a.x * (int)b.x, a.y * (int)b.y};
}
__device__ __host__ int2 operator/(const int2 &a, const uint2 &b){
  return {a.x / (int)b.x, a.y / (int)b.y};
}
__device__ __host__ int2 operator+(const uint2 &a, const int2 &b){
  return {(int)a.x + b.x, (int)a.y + b.y};
}
__device__ __host__ int2 operator-(const uint2 &a, const int2 &b){
  return {(int)a.x - (int)b.x, (int)a.y - (int)b.y};
}
__device__ __host__ int2 operator*(const uint2 &a, const int2 &b){
  return {(int)a.x * b.x, (int)a.y * b.y};
}
__device__ __host__ int2 operator/(const uint2 &a, const int2 &b){
  return {(int)a.x /b.x, (int)a.y / b.y};
}
__device__ __host__ ulong2 operator+(const ulong2 &a, const int2 &b){
  return {a.x + (unsigned long)b.x, a.y + (unsigned long)b.y};
}
__device__ __host__ ulong2 operator-(const ulong2 &a, const int2 &b){
  return  {a.x - (unsigned long)b.x, a.y - (unsigned long)b.y};
}
__device__ __host__ ulong2 operator*(const ulong2 &a, const int2 &b){
  return  {a.x * (unsigned long)b.x, a.y * (unsigned long)b.y};
}
__device__ __host__ ulong2 operator+(const int2 &a, const ulong2 &b){
  return {(unsigned long)a.x + b.x, (unsigned long)a.y + b.y};
}
__device__ __host__ ulong2 operator-(const int2 &a, const ulong2 &b){
  return {(unsigned long)a.x - b.x, (unsigned long)a.y - b.y};
}
__device__ __host__ ulong2 operator*(const int2 &a, const ulong2 &b){
  return {(unsigned long)a.x * b.x, (unsigned long)a.y * b.y};
}
__device__ __host__ float2 operator*(const int2 &a, const float &b){
  return {((float)a.x) * b, ((float)a.y) * b};
}
__device__ __host__ bool operator>(const float2 &a, const float &b){
  return (a.x > b) && (a.y > b);
}
__device__ __host__ bool operator<(const float2 &a, const float &b){
  return (a.x < b) && (a.y < b);
}
__device__ __host__ bool operator>(const float2 &a, const int2 &b){
  return (a.x > b.x) && (a.y > b.y);
}
__device__ __host__ bool operator<(const float2 &a, const int2 &b){
  return (a.x < b.x) && (a.y < b.y);
}
