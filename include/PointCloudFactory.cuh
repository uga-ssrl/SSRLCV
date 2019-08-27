#ifndef POINTCLOUDFACTORY_CUH
#define POINTCLOUDFACTORY_CUH

#include "common_includes.h"
#include "MatrixUtil.cuh"
#include "Image.cuh"
#include "MatchFactory.cuh"
#include "Unity.cuh"


namespace ssrlcv{
  class PointCloudFactory {

  public:
  	PointCloudFactory();

    Unity<float3>* reproject(Unity<Match>* matches, Image* target, Image* query);

    float3* stereo_disparity(float2* matches0, float2* matches1, float3* points, int n, float scale);

  };

  __global__ void h_stereo_disparity(float2* matches0, float2* matches1, float3* points,
    int n, float scale);

  __global__ void two_view_reproject(int numMatches, float4* matches, float cam1C[3],
  	float cam1V[3],float cam2C[3], float cam2V[3], float K_inv[9],
  	float rotationTranspose1[9], float rotationTranspose2[9], float3* points);

}


#endif /* POINTCLOUDFACTORY_CUH */
