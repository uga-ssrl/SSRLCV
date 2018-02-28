#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <cuda.h>
#include <iostream>
#include <string>
#include <stdio.h>
#include <ctype.h>
#include <vector>
#include <algorithm>
#include <thrust/sort.h>
#include <thrust/execution_policy.h>
#include <thrust/pair.h>
#include <thrust/unique_by_key.h>


//pretty much just a binary search in each dimension performed by threads
__global__ void getShuffledXYZ(float3* p, int* keys, float3* centers, float3 c, float W, int N, int D){

    int tx = threadIdx.x;
    int bx = blockIdx.x;
    int globalID = bx * blockDim.x + tx;
    if(globalID < N){
      float x = p[i].x;
      float y = p[i].y;
      float z = p[i].z;
      float leftx = c.x-W/2.0f, rightx = c.x + W/2.0f;
      float lefty = c.y-W/2.0f, righty = c.y + W/2.0f;
      float leftz = c.z-W/2.0f, rightz = c.z + W/2.0f;
      int key = 0;
      int depth = 1;
      while(depth <= D){
        if(x < c.x){
          key <<= 1;
          rightx = c.x;
          c.x = (leftx + rightx)/2.0f;
        }
        else{
          key = (key << 1) + 1;
          leftx = c.x;
          c.x = (leftx + rightx)/2.0f;
        }
        if(y < c.y){
          key <<= 1;
          righty = c.y;
          c.y = (lefty + righty)/2.0f;
        }
        else{
          key = (key << 1) + 1;
          lefty = c.y;
          c.y = (lefty + righty)/2.0f;
        }
        if(z < c.z){
          key <<= 1;
          rightz = c.z;
          c.z = (leftz + rightz)/2.0f;
        }
        else{
          key = (key << 1) + 1;
          leftz = c.z;
          c.z = (leftz + rightz)/2.0f;
        }
        depth++;
      }
      keys[i] = key;
      centers[i].x = c.x;
      centers[i].y = c.y;
      centers[i].z = c.z;
    }
  }

}

struct Octree{
  float3* points;
  float3* centers;
  float3* normals;
  int* keys;
  int numPoints;
  Octree();
  Octree(float3* points, float3* normals, int numPoints){
      this->points = points;
      this->normals = normals;
      this->centers = new float3[numPoints];
      this->keys = new int[numPoints];
      this->numPoints = numPoints;
  }
  void calculateNormals(){
    for(int i = 0; i < numPoints; ++i){
      //
    }
  }

};

int main(){
  //1. find keys
  //2. sort keys, points, normals, and device_launch_parameters
  //3. compact the keys





  return 0;
}
