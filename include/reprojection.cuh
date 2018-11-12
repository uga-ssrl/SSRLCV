#ifndef REPROJECTION_CUH
#define REPROJECTION_CUH

//#======================================#//
//# UGA SSRL Reprojection                #//
//# Author: James Roach                  #//
//# Contact: Jhroach14@gmail.com         #//
//# Citation: Caleb Adams wrote the      #//
//# prototype/proof of concept this is   #//
//# based on.                            #//
//#======================================#//
// This program is only meant to perform a
// small portion of MOCI's science pipeline
//

//standard headers
#include <new>
#include <sstream>
#include "common_includes.h"
#include "cuda_util.cuh"

__device__ int getIndex_gpu(int x, int y);
__device__ void multiply3x3x1_gpu(float A[9], float B[3], float (&C)[3]);
__device__ void multiply3x3x1_gpu(float A[3][3], float B[3], float (&C)[3]);
__device__ float dot_product_gpu(float a[3], float b[3]);
__device__ float magnitude_gpu(float v[3]);
__device__ void normalize_gpu(float (&v)[3]);
__device__ int getGlobalIdx_1D_1D();
__device__ void inverse3x3_gpu(float M[3][3], float (&Minv)[3][3]);

__global__ void two_view_reproject(float4* matches, float cam1C[3],
	float cam1V[3],float cam2C[3], float cam2V[3], float K_inv[9],
	float rotationTranspose1[9], float rotationTranspose2[9], float3* points);

//struct to allow for easy configuration. Using this instead of global vars
struct ConfigVals
{
	bool isDebug;
	int numCameras;

	ConfigVals()
	: isDebug(true),
	  numCameras(2)
	{}
};
typedef struct ConfigVals ConfigVals;
/*
struct Match
{
	float4 matchPoints;  // x,y of pic 1 + x,y of pic2
	//uchar3 matchColorsL;	 // rgb of left image point
	//uchar3 matchColorsR;	 // rgb of right image point

};
typedef struct Match Match;
*/
struct FeatureMatches
{
	int numMatches;
	float4* matches;
};
typedef struct FeatureMatches FeatureMatches;

struct Camera
{
	float val1;  //need to find out what these values mean
	float val2;
	float val3;
	float val4;
	float val5;
	float val6;
};
typedef struct Camera Camera;

struct CameraData
{
	int numCameras;
	Camera* cameras;
};
typedef struct CameraData CameraData;
/*
struct Point
{
	float3 location;  //x,y,z for 3d point
	//uchar3 color;			//rgb average of the 2 matched point's color
};
typedef struct Point Point;
*/

struct PointCloud
{
	int numPoints;
	float3* points;
};
typedef struct PointCloud PointCloud;

void transpose_cpu(float M[3][3], float (&M_t)[3][3]);
void inverse3x3_cpu(float M[3][3], float (&Minv)[3][3]);
void multiply3x3_cpu(float A[3][3], float B[3][3], float (&C)[3][3]);
void multiply3x3x1_cpu(float A[3][3], float B[3], float (&C)[3]);
void debugPrint(std::string str);
void savePly(PointCloud* pCloud);
void parseMatchData(float4* &currentMatch, std::string line, int index);
void loadMatchData(FeatureMatches* &matches, std::string pathToMatches);
void parseCameraData(Camera* &currentCamera, std::string line);
void loadCameraData(CameraData* &cameras, std::string pathToCameras);

void twoViewReprojection(FeatureMatches* fMatches, CameraData* cData, PointCloud* &pointCloud);

#endif // REPROJECTION_CUH
