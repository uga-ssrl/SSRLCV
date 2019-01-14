//#======================================#//
//# UGA SSRL Reprojection                #//
//# Author: James Roach                  #//
//# Contact: Jhroach14@gmail.com         #//
//#                           					 #//
//#======================================#//
// This file contains implementations for utilities used elsewhere in reprojection
//

//#include "cpuUtilities.cuh"
//standard headers
#include <fstream>
#include <iostream>
#include <sstream>
#include <string>

//nvidia headers
#include <cuda_runtime.h>
#include <cuda.h>

//custom headers
#include "reprojection.cuh"


void transpose_cpu(float M[3][3], float (&M_t)[3][3])
{
  for(int r = 0; r < 3; ++r)
  {
    for(int c = 0; c < 3; ++c)
    {
      M_t[r][c] = M[c][r];
    }
  }
}

void inverse3x3_cpu(float M[3][3], float (&Minv)[3][3])
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
}

void multiply3x3_cpu(float A[3][3], float B[3][3], float (&C)[3][3])
{
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

void multiply3x3x1_cpu(float A[3][3], float B[3], float (&C)[3])
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

//using this to make the code prettier and not have a bunch of if statements
void debugPrint(std::string str)
{
	ConfigVals config(); //used for configuration
	if (true) //fix later
	{
		std::cout << str << std::endl;
	}
}

//takes in point cloud outputted from gpu and saves it in ply format
void savePly(PointCloud* pCloud)
{
	std::ofstream outputFile1("output.ply");
	outputFile1 << "ply\nformat ascii 1.0\nelement vertex ";
	outputFile1 << pCloud->numPoints << "\n";
	outputFile1 << "property float x\nproperty float y\nproperty float z\nproperty uchar red\nproperty uchar green\nproperty uchar blue\n";
	outputFile1 << "end_header\n";

	float3* currentPoint;
	for(int i = 0; i < pCloud->numPoints; i++)
	{
		currentPoint = &(pCloud->points[i]);
		outputFile1 << currentPoint->x << " " << currentPoint->y << " " << currentPoint->z << " " << /*currentPoint->color.x << " " << currentPoint->color.y << " " << currentPoint->color.z <<*/ "\n";
	}
}

//used inside of loadMatchData to fill in match data for each line
 void parseMatchData(float4* &currentMatch, std::string line, int index)
{
	std::istringstream ss(line);
	std::string token;
	std::vector<std::string> data;
	while (getline(ss, token, ','))
	{
		debugPrint(token);
		data.push_back(token);
	}
	//debugPrint("ATTEMPTING TO RELLOCATE 4 ELEMENTS IN VECTOR LENGTH: " + data.size());
	//store match data
	currentMatch->x = stof(data[2]);
	currentMatch->y = stof(data[3]);
	currentMatch->z = stof(data[4]);
	currentMatch->w = stof(data[5]);
	//store match colors
	/*currentMatch->matchColorsL.x = stoi(data[6]);
	currentMatch->matchColorsL.y = stoi(data[7]);
	currentMatch->matchColorsL.z = stoi(data[8]);
	currentMatch->matchColorsR.x = stoi(data[9]);
	currentMatch->matchColorsR.y = stoi(data[10]);
	currentMatch->matchColorsR.z = stoi(data[11]);*/
}

//this function is used to load the feaure matches file and fill in the needed values for it
 void loadMatchData(FeatureMatches* &matches, std::string pathToMatches)
{
	matches = new FeatureMatches();
	std::ifstream inFile(pathToMatches);
	std::string line;
	bool first = 1;
	int index = 0;
	while (getline( inFile, line))
	{
		debugPrint(line);
		if (first)
		{
			first = 0;
			//allocate mem needed for matched point data
			matches->numMatches = (std::stoi(line) + 1);
			matches->matches = new float4[matches->numMatches];
			//debugPrint("DYNAMICALLY ALLOCATING ON HOST... ");
			//debugPrint("read: " << line << ", generating: " << matches->numMatches);
			// TODO potentially average the colors instead of just picking the first image?
		}
		else
		{
			float4* match = &(matches->matches[index]);
			parseMatchData(match, line, index);
			index++;
		}
	} //end while
}

void parseCameraData(Camera* &currentCamera, std::string line)
{
	std::istringstream ss(line);
	std::string token;
	std::vector<std::string> data;
	while (getline(ss, token, ','))
	{
		debugPrint(token);
		data.push_back(token);
	}
	currentCamera->val1 = std::stof(data[1]);
	currentCamera->val2 = std::stof(data[2]);
	currentCamera->val3 = std::stof(data[3]);
	currentCamera->val4 = std::stof(data[4]);
	currentCamera->val5 = std::stof(data[5]);
	currentCamera->val6 = std::stof(data[6]);
}

void loadCameraData(CameraData* &cameras, std::string pathToCameras)
{
	ConfigVals config();
	cameras = new CameraData();
	cameras->numCameras = 2; //automate
	cameras->cameras = new Camera[2]; //automate

	std::ifstream inFile(pathToCameras);
	std::string line;
	int index = 0;
	while (getline(inFile, line))
	{
		debugPrint(line);
		Camera* currentCam = &(cameras->cameras[index]);
		parseCameraData(currentCam, line);
		index++;
	}
}

//arg parser used for easy argument handling. Not super helpful for just 2
//input arguments, but if more are added this thing will shine
struct ArgParser
{
        std::string cameraPath;
        std::string matchesPath;

        ArgParser( int argc, char* argv[]);  //constructor
};


//arg parser methods
ArgParser::ArgParser( int argc, char* argv[])
{
	std::cout << "*===================* REPROJECTION *===================*" << std::endl;
	if (argc < 3)
	{
		std::cout << "ERROR: not enough arguments ... " << std::endl;
    std::cout << "USAGE: " << std::endl;
		std::cout << "./reprojection.x path/to/cameras.txt path/to/matches.txt" << std::endl;
		std::cout << "*=========================================================*" << std::endl;
		exit(-1); //goodbye cruel world
	}
	std::cout << "*                                                      *" << std::endl;
  std::cout << "*                     ~ UGA SSRL ~                     *" << std::endl;
  std::cout << "*        Multiview Onboard Computational Imager        *" << std::endl;
  std::cout << "*                                                      *" << std::endl;
	std::cout << "*=========================================================*" << std::endl;

	this->cameraPath = argv[1];
	this->matchesPath = argv[2];
}

//prints tx2 info relevant to cuda devlopment
//citation: this method comes from the nvidia formums, and has been modified slightly
void printDeviceProperties()
{
	std::cout<<"\n---------------START OF DEVICE PROPERTIES---------------\n"<<std::endl;

  int nDevices;
  cudaGetDeviceCount(&nDevices);      //find num of devices on tx2

  for (int i = 0; i < nDevices; i++)  //print info on each device
	{
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, i);

    printf("Device Number: %d\n", i);
    printf(" -Device name: %s\n\n", prop.name);
    printf(" -Memory\n  -Memory Clock Rate (KHz): %d\n", prop.memoryClockRate);
    printf("  -Memory Bus Width (bits): %d\n",prop.memoryBusWidth);
    printf("  -Peak Memory Bandwidth (GB/s): %f\n",2.0 * prop.memoryClockRate * (prop.memoryBusWidth / 8) / 1.0e6);
    printf("  -Total Global Memory (bytes): %lo\n", prop.totalGlobalMem);
    printf("  -Total Const Memory (bytes): %lo\n", prop.totalConstMem);
    printf("  -Max pitch allowed for memcpy in regions allocated by cudaMallocPitch() (bytes): %lo\n\n", prop.memPitch);
    printf("  -Shared Memory per block (bytes): %lo\n", prop.sharedMemPerBlock);
    printf("  -Max number of threads per block: %d\n",prop.maxThreadsPerBlock);
    printf("  -Max number of blocks: %dx%dx%d\n",prop.maxGridSize[0], prop.maxGridSize[1], prop.maxGridSize[2]);
    printf("  -32bit Registers per block: %d\n", prop.regsPerBlock);
    printf("  -Threads per warp: %d\n\n", prop.warpSize);
    printf("  -Total number of Multiprocessors: %d\n",prop.multiProcessorCount);
    printf("  -Shared Memory per Multiprocessor (bytes): %lo\n",prop.sharedMemPerMultiprocessor);
    printf("  -32bit Registers per Multiprocessor: %d\n\n", prop.regsPerMultiprocessor);
    printf("  -Number of asynchronous engines: %d\n", prop.asyncEngineCount);
    printf("  -Texture alignment requirement (bytes): %lo\n  -Texture base addresses that are aligned to "
    "textureAlignment bytes do not need an offset applied to texture fetches.\n\n", prop.textureAlignment);
    printf(" -Device Compute Capability:\n  -Major revision #: %d\n  -Minor revision #: %d\n", prop.major, prop.minor);

		printf(" -Run time limit for kernels that get executed on this device: ");
		if(prop.kernelExecTimeoutEnabled)
		{
      printf("YES\n");
    }
    else
		{
      printf("NO\n");
    }

    printf(" -Device is ");
    if(prop.integrated)
		{
      printf("integrated. (motherboard)\n");
    }
    else
		{
      printf("discrete. (card)\n\n");
    }

    if(prop.isMultiGpuBoard)
		{
      printf(" -Device is on a MultiGPU configurations.\n\n");
    }

    switch(prop.computeMode)
		{
      case(0):
        printf(" -Default compute mode (Multiple threads can use cudaSetDevice() with this device)\n");
        break;
      case(1):
        printf(" -Compute-exclusive-thread mode (Only one thread in one processwill be able to use\n cudaSetDevice() with this device)\n");
        break;
      case(2):
        printf(" -Compute-prohibited mode (No threads can use cudaSetDevice() with this device)\n");
        break;
      case(3):
        printf(" -Compute-exclusive-process mode (Many threads in one process will be able to use\n cudaSetDevice() with this device)\n");
        break;
      default:
        printf(" -GPU in unknown compute mode.\n");
        break;
    }

    if(prop.canMapHostMemory)
		{
      printf("\n -The device can map host memory into the CUDA address space for use with\n cudaHostAlloc() or cudaHostGetDevicePointer().\n\n");
    }
    else
		{
      printf("\n -The device CANNOT map host memory into the CUDA address space.\n\n");
    }

    printf(" -ECC support: ");
    if(prop.ECCEnabled)
		{
      printf(" ON\n");
    }
    else
		{
      printf(" OFF\n");
    }

    printf(" -PCI Bus ID: %d\n", prop.pciBusID);
    printf(" -PCI Domain ID: %d\n", prop.pciDomainID);
    printf(" -PCI Device (slot) ID: %d\n", prop.pciDeviceID);

    printf(" -Using a TCC Driver: ");
    if(prop.tccDriver)
		{
      printf("YES\n");
    }
    else
		{
      printf("NO\n");
    }
  }
  std::cout<<"\n----------------END OF DEVICE PROPERTIES----------------\n"<<std::endl;
}
