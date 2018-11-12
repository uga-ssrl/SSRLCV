//#======================================#//
//# UGA SSRL Reprojection                #//
//# Author: James Roach                  #//
//# Contact: Jhroach14@gmail.com         #//
//# Citation: Caleb Adams wrote the      #//
//# prototype/proof of concept this is   #//
//# built on.                            #//
//#======================================#//
// A seriously good source:
// https://developer.nvidia.com/sites/default/files/akamai/cuda/files/Misc/mygpu.pdf
//
// This program is only meant to perform a
// small portion of MOCI's science pipeline
//

//custom headers
#include "common_includes.h"
#include "reprojection.cuh"

// Main Method for 2 view reprojection
int main(int argc, char* argv[])
{
  ConfigVals config();					 //stores configuration values

	if (true)
	{
		printDeviceProperties();
	}

	//load and structure reprojection input
	FeatureMatches* fMatches =NULL;
	loadMatchData(fMatches, "/home/nvidia/Development/reprojection/data/matches.txt");
	std::cout<<"made it here 1\n\n";
	CameraData* cData = NULL;
	loadCameraData(cData, "/home/nvidia/Development/reprojection/data/cameras.txt");

	std::cout<<"made it here 2\n\n";

	//execute 2 view reprojection on gpu
	PointCloud* pCloud = NULL;
	twoViewReprojection(fMatches, cData, pCloud);

	//save output point cloud as a .ply file
  savePly(pCloud);
}
