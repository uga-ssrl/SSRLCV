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

	//load and structure reprojection input
	FeatureMatches* fMatches =NULL;
	loadMatchData(fMatches, "./data/repro_test/nk_matches.txt");
	std::cout<<"made it here 1\n\n";
	CameraData* cData = NULL;
	loadCameraData(cData, "./data/repro_test/nk_cameras.txt");

	std::cout<<"made it here 2\n\n";

	//execute 2 view reprojection on gpu
	PointCloud* pCloud = NULL;
	twoViewReprojection(fMatches, cData, pCloud);

	//save output point cloud as a .ply file
  savePly(pCloud);
}
