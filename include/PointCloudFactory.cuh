#ifndef PCFACTORY_CUH
#define PCFACTORY_CUH


/**
	Point Cloud Factory - Factory pattern implementation of James Roach's reprojection code
	@author Jake Conley

 */

#include "reprojection.cuh" 
#include "Image.cuh" 
#include "MatchFactory.cuh"

class PointCloudFactory { 

public:
	PointCloudFactory(); 

	void generatePointCloud(PointCloud * out, Image * images, int numImages, SubPixelMatchSet * matchSet);	


};

#endif 