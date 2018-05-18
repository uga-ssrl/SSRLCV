#ifndef OCTREE_CUH
#define OCTREE_CUH

#include "common_includes.h"
#include "octree.cuh"

//put kernels here

/*
POISSON RECONSTRUCTION METHODS
*/
void computeDivergenceVector();
void computeImplicitFunction();
void marchingCubes();
void isosurfaceExtraction();


#endif /* OCTREE_CUH */
