/** \file common_includes.h
* \brief common location for global includes
* \todo remove non-global includes and those that are not
*/

#ifndef COMMON_INCLUDES_HPP
#define COMMON_INCLUDES_HPP

// our boiz \ nvidia
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <cuda.h>
#include <cuda_occupancy.h>

//util
#include "cuda_util.cuh"
#include "cuda_vec_util.cuh"
#include "matrix_util.cuh"
#include "Logger.hpp"
#include "Unity.cuh"
#include "Exceptions.hpp"

//cpp includes
#include <iostream>
#include <stdio.h>
#include <algorithm>
#include <iterator>
#include <fstream>
#include <cfloat>
#include <cmath>
#include <sstream>
#include <time.h>
#include <dirent.h>
#include <iomanip>
#include <random>
#include <csignal>
#include <chrono>

//data structures
#include <map>
#include <string>
#include <cstring>
#include <vector>
#include <type_traits>

#define PI 3.14159265358979323846264338327950288

// 0 based indexing for cuBLAS
#define IDX2C(i,j,ld) (((j)*(ld))+(i))

#define EARTH_MAX_KM_FROM_CENT 6384.4
#define EARTH_MIN_KM_FROM_CENT 6356.77

namespace ssrlcv{
    /**
     * \brief Direction enumeration for algorithms that involve heading. 
     * \details Not all directions are currently used, they were defined 
     * verbosly for the purpose of future proofing. 
     */ 
    typedef enum Direction{
        up,
        down,
        left,
        right,
        forward,
        backward,
        in,
        out,
        north,
        south,
        east,
        west,
        northwest,
        northeast,
        southwest,
        southeast,
        undefined
    } Direction;
}

/**
 * \defgroup cuda_kernels
 * \defgroup cuda_util
 * \defgroup trees
 */

#endif /* COMMON_INCLUDES_H */
