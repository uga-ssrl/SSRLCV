/** \file common_includes.h
* \brief common location for global includes
* \todo remove non-global includes
*/

#ifndef COMMON_INCLUDES_H
#define COMMON_INCLUDES_H

// our boiz \ nvidia
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <cuda.h>
#include <cuda_occupancy.h>
#include <cublas_v2.h>

//util
#include "cuda_util.cuh"
#include "cuda_vec_util.cuh"
#include "matrix_util.cuh"
#include "Unity.cuh"

//cpp includes
#include <iostream>
#include <stdio.h>
#include <algorithm>
#include <iterator>
#include <fstream>
#include <cmath>
#include <sstream>
#include <time.h>
#include <dirent.h>
#include <iomanip>
#include <random>
#include <locale>
#include <csignal>
#include <thread>
#include <mutex>
#include <unistd.h>

//data structures
#include <map>
#include <string>
#include <cstring>
#include <ctype.h>
#include <array>
#include <vector>
#include <queue>
#include <stack>
#include <limits>
#include <type_traits>
#include <cfloat>

//image io
#include <png.h>
#include "tiffio.h"
#include <jpeglib.h>
#include <jerror.h>



#include "CVExceptions.hpp"

#define PI 3.1415926535897932384626433832795028841971693993

// 0 based indexing for cuBLAS
#define IDX2C(i,j,ld) (((j)*(ld))+(i))

namespace ssrlcv{
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
