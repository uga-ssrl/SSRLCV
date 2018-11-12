#ifndef COMMON_INCLUDES_H
#define COMMON_INCLUDES_H

// our boiz @ nvidia
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <cuda.h>

#include <iostream>
#include <string>
#include <stdio.h>
#include <ctype.h>
#include <array>
#include <vector>
#include <queue>
#include <stack>
#include <algorithm>
#include <fstream>
#include <cmath>
#include <limits>
#include <sstream>
#include <time.h>
#include <string>
#include <fstream>

#include <png.h>

#include <cstring>
#include <cmath>
#include <time.h>
#include <dirent.h>
#include <cfloat>
#include <cstring>

//TODO change enum to 1 and 2 so both = cpu + gpu
#define CPU 1
#define GPU 2
typedef enum MemoryState{
  cpu = 0,
  gpu = 1,
  both = 2
} MemoryState;




#endif /* COMMON_INCLUDES_H */
