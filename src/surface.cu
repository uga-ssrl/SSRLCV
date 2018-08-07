#include "surface.cuh"

// Define this to turn on error checking
#define CUDA_ERROR_CHECK

#define CudaSafeCall( err ) __cudaSafeCall( err, __FILE__, __LINE__ )
#define CudaCheckError()    __cudaCheckError( __FILE__, __LINE__ )

inline void __cudaSafeCall(cudaError err, const char *file, const int line) {
#ifdef CUDA_ERROR_CHECK
  if (cudaSuccess != err) {
      fprintf(stderr, "cudaSafeCall() failed at %s:%i : %s\n",
      file, line, cudaGetErrorString(err));
      exit(-1);
  }
#endif

  return;
}
inline void __cudaCheckError(const char *file, const int line) {
#ifdef CUDA_ERROR_CHECK
  cudaError err = cudaGetLastError();
  if (cudaSuccess != err) {
    fprintf(stderr, "cudaCheckError() failed at %s:%i : %s\n",
    file, line, cudaGetErrorString(err));
    exit(-1);
  }

  // More careful checking. However, this will affect performance.
  // Comment away if needed.
  //err = cudaDeviceSynchronize();
  if (cudaSuccess != err) {
    fprintf(stderr, "cudaCheckError() with sync failed at %s:%i : %s\n",
    file, line, cudaGetErrorString(err));
    exit(-1);
  }
#endif

  return;
}

/*
my edges - everyone elses
0-0
1-8
2-9
3-4
4-3
5-1
6-7
7-5
8-2
9-11
10-10
11-6


*/
__constant__ int cubeCategoryTrianglesFromEdges[256][15] = {
  {-1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
  {0, 1, 4, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
  {0, 5, 2, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
  {5, 1, 4, 2, 1, 5, -1, -1, -1, -1, -1, -1, -1, -1, -1},
  {5, 8, 10, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
  {0, 1, 4, 5, 8, 10, -1, -1, -1, -1, -1, -1, -1, -1, -1},
  {2, 8, 10, 0, 8, 2, -1, -1, -1, -1, -1, -1, -1, -1, -1},
  {8, 1, 4, 8, 10, 1, 10, 2, 1, -1, -1, -1, -1, -1, -1},
  {4, 9, 8, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
  {0, 9, 8, 1, 9, 0, -1, -1, -1, -1, -1, -1, -1, -1, -1},
  {5, 2, 0, 8, 4, 9, -1, -1, -1, -1, -1, -1, -1, -1, -1},
  {5, 9, 8, 5, 2, 9, 2, 1, 9, -1, -1, -1, -1, -1, -1},
  {4, 10, 5, 9, 10, 4, -1, -1, -1, -1, -1, -1, -1, -1, -1},
  {0, 10, 5, 0, 1, 10, 1, 9, 10, -1, -1, -1, -1, -1, -1},
  {4, 2, 0, 4, 9, 2, 9, 10, 2, -1, -1, -1, -1, -1, -1},
  {2, 1, 10, 10, 1, 9, -1, -1, -1, -1, -1, -1, -1, -1, -1},
  {3, 6, 1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
  {3, 4, 0, 6, 4, 3, -1, -1, -1, -1, -1, -1, -1, -1, -1},
  {0, 5, 2, 1, 3, 6, -1, -1, -1, -1, -1, -1, -1, -1, -1},
  {3, 5, 2, 3, 6, 5, 6, 4, 5, -1, -1, -1, -1, -1, -1},
  {5, 8, 10, 1, 3, 6, -1, -1, -1, -1, -1, -1, -1, -1, -1},
  {4, 3, 6, 4, 0, 3, 5, 8, 10, -1, -1, -1, -1, -1, -1},
  {2, 8, 10, 2, 0, 8, 1, 3, 6, -1, -1, -1, -1, -1, -1},
  {8, 10, 2, 8, 2, 6, 8, 6, 4, 6, 2, 3, -1, -1, -1},
  {1, 3, 6, 4, 9, 8, -1, -1, -1, -1, -1, -1, -1, -1, -1},
  {9, 3, 6, 9, 8, 3, 8, 0, 3, -1, -1, -1, -1, -1, -1},
  {2, 0, 5, 1, 3, 6, 8, 4, 9, -1, -1, -1, -1, -1, -1},
  {3, 6, 9, 2, 3, 9, 2, 9, 8, 2, 8, 5, -1, -1, -1},
  {4, 10, 5, 4, 9, 10, 6, 1, 3, -1, -1, -1, -1, -1, -1},
  {5, 9, 10, 5, 3, 9, 5, 0, 3, 6, 9, 3, -1, -1, -1},
  {3, 6, 1, 2, 0, 9, 2, 9, 10, 9, 0, 4, -1, -1, -1},
  {3, 6, 9, 3, 9, 2, 2, 9, 10, -1, -1, -1, -1, -1, -1},
  {2, 7, 3, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
  {2, 7, 3, 0, 1, 4, -1, -1, -1, -1, -1, -1, -1, -1, -1},
  {0, 7, 3, 5, 7, 0, -1, -1, -1, -1, -1, -1, -1, -1, -1},
  {1, 7, 3, 1, 4, 7, 4, 5, 7, -1, -1, -1, -1, -1, -1},
  {5, 8, 10, 2, 7, 3, -1, -1, -1, -1, -1, -1, -1, -1, -1},
  {4, 0, 1, 5, 8, 10, 3, 2, 7, -1, -1, -1, -1, -1, -1},
  {7, 8, 10, 7, 3, 8, 3, 0, 8, -1, -1, -1, -1, -1, -1},
  {8, 10, 7, 4, 8, 7, 4, 7, 3, 4, 3, 1, -1, -1, -1},
  {2, 7, 3, 8, 4, 9, -1, -1, -1, -1, -1, -1, -1, -1, -1},
  {0, 9, 8, 0, 1, 9, 3, 2, 7, -1, -1, -1, -1, -1, -1},
  {0, 7, 3, 0, 5, 7, 8, 4, 9, -1, -1, -1, -1, -1, -1},
  {8, 5, 7, 8, 7, 1, 8, 1, 9, 3, 1, 7, -1, -1, -1},
  {10, 4, 9, 10, 5, 4, 2, 7, 3, -1, -1, -1, -1, -1, -1},
  {3, 2, 7, 0, 1, 5, 1, 10, 5, 1, 9, 10, -1, -1, -1},
  {7, 3, 0, 7, 0, 9, 7, 9, 10, 9, 0, 4, -1, -1, -1},
  {7, 3, 1, 7, 1, 10, 10, 1, 9, -1, -1, -1, -1, -1, -1},
  {2, 6, 1, 7, 6, 2, -1, -1, -1, -1, -1, -1, -1, -1, -1},
  {2, 4, 0, 2, 7, 4, 7, 6, 4, -1, -1, -1, -1, -1, -1},
  {0, 6, 1, 0, 5, 6, 5, 7, 6, -1, -1, -1, -1, -1, -1},
  {5, 7, 4, 4, 7, 6, -1, -1, -1, -1, -1, -1, -1, -1, -1},
  {2, 6, 1, 2, 7, 6, 10, 5, 8, -1, -1, -1, -1, -1, -1},
  {10, 5, 8, 2, 7, 0, 7, 4, 0, 7, 6, 4, -1, -1, -1},
  {1, 0, 8, 1, 8, 7, 1, 7, 6, 10, 7, 8, -1, -1, -1},
  {8, 10, 7, 8, 7, 4, 4, 7, 6, -1, -1, -1, -1, -1, -1},
  {6, 2, 7, 6, 1, 2, 4, 9, 8, -1, -1, -1, -1, -1, -1},
  {2, 7, 6, 2, 6, 8, 2, 8, 0, 8, 6, 9, -1, -1, -1},
  {8, 4, 9, 0, 5, 1, 5, 6, 1, 5, 7, 6, -1, -1, -1},
  {9, 8, 5, 9, 5, 6, 6, 5, 7, -1, -1, -1, -1, -1, -1},
  {2, 7, 1, 1, 7, 6, 10, 5, 4, 10, 4, 9, -1, -1, -1},
  {7, 6, 0, 7, 0, 2, 6, 9, 0, 5, 0, 10, 9, 10, 0},
  {9, 10, 0, 9, 0, 4, 10, 7, 0, 1, 0, 6, 7, 6, 0},
  {9, 10, 7, 6, 9, 7, -1, -1, -1, -1, -1, -1, -1, -1, -1},
  {10, 11, 7, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
  {0, 1, 4, 7, 10, 11, -1, -1, -1, -1, -1, -1, -1, -1, -1},
  {2, 0, 5, 7, 10, 11, -1, -1, -1, -1, -1, -1, -1, -1, -1},
  {5, 1, 4, 5, 2, 1, 7, 10, 11, -1, -1, -1, -1, -1, -1},
  {5, 11, 7, 8, 11, 5, -1, -1, -1, -1, -1, -1, -1, -1, -1},
  {5, 11, 7, 5, 8, 11, 4, 0, 1, -1, -1, -1, -1, -1, -1},
  {2, 11, 7, 2, 0, 11, 0, 8, 11, -1, -1, -1, -1, -1, -1},
  {7, 2, 1, 7, 1, 8, 7, 8, 11, 4, 8, 1, -1, -1, -1},
  {8, 4, 9, 10, 11, 7, -1, -1, -1, -1, -1, -1, -1, -1, -1},
  {9, 0, 1, 9, 8, 0, 10, 11, 7, -1, -1, -1, -1, -1, -1},
  {0, 5, 2, 8, 4, 9, 7, 10, 11, -1, -1, -1, -1, -1, -1},
  {7, 10, 11, 5, 2, 8, 2, 9, 8, 2, 1, 9, -1, -1, -1},
  {11, 4, 9, 11, 7, 4, 7, 5, 4, -1, -1, -1, -1, -1, -1},
  {0, 1, 9, 0, 9, 7, 0, 7, 5, 7, 9, 11, -1, -1, -1},
  {4, 9, 11, 0, 4, 11, 0, 11, 7, 0, 7, 2, -1, -1, -1},
  {11, 7, 2, 11, 2, 9, 9, 2, 1, -1, -1, -1, -1, -1, -1},
  {7, 10, 11, 3, 6, 1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
  {3, 4, 0, 3, 6, 4, 11, 7, 10, -1, -1, -1, -1, -1, -1},
  {5, 2, 0, 7, 10, 11, 1, 3, 6, -1, -1, -1, -1, -1, -1},
  {10, 11, 7, 5, 2, 6, 5, 6, 4, 6, 2, 3, -1, -1, -1},
  {11, 5, 8, 11, 7, 5, 3, 6, 1, -1, -1, -1, -1, -1, -1},
  {5, 8, 7, 7, 8, 11, 4, 0, 3, 4, 3, 6, -1, -1, -1},
  {1, 3, 6, 2, 0, 7, 0, 11, 7, 0, 8, 11, -1, -1, -1},
  {6, 4, 2, 6, 2, 3, 4, 8, 2, 7, 2, 11, 8, 11, 2},
  {4, 9, 8, 6, 1, 3, 10, 11, 7, -1, -1, -1, -1, -1, -1},
  {7, 10, 11, 3, 6, 8, 3, 8, 0, 8, 6, 9, -1, -1, -1},
  {0, 5, 2, 3, 6, 1, 8, 4, 9, 7, 10, 11, -1, -1, -1},
  {2, 8, 5, 2, 9, 8, 2, 3, 9, 6, 9, 3, 7, 10, 11},
  {1, 3, 6, 4, 9, 7, 4, 7, 5, 7, 9, 11, -1, -1, -1},
  {7, 5, 9, 7, 9, 11, 5, 0, 9, 6, 9, 3, 0, 3, 9},
  {0, 7, 2, 0, 11, 7, 0, 4, 11, 9, 11, 4, 1, 3, 6},
  {11, 7, 2, 11, 2, 9, 3, 6, 2, 6, 9, 2, -1, -1, -1},
  {10, 3, 2, 11, 3, 10, -1, -1, -1, -1, -1, -1, -1, -1, -1},
  {3, 10, 11, 3, 2, 10, 0, 1, 4, -1, -1, -1, -1, -1, -1},
  {10, 0, 5, 10, 11, 0, 11, 3, 0, -1, -1, -1, -1, -1, -1},
  {1, 4, 5, 1, 5, 11, 1, 11, 3, 11, 5, 10, -1, -1, -1},
  {5, 3, 2, 5, 8, 3, 8, 11, 3, -1, -1, -1, -1, -1, -1},
  {4, 0, 1, 5, 8, 2, 8, 3, 2, 8, 11, 3, -1, -1, -1},
  {0, 8, 3, 3, 8, 11, -1, -1, -1, -1, -1, -1, -1, -1, -1},
  {1, 4, 8, 1, 8, 3, 3, 8, 11, -1, -1, -1, -1, -1, -1},
  {10, 3, 2, 10, 11, 3, 9, 8, 4, -1, -1, -1, -1, -1, -1},
  {0, 1, 8, 8, 1, 9, 3, 2, 10, 3, 10, 11, -1, -1, -1},
  {4, 9, 8, 0, 5, 11, 0, 11, 3, 11, 5, 10, -1, -1, -1},
  {11, 3, 5, 11, 5, 10, 3, 1, 5, 8, 5, 9, 1, 9, 5},
  {2, 11, 3, 2, 4, 11, 2, 5, 4, 9, 11, 4, -1, -1, -1},
  {1, 9, 5, 1, 5, 0, 9, 11, 5, 2, 5, 3, 11, 3, 5},
  {4, 9, 11, 4, 11, 0, 0, 11, 3, -1, -1, -1, -1, -1, -1},
  {11, 3, 1, 9, 11, 1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
  {6, 10, 11, 6, 1, 10, 1, 2, 10, -1, -1, -1, -1, -1, -1},
  {0, 6, 4, 0, 10, 6, 0, 2, 10, 11, 6, 10, -1, -1, -1},
  {10, 11, 6, 5, 10, 6, 5, 6, 1, 5, 1, 0, -1, -1, -1},
  {10, 11, 6, 10, 6, 5, 5, 6, 4, -1, -1, -1, -1, -1, -1},
  {5, 8, 11, 5, 11, 1, 5, 1, 2, 1, 11, 6, -1, -1, -1},
  {8, 11, 2, 8, 2, 5, 11, 6, 2, 0, 2, 4, 6, 4, 2},
  {6, 1, 0, 6, 0, 11, 11, 0, 8, -1, -1, -1, -1, -1, -1},
  {6, 4, 8, 11, 6, 8, -1, -1, -1, -1, -1, -1, -1, -1, -1},
  {8, 4, 9, 10, 11, 1, 10, 1, 2, 1, 11, 6, -1, -1, -1},
  {8, 0, 6, 8, 6, 9, 0, 2, 6, 11, 6, 10, 2, 10, 6},
  {5, 1, 0, 5, 6, 1, 5, 10, 6, 11, 6, 10, 8, 4, 9},
  {9, 8, 5, 9, 5, 6, 10, 11, 5, 11, 6, 5, -1, -1, -1},
  {1, 2, 11, 1, 11, 6, 2, 5, 11, 9, 11, 4, 5, 4, 11},
  {0, 2, 5, 9, 11, 6, -1, -1, -1, -1, -1, -1, -1, -1, -1},
  {6, 1, 0, 6, 0, 11, 4, 9, 0, 9, 11, 0, -1, -1, -1},
  {6, 9, 11, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
  {6, 11, 9, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
  {4, 0, 1, 9, 6, 11, -1, -1, -1, -1, -1, -1, -1, -1, -1},
  {0, 5, 2, 9, 6, 11, -1, -1, -1, -1, -1, -1, -1, -1, -1},
  {1, 5, 2, 1, 4, 5, 9, 6, 11, -1, -1, -1, -1, -1, -1},
  {10, 5, 8, 11, 9, 6, -1, -1, -1, -1, -1, -1, -1, -1, -1},
  {5, 8, 10, 4, 0, 1, 11, 9, 6, -1, -1, -1, -1, -1, -1},
  {8, 2, 0, 8, 10, 2, 11, 9, 6, -1, -1, -1, -1, -1, -1},
  {11, 9, 6, 8, 10, 4, 10, 1, 4, 10, 2, 1, -1, -1, -1},
  {6, 8, 4, 11, 8, 6, -1, -1, -1, -1, -1, -1, -1, -1, -1},
  {6, 0, 1, 6, 11, 0, 11, 8, 0, -1, -1, -1, -1, -1, -1},
  {8, 6, 11, 8, 4, 6, 0, 5, 2, -1, -1, -1, -1, -1, -1},
  {5, 11, 8, 5, 1, 11, 5, 2, 1, 1, 6, 11, -1, -1, -1},
  {10, 6, 11, 10, 5, 6, 5, 4, 6, -1, -1, -1, -1, -1, -1},
  {10, 6, 11, 5, 6, 10, 5, 1, 6, 5, 0, 1, -1, -1, -1},
  {0, 4, 6, 0, 6, 10, 0, 10, 2, 11, 10, 6, -1, -1, -1},
  {6, 11, 10, 6, 10, 1, 1, 10, 2, -1, -1, -1, -1, -1, -1},
  {11, 1, 3, 9, 1, 11, -1, -1, -1, -1, -1, -1, -1, -1, -1},
  {4, 11, 9, 4, 0, 11, 0, 3, 11, -1, -1, -1, -1, -1, -1},
  {1, 11, 9, 1, 3, 11, 2, 0, 5, -1, -1, -1, -1, -1, -1},
  {2, 3, 11, 2, 11, 4, 2, 4, 5, 9, 4, 11, -1, -1, -1},
  {11, 1, 3, 11, 9, 1, 8, 10, 5, -1, -1, -1, -1, -1, -1},
  {5, 8, 10, 4, 0, 9, 0, 11, 9, 0, 3, 11, -1, -1, -1},
  {3, 9, 1, 3, 11, 9, 0, 8, 2, 8, 10, 2, -1, -1, -1},
  {10, 2, 4, 10, 4, 8, 2, 3, 4, 9, 4, 11, 3, 11, 4},
  {1, 8, 4, 1, 3, 8, 3, 11, 8, -1, -1, -1, -1, -1, -1},
  {0, 3, 8, 3, 11, 8, -1, -1, -1, -1, -1, -1, -1, -1, -1},
  {5, 2, 0, 8, 4, 3, 8, 3, 11, 3, 4, 1, -1, -1, -1},
  {5, 2, 3, 5, 3, 8, 8, 3, 11, -1, -1, -1, -1, -1, -1},
  {1, 5, 4, 1, 11, 5, 1, 3, 11, 11, 10, 5, -1, -1, -1},
  {10, 5, 0, 10, 0, 11, 11, 0, 3, -1, -1, -1, -1, -1, -1},
  {3, 11, 4, 3, 4, 1, 11, 10, 4, 0, 4, 2, 10, 2, 4},
  {10, 2, 3, 11, 10, 3, -1, -1, -1, -1, -1, -1, -1, -1, -1},
  {3, 2, 7, 6, 11, 9, -1, -1, -1, -1, -1, -1, -1, -1, -1},
  {0, 1, 4, 3, 2, 7, 9, 6, 11, -1, -1, -1, -1, -1, -1},
  {7, 0, 5, 7, 3, 0, 6, 11, 9, -1, -1, -1, -1, -1, -1},
  {9, 6, 11, 1, 4, 3, 4, 7, 3, 4, 5, 7, -1, -1, -1},
  {2, 7, 3, 10, 5, 8, 6, 11, 9, -1, -1, -1, -1, -1, -1},
  {11, 9, 6, 5, 8, 10, 0, 1, 4, 3, 2, 7, -1, -1, -1},
  {6, 11, 9, 7, 3, 10, 3, 8, 10, 3, 0, 8, -1, -1, -1},
  {4, 3, 1, 4, 7, 3, 4, 8, 7, 10, 7, 8, 9, 6, 11},
  {6, 8, 4, 6, 11, 8, 7, 3, 2, -1, -1, -1, -1, -1, -1},
  {2, 7, 3, 0, 1, 11, 0, 11, 8, 11, 1, 6, -1, -1, -1},
  {4, 11, 8, 4, 6, 11, 5, 7, 0, 7, 3, 0, -1, -1, -1},
  {11, 8, 1, 11, 1, 6, 8, 5, 1, 3, 1, 7, 5, 7, 1},
  {2, 7, 3, 10, 5, 11, 5, 6, 11, 5, 4, 6, -1, -1, -1},
  {5, 11, 10, 5, 6, 11, 5, 0, 6, 1, 6, 0, 2, 7, 3},
  {3, 0, 10, 3, 10, 7, 0, 4, 10, 11, 10, 6, 4, 6, 10},
  {6, 11, 10, 6, 10, 1, 7, 3, 10, 3, 1, 10, -1, -1, -1},
  {11, 2, 7, 11, 9, 2, 9, 1, 2, -1, -1, -1, -1, -1, -1},
  {4, 11, 9, 0, 11, 4, 0, 7, 11, 0, 2, 7, -1, -1, -1},
  {0, 9, 1, 0, 7, 9, 0, 5, 7, 7, 11, 9, -1, -1, -1},
  {11, 9, 4, 11, 4, 7, 7, 4, 5, -1, -1, -1, -1, -1, -1},
  {5, 8, 10, 2, 7, 9, 2, 9, 1, 9, 7, 11, -1, -1, -1},
  {0, 9, 4, 0, 11, 9, 0, 2, 11, 7, 11, 2, 5, 8, 10},
  {9, 1, 7, 9, 7, 11, 1, 0, 7, 10, 7, 8, 0, 8, 7},
  {11, 9, 4, 11, 4, 7, 8, 10, 4, 10, 7, 4, -1, -1, -1},
  {7, 1, 2, 7, 8, 1, 7, 11, 8, 4, 1, 8, -1, -1, -1},
  {2, 7, 11, 2, 11, 0, 0, 11, 8, -1, -1, -1, -1, -1, -1},
  {5, 7, 1, 5, 1, 0, 7, 11, 1, 4, 1, 8, 11, 8, 1},
  {5, 7, 11, 8, 5, 11, -1, -1, -1, -1, -1, -1, -1, -1, -1},
  {5, 4, 11, 5, 11, 10, 4, 1, 11, 7, 11, 2, 1, 2, 11},
  {10, 5, 0, 10, 0, 11, 2, 7, 0, 7, 11, 0, -1, -1, -1},
  {0, 4, 1, 7, 11, 10, -1, -1, -1, -1, -1, -1, -1, -1, -1},
  {10, 7, 11, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
  {9, 7, 10, 6, 7, 9, -1, -1, -1, -1, -1, -1, -1, -1, -1},
  {9, 7, 10, 9, 6, 7, 1, 4, 0, -1, -1, -1, -1, -1, -1},
  {7, 9, 6, 7, 10, 9, 5, 2, 0, -1, -1, -1, -1, -1, -1},
  {10, 6, 7, 10, 9, 6, 2, 1, 5, 1, 4, 5, -1, -1, -1},
  {9, 5, 8, 9, 6, 5, 6, 7, 5, -1, -1, -1, -1, -1, -1},
  {0, 1, 4, 5, 8, 6, 5, 6, 7, 6, 8, 9, -1, -1, -1},
  {2, 6, 7, 2, 8, 6, 2, 0, 8, 8, 9, 6, -1, -1, -1},
  {6, 7, 8, 6, 8, 9, 7, 2, 8, 4, 8, 1, 2, 1, 8},
  {8, 7, 10, 8, 4, 7, 4, 6, 7, -1, -1, -1, -1, -1, -1},
  {1, 8, 0, 1, 7, 8, 1, 6, 7, 10, 8, 7, -1, -1, -1},
  {2, 0, 5, 7, 10, 4, 7, 4, 6, 4, 10, 8, -1, -1, -1},
  {2, 1, 8, 2, 8, 5, 1, 6, 8, 10, 8, 7, 6, 7, 8},
  {5, 4, 7, 4, 6, 7, -1, -1, -1, -1, -1, -1, -1, -1, -1},
  {0, 1, 6, 0, 6, 5, 5, 6, 7, -1, -1, -1, -1, -1, -1},
  {2, 0, 4, 2, 4, 7, 7, 4, 6, -1, -1, -1, -1, -1, -1},
  {2, 1, 6, 7, 2, 6, -1, -1, -1, -1, -1, -1, -1, -1, -1},
  {7, 1, 3, 7, 10, 1, 10, 9, 1, -1, -1, -1, -1, -1, -1},
  {7, 0, 3, 7, 9, 0, 7, 10, 9, 9, 4, 0, -1, -1, -1},
  {0, 5, 2, 1, 3, 10, 1, 10, 9, 10, 3, 7, -1, -1, -1},
  {10, 9, 3, 10, 3, 7, 9, 4, 3, 2, 3, 5, 4, 5, 3},
  {8, 7, 5, 8, 1, 7, 8, 9, 1, 3, 7, 1, -1, -1, -1},
  {0, 3, 9, 0, 9, 4, 3, 7, 9, 8, 9, 5, 7, 5, 9},
  {0, 8, 7, 0, 7, 2, 8, 9, 7, 3, 7, 1, 9, 1, 7},
  {2, 3, 7, 8, 9, 4, -1, -1, -1, -1, -1, -1, -1, -1, -1},
  {8, 7, 10, 4, 7, 8, 4, 3, 7, 4, 1, 3, -1, -1, -1},
  {7, 10, 8, 7, 8, 3, 3, 8, 0, -1, -1, -1, -1, -1, -1},
  {4, 10, 8, 4, 7, 10, 4, 1, 7, 3, 7, 1, 0, 5, 2},
  {7, 10, 8, 7, 8, 3, 5, 2, 8, 2, 3, 8, -1, -1, -1},
  {1, 3, 7, 1, 7, 4, 4, 7, 5, -1, -1, -1, -1, -1, -1},
  {0, 3, 7, 5, 0, 7, -1, -1, -1, -1, -1, -1, -1, -1, -1},
  {1, 3, 7, 1, 7, 4, 2, 0, 7, 0, 4, 7, -1, -1, -1},
  {2, 3, 7, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
  {3, 9, 6, 3, 2, 9, 2, 10, 9, -1, -1, -1, -1, -1, -1},
  {0, 1, 4, 3, 2, 6, 2, 9, 6, 2, 10, 9, -1, -1, -1},
  {5, 10, 9, 5, 9, 3, 5, 3, 0, 6, 3, 9, -1, -1, -1},
  {4, 5, 3, 4, 3, 1, 5, 10, 3, 6, 3, 9, 10, 9, 3},
  {3, 9, 6, 2, 9, 3, 2, 8, 9, 2, 5, 8, -1, -1, -1},
  {2, 6, 3, 2, 9, 6, 2, 5, 9, 8, 9, 5, 0, 1, 4},
  {9, 6, 3, 9, 3, 8, 8, 3, 0, -1, -1, -1, -1, -1, -1},
  {9, 6, 3, 9, 3, 8, 1, 4, 3, 4, 8, 3, -1, -1, -1},
  {8, 2, 10, 8, 6, 2, 8, 4, 6, 6, 3, 2, -1, -1, -1},
  {2, 10, 6, 2, 6, 3, 10, 8, 6, 1, 6, 0, 8, 0, 6},
  {4, 6, 10, 4, 10, 8, 6, 3, 10, 5, 10, 0, 3, 0, 10},
  {5, 10, 8, 1, 6, 3, -1, -1, -1, -1, -1, -1, -1, -1, -1},
  {3, 2, 5, 3, 5, 6, 6, 5, 4, -1, -1, -1, -1, -1, -1},
  {3, 2, 5, 3, 5, 6, 0, 1, 5, 1, 6, 5, -1, -1, -1},
  {3, 0, 4, 6, 3, 4, -1, -1, -1, -1, -1, -1, -1, -1, -1},
  {3, 1, 6, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
  {2, 10, 1, 10, 9, 1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
  {4, 0, 2, 4, 2, 9, 9, 2, 10, -1, -1, -1, -1, -1, -1},
  {0, 5, 10, 0, 10, 1, 1, 10, 9, -1, -1, -1, -1, -1, -1},
  {4, 5, 10, 9, 4, 10, -1, -1, -1, -1, -1, -1, -1, -1, -1},
  {5, 8, 9, 5, 9, 2, 2, 9, 1, -1, -1, -1, -1, -1, -1},
  {4, 0, 2, 4, 2, 9, 5, 8, 2, 8, 9, 2, -1, -1, -1},
  {0, 8, 9, 1, 0, 9, -1, -1, -1, -1, -1, -1, -1, -1, -1},
  {4, 8, 9, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
  {8, 4, 1, 8, 1, 10, 10, 1, 2, -1, -1, -1, -1, -1, -1},
  {2, 10, 8, 0, 2, 8, -1, -1, -1, -1, -1, -1, -1, -1, -1},
  {8, 4, 1, 8, 1, 10, 0, 5, 1, 5, 10, 1, -1, -1, -1},
  {5, 10, 8, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
  {5, 4, 1, 2, 5, 1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
  {0, 2, 5, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
  {0, 4, 1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
  {-1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1}
};

__constant__ int cubeCategoryEdgeIdentity[256] = {0, 19, 37, 54, 1312, 1331, 1285, 1302,
  784, 771, 821, 806, 1584, 1571, 1557, 1542, 74, 89, 111, 124, 1386, 1401, 1359, 1372,
  858, 841, 895, 876, 1658, 1641, 1631, 1612, 140, 159, 169, 186, 1452, 1471, 1417, 1434,
  924, 911, 953, 938, 1724, 1711, 1689, 1674, 198, 213, 227, 240, 1510, 1525, 1475, 1488,
  982, 965, 1011, 992, 1782, 1765, 1747, 1728, 3200, 3219, 3237, 3254, 2464, 2483, 2437,
  2454, 3984, 3971, 4021, 4006, 2736, 2723, 2709, 2694, 3274, 3289, 3311, 3324, 2538,
  2553, 2511, 2524, 4058, 4041, 4095, 876, 2810, 2793, 2709, 2764, 3084, 3103, 3113,
  3130, 2348, 2367, 2313, 2330, 3868, 3855, 3897, 3882, 2620, 2607, 2585, 2570, 3142,
  3157, 3171, 3184, 2406, 2421, 2371, 2384, 3926, 3909, 3171, 3936, 2678, 2661, 2643,
  2624, 2624, 2643, 2661, 2678, 3936, 3955, 3909, 3926, 2384, 2371, 2421, 2406, 3184,
  3171, 3157, 3142, 2570, 2585, 2607, 2620, 3882, 3897, 3855, 3868, 2330, 2313, 2367,
  2348, 3130, 3113, 3103, 3084, 2764, 2783, 2793, 2810, 4076, 4095, 4041, 1434, 2524,
  2511, 2553, 2538, 3324, 3171, 3289, 3274, 2694, 2709, 2723, 2736, 4006, 2709, 3971,
  3984, 2454, 2437, 2483, 2464, 3254, 3237, 3219, 3200, 1728, 1747, 1765, 1782, 992,
  1011, 965, 982, 1488, 1475, 1525, 1510, 240, 227, 213, 198, 1674, 1689, 1711, 1724,
  938, 953, 911, 924, 1434, 1417, 1434, 1452, 186, 169, 159, 140, 1612, 1631, 1641,
  1658, 876, 876, 841, 858, 1372, 1359, 1401, 1386, 124, 111, 89, 74, 1542, 1557,
  1571, 1584, 806, 821, 771, 784, 1302, 1285, 1331, 1312, 54, 37, 19, 0};


__constant__ int numTrianglesInCubeCategory[256] = {0, 1, 1, 2, 1, 2, 2, 3, 1, 2,
  2, 3, 2, 3, 3, 2, 1, 2, 2, 3, 2, 3, 3, 4, 2, 3, 3, 4, 3, 4, 4, 3, 1, 2, 2, 3, 2,
  3, 3, 4, 2, 3, 3, 4, 3, 4, 4, 3, 2, 3, 3, 2, 3, 4, 4, 3, 3, 4, 4, 3, 4, 5, 5, 2,
  1, 2, 2, 3, 2, 3, 3, 4, 2, 3, 3, 4, 3, 4, 4, 3, 2, 3, 3, 4, 3, 4, 4, 5, 3, 4, 4,
  5, 4, 5, 5, 4, 2, 3, 3, 4, 3, 4, 2, 3, 3, 4, 4, 5, 4, 5, 3, 2, 3, 4, 4, 3, 4, 5,
  3, 2, 4, 5, 5, 4, 5, 2, 4, 1, 1, 2, 2, 3, 2, 3, 3, 4, 2, 3, 3, 4, 3, 4, 4, 3, 2,
  3, 3, 4, 3, 4, 4, 5, 3, 2, 4, 3, 4, 3, 5, 2, 2, 3, 3, 4, 3, 4, 4, 5, 3, 4, 4, 5,
  4, 5, 5, 4, 3, 4, 4, 3, 4, 5, 5, 4, 4, 3, 5, 2, 5, 4, 2, 1, 2, 3, 3, 4, 3, 4, 4,
  5, 3, 4, 4, 5, 2, 3, 3, 2, 3, 4, 4, 5, 4, 5, 5, 2, 4, 3, 5, 4, 3, 2, 4, 1, 3, 4,
  4, 5, 4, 5, 3, 4, 4, 5, 5, 2, 3, 4, 2, 1, 2, 3, 3, 2, 3, 4, 2, 1, 3, 2, 4, 1, 2,
  1, 1, 0};

struct is_not_neg_int{
  __host__ __device__
  bool operator()(const int x)
  {
    return (x >= 0);
  }
};
struct is_not_zero_float{
  __host__ __device__
  bool operator()(const float x)
  {
    return (x != 0.0f);
  }
};

//TODO maybe get the third convolution to get closer to gausian filter
__device__ __host__ float3 blender(const float3 &a, const float3 &b, const float &bw){
  float t[3] = {(a.x-b.x)/bw,(a.y-b.y)/bw,(a.z-b.z)/bw};
  float result[3] = {0.0f};
  for(int i = 0; i < 3; ++i){
    if(t[i] > 0.5 && t[i] <= 1.5){
      result[i] = (t[i]-1.5)*(t[i]-1.5)/(2.0f);//*bw*bw*bw);
    }
    else if(t[i] < -0.5 && t[i] >= -1.5){
      result[i] = (t[i]+1.5)*(t[i]+1.5)/(2.0f);//*bw*bw*bw);
    }
    else if(t[i] <= 0.5 && t[i] >= -0.5){
      result[i] = (1.5-(t[i]*t[i]))/(2.0f);//*bw*bw*bw);
    }
    else return {0.0f,0.0f,0.0f};
  }
  return {result[0],result[1],result[2]};
}
__device__ __host__ float3 blenderPrime(const float3 &a, const float3 &b, const float &bw){
  float t[3] = {(a.x-b.x)/bw,(a.y-b.y)/bw,(a.z-b.z)/bw};
  float result[3] = {0.0f};
  for(int i = 0; i < 3; ++i){
    if(t[i] > 0.5 && t[i] <= 1.5){
      result[i] = (2.0f*t[i] + 3.0f)/(2.0f);//*bw*bw*bw);
    }
    else if(t[i] < -0.5 && t[i] >= -1.5){
      result[i] = (2.0f*t[i] - 3.0f)/(2.0f);//*bw*bw*bw);
    }
    else if(t[i] <= 0.5 && t[i] >= -0.5){
      result[i] = (-1.0f*t[i]);//(bw*bw*bw);
    }
    else return {0.0f,0.0f,0.0f};
  }
  return {result[0],result[1],result[2]};
}
__device__ __host__ float3 blenderPrimePrime(const float3 &a, const float3 &b, const float &bw){
  float t[3] = {(a.x-b.x)/bw,(a.y-b.y)/bw,(a.z-b.z)/bw};
  float result[3] = {0.0f};
  for(int i = 0; i < 3; ++i){
    if((t[i] > 0.5 && t[i] <= 1.5)||(t[i] < -0.5 && t[i] >= -1.5)){
      result[i] = 1.0f;//(bw*bw*bw);
    }
    else if(t[i] <= 0.5 && t[i] >= -0.5){
      result[i] = -1.0f;//(bw*bw*bw);
    }
    else return {0.0f,0.0f,0.0f};
  }
  return {result[0],result[1],result[2]};
}

__device__ __host__ int3 splitCrunchBits3(const unsigned int &size, const int &key){
  int3 xyz = {0,0,0};
  for(int i = size - 1;i >= 0;){
    xyz.x = (xyz.x << 1) + ((key >> i) & 1);
    --i;
    xyz.y = (xyz.y << 1) + ((key >> i) & 1);
    --i;
    xyz.z = (xyz.z << 1) + ((key >> i) & 1);
    --i;
  }
  return xyz;
}

__global__ void computeVectorFeild(Node* nodeArray, int numFinestNodes, float3* vectorField, float3* normals, float3* points){
  int blockID = blockIdx.y * gridDim.x + blockIdx.x;
  if(blockID < numFinestNodes){
    __shared__ float3 vec;
    vec = {0.0f, 0.0f, 0.0f};
    __syncthreads();
    int neighborIndex = nodeArray[blockID].neighbors[threadIdx.x];
    if(neighborIndex != -1){
      int currentPoint = nodeArray[neighborIndex].pointIndex;
      int stopIndex = nodeArray[neighborIndex].numPoints + currentPoint;
      float3 blend = {0.0f,0.0f,0.0f};
      float width = nodeArray[blockID].width;
      float3 center = nodeArray[blockID].center;
      for(int i = currentPoint; i < stopIndex; ++i){
        //n = 2 Fo(q) make bounds {0.0f, 1.0f}
          //blend = 1.0f - blend;
        //n = 2 Fo(q) make bounds {-1.0f, 0.0f}
          //blend = blend + 1.0f;
        //n currently = 3
        blend = blender(points[i],center,width)*normals[i];
        if(blend.x == 0.0f && blend.y == 0.0f && blend.z == 0.0f) continue;
        atomicAdd(&vec.x, blend.x);
        atomicAdd(&vec.y, blend.y);
        atomicAdd(&vec.z, blend.z);
      }
    }
    __syncthreads();
    if(threadIdx.x != 0) return;
    else vectorField[blockID] = vec;
  }
}
__global__ void computeDivergenceFine(int depthOfOctree, Node* nodeArray, int numNodes, int depthIndex, float3* vectorField, float* divCoeff, float* fPrimeLUT){
  int blockID = blockIdx.y * gridDim.x + blockIdx.x;
  if(blockID < numNodes){
    __shared__ float coeff;
    int neighborIndex = nodeArray[blockID + depthIndex].neighbors[threadIdx.x];
    if(neighborIndex != -1){
      int numFinestChildren = nodeArray[neighborIndex].numFinestChildren;
      int finestChildIndex = nodeArray[neighborIndex].finestChildIndex;
      int3 xyz1;
      int3 xyz2;
      xyz1 = splitCrunchBits3(depthOfOctree*3, nodeArray[blockID + depthIndex].key);
      int mult = pow(2,depthOfOctree + 1) - 1;
      for(int i = finestChildIndex; i < finestChildIndex + numFinestChildren; ++i){
        xyz2 = splitCrunchBits3(depthOfOctree*3, nodeArray[i].key);
        atomicAdd(&coeff, dotProduct(vectorField[i], {fPrimeLUT[xyz1.x*mult + xyz2.x],fPrimeLUT[xyz1.y*mult + xyz2.y],fPrimeLUT[xyz1.z*mult + xyz2.z]}));
      }
      __syncthreads();
      //may want only one thread doing this should not matter though
      divCoeff[blockID + depthIndex] = coeff;
    }
  }
}
__global__ void findRelatedChildren(Node* nodeArray, int numNodes, int depthIndex, int2* relativityIndicators){
  int blockID = blockIdx.y * gridDim.x + blockIdx.x;
  if(blockID < numNodes){
    __shared__ int numRelativeChildren;
    __shared__ int firstRelativeChild;
    numRelativeChildren = 0;
    firstRelativeChild = 2147483647;//max int
    int neighborIndex = nodeArray[blockID + depthIndex].neighbors[threadIdx.x];
    if(neighborIndex != -1){
      //may not be helping anything by doing this but it prevents 2 accesses to global memory
      int registerChildChecker = nodeArray[neighborIndex].numFinestChildren;
      int registerChildIndex = nodeArray[neighborIndex].finestChildIndex;
      if(registerChildIndex != -1 && registerChildChecker != 0){
        atomicAdd(&numRelativeChildren, nodeArray[neighborIndex].numFinestChildren);
        atomicMin(&firstRelativeChild, nodeArray[neighborIndex].finestChildIndex);
      }
    }
    __syncthreads();
    //may want only one thread doing this should not matter though
    relativityIndicators[blockID].x = firstRelativeChild;
    relativityIndicators[blockID].y = numRelativeChildren;
  }
}
//TODO optimize with warp aggregated atomics
__global__ void computeDivergenceCoarse(int depthOfOctree, Node* nodeArray, int2* relativityIndicators, int currentNode, int depthIndex, float3* vectorField, float* divCoeff, float* fPrimeLUT){
  int globalID = blockIdx.x *blockDim.x + threadIdx.x;
  if(globalID < relativityIndicators[currentNode].y){
    globalID += relativityIndicators[currentNode].x;
    int3 xyz1;
    int3 xyz2;
    xyz1 = splitCrunchBits3(depthOfOctree*3, nodeArray[currentNode + depthIndex].key);
    xyz2 = splitCrunchBits3(depthOfOctree*3, nodeArray[globalID].key);
    int mult = pow(2,depthOfOctree + 1) - 1;
    //TODO try and find a way to optimize this so that it is not using atomics and global memory
    float fx,fy,fz;
    fx = fPrimeLUT[xyz1.x*mult + xyz2.x];
    fy = fPrimeLUT[xyz1.y*mult + xyz2.y];
    fz = fPrimeLUT[xyz1.z*mult + xyz2.z];
    float divergenceContributer = dotProduct(vectorField[globalID], {fx,fy,fz});
    atomicAdd(&divCoeff[currentNode + depthIndex], divergenceContributer);
  }
}

__global__ void computeLd(int depthOfOctree, Node* nodeArray, int numNodes, int depthIndex, float* laplacianValues, int* laplacianIndices, float* fLUT, float* fPrimePrimeLUT){
  int blockID = blockIdx.y * gridDim.x + blockIdx.x;
  if(blockID < numNodes){
    int neighborIndex = nodeArray[blockID + depthIndex].neighbors[threadIdx.x];
    if(neighborIndex != -1){
      int3 xyz1;
      int3 xyz2;
      xyz1 = splitCrunchBits3(depthOfOctree*3, nodeArray[blockID + depthIndex].key);
      xyz2 = splitCrunchBits3(depthOfOctree*3, nodeArray[neighborIndex].key);
      int mult = pow(2,depthOfOctree + 1) - 1;
      float laplacianValue = (fPrimePrimeLUT[xyz1.x*mult + xyz2.x]*fLUT[xyz1.y*mult + xyz2.y]*fLUT[xyz1.z*mult + xyz2.z])+
      (fLUT[xyz1.x*mult + xyz2.x]*fPrimePrimeLUT[xyz1.y*mult + xyz2.y]*fLUT[xyz1.z*mult + xyz2.z])+
      (fLUT[xyz1.x*mult + xyz2.x]*fLUT[xyz1.y*mult + xyz2.y]*fPrimePrimeLUT[xyz1.z*mult + xyz2.z]);
      if(laplacianValue != 0.0f){
        laplacianValues[blockID*27 + threadIdx.x] = laplacianValue;
        laplacianIndices[blockID*27 + threadIdx.x] = neighborIndex - depthIndex;
      }
      else{
        laplacianIndices[blockID*27 + threadIdx.x] = -1;
      }
    }
    else{
      laplacianIndices[blockID*27 + threadIdx.x] = -1;
    }
  }
}
__global__ void computeLdCSR(int depthOfOctree, Node* nodeArray, int numNodes, int depthIndex, float* laplacianValues, int* laplacianIndices, int* numNonZero, float* fLUT, float* fPrimePrimeLUT){
  int blockID = blockIdx.y * gridDim.x + blockIdx.x;
  if(blockID < numNodes){
    int neighborIndex = nodeArray[blockID + depthIndex].neighbors[threadIdx.x];
    if(neighborIndex != -1){
      __shared__ int numNonZeroRow;
      numNonZeroRow = 0;
      __syncthreads();
      int3 xyz1;
      int3 xyz2;
      xyz1 = splitCrunchBits3(depthOfOctree*3, nodeArray[blockID + depthIndex].key);
      xyz2 = splitCrunchBits3(depthOfOctree*3, nodeArray[neighborIndex].key);
      int mult = pow(2,depthOfOctree + 1) - 1;
      float laplacianValue = (fPrimePrimeLUT[xyz1.x*mult + xyz2.x]*fLUT[xyz1.y*mult + xyz2.y]*fLUT[xyz1.z*mult + xyz2.z])+
      (fLUT[xyz1.x*mult + xyz2.x]*fPrimePrimeLUT[xyz1.y*mult + xyz2.y]*fLUT[xyz1.z*mult + xyz2.z])+
      (fLUT[xyz1.x*mult + xyz2.x]*fLUT[xyz1.y*mult + xyz2.y]*fPrimePrimeLUT[xyz1.z*mult + xyz2.z]);
      if(laplacianValue != 0.0f){
        laplacianValues[blockID*27 + threadIdx.x] = laplacianValue;
        laplacianIndices[blockID*27 + threadIdx.x] = neighborIndex - depthIndex;
        atomicAdd(&numNonZeroRow, 1);
      }
      else{
        laplacianIndices[blockID*27 + threadIdx.x] = -1;
      }
      __syncthreads();
      atomicAdd(&numNonZero[blockID + 1], 1);
    }
    else{
      laplacianIndices[blockID*27 + threadIdx.x] = -1;
    }
  }
}
__global__ void updateDivergence(int depthOfOctree, Node* nodeArray, int numNodes, int depthIndex, float* divCoeff, float* fLUT, float* fPrimePrimeLUT, float* nodeImplicit){
  int blockID = blockIdx.y * gridDim.x + blockIdx.x;
  if(blockID < numNodes){
    __shared__ float update;
    update = 0.0f;
    __syncthreads();
    int parent = nodeArray[blockID + depthIndex].parent;
    float laplacianValue = 0.0f;
    int mult = pow(2,depthOfOctree + 1) - 1;
    float nodeImplicitValue = 0.0f;
    int3 xyz1 = splitCrunchBits3(depthOfOctree*3, nodeArray[blockID + depthIndex].key);
    int3 xyz2 = {0,0,0};
    while(parent != -1){
      int parentNeighbor = nodeArray[parent].neighbors[threadIdx.x];
      if(parentNeighbor != -1){
        nodeImplicitValue = nodeImplicit[parentNeighbor];
        laplacianValue = 0.0f;
        xyz2 = splitCrunchBits3(depthOfOctree*3, nodeArray[parentNeighbor].key);
        laplacianValue = (fPrimePrimeLUT[xyz1.x*mult + xyz2.x]*fLUT[xyz1.y*mult + xyz2.y]*fLUT[xyz1.z*mult + xyz2.z])+
        (fLUT[xyz1.x*mult + xyz2.x]*fPrimePrimeLUT[xyz1.y*mult + xyz2.y]*fLUT[xyz1.z*mult + xyz2.z])+
        (fLUT[xyz1.x*mult + xyz2.x]*fLUT[xyz1.y*mult + xyz2.y]*fPrimePrimeLUT[xyz1.z*mult + xyz2.z]);
        if(laplacianValue != 0.0f) atomicAdd(&update, laplacianValue*nodeImplicitValue);
      }
      parent = nodeArray[parent].parent;
    }
    __syncthreads();
    if(threadIdx.x == 0){
      divCoeff[blockID + depthIndex] -= update;
      if(!isfinite(divCoeff[blockID + depthIndex])){
        printf("BROKEN %d,%f\n",blockID + depthIndex, update);
      }
    }
  }
}

__global__ void multiplyLdAnd1D(int numNodesAtDepth, float* laplacianValues, int* laplacianIndices, float* matrix1D, float* result){
  int blockID = blockIdx.y * gridDim.x + blockIdx.x;
  if(blockID < numNodesAtDepth){
    __shared__ float resultElement;
    // may need to do the following
    // resultElement = 0.0f;
    // __syncthreads();
    int laplacianIndex = laplacianIndices[blockID*27 + threadIdx.x];
    if(laplacianIndex != -1){
      //printf("%f,%f\n", laplacianValues[blockID*27 + threadIdx.x],matrix1D[laplacianIndex]);
      atomicAdd(&resultElement, laplacianValues[blockID*27 + threadIdx.x]*matrix1D[laplacianIndex]);
    }
    __syncthreads();
    if(threadIdx.x == 0 && resultElement != 0.0f){
      atomicAdd(&result[blockID], resultElement);
    }
  }
}
__global__ void computeAlpha(int numNodesAtDepth, float* r, float* pTL, float* p, float* numerator, float* denominator){
  *numerator = 0.0f;
  *denominator = 0.0f;
  __syncthreads();
  int globalID = blockIdx.x * blockDim.x + threadIdx.x;
  if(globalID < numNodesAtDepth){
    __shared__ float numeratorPartial;
    __shared__ float denominatorPartial;
    numeratorPartial = 0.0f;
    denominatorPartial = 0.0f;
    __syncthreads();
    atomicAdd(&numeratorPartial, r[globalID]*r[globalID]);
    atomicAdd(&denominatorPartial, pTL[globalID]*p[globalID]);
    __syncthreads();
    if(threadIdx.x == 0 && numeratorPartial != 0.0f && denominatorPartial != 0.0f){
      atomicAdd(numerator, numeratorPartial);
      atomicAdd(denominator, denominatorPartial);
    }
  }
}
__global__ void updateX(int numNodesAtDepth, int depthIndex, float* x, float alpha, float* p){
  int globalID = blockIdx.x *blockDim.x + threadIdx.x;
  if(globalID < numNodesAtDepth){
    x[globalID + depthIndex] = alpha*p[globalID] + x[globalID + depthIndex];
  }
}
__global__ void computeRNew(int numNodesAtDepth, float* r, float alpha, float* temp){
  int globalID = blockIdx.x *blockDim.x + threadIdx.x;
  if(globalID < numNodesAtDepth){
    float registerPlaceHolder = 0.0f;
    registerPlaceHolder = -1.0f*alpha*temp[globalID] + r[globalID];
    temp[globalID] = registerPlaceHolder;

  }
}
__global__ void computeBeta(int numNodesAtDepth, float* r, float* rNew, float* numerator, float* denominator){
  *numerator = 0.0f;
  *denominator = 0.0f;
  __syncthreads();
  int globalID = blockIdx.x *blockDim.x + threadIdx.x;
  if(globalID < numNodesAtDepth){
    __shared__ float numeratorPartial;
    __shared__ float denominatorPartial;
    numeratorPartial = 0.0f;
    denominatorPartial = 0.0f;
    __syncthreads();
    atomicAdd(&numeratorPartial, rNew[globalID]*rNew[globalID]);
    atomicAdd(&denominatorPartial, r[globalID]*r[globalID]);
    __syncthreads();
    if(threadIdx.x == 0){
      atomicAdd(numerator, numeratorPartial);
      atomicAdd(denominator, denominatorPartial);
    }
  }
}
__global__ void updateP(int numNodesAtDepth, float* rNew, float beta, float* p){
  int globalID = blockIdx.x *blockDim.x + threadIdx.x;
  if(globalID < numNodesAtDepth){
    p[globalID] = beta*p[globalID] + rNew[globalID];
  }
}

__global__ void pointSumImplicitTraversal(int numPoints, float3* points, Node* nodeArray, int root, float* nodeImplicit, float* sumImplicit){
  int globalID = blockIdx.x *blockDim.x + threadIdx.x;
  __shared__ float blockSumImplicit;
  blockSumImplicit = 0.0f;
  __syncthreads();
  if(globalID < numPoints){
    int nodeIndex = root;
    bool noChildren = false;
    int childIndex = -1;
    int currentNodePointIndex = -1;
    float regPointImplicit = 0.0f;
    float currentImplicit = 0.0f;
    while(!noChildren){
      currentImplicit = nodeImplicit[nodeIndex];

      //LOOOKKKKKK
      regPointImplicit += currentImplicit;
      //printf("%d,%f\n",nodeIndex,nodeImplicit[nodeIndex]);

      for(int i = 0; i < 8; ++i){
        childIndex = nodeArray[nodeIndex].children[i];
        if(childIndex == -1) continue;
        currentNodePointIndex = nodeArray[childIndex].pointIndex;
        if(globalID >= currentNodePointIndex && globalID < currentNodePointIndex + nodeArray[childIndex].numPoints){
          nodeIndex = childIndex;
        }
      }
      if(childIndex == -1){
        //printf("%d = %f\n",globalID,regPointImplicit);
        atomicAdd(&blockSumImplicit, regPointImplicit);
        break;
      }
    }
    __syncthreads();
    if(threadIdx.x == 0){
      atomicAdd(sumImplicit, blockSumImplicit);
    }
  }
}
__global__ void vertexSumImplicitTraversal(int numVertices, Vertex* vertexArray, float* nodeImplicit, float* vertexImplicit, float* sumImplicit, int numPoints){
  int blockID = blockIdx.y * gridDim.x + blockIdx.x;
  if(blockID < numVertices){
    __shared__ float blockImplicit;
    blockImplicit = 0.0f;
    __syncthreads();
    int associatedNode = vertexArray[blockID].nodes[threadIdx.x];
    if(associatedNode != -1){
      atomicAdd(&blockImplicit, nodeImplicit[associatedNode]);
    }
    __syncthreads();
    if(threadIdx.x == 0){
      float regAVGImp = (*sumImplicit)/numPoints;
      vertexImplicit[blockID] = blockImplicit - regAVGImp;
    }
  }
}

__global__ void vertexImplicitFromNormals(int numVertices, Vertex* vertexArray, Node* nodeArray, float3* normals, float3* points, float* vertexImplicit){
  int blockID = blockIdx.y * gridDim.x + blockIdx.x;
  if(blockID < numVertices){
    __shared__ float imp;
    imp = 0.0f;
    __syncthreads();
    int node = vertexArray[blockID].nodes[threadIdx.x];
    if(node != -1){
      float3 vertex = vertexArray[blockID].coord;
      int numPoints = 0;
      int pointIndex = 0;
      float3 currentNormal = {0.0f,0.0f,0.0f};
      float3 currentVector = {0.0f,0.0f,0.0f};
      float dot = 0.0f;
      int neighbor = 0;
      for(int n = 0; n < 27; ++n){
        neighbor = nodeArray[node].neighbors[n];
        if(neighbor == -1) continue;
        numPoints = nodeArray[neighbor].numPoints;
        pointIndex = nodeArray[neighbor].pointIndex;
        for(int p = pointIndex; p < pointIndex + numPoints; ++p){
          dot = 0.0f;
          currentNormal = normals[pointIndex];
          currentNormal = currentNormal/sqrtf(dotProduct(currentNormal,currentNormal));
          currentVector = vertex - points[pointIndex];
          currentVector = currentVector/sqrtf(dotProduct(currentVector,currentVector));
          dot = dotProduct(currentNormal,currentVector);
          atomicAdd(&imp, dot);
        }
      }
    }
    __syncthreads();
    vertexImplicit[blockID] = imp;
  }
}

__global__ void calcVertexNumbers(int numEdges, Edge* edgeArray, float* vertexImplicit, int* vertexNumbers){
  int globalID = blockIdx.x * blockDim.x + threadIdx.x;
  if(globalID < numEdges){
    float impV1 = 0;
    float impV2 = 0;
    impV1 = vertexImplicit[edgeArray[globalID].v1];
    impV2 = vertexImplicit[edgeArray[globalID].v2];
    if(impV1 > 0.0f && impV2 < 0.0f || impV1 < 0.0f && impV2 > 0.0f){
      vertexNumbers[globalID] = 1;
    }
    else{
      vertexNumbers[globalID] = 0;
    }
  }
}
__global__ void determineCubeCategories(int numNodes, Node* nodeArray, int* vertexNumbers, int* cubeCategory, int* triangleNumbers){
  int globalID = blockIdx.x * blockDim.x + threadIdx.x;
  if(globalID < numNodes){
    int edgeBasedCategory = 0;
    int regEdge = 0;
    for(int i = 11; i >= 0; --i){
      regEdge = nodeArray[globalID].edges[i];
      if(vertexNumbers[regEdge]){
        edgeBasedCategory = (edgeBasedCategory << 1) + 1;
      }
      else{
        edgeBasedCategory <<= 1;
      }
    }
    for(int i = 0; i < 256; ++i){
      if(edgeBasedCategory == cubeCategoryEdgeIdentity[i]){
        triangleNumbers[globalID] = numTrianglesInCubeCategory[i];
        cubeCategory[globalID] = i;
        break;
      }
    }
  }
}
__global__ void generateSurfaceVertices(int numEdges, Edge* edgeArray, Vertex* vertexArray, int* vertexNumbers, int* vertexAddresses, float3* surfaceVertices){
  int globalID = blockIdx.x * blockDim.x + threadIdx.x;
  if(globalID < numEdges){
    if(vertexNumbers[globalID] == 1){
      int v1 = edgeArray[globalID].v1;
      int v2 = edgeArray[globalID].v2;
      float3 midPoint = vertexArray[v1].coord + vertexArray[v2].coord;
      midPoint = midPoint/2.0f;
      int vertAddress = (globalID == 0) ? 0 : vertexAddresses[globalID - 1];
      surfaceVertices[vertAddress] = midPoint;
    }
  }
}
__global__ void generateSurfaceTriangles(int numNodes, Node* nodeArray, int* vertexAddresses, int* triangleNumbers, int* triangleAddresses, int* cubeCategory, int3* surfaceTriangles){
  int blockID = blockIdx.y * gridDim.x + blockIdx.x;
  if(blockID < numNodes){
    int numTrianglesInNode = triangleNumbers[blockID];
    if(threadIdx.x < numTrianglesInNode){
      int3 nodeTriangle = {cubeCategoryTrianglesFromEdges[cubeCategory[blockID]][threadIdx.x*3],
        cubeCategoryTrianglesFromEdges[cubeCategory[blockID]][threadIdx.x*3 + 1],
        cubeCategoryTrianglesFromEdges[cubeCategory[blockID]][threadIdx.x*3 + 2]};
      int3 surfaceTriangle = {nodeArray[blockID].edges[nodeTriangle.x],
        nodeArray[blockID].edges[nodeTriangle.y],
        nodeArray[blockID].edges[nodeTriangle.z]};
      int triAddress = (blockID == 0) ? 0 : triangleAddresses[blockID - 1] + threadIdx.x;
      int3 vertAddress = {-1,-1,-1};
      vertAddress.x = (surfaceTriangle.x == 0) ? 0 : vertexAddresses[surfaceTriangle.x - 1];
      vertAddress.y = (surfaceTriangle.y == 0) ? 0 : vertexAddresses[surfaceTriangle.y - 1];
      vertAddress.z = (surfaceTriangle.z == 0) ? 0 : vertexAddresses[surfaceTriangle.z - 1];
      surfaceTriangles[triAddress] = vertAddress;
    }
  }
}

Surface::Surface(std::string pathToPLY, int depthOfOctree){

  std::cout<<"---------------------------------------------------"<<std::endl;
  std::cout<<"COMPUTING OCTREE\n"<<std::endl;

  this->octree = new Octree(pathToPLY, depthOfOctree);
  this->octree->init_octree_gpu();
  this->octree->generateKeys();
  this->octree->prepareFinestUniquNodes();
  this->octree->createFinalNodeArray();
  this->octree->freePrereqArrays();
  this->octree->fillLUTs();
  this->octree->fillNeighborhoods();
  if(!this->octree->normalsComputed){
    this->octree->computeNormals(3, 20);
  }
  this->octree->computeVertexArray();
  this->octree->computeEdgeArray();
  this->octree->computeFaceArray();
  std::cout<<"---------------------------------------------------"<<std::endl;

  float* divergenceVector = new float[this->octree->totalNodes];
  for(int i = 0; i < this->octree->totalNodes; ++i){
    divergenceVector[i] = 0.0f;
  }
  CudaSafeCall(cudaMalloc((void**)&this->divergenceVectorDevice, this->octree->totalNodes*sizeof(float)));
  CudaSafeCall(cudaMemcpy(this->divergenceVectorDevice, divergenceVector, this->octree->totalNodes*sizeof(float), cudaMemcpyHostToDevice));
  if(!this->octree->pointsDeviceReady) this->octree->copyPointsToDevice();
  if(!this->octree->normalsDeviceReady) this->octree->copyNormalsToDevice();
}

Surface::Surface(){

}

Surface::~Surface(){
  delete this->octree;
}

//TODO OPTMIZE THIS YOU FUCK TARD
void Surface::computeLUTs(){
  clock_t timer;
  timer = clock();

  float currentWidth = this->octree->width;
  float3 currentCenter = this->octree->center;
  float3 tempCenter = {0.0f,0.0f,0.0f};
  int pow2 = 1;
  std::vector<float3> centers;
  std::queue<float3> centersTemp;
  centersTemp.push(currentCenter);
  for(int d = 0; d <= this->octree->depth; ++d){
    for(int i = 0; i < pow2; ++i){
      tempCenter = centersTemp.front();
      centersTemp.pop();
      centers.push_back(tempCenter);
      centersTemp.push(tempCenter - (currentWidth/4));
      centersTemp.push(tempCenter + (currentWidth/4));
    }
    currentWidth /= 2;
    pow2 *= 2;
  }
  int numCenters = centers.size();
  //printf("number of absolute unique centers = %d\n\n",numCenters);

  unsigned int size = (pow(2, this->octree->depth + 1) - 1);
  float** f = new float*[size];
  float** ff = new float*[size];
  float** fff = new float*[size];
  for(int i = 0; i < size; ++i){
    f[i] = new float[size];
    ff[i] = new float[size];
    fff[i] = new float[size];
  }

  int pow2i = 1;
  int offseti = 0;
  int pow2j = 1;
  int offsetj = 0;
  for(int i = 0; i <= this->octree->depth; ++i){
    offseti = pow2i - 1;
    pow2j = 1;
    for(int j = 0; j <= this->octree->depth; ++j){
      offsetj = pow2j - 1;
      for(int k = offseti; k < offseti + pow2i; ++k){
        for(int l = offsetj; l < offsetj + pow2j; ++l){
          f[k][l] = dotProduct(blender(centers[l],centers[k],this->octree->width/pow2i),blender(centers[k],centers[l],this->octree->width/pow2j));
          ff[k][l] = dotProduct(blender(centers[l],centers[k],this->octree->width/pow2i),blenderPrime(centers[k],centers[l],this->octree->width/pow2j));
          fff[k][l] = dotProduct(blender(centers[l],centers[k],this->octree->width/pow2i),blenderPrimePrime(centers[k],centers[l],this->octree->width/pow2j));
          // if(f[k][l] == 0.0f && !(f[k][l] == 0.0f && ff[k][l] == 0.0f && fff[k][l] == 0.0f)){
          //   printf("%d,%d -> %.9f,%.9f,%.9f\n",k,l,f[k][l],ff[k][l],fff[k][l]);
          // }
          if(isfinite(f[k][l]) == 0|| isfinite(ff[k][l]) == 0|| isfinite(fff[k][l]) == 0){
            printf("FAILURE @ %d,%d -> %.9f,%.9f,%.9f\n",k,l,f[k][l],ff[k][l],fff[k][l]);
            exit(-1);
          }
        }
      }
      pow2j *= 2;
    }
    pow2i *= 2;
  }
  this->fLUT = new float[size*size];
  this->fPrimeLUT = new float[size*size];
  this->fPrimePrimeLUT = new float[size*size];
  for(int i = 0; i < size; ++i){
    for(int j = 0; j < size; ++j){
      this->fLUT[i*size + j] = f[i][j];
      this->fPrimeLUT[i*size + j] = ff[i][j];
      this->fPrimePrimeLUT[i*size + j] = fff[i][j];
    }
  }
  timer = clock() - timer;
  printf("blending LUT generation took %f seconds fully on the CPU.\n",((float) timer)/CLOCKS_PER_SEC);
}
//TODO should optimize computeDivergenceCoarse
//TODO THERE ARE MEMORY ACCESS PROBLEMS ORIGINATING PROBABLY FROM LUT STUFF!!!!!!!!!!!!!! FIXXXXXXXXx
void Surface::computeDivergenceVector(){
  clock_t cudatimer;
  cudatimer = clock();
  /*
  FIRST COMPUTE VECTOR FIELD
  */

  int numNodesAtDepth = 0;
  dim3 grid = {1,1,1};
  dim3 block = {1,1,1};
  numNodesAtDepth = this->octree->depthIndex[1];
  if(numNodesAtDepth < 65535) grid.x = (unsigned int) numNodesAtDepth;
  else{
    grid.x = 65535;
    while(grid.x*grid.y < numNodesAtDepth){
      ++grid.y;
    }
    while(grid.x*grid.y > numNodesAtDepth){
      --grid.x;

    }
    if(grid.x*grid.y < numNodesAtDepth){
      ++grid.x;
    }
  }
  block.x = 27;
  float3* vectorField = new float3[numNodesAtDepth];
  for(int i = 0; i < numNodesAtDepth; ++i){
    vectorField[i] = {0.0f,0.0f,0.0f};
  }
  float3* vectorFieldDevice;
  CudaSafeCall(cudaMalloc((void**)&vectorFieldDevice, numNodesAtDepth*sizeof(float3)));
  CudaSafeCall(cudaMemcpy(vectorFieldDevice, vectorField, numNodesAtDepth*sizeof(float3), cudaMemcpyHostToDevice));
  computeVectorFeild<<<grid,block>>>(this->octree->finalNodeArrayDevice, numNodesAtDepth, vectorFieldDevice, this->octree->normalsDevice, this->octree->pointsDevice);
  cudaDeviceSynchronize();//force this to finish as it is necessary for next kernels
  CudaCheckError();
  /*
  CudaSafeCall(cudaMemcpy(vectorField, vectorFieldDevice, numNodesAtDepth*sizeof(float3), cudaMemcpyDeviceToHost));
  for(int i = 0; i < numNodesAtDepth; ++i){
    if(vectorField[i].x != 0.0f && vectorField[i].y != 0.0f && vectorField[i].z != 0.0f){
      std::cout<<vectorField[i].x<<","<<vectorField[i].y<<","<<vectorField[i].z<<std::endl;
    }
  }
  */
  delete[] vectorField;
  cudatimer = clock() - cudatimer;
  printf("Vector field generation kernel took %f seconds.\n",((float) cudatimer)/CLOCKS_PER_SEC);
  cudatimer = clock();
  /*
  NOW COMPUTE DIVERGENCE VECTOR AFTER FINDING VECTOR FIELD
  */

  unsigned int size = (pow(2, this->octree->depth + 1) - 1);
  CudaSafeCall(cudaMalloc((void**)&this->fPrimeLUTDevice, size*size*sizeof(float)));
  CudaSafeCall(cudaMemcpy(this->fPrimeLUTDevice, this->fPrimeLUT, size*size*sizeof(float), cudaMemcpyHostToDevice));

  int2* relativityIndicators;
  int2* relativityIndicatorsDevice;
  for(int d = 0; d <= this->octree->depth; ++d){
    block = {27,1,1};
    grid = {1,1,1};
    if(d != this->octree->depth){
      numNodesAtDepth = this->octree->depthIndex[d + 1] - this->octree->depthIndex[d];
    }
    else numNodesAtDepth = 1;

    if(numNodesAtDepth < 65535) grid.x = (unsigned int) numNodesAtDepth;
    else{
      grid.x = 65535;
      while(grid.x*grid.y < numNodesAtDepth){
        ++grid.y;
      }
      while(grid.x*grid.y > numNodesAtDepth){
        --grid.x;
        if(grid.x*grid.y < numNodesAtDepth){
          ++grid.x;//to ensure that numThreads > nodes
          break;
        }
      }
    }
    if(d <= 5){//evaluate divergence coefficients at finer depths
      computeDivergenceFine<<<grid, block>>>(this->octree->depth, this->octree->finalNodeArrayDevice, numNodesAtDepth, this->octree->depthIndex[d], vectorFieldDevice, this->divergenceVectorDevice, this->fPrimeLUTDevice);
      CudaCheckError();
    }
    else{//evaluate divergence coefficients at coarser depths
      relativityIndicators = new int2[numNodesAtDepth];
      for(int i = 0; i < numNodesAtDepth; ++i){
        relativityIndicators[i] = {0,0};
      }
      CudaSafeCall(cudaMalloc((void**)&relativityIndicatorsDevice, numNodesAtDepth*sizeof(int2)));
      CudaSafeCall(cudaMemcpy(relativityIndicatorsDevice, relativityIndicators, numNodesAtDepth*sizeof(int2), cudaMemcpyHostToDevice));
      findRelatedChildren<<<grid, block>>>(this->octree->finalNodeArrayDevice, numNodesAtDepth, this->octree->depthIndex[d], relativityIndicatorsDevice);
      cudaDeviceSynchronize();
      CudaCheckError();
      CudaSafeCall(cudaMemcpy(relativityIndicators, relativityIndicatorsDevice, numNodesAtDepth*sizeof(int2), cudaMemcpyDeviceToHost));
      for(int currentNode = 0; currentNode < numNodesAtDepth; ++currentNode){
        block.x = 1;
        grid.y = 1;
        if(relativityIndicators[currentNode].y == 0) continue;//TODO ensure this assumption is valid
        else if(relativityIndicators[currentNode].y < 65535) grid.x = (unsigned int) relativityIndicators[currentNode].y;
        else{
          grid.x = 65535;
          while(grid.x*block.x < relativityIndicators[currentNode].y){
            ++block.x;
          }
          while(grid.x*block.x > relativityIndicators[currentNode].y){
            --grid.x;
            if(grid.x*block.x < relativityIndicators[currentNode].y){
              ++grid.x;//to ensure that numThreads > nodes
              break;
            }
          }
        }
        computeDivergenceCoarse<<<grid, block>>>(this->octree->depth, this->octree->finalNodeArrayDevice, relativityIndicatorsDevice, currentNode, this->octree->depthIndex[d], vectorFieldDevice, this->divergenceVectorDevice, this->fPrimeLUTDevice);
        CudaCheckError();
      }
      CudaSafeCall(cudaFree(relativityIndicatorsDevice));
      delete[] relativityIndicators;
    }
  }
  CudaSafeCall(cudaFree(vectorFieldDevice));
  CudaSafeCall(cudaFree(this->fPrimeLUTDevice));

  CudaSafeCall(cudaFree(this->octree->normalsDevice));
  this->octree->normalsDeviceReady = false;


  delete[] this->fPrimeLUT;

  cudatimer = clock() - cudatimer;
  printf("Divergence vector generation kernel took %f seconds.\n",((float) cudatimer)/CLOCKS_PER_SEC);
}

void Surface::computeImplicitFunction(){
  this->computeLUTs();
  this->computeDivergenceVector();

  clock_t timer;
  timer = clock();
  clock_t cudatimer;
  cudatimer = clock();

  unsigned int size = (pow(2, this->octree->depth + 1) - 1);
  float* nodeImplicit = new float[this->octree->totalNodes];
  for(int i = 0; i < this->octree->totalNodes; ++i){
    nodeImplicit[i] = 0.0f;
  }

  int numNodesAtDepth = 0;
  float* temp;
  int* tempInt;
  float* laplacianValuesDevice;
  int* laplacianIndicesDevice;
  float* rDevice;
  float* pDevice;
  float* temp1DDevice;
  dim3 grid;
  dim3 block;
  dim3 grid1D;
  dim3 block1D;
  float alpha = 0.0f;
  float beta = 0.0f;
  float* numeratorDevice;
  float* denominatorDevice;
  float denominator = 0.0f;
  CudaSafeCall(cudaMalloc((void**)&numeratorDevice, sizeof(float)));
  CudaSafeCall(cudaMalloc((void**)&denominatorDevice, sizeof(float)));
  CudaSafeCall(cudaMalloc((void**)&this->fLUTDevice, size*size*sizeof(float)));
  CudaSafeCall(cudaMalloc((void**)&this->fPrimePrimeLUTDevice, size*size*sizeof(float)));
  CudaSafeCall(cudaMalloc((void**)&this->nodeImplicitDevice, this->octree->totalNodes*sizeof(float)));
  CudaSafeCall(cudaMemcpy(numeratorDevice, &alpha, sizeof(float), cudaMemcpyHostToDevice));
  CudaSafeCall(cudaMemcpy(denominatorDevice, &alpha, sizeof(float), cudaMemcpyHostToDevice));
  CudaSafeCall(cudaMemcpy(this->fLUTDevice, this->fLUT, size*size*sizeof(float), cudaMemcpyHostToDevice));
  CudaSafeCall(cudaMemcpy(this->fPrimePrimeLUTDevice, this->fPrimePrimeLUT, size*size*sizeof(float), cudaMemcpyHostToDevice));
  CudaSafeCall(cudaMemcpy(this->nodeImplicitDevice, nodeImplicit, this->octree->totalNodes*sizeof(float), cudaMemcpyHostToDevice));

  for(int d = this->octree->depth; d >= 0; --d){
    //update divergence coefficients based on solutions at coarser depths
    grid = {1,1,1};
    block = {27,1,1};
    if(d != this->octree->depth){
      numNodesAtDepth = this->octree->depthIndex[d + 1] - this->octree->depthIndex[d];
      if(numNodesAtDepth < 65535) grid.x = (unsigned int) numNodesAtDepth;
      else{
        grid.x = 65535;
        while(grid.x*grid.y < numNodesAtDepth){
          ++grid.y;
        }
        while(grid.x*grid.y > numNodesAtDepth){
          --grid.x;
        }
        if(grid.x*grid.y < numNodesAtDepth){
          ++grid.x;
        }
      }
      for(int dcoarse = this->octree->depth; dcoarse >= d + 1; --dcoarse){
        updateDivergence<<<grid, block>>>(this->octree->depth, this->octree->finalNodeArrayDevice, numNodesAtDepth,
          this->octree->depthIndex[d], this->divergenceVectorDevice,
          this->fLUTDevice, this->fPrimePrimeLUTDevice, this->nodeImplicitDevice);
        CudaCheckError();
        exit(0);
      }
    }
    else{
      numNodesAtDepth = 1;
    }

    temp = new float[numNodesAtDepth*27];
    tempInt = new int[numNodesAtDepth*27];
    for(int i = 0; i < numNodesAtDepth*27; ++i){
      temp[i] = 0.0f;
      tempInt[i] = -1;
    }

    CudaSafeCall(cudaMalloc((void**)&laplacianValuesDevice, numNodesAtDepth*27*sizeof(float)));
    CudaSafeCall(cudaMalloc((void**)&laplacianIndicesDevice, numNodesAtDepth*27*sizeof(int)));
    CudaSafeCall(cudaMemcpy(laplacianValuesDevice, temp, numNodesAtDepth*27*sizeof(float), cudaMemcpyHostToDevice));
    CudaSafeCall(cudaMemcpy(laplacianIndicesDevice, tempInt, numNodesAtDepth*27*sizeof(int), cudaMemcpyHostToDevice));
    computeLd<<<grid, block>>>(this->octree->depth, this->octree->finalNodeArrayDevice, numNodesAtDepth, this->octree->depthIndex[d],
      laplacianValuesDevice, laplacianIndicesDevice, this->fLUTDevice, this->fPrimePrimeLUTDevice);
    CudaCheckError();

    CudaSafeCall(cudaMalloc((void**)&temp1DDevice, numNodesAtDepth*sizeof(float)));
    CudaSafeCall(cudaMalloc((void**)&pDevice, numNodesAtDepth*sizeof(float)));
    CudaSafeCall(cudaMalloc((void**)&rDevice, numNodesAtDepth*sizeof(float)));
    CudaSafeCall(cudaMemcpy(temp1DDevice, temp, numNodesAtDepth*sizeof(float),cudaMemcpyHostToDevice));
    CudaSafeCall(cudaMemcpy(rDevice, this->divergenceVectorDevice + this->octree->depthIndex[d], numNodesAtDepth*sizeof(float),cudaMemcpyDeviceToDevice));
    CudaSafeCall(cudaMemcpy(pDevice, this->divergenceVectorDevice + this->octree->depthIndex[d], numNodesAtDepth*sizeof(float),cudaMemcpyDeviceToDevice));

    delete[] temp;
    delete[] tempInt;

    //gradient solver r = b - Lx
    //r is instantiated as b
    //will converge in n iterations
    grid1D = {1,1,1};
    block1D = {1,1,1};
    //this will allow for at most ~67,000,000 numNodesAtDepth
    if(numNodesAtDepth < 65535) grid.x = (unsigned int) numNodesAtDepth;
    else{
      grid1D.x = 65535;
      while(grid1D.x*block1D.x < numNodesAtDepth){
        ++block1D.x;
      }
      while(grid1D.x*block1D.x > numNodesAtDepth){
        --grid1D.x;
      }
      if(grid1D.x*block1D.x < numNodesAtDepth){
        ++grid1D.x;
      }
    }
    // 1.temp = pT * Ld
    // 2.alpha = dot(r,r)/dot(temp,p)
    // 3.x = x + alpha*p
    // 4.temp = Ld * p
    // 5.temp = r - alpha*temp
    // 6.if(rValues == 0.0f) gradientSolverConverged = true, break
    // 7.p = temp + (dot(temp,temp)/dot(r,r))*p
    // STEPS 1 and 4 MAY RESULT IN THE SAME THING
    beta = 0.0f;
    alpha = 0.0f;
    for(int i = 0; i < numNodesAtDepth; ++i){
      multiplyLdAnd1D<<<grid, block>>>(numNodesAtDepth, laplacianValuesDevice, laplacianIndicesDevice, pDevice, temp1DDevice);
      CudaCheckError();
      cudaDeviceSynchronize();
      computeAlpha<<<grid1D, block1D>>>(numNodesAtDepth, rDevice, temp1DDevice, pDevice, numeratorDevice, denominatorDevice);
      CudaCheckError();
      CudaSafeCall(cudaMemcpy(&alpha, numeratorDevice, sizeof(float), cudaMemcpyDeviceToHost));
      CudaSafeCall(cudaMemcpy(&denominator, denominatorDevice, sizeof(float), cudaMemcpyDeviceToHost));
      alpha /= denominator;
      updateX<<<grid1D, block1D>>>(numNodesAtDepth, this->octree->depthIndex[d], this->nodeImplicitDevice, alpha, pDevice);
      CudaCheckError();
      computeRNew<<<grid1D, block1D>>>(numNodesAtDepth, rDevice, alpha, temp1DDevice);
      CudaCheckError();
      cudaDeviceSynchronize();
      computeBeta<<<grid1D, block1D>>>(numNodesAtDepth, rDevice, temp1DDevice, numeratorDevice, denominatorDevice);
      CudaCheckError();
      CudaSafeCall(cudaMemcpy(&beta, numeratorDevice, sizeof(float), cudaMemcpyDeviceToHost));
      CudaSafeCall(cudaMemcpy(&denominator, denominatorDevice, sizeof(float), cudaMemcpyDeviceToHost));

      beta /= denominator;
      //std::cout<<beta<<std::endl;
      if(denominator == 0.0f || .01 > sqrtf(beta)){
        printf("CONVERGED AT %d\n",i);
        break;
      }
      if(isfinite(beta) == 0){
        exit(0);
      }
      updateP<<<grid1D, block1D>>>(numNodesAtDepth, rDevice, beta, pDevice);
      CudaCheckError();
      CudaSafeCall(cudaMemcpy(rDevice, temp1DDevice, numNodesAtDepth*sizeof(float), cudaMemcpyDeviceToDevice));
    }
    CudaSafeCall(cudaFree(laplacianValuesDevice));
    CudaSafeCall(cudaFree(laplacianIndicesDevice));
    CudaSafeCall(cudaFree(temp1DDevice));
    CudaSafeCall(cudaFree(rDevice));
    CudaSafeCall(cudaFree(pDevice));
    cudatimer = clock() - cudatimer;
    //printf("beta was %f at convergence\n",beta);
    printf("Node Implicit computation for depth %d took %f seconds w/%d nodes.\n", this->octree->depth - d,((float) cudatimer)/CLOCKS_PER_SEC, numNodesAtDepth);
    cudatimer = clock();
  }
  CudaSafeCall(cudaFree(numeratorDevice));
  CudaSafeCall(cudaFree(denominatorDevice));
  CudaSafeCall(cudaFree(this->fLUTDevice));
  CudaSafeCall(cudaFree(this->fPrimePrimeLUTDevice));
  CudaSafeCall(cudaFree(this->divergenceVectorDevice));
  delete[] this->fLUT;
  delete[] this->fPrimePrimeLUT;
  delete[] nodeImplicit;
  timer = clock() - timer;
  printf("Node Implicit compuation took a total of %f seconds.\n\n",((float) timer)/CLOCKS_PER_SEC);
}
void Surface::computeImplicitMagma(){
  this->computeLUTs();
  this->computeDivergenceVector();

  clock_t timer;
  timer = clock();
  clock_t cudatimer;
  cudatimer = clock();

  unsigned int size = (pow(2, this->octree->depth + 1) - 1);
  int numNodesAtDepth = 0;
  float* temp;
  int* tempInt;
  int* numNonZero;
  int* numNonZeroDevice;
  float* laplacianValuesDevice;
  int* laplacianIndicesDevice;
  float* csrValues;
  int* csrIndices;
  float* csrValuesDevice;
  int* csrIndicesDevice;
  int totalNonZero = 0;
  float* partialDivergence;
  float* partialImplicit;
  int m;
  int n = 1;
  dim3 grid;
  dim3 block;

  CudaSafeCall(cudaMalloc((void**)&this->nodeImplicitDevice, this->octree->totalNodes*sizeof(float)));
  CudaSafeCall(cudaMalloc((void**)&this->fLUTDevice, size*size*sizeof(float)));
  CudaSafeCall(cudaMalloc((void**)&this->fPrimePrimeLUTDevice, size*size*sizeof(float)));
  CudaSafeCall(cudaMemcpy(this->fLUTDevice, this->fLUT, size*size*sizeof(float), cudaMemcpyHostToDevice));
  CudaSafeCall(cudaMemcpy(this->fPrimePrimeLUTDevice, this->fPrimePrimeLUT, size*size*sizeof(float), cudaMemcpyHostToDevice));

  for(int d = this->octree->depth; d >= 0; --d){
    //update divergence coefficients based on solutions at coarser depths
    grid = {1,1,1};
    block = {27,1,1};
    if(d != this->octree->depth){
      numNodesAtDepth = this->octree->depthIndex[d + 1] - this->octree->depthIndex[d];
      if(numNodesAtDepth < 65535) grid.x = (unsigned int) numNodesAtDepth;
      else{
        grid.x = 65535;
        while(grid.x*grid.y < numNodesAtDepth){
          ++grid.y;
        }
        while(grid.x*grid.y > numNodesAtDepth){
          --grid.x;
        }
        if(grid.x*grid.y < numNodesAtDepth){
          ++grid.x;
        }
      }
      updateDivergence<<<grid, block>>>(this->octree->depth, this->octree->finalNodeArrayDevice, numNodesAtDepth,
        this->octree->depthIndex[d], this->divergenceVectorDevice,
        this->fLUTDevice, this->fPrimePrimeLUTDevice, this->nodeImplicitDevice);
      cudaDeviceSynchronize();
      CudaCheckError();
    }
    else{
      numNodesAtDepth = 1;
    }

    temp = new float[numNodesAtDepth*27];
    tempInt = new int[numNodesAtDepth*27];
    numNonZero = new int[numNodesAtDepth + 1];
    numNonZero[0] = 0;
    for(int i = 0; i < numNodesAtDepth*27; ++i){
      temp[i] = 0.0f;
      tempInt[i] = -1;
      if(i % 27 == 0){
        numNonZero[(i/27) + 1] = 0;
      }
    }
    CudaSafeCall(cudaMalloc((void**)&numNonZeroDevice, (numNodesAtDepth+1)*sizeof(int)));
    CudaSafeCall(cudaMalloc((void**)&laplacianValuesDevice, numNodesAtDepth*27*sizeof(float)));
    CudaSafeCall(cudaMalloc((void**)&laplacianIndicesDevice, numNodesAtDepth*27*sizeof(int)));
    CudaSafeCall(cudaMemcpy(numNonZeroDevice, numNonZero, (numNodesAtDepth+1)*sizeof(int), cudaMemcpyHostToDevice));
    CudaSafeCall(cudaMemcpy(laplacianValuesDevice, temp, numNodesAtDepth*27*sizeof(float), cudaMemcpyHostToDevice));
    CudaSafeCall(cudaMemcpy(laplacianIndicesDevice, tempInt, numNodesAtDepth*27*sizeof(int), cudaMemcpyHostToDevice));

    computeLdCSR<<<grid, block>>>(this->octree->depth, this->octree->finalNodeArrayDevice, numNodesAtDepth, this->octree->depthIndex[d],
      laplacianValuesDevice, laplacianIndicesDevice, numNonZeroDevice, this->fLUTDevice, this->fPrimePrimeLUTDevice);
    cudaDeviceSynchronize();
    CudaCheckError();

    thrust::device_ptr<int> nN(numNonZeroDevice);
    thrust::inclusive_scan(nN, nN + numNodesAtDepth + 1, nN);
    CudaCheckError();
    CudaSafeCall(cudaMemcpy(numNonZero, numNonZeroDevice, (numNodesAtDepth+1)*sizeof(int), cudaMemcpyDeviceToHost));

    totalNonZero = numNonZero[numNodesAtDepth];

    delete[] temp;
    delete[] tempInt;
    csrValues = new float[totalNonZero];
    csrIndices = new int[totalNonZero];
    for(int i = 0; i < totalNonZero; ++i){
      csrValues[i] = 0.0f;
      csrIndices[i] = 0;
    }

    CudaSafeCall(cudaMalloc((void**)&csrValuesDevice, totalNonZero*sizeof(float)));
    CudaSafeCall(cudaMalloc((void**)&csrIndicesDevice, totalNonZero*sizeof(int)));

    thrust::device_ptr<float> arrayToCompact(laplacianValuesDevice);
    thrust::device_ptr<float> arrayOut(csrValuesDevice);
    thrust::device_ptr<int> placementToCompact(laplacianIndicesDevice);
    thrust::device_ptr<int> placementOut(csrIndicesDevice);

    thrust::copy_if(arrayToCompact, arrayToCompact + (numNodesAtDepth*27), arrayOut, is_not_zero_float());
    CudaCheckError();
    thrust::copy_if(placementToCompact, placementToCompact + (numNodesAtDepth*27), placementOut, is_not_neg_int());
    CudaCheckError();

    CudaSafeCall(cudaFree(laplacianValuesDevice));
    CudaSafeCall(cudaFree(laplacianIndicesDevice));
    CudaSafeCall(cudaMemcpy(csrValues, csrValuesDevice, totalNonZero*sizeof(float),cudaMemcpyDeviceToHost));
    CudaSafeCall(cudaMemcpy(csrIndices, csrIndicesDevice, totalNonZero*sizeof(int),cudaMemcpyDeviceToHost));

    partialDivergence = new float[numNodesAtDepth];
    CudaSafeCall(cudaMemcpy(partialDivergence, this->divergenceVectorDevice + this->octree->depthIndex[d], numNodesAtDepth*sizeof(float), cudaMemcpyDeviceToHost));
    partialImplicit = new float[numNodesAtDepth];
    for(int i = 0; i < numNodesAtDepth; ++i){
      partialImplicit[i] = 0.0f;
    }
    if(d != this->octree->depth){

      m = numNodesAtDepth;

      //DO SPARSE LINEAR SOLVER WITH A IN CSR FORMAT
      magma_init();
      magma_sopts opts;
      magma_queue_t queue;
      magma_queue_create(0 , &queue);

      magma_s_matrix A={Magma_CSR}, dA={Magma_CSR};
      magma_s_matrix b={Magma_CSR}, db={Magma_CSR};
      magma_s_matrix x={Magma_CSR}, dx={Magma_CSR};
      magma_scsrset(m, m, numNonZero, csrIndices, csrValues, &A, queue);
      magma_svset(m, n, partialDivergence, &b, queue);
      magma_svset(m, n, partialImplicit, &x, queue);

      opts.solver_par.solver     = Magma_CG;
      opts.solver_par.maxiter    = m;

      magma_smtransfer( A, &dA, Magma_CPU, Magma_DEV, queue );
      magma_smtransfer( b, &db, Magma_CPU, Magma_DEV, queue );
      magma_smtransfer( x, &dx, Magma_CPU, Magma_DEV, queue );

      //magma_scg_res(dA,db,&dx,&opts.solver_par,queue);//preconditioned cojugate gradient solver
      //magma_scg_merge(dA,db,&dx,&opts.solver_par,queue);//cojugate gradient in variant solver merge
      magma_scg(dA,db,&dx,&opts.solver_par,queue);//cojugate gradient solver
      //magma_s_solver(dA,db,&dx,&opts,queue);//cojugate gradient solver

      magma_smfree( &x, queue );
      magma_smtransfer( dx, &x, Magma_CPU, Magma_DEV, queue );

      magma_svget( x, &m, &n, &partialImplicit, queue );

      magma_smfree( &dx, queue );
      magma_smfree( &db, queue );
      magma_smfree( &dA, queue );

      magma_queue_destroy(queue);
      magma_finalize();
    }
    else{
      partialImplicit[0] = csrValues[0]/partialDivergence[0];
    }

    //copy partial implicit into the final nodeImplicitFunction array
    CudaSafeCall(cudaMemcpy(this->nodeImplicitDevice + this->octree->depthIndex[d], partialImplicit, numNodesAtDepth*sizeof(float), cudaMemcpyHostToDevice));

    delete[] csrValues;
    delete[] csrIndices;
    delete[] numNonZero;
    delete[] partialDivergence;
    delete[] partialImplicit;
    CudaSafeCall(cudaFree(csrValuesDevice));
    CudaSafeCall(cudaFree(csrIndicesDevice));
    CudaSafeCall(cudaFree(numNonZeroDevice));

    cudatimer = clock() - cudatimer;
    printf("Node Implicit computation for depth %d took %f seconds w/%d nodes.\n", this->octree->depth - d,((float) cudatimer)/CLOCKS_PER_SEC, numNodesAtDepth);
    cudatimer = clock();
  }
  CudaSafeCall(cudaFree(this->fLUTDevice));
  CudaSafeCall(cudaFree(this->fPrimePrimeLUTDevice));
  CudaSafeCall(cudaFree(this->divergenceVectorDevice));
  delete[] this->fLUT;
  delete[] this->fPrimePrimeLUT;
  timer = clock() - timer;
  printf("Node Implicit compuation took a total of %f seconds.\n\n",((float) timer)/CLOCKS_PER_SEC);

}
//TODO precondition with cusparseScsric0
void Surface::computeImplicitCuSPSolver(){
  this->computeLUTs();
  this->computeDivergenceVector();

  clock_t timer;
  timer = clock();
  clock_t cudatimer;
  cudatimer = clock();

  unsigned int size = (pow(2, this->octree->depth + 1) - 1);
  int numNodesAtDepth = 0;
  float* temp;
  int* tempInt;
  int* numNonZero;
  int* numNonZeroDevice;
  float* laplacianValuesDevice;
  int* laplacianIndicesDevice;
  float* csrValues;
  int* csrIndices;
  float* csrValuesDevice;
  int* csrIndicesDevice;
  int totalNonZero = 0;
  float* partialDivergence;
  float* partialImplicit;
  dim3 grid;
  dim3 block;
  const float tol = 1e-5f;
  int max_iter = size;
  float a, b, na, r0, r1;
  float dot, m;
  float *d_p, *d_Ax;
  int k;
  float alpha, beta, alpham1;

  CudaSafeCall(cudaMalloc((void**)&this->nodeImplicitDevice, this->octree->totalNodes*sizeof(float)));
  CudaSafeCall(cudaMalloc((void**)&this->fLUTDevice, size*size*sizeof(float)));
  CudaSafeCall(cudaMalloc((void**)&this->fPrimePrimeLUTDevice, size*size*sizeof(float)));
  CudaSafeCall(cudaMemcpy(this->fLUTDevice, this->fLUT, size*size*sizeof(float), cudaMemcpyHostToDevice));
  CudaSafeCall(cudaMemcpy(this->fPrimePrimeLUTDevice, this->fPrimePrimeLUT, size*size*sizeof(float), cudaMemcpyHostToDevice));


  for(int d = this->octree->depth; d >= 0; --d){
    //update divergence coefficients based on solutions at coarser depths
    grid = {1,1,1};
    block = {27,1,1};
    if(d != this->octree->depth){
      numNodesAtDepth = this->octree->depthIndex[d + 1] - this->octree->depthIndex[d];
      if(numNodesAtDepth < 65535) grid.x = (unsigned int) numNodesAtDepth;
      else{
        grid.x = 65535;
        while(grid.x*grid.y < numNodesAtDepth){
          ++grid.y;
        }
        while(grid.x*grid.y > numNodesAtDepth){
          --grid.x;
        }
        if(grid.x*grid.y < numNodesAtDepth){
          ++grid.x;
        }
      }
      for(int dcoarse = d + 1; dcoarse <= this->octree->depth; ++dcoarse){
        updateDivergence<<<grid, block>>>(this->octree->depth, this->octree->finalNodeArrayDevice, numNodesAtDepth,
          this->octree->depthIndex[d], this->divergenceVectorDevice,
          this->fLUTDevice, this->fPrimePrimeLUTDevice, this->nodeImplicitDevice);
        cudaDeviceSynchronize();
        CudaCheckError();
        //if(d == this->octree->depth - 8) exit(0);
      }
    }
    else{

      numNodesAtDepth = 1;
    }

    temp = new float[numNodesAtDepth*27];
    tempInt = new int[numNodesAtDepth*27];
    numNonZero = new int[numNodesAtDepth + 1];
    numNonZero[0] = 0;
    for(int i = 0; i < numNodesAtDepth*27; ++i){
      temp[i] = 0.0f;
      tempInt[i] = -1;
      if(i % 27 == 0){
        numNonZero[(i/27) + 1] = 0;
      }
    }
    CudaSafeCall(cudaMalloc((void**)&numNonZeroDevice, (numNodesAtDepth+1)*sizeof(int)));
    CudaSafeCall(cudaMalloc((void**)&laplacianValuesDevice, numNodesAtDepth*27*sizeof(float)));
    CudaSafeCall(cudaMalloc((void**)&laplacianIndicesDevice, numNodesAtDepth*27*sizeof(int)));
    CudaSafeCall(cudaMemcpy(numNonZeroDevice, numNonZero, (numNodesAtDepth+1)*sizeof(int), cudaMemcpyHostToDevice));
    CudaSafeCall(cudaMemcpy(laplacianValuesDevice, temp, numNodesAtDepth*27*sizeof(float), cudaMemcpyHostToDevice));
    CudaSafeCall(cudaMemcpy(laplacianIndicesDevice, tempInt, numNodesAtDepth*27*sizeof(int), cudaMemcpyHostToDevice));

    computeLdCSR<<<grid, block>>>(this->octree->depth, this->octree->finalNodeArrayDevice, numNodesAtDepth, this->octree->depthIndex[d],
      laplacianValuesDevice, laplacianIndicesDevice, numNonZeroDevice, this->fLUTDevice, this->fPrimePrimeLUTDevice);
    cudaDeviceSynchronize();
    CudaCheckError();

    thrust::device_ptr<int> nN(numNonZeroDevice);
    thrust::inclusive_scan(nN, nN + numNodesAtDepth + 1, nN);
    CudaCheckError();
    CudaSafeCall(cudaMemcpy(numNonZero, numNonZeroDevice, (numNodesAtDepth+1)*sizeof(int), cudaMemcpyDeviceToHost));

    totalNonZero = numNonZero[numNodesAtDepth];

    delete[] tempInt;
    csrValues = new float[totalNonZero];
    csrIndices = new int[totalNonZero];
    for(int i = 0; i < totalNonZero; ++i){
      csrValues[i] = 0.0f;
      csrIndices[i] = 0;
    }

    CudaSafeCall(cudaMalloc((void**)&csrValuesDevice, totalNonZero*sizeof(float)));
    CudaSafeCall(cudaMalloc((void**)&csrIndicesDevice, totalNonZero*sizeof(int)));

    thrust::device_ptr<float> arrayToCompact(laplacianValuesDevice);
    thrust::device_ptr<float> arrayOut(csrValuesDevice);
    thrust::device_ptr<int> placementToCompact(laplacianIndicesDevice);
    thrust::device_ptr<int> placementOut(csrIndicesDevice);

    thrust::copy_if(arrayToCompact, arrayToCompact + (numNodesAtDepth*27), arrayOut, is_not_zero_float());
    CudaCheckError();
    thrust::copy_if(placementToCompact, placementToCompact + (numNodesAtDepth*27), placementOut, is_not_neg_int());
    CudaCheckError();

    CudaSafeCall(cudaFree(laplacianValuesDevice));
    CudaSafeCall(cudaFree(laplacianIndicesDevice));
    CudaSafeCall(cudaMemcpy(csrValues, csrValuesDevice, totalNonZero*sizeof(float),cudaMemcpyDeviceToHost));
    CudaSafeCall(cudaMemcpy(csrIndices, csrIndicesDevice, totalNonZero*sizeof(int),cudaMemcpyDeviceToHost));

    CudaSafeCall(cudaMalloc((void**)&partialImplicit, numNodesAtDepth*sizeof(float)));
    CudaSafeCall(cudaMalloc((void**)&partialDivergence, numNodesAtDepth*sizeof(float)));

    CudaSafeCall(cudaMemcpy(partialDivergence, this->divergenceVectorDevice + this->octree->depthIndex[d], numNodesAtDepth*sizeof(float), cudaMemcpyDeviceToDevice));
    CudaSafeCall(cudaMemcpy(partialImplicit, temp + (numNodesAtDepth - 1), numNodesAtDepth*sizeof(float), cudaMemcpyHostToDevice));
    delete[] temp;

    m = numNodesAtDepth;
    max_iter = numNodesAtDepth;

    //DO SPARSE LINEAR SOLVER WITH A IN CSR FORMAT
    /* Get handle to the CUBLAS context */
    cublasHandle_t cublasHandle = 0;
    cublasStatus_t cublasStatus;
    cublasStatus = cublasCreate(&cublasHandle);

    //TODO check those status
    /* Get handle to the CUSPARSE context */
    cusparseHandle_t cusparseHandle = 0;
    cusparseStatus_t cusparseStatus;
    cusparseStatus = cusparseCreate(&cusparseHandle);

    cusparseMatDescr_t descr = 0;
    cusparseStatus = cusparseCreateMatDescr(&descr);

    cusparseSetMatType(descr,CUSPARSE_MATRIX_TYPE_GENERAL);
    cusparseSetMatIndexBase(descr,CUSPARSE_INDEX_BASE_ZERO);

    CudaSafeCall(cudaMalloc((void **)&d_p, m*sizeof(float)));
    CudaSafeCall(cudaMalloc((void **)&d_Ax, m*sizeof(float)));

    alpha = 1.0;
    alpham1 = -1.0;
    beta = 0.0;
    r0 = 0.;

    cusparseScsrmv(cusparseHandle,CUSPARSE_OPERATION_NON_TRANSPOSE, m, m, totalNonZero,
      &alpha, descr, csrValuesDevice, numNonZeroDevice, csrIndicesDevice, partialImplicit, &beta, d_Ax);

    cublasSaxpy(cublasHandle, m, &alpham1, d_Ax, 1, partialDivergence, 1);
    cublasStatus = cublasSdot(cublasHandle, m, partialDivergence, 1, partialDivergence, 1, &r1);

    k = 1;

    while (r1 > tol*tol && k <= max_iter){
        if (k > 1){
            b = r1 / r0;
            cublasStatus = cublasSscal(cublasHandle, m, &b, d_p, 1);
            cublasStatus = cublasSaxpy(cublasHandle, m, &alpha, partialDivergence, 1, d_p, 1);
        }
        else{
            cublasStatus = cublasScopy(cublasHandle, m, partialDivergence, 1, d_p, 1);
        }

        cusparseScsrmv(cusparseHandle, CUSPARSE_OPERATION_NON_TRANSPOSE, m, m, totalNonZero,
          &alpha, descr, csrValuesDevice, numNonZeroDevice, csrIndicesDevice, d_p, &beta, d_Ax);

        cublasStatus = cublasSdot(cublasHandle, m, d_p, 1, d_Ax, 1, &dot);
        a = r1 / dot;

        cublasStatus = cublasSaxpy(cublasHandle, m, &a, d_p, 1, partialImplicit, 1);
        na = -a;
        cublasStatus = cublasSaxpy(cublasHandle, m, &na, d_Ax, 1, partialDivergence, 1);

        r0 = r1;
        cublasStatus = cublasSdot(cublasHandle, m, partialDivergence, 1, partialDivergence, 1, &r1);
        cudaDeviceSynchronize();
        printf("iteration = %3d, residual = %e\n", k, sqrt(r1));
        k++;
    }
    printf("solver at depth %d was solved in %d iterations with a beta value of %f\n",this->octree->depth - d, k, beta);
    // float rsum, diff, err = 0.0;
    //
    // for (int i = 0; i < N; i++)
    // {
    //     rsum = 0.0;
    //
    //     for (int j = I[i]; j < I[i+1]; j++)
    //     {
    //         rsum += val[j]*x[J[j]];
    //     }
    //
    //     diff = fabs(rsum - rhs[i]);
    //
    //     if (diff > err)
    //     {
    //         err = diff;
    //     }
    // }
    // printf("Test Summary:  Error amount = %f\n", err);
    // exit((k <= max_iter) ? 0 : 1);

    cusparseDestroy(cusparseHandle);
    cublasDestroy(cublasHandle);

    //copy partial implicit into the final nodeImplicitFunction array
    CudaSafeCall(cudaMemcpy(this->nodeImplicitDevice + this->octree->depthIndex[d], partialImplicit, numNodesAtDepth*sizeof(float), cudaMemcpyHostToDevice));

    delete[] csrValues;
    delete[] csrIndices;
    delete[] numNonZero;
    cudaFree(d_p);
    cudaFree(d_Ax);
    CudaSafeCall(cudaFree(csrValuesDevice));
    CudaSafeCall(cudaFree(csrIndicesDevice));
    CudaSafeCall(cudaFree(numNonZeroDevice));
    CudaSafeCall(cudaFree(partialImplicit));
    CudaSafeCall(cudaFree(partialDivergence));

    cudatimer = clock() - cudatimer;
    printf("Node Implicit computation for depth %d took %f seconds w/%d nodes.\n", d,((float) cudatimer)/CLOCKS_PER_SEC, numNodesAtDepth);
    cudatimer = clock();
  }
  CudaSafeCall(cudaFree(this->fLUTDevice));
  CudaSafeCall(cudaFree(this->fPrimePrimeLUTDevice));
  CudaSafeCall(cudaFree(this->divergenceVectorDevice));
  delete[] this->fLUT;
  delete[] this->fPrimePrimeLUT;
  timer = clock() - timer;
  printf("Node Implicit computation took a total of %f seconds.\n\n",((float) timer)/CLOCKS_PER_SEC);

}
void Surface::computeImplicitEasy(){
  clock_t timer;
  timer = clock();

  float* easyVertexImplicit = new float[this->octree->totalNodes];
  int currentNeighbor = 0;
  float currentImp = 0.0f;
  int currentDepth = 0;
  int numFinestVertices = this->octree->vertexIndex[1];
  if(!this->octree->vertexArrayDeviceReady) this->octree->copyVerticesToDevice();
  if(!this->octree->normalsDeviceReady) this->octree->copyNormalsToDevice();
  if(!this->octree->pointsDeviceReady) this->octree->copyPointsToDevice();
  CudaSafeCall(cudaMalloc((void**)&this->vertexImplicitDevice, numFinestVertices*sizeof(float)));

  dim3 grid = {1,1,1};
  dim3 block = {8,1,1};
  if(numFinestVertices < 65535) grid.x = (unsigned int) numFinestVertices;
  else{
    grid.x = 65535;
    while(grid.x*grid.y < numFinestVertices){
      ++grid.y;
    }
    while(grid.x*grid.y > numFinestVertices){
      --grid.x;
    }
    if(grid.x*grid.y < numFinestVertices){
      ++grid.x;
    }
  }
  vertexImplicitFromNormals<<<grid,block>>>(numFinestVertices, this->octree->vertexArrayDevice, this->octree->finalNodeArrayDevice, this->octree->normalsDevice, this->octree->pointsDevice, this->vertexImplicitDevice);
  cudaDeviceSynchronize();//may not be necessary
  CudaCheckError();
  CudaSafeCall(cudaFree(this->octree->pointsDevice));
  CudaSafeCall(cudaFree(this->octree->normalsDevice));
  CudaSafeCall(cudaFree(this->octree->vertexArrayDevice));
  this->octree->pointsDeviceReady = false;
  this->octree->normalsDeviceReady = false;
  this->octree->vertexArrayDeviceReady = false;
  timer = clock() - timer;
  printf("Computing Vertex Implicit Values with normals took a total of %f seconds.\n\n",((float) timer)/CLOCKS_PER_SEC);
}

void Surface::computeVertexImplicit(){
  clock_t timer;
  timer = clock();

  /*Vertices*/

  dim3 grid = {1,1,1};
  dim3 block = {1,1,1};
  if(this->octree->numPoints < 65535) grid.x = (unsigned int) this->octree->numPoints;
  else{
    grid.x = 65535;
    while(grid.x*block.x < this->octree->numPoints){
      ++block.x;
    }
    while(grid.x*block.x > this->octree->numPoints){
      --grid.x;
    }
    if(grid.x*block.x < this->octree->numPoints){
      ++grid.x;
    }
  }
  float* sumImplicitDevice;
  CudaSafeCall(cudaMalloc((void**)&sumImplicitDevice, sizeof(float)));
  pointSumImplicitTraversal<<<grid,block>>>(this->octree->numPoints, this->octree->pointsDevice, this->octree->finalNodeArrayDevice, this->octree->depthIndex[this->octree->depth], this->nodeImplicitDevice, sumImplicitDevice);
  cudaDeviceSynchronize();//may not be necessary
  CudaCheckError();
  CudaSafeCall(cudaFree(this->octree->pointsDevice));
  this->octree->pointsDeviceReady = false;
  if(!this->octree->vertexArrayDeviceReady) this->octree->copyVerticesToDevice();
  int numFinestVertices = this->octree->vertexIndex[1];
  CudaSafeCall(cudaMalloc((void**)&this->vertexImplicitDevice, numFinestVertices*sizeof(float)));
  block = {8,1,1};
  if(numFinestVertices < 65535) grid.x = (unsigned int) numFinestVertices;
  else{
    grid.x = 65535;
    while(grid.x*grid.y < numFinestVertices){
      ++grid.y;
    }
    while(grid.x*grid.y > numFinestVertices){
      --grid.x;
    }
    if(grid.x*grid.y < numFinestVertices){
      ++grid.x;
    }
  }
  vertexSumImplicitTraversal<<<grid,block>>>(numFinestVertices, this->octree->vertexArrayDevice, this->nodeImplicitDevice, this->vertexImplicitDevice, sumImplicitDevice, this->octree->numPoints);
  CudaCheckError();
  CudaSafeCall(cudaFree(sumImplicitDevice));
  CudaSafeCall(cudaFree(this->nodeImplicitDevice));
  timer = clock() - timer;
  printf("Computing Vertex Implicit Values took a total of %f seconds.\n\n",((float) timer)/CLOCKS_PER_SEC);

}

void Surface::marchingCubes(){
  clock_t timer;
  timer = clock();

  if(!this->octree->edgeArrayDeviceReady) this->octree->copyEdgesToDevice();
  int numFinestEdges = this->octree->edgeIndex[1];
  int* vertexNumbersDevice;
  CudaSafeCall(cudaMalloc((void**)&vertexNumbersDevice, numFinestEdges*sizeof(int)));
  dim3 gridEdge = {1,1,1};
  dim3 blockEdge = {1,1,1};
  if(numFinestEdges < 65535) gridEdge.x = (unsigned int) numFinestEdges;
  else{
    gridEdge.x = 65535;
    while(gridEdge.x*blockEdge.x < numFinestEdges){
      ++blockEdge.x;
    }
    while(gridEdge.x*blockEdge.x > numFinestEdges){
      --gridEdge.x;
    }
    if(gridEdge.x*blockEdge.x < numFinestEdges){
      ++gridEdge.x;
    }
  }
  calcVertexNumbers<<<gridEdge,blockEdge>>>(numFinestEdges, this->octree->edgeArrayDevice, this->vertexImplicitDevice, vertexNumbersDevice);
  cudaDeviceSynchronize();
  CudaCheckError();
  CudaSafeCall(cudaFree(this->vertexImplicitDevice));
  int* vertexAddressesDevice;
  CudaSafeCall(cudaMalloc((void**)&vertexAddressesDevice, numFinestEdges*sizeof(int)));
  thrust::device_ptr<int> vN(vertexNumbersDevice);
  thrust::device_ptr<int> vA(vertexAddressesDevice);
  thrust::inclusive_scan(vN, vN + numFinestEdges, vA);

  /*Triangles*/
  //surround vertices with values less than 0

  int numFinestNodes = this->octree->depthIndex[1];
  int* triangleNumbersDevice;
  int* cubeCategoryDevice;
  CudaSafeCall(cudaMalloc((void**)&triangleNumbersDevice, numFinestNodes*sizeof(int)));
  CudaSafeCall(cudaMalloc((void**)&cubeCategoryDevice, numFinestNodes*sizeof(int)));

  dim3 grid = {1,1,1};
  dim3 block = {1,1,1};
  if(numFinestNodes < 65535) grid.x = (unsigned int) numFinestNodes;
  else{
    grid.x = 65535;
    while(grid.x*block.x < numFinestNodes){
      ++block.x;
    }
    while(grid.x*block.x > numFinestNodes){
      --grid.x;
    }
    if(grid.x*block.x < numFinestNodes){
      ++grid.x;
    }
  }
  determineCubeCategories<<<grid,block>>>(numFinestNodes, this->octree->finalNodeArrayDevice, vertexNumbersDevice, cubeCategoryDevice, triangleNumbersDevice);
  cudaDeviceSynchronize();
  CudaCheckError();

  int* triangleAddressesDevice;
  CudaSafeCall(cudaMalloc((void**)&triangleAddressesDevice, numFinestNodes*sizeof(int)));
  thrust::device_ptr<int> tN(triangleNumbersDevice);
  thrust::device_ptr<int> tA(triangleAddressesDevice);
  thrust::inclusive_scan(tN, tN + numFinestNodes, tA);

  CudaSafeCall(cudaMemcpy(&this->numSurfaceVertices, vertexAddressesDevice + (numFinestEdges - 1), sizeof(int), cudaMemcpyDeviceToHost));
  CudaSafeCall(cudaMemcpy(&this->numSurfaceTriangles, triangleAddressesDevice + (numFinestNodes - 1), sizeof(int), cudaMemcpyDeviceToHost));
  printf("%d vertices and %d triangles\n",this->numSurfaceVertices, this->numSurfaceTriangles);


  float3* surfaceVerticesDevice;
  CudaSafeCall(cudaMalloc((void**)&surfaceVerticesDevice, this->numSurfaceVertices*sizeof(float3)));


  if(!this->octree->vertexArrayDeviceReady) this->octree->copyVerticesToDevice();

  /* generate vertices */
  generateSurfaceVertices<<<gridEdge,blockEdge>>>(numFinestEdges, this->octree->edgeArrayDevice, this->octree->vertexArrayDevice, vertexNumbersDevice, vertexAddressesDevice, surfaceVerticesDevice);
  CudaCheckError();
  this->surfaceVertices = new float3[this->numSurfaceVertices];
  CudaSafeCall(cudaMemcpy(this->surfaceVertices, surfaceVerticesDevice, this->numSurfaceVertices*sizeof(float3),cudaMemcpyDeviceToHost));
  CudaSafeCall(cudaFree(surfaceVerticesDevice));
  CudaSafeCall(cudaFree(vertexNumbersDevice));
  CudaSafeCall(cudaFree(this->octree->edgeArrayDevice));
  this->octree->edgeArrayDeviceReady = false;
  CudaSafeCall(cudaFree(this->octree->vertexArrayDevice));
  this->octree->vertexArrayDeviceReady = false;


  int3* surfaceTrianglesDevice;

  CudaSafeCall(cudaMalloc((void**)&surfaceTrianglesDevice, this->numSurfaceTriangles*sizeof(int3)));

  /* generate triangles */
  //grid is already numFinestNodes
  if(numFinestNodes < 65535) grid.x = (unsigned int) numFinestNodes;
  else{
    grid.x = 65535;
    while(grid.x*grid.y < numFinestNodes){
      ++grid.y;
    }
    while(grid.x*grid.y > numFinestNodes){
      --grid.x;
    }
    if(grid.x*grid.y < numFinestNodes){
      ++grid.x;
    }
  }
  block = {5,1,1};
  generateSurfaceTriangles<<<grid,block>>>(numFinestNodes, this->octree->finalNodeArrayDevice, vertexAddressesDevice, triangleNumbersDevice, triangleAddressesDevice, cubeCategoryDevice, surfaceTrianglesDevice);
  CudaCheckError();

  this->surfaceTriangles = new int3[this->numSurfaceTriangles];
  CudaSafeCall(cudaMemcpy(this->surfaceTriangles, surfaceTrianglesDevice, this->numSurfaceTriangles*sizeof(int3),cudaMemcpyDeviceToHost));
  CudaSafeCall(cudaFree(surfaceTrianglesDevice));
  CudaSafeCall(cudaFree(vertexAddressesDevice));
  CudaSafeCall(cudaFree(triangleNumbersDevice));
  CudaSafeCall(cudaFree(triangleAddressesDevice));
  CudaSafeCall(cudaFree(cubeCategoryDevice));
  timer = clock() - timer;
  printf("Marching cubes took a total of %f seconds.\n\n",((float) timer)/CLOCKS_PER_SEC);

}

void Surface::generateMesh(){
  std::string newFile = "out" + this->octree->pathToFile.substr(4, this->octree->pathToFile.length() - 4) + "_mesh.ply";
  std::ofstream plystream(newFile);
  if (plystream.is_open()) {
    std::ostringstream stringBuffer = std::ostringstream("");
    stringBuffer << "ply\nformat ascii 1.0\ncomment object: SSRL test\n";
    stringBuffer << "element vertex ";
    stringBuffer << this->numSurfaceVertices;
    stringBuffer << "\nproperty float x\nproperty float y\nproperty float z\n";
    stringBuffer << "element face ";
    stringBuffer << this->numSurfaceTriangles;
    stringBuffer << "\nproperty list uchar int vertex_index\n";
    stringBuffer << "end_header\n";
    plystream << stringBuffer.str();
    for(int i = 0; i < this->numSurfaceVertices; ++i){
      stringBuffer = std::ostringstream("");
      stringBuffer << this->surfaceVertices[i].x;
      stringBuffer << " ";
      stringBuffer << this->surfaceVertices[i].y;
      stringBuffer << " ";
      stringBuffer << this->surfaceVertices[i].z;
      stringBuffer << "\n";
      plystream << stringBuffer.str();
    }
    for(int i = 0; i < this->numSurfaceTriangles; ++i){
      stringBuffer = std::ostringstream("");
      stringBuffer << "3 ";
      stringBuffer << this->surfaceTriangles[i].x;
      stringBuffer << " ";
      stringBuffer << this->surfaceTriangles[i].y;
      stringBuffer << " ";
      stringBuffer << this->surfaceTriangles[i].z;
      stringBuffer << "\n";
      plystream << stringBuffer.str();
    }
    std::cout<<newFile + " has been created.\n"<<std::endl;
  }
  else{
    std::cout << "Unable to open: " + newFile<< std::endl;
    exit(1);
  }
}
