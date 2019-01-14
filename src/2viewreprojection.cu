//#======================================#//
//# UGA SSRL Reprojection                #//
//# Author: Caleb Adams                  #//
//# Contact: CalebAshmoreAdams@gmail.com #//
//#======================================#//
// A seriously good source:
// https://developer.nvidia.com/sites/default/files/akamai/cuda/files/Misc/mygpu.pdf
//
// This program is only meant to perform a
// small portion of MOCI's science pipeline
//

#include <array>
#include <cmath>
#include <fstream>
#include <iostream>
#include <limits>
#include <sstream>
#include <string>
#include <new>
#include <vector>
// my boiz @ nvidia
#include <cuda.h>
#include <cuda_runtime.h>
//#include "cublas.h"
#include "cublas_v2.h"
//#include "cusolver.h"
// to remove eventually
#include <time.h>

// alsp remove eventually
#define N 8192
#define BILLION  1000000000L;

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
    err = cudaDeviceSynchronize();
    if (cudaSuccess != err) {
        fprintf(stderr, "cudaCheckError() with sync failed at %s:%i : %s\n",
                file, line, cudaGetErrorString(err));
        exit(-1);
    }
#endif

    return;
}


using namespace std;

//Just to print all properties on TX2 pertinent to cuda development
// TODO move this to a different file
void printDeviceProperties() {
  cout<<"\n---------------START OF DEVICE PROPERTIES---------------\n"<<endl;

  int nDevices;
  cudaGetDeviceCount(&nDevices);
  for (int i = 0; i < nDevices; i++) {
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
    if(prop.kernelExecTimeoutEnabled){
      printf("YES\n");
    }
    else{
      printf("NO\n");
    }
    printf(" -Device is ");
    if(prop.integrated){
      printf("integrated. (motherboard)\n");
    }
    else{
      printf("discrete. (card)\n\n");
    }
    if(prop.isMultiGpuBoard){
      printf(" -Device is on a MultiGPU configurations.\n\n");
    }
    switch(prop.computeMode){
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
      if(prop.canMapHostMemory){
        printf("\n -The device can map host memory into the CUDA address space for use with\n cudaHostAlloc() or cudaHostGetDevicePointer().\n\n");
      }
      else{
        printf("\n -The device CANNOT map host memory into the CUDA address space.\n\n");
      }
      printf(" -ECC support: ");
      if(prop.ECCEnabled){
        printf(" ON\n");
      }
      else{
        printf(" OFF\n");
      }
      printf(" -PCI Bus ID: %d\n", prop.pciBusID);
      printf(" -PCI Domain ID: %d\n", prop.pciDomainID);
      printf(" -PCI Device (slot) ID: %d\n", prop.pciDeviceID);
      printf(" -Using a TCC Driver: ");
      if(prop.tccDriver){
        printf("YES\n");
      }
      else{
        printf("NO\n");
      }
    }
    cout<<"\n----------------END OF DEVICE PROPERTIES----------------\n"<<endl;
}


// == GLOBAL VARIABLES == //
bool   verbose = 1;
bool   debug   = 1;
bool   simple  = 0;
int    gpu_acc = 1; // is GPU accellerated?

bool line_intersection = 1;

__constant__ bool   d_debug             = 1;
__constant__ bool   d_verbose           = 0;

// only one of these should be active at a time
__constant__ bool   d_line_intersection = 1;
__constant__ bool   d_least_squares     = 0;

string cameras_path;
string matches_path;
int    to_pan;
unsigned short match_count;
unsigned short camera_count;

// TODO (some of) this stuff should be set by camera calibration
// TODO have this stuff sent in with camera parameter files
// This was for the test cases only
__constant__ int   d_res      = 3024;//1024;
__constant__ float d_foc      = 0.00399;//0.035;
__constant__ float d_fov      = 1.1089822; // 64.54 degrees 0.8575553107;//0.0593412; //3.4 degrees to match the blender sim //0.8575553107; // 49.1343 degrees  // 0.785398163397; // 45 degrees
__constant__ float d_PI       = 3.1415926535;
__constant__ float d_dpix     = 0.00000163426;//0.00003124996;//0.00000103877;// 0.00003124996;///(d_foc*tan(d_fov/2))/(d_res/2);
__constant__ float d_stepsize = 0.1; // the step size of the iterative solution

unsigned int   res  = 1024;
float          foc  = 0.035;
float          fov  = 0.8575553107;//0.0593412; //3.4 degrees to match the blender sim //0.8575553107; // 49.1343 degrees  // 0.785398163397; // 45 degrees
float          PI   = 3.1415926535;
float          dpix = (foc*tan(fov/2))/(res/2); //float          dpix = 0.00002831538; //(foc*tan(fov/2))/(res/2)

// for debugging
float          max_angle = -1000.0;
float          min_angle =  1000.0;

// for the CPU
vector< vector<string> > matches;
vector< vector<string> > cameras;
vector< vector<string> > projections;
vector< vector<float> >  points;
vector< vector<float> >  matchesr3;
vector< vector<int> >    colors;

// for the GPU, Host -> Device memory things
int MATCH_POINTS_SIZE;
int MATCH_COLORS_SIZE;
int CAMERA_DATA_SIZE;

float         *match_points;
int           *match_colors;
float         *camera_data;
float         *point_cloud;

// ============================================ //
// =========== All Device Functions =========== //
// ============================================ //

// dot product of 2 vectors!
__device__ float dot_product(float *A, float *B, int size){
  float product = 0;
  for (int i = 0; i < size; i++){
    product += A[i] * B[i];
  }
  return product;
}

// returns the angle between the input vector and the x unit vector
__device__ float get_vector_x_angle(float cam[6]){
  float w[3] = {1.0,0.0,0.0};
  float v[3] = {cam[3],cam[4],cam[5]};
  // find the dot product
  float dot_v_w = dot_product(v,w,3);
  // calculate magnitude for v
  float v_mag = dot_product(v,v,3);
  // make the fraction:
  float fract = (dot_v_w)/(sqrtf(v_mag));
  // find the angle
  float angle = acosf(fract);
  // check to see if outside of second quad
  //if (cam[4] < 0.0) angle = 2 * d_PI - angle;
  return angle;
}

/////////////////////////////////////////////////////////////////
//returns the angle between the input vector and the z unit vector
__device__ float get_vector_z_angle(float cam[6]){
  float w[3] = {0.0,0.0,1.0};
  float v[3] = {cam[3],cam[4],cam[5]};
  // dot product
  float dot_v_w = dot_product(v,w,3);
  float v_mag = dot_product(v,v,3);
  float fract = (dot_v_w)/(sqrtf(v_mag));
  float angle = acosf(fract);
  return angle;
}

// rotates a 3x1 vector in the x plane
//__device__ float* rotate_projection_x(float *v, float angle){
__device__ void rotate_projection_x(float *v, float angle){
  float x_n = v[0];
  float y_n = cosf(angle)*v[1] + -1*sinf(angle)*v[2];
  float z_n = sinf(angle)*v[1] + cosf(angle)*v[2];
  // float w[3] = {x_n,y_n,z_n};
  // return w;
  v[0] = x_n;
  v[1] = y_n;
  v[2] = z_n;
}

// rotates a 3x1 vector in the z plane
//__device__ float* rotate_projection_z(float *v, float angle){
__device__ void rotate_projection_z(float *v, float angle){
  float x_n = cosf(angle)*v[0] + -1*sinf(angle)*v[1];
  float y_n = sinf(angle)*v[0] + cosf(angle)*v[1];
  float z_n = v[2];
  v[0] = x_n;
  v[1] = y_n;
  v[2] = z_n;
}

// returns the cross product of 3x1 vectors
__device__ void cross_product(float *a, float *b, float *n){
  n[0] = a[1]*b[2] - a[2]*b[1];
  n[1] = a[2]*b[0] - a[0]*b[2];
  n[2] = a[0]*b[1] - a[1]*b[0];
}

// subtract 2 3x1 vectors
__device__ void sub(float *a, float *b, float *n){
  n[0] = a[0] - b[0];
  n[1] = a[1] - b[1];
  n[2] = a[2] - b[2];
}

// subtract 2 3x1 vectors
__device__ void add(float *a, float *b, float *n){
  n[0] = a[0] + b[0];
  n[1] = a[1] + b[1];
  n[2] = a[2] + b[2];
}

// multiplies a by 3x1 vector b
__device__ void mul(float a, float *b, float *n){
  n[0] = a * b[0];
  n[1] = a * b[1];
  n[2] = a * b[2];
}

__device__ void zero3(float*n){
  n[0] = 0.0;
  n[1] = 0.0;
  n[2] = 0.0;
}

__device__ float squared(float x){
  return x*x;
}

__device__ float euclid(float p1[3], float p2[3]){
  return sqrtf(((p2[0] - p1[0])*(p2[0] - p1[0])) + ((p2[1] - p1[1])*(p2[1] - p1[1])) + ((p2[2] - p1[2])*(p2[2] - p1[2])));
  //return sqrtf((squared));
}

// normalize a vector
__device__ void normalize(float *v){
  float len = sqrtf(v[0]*v[0] + v[1]*v[1] + v[2]*v[2]);
  v[0] /= len;
  v[1] /= len;
  v[2] /= len;
}

//
// CUDA kernel for performing a two view reprojection
//
__global__ void two_view_reproject(float *r2points, float *r3cameras,float *point_cloud){

  // get index
  int i_m = blockIdx.x * blockDim.x + threadIdx.x;
  int i_p = 3*(i_m);
  i_m *= 4;

  // grab the camera data, not nessesary but this makes it conseptually easier
  float camera0[6] = {r3cameras[0],r3cameras[1],r3cameras[2],r3cameras[3],r3cameras[4],r3cameras[5]};
  float camera1[6] = {r3cameras[6],r3cameras[7],r3cameras[8],r3cameras[9],r3cameras[10],r3cameras[11]};

  // scale my dudes
  float x0 = d_dpix * ((     r2points[i_m]  ) - d_res/2.0);
  float y0 = d_dpix * ((-1.0*r2points[i_m+1]) + d_res/2.0);
  float x1 = d_dpix * ((     r2points[i_m+2]) - d_res/2.0);
  float y1 = d_dpix * ((-1.0*r2points[i_m+3]) + d_res/2.0);

  float kp0[3] = {x0,y0,0.0};
  float kp1[3] = {x1,y1,0.0};

  // get the camera angles
  float r0x = get_vector_x_angle(camera0);
  float r1x = get_vector_x_angle(camera1);
  float r0z = get_vector_z_angle(camera0);
  float r1z = get_vector_z_angle(camera1);

  rotate_projection_x(kp0,d_PI/2.0); // was d_PI/2.0
  rotate_projection_x(kp1,d_PI/2.0);

  rotate_projection_z(kp0,r0x + d_PI/2.0); // + d_PI/2.0
  rotate_projection_z(kp1,r1x + d_PI/2.0); // + d_PI/2.0

  // adjust the kp's location in a plane
  kp0[0] = camera0[0] - (kp0[0] + (camera0[3] * d_foc));
  kp0[1] = camera0[1] - (kp0[1] + (camera0[4] * d_foc));

  kp1[0] = camera1[0] - (kp1[0] + (camera1[3] * d_foc));
  kp1[1] = camera1[1] - (kp1[1] + (camera1[4] * d_foc));

  float points0[6] = {camera0[0],camera0[1],camera0[2],kp0[0],kp0[1],kp0[2]};
  float points1[6] = {camera1[0],camera1[1],camera1[2],kp1[0],kp1[1],kp1[2]};

  // calculate the vectors
  float v0[3]    = {points0[3] - points0[0],points0[4] - points0[1],points0[5] - points0[2]};
  float v1[3]    = {points1[3] - points1[0],points1[4] - points1[1],points1[5] - points1[2]};

  // TODO here is where a better method could be used to find the closest
  // point of intescetion between the two lines
  float smallest = 800000000.0; // this needs to be really big
  float t_holder = 0;
  float p0[3]; //= {0.0,0.0,0.0};
  float p1[3]; //= {0.0,0.0,0.0};
  float point[4];
  // TODO add a least squares version
  if (d_least_squares){
    //=====================//
    //=                   =//
    //=    Here be a      =//
    //=   least squares   =//
    //=                   =//
    //=====================//

  } else if (d_line_intersection){
    //=====================//
    //=                   =//
    //=    Here be an     =//
    //=   algibra dudes   =//
    //=                   =//
    //=====================//
    // https://en.wikipedia.org/wiki/Skew_lines#Nearest_Points
    float temp[3];
    float solution0[3];
    float solution1[3];
    //normalize(v0);
    //normalize(v1);
    zero3(temp);
    zero3(solution0);
    zero3(solution1);
    // starting points
    p0[0] = points0[3];
    p0[1] = points0[4];
    p0[2] = points0[5];
    p1[0] = points1[3];
    p1[1] = points1[4];
    p1[2] = points1[5];
    // the crossy dudes
    float n0[3];

    float n1[3];
    float n2[3];

    zero3(n0);
    zero3(n1);
    zero3(n2);

    cross_product(v0,v1,n0);
    cross_product(v1,n0,n1);
    // calculate the points
    // calculate the numorator
    sub(p1,p0,temp);
    float numor0 = dot_product(temp,n1,3);
    float denom0 = dot_product(v0,n1,3);
    float frac0  = numor0/denom0;
    // // clear temp
    zero3(temp);
    // // calculate solution
    mul(frac0,v0,temp);
    add(p0,temp,solution0);

    // // do that again!

    zero3(temp);
    cross_product(v1,v0,temp);
    cross_product(v0,temp,n2);
    //
    zero3(temp);
    sub(p0,p1,temp);
    float numor1 = dot_product(temp,n2,3);
    float denom1 = dot_product(v1,n2,3);
    float frac1  = numor1/denom1;
    // // clear temp
    zero3(temp);
    // // calculate solution
    mul(frac1,v1,temp);
    add(p1,temp,solution1);
    //
    // // we found the solutions! now, find their midpoint
    point[0] = (solution0[0]+solution1[0])/2.0;
    point[1] = (solution0[1]+solution1[1])/2.0;
    point[2] = (solution0[2]+solution1[2])/2.0;

    // supposed to be perp
    float d[3];
    d[0] = solution0[0] - solution1[0];
    d[1] = solution0[1] - solution1[1];
    d[2] = solution0[2] - solution1[2];

    float dot0 = dot_product(v0,d,3);
    float dot1 = dot_product(v1,d,3);
    if (d_debug){
      float threshold = 0.000001;
      if (!(dot0 <= threshold)) printf("ASS0 %f",dot0);
      if (!(dot1 <= threshold)) printf("ASS1 %f",dot1);
    }

  } else {
    //=====================//
    //=                   =//
    //=    Here be a      =//
    //=   brute force     =//
    //=                   =//
    //=====================//
    for (float t = 0.0f; t < 8000.0f; t += d_stepsize){
      t_holder = t;

      p0[0] = points0[3] + (v0[0]*t);
      p0[1] = points0[4] + (v0[1]*t);
      p0[2] = points0[5] + (v0[2]*t);

      p1[0] = points1[3] + (v1[0]*t);
      p1[1] = points1[4] + (v1[1]*t);
      p1[2] = points1[5] + (v1[2]*t);

      //float dist = euclid(p0,p1);
      float dist = norm3df((p0[0]-p1[0]),(p0[1]-p1[1]),(p0[2]-p0[2]));
      if (dist <= smallest){
        smallest = dist;
        point[0] = (p0[0]+p1[0])/2.0;
        point[1] = (p0[1]+p1[1])/2.0;
        point[2] = (p0[2]+p1[2])/2.0;
        point[3] = smallest;
      } else break;
    }
  }
  // this is the standard output for the result
  point_cloud[i_p]   = point[0];
  point_cloud[i_p+1] = point[1];
  point_cloud[i_p+2] = point[2];

  if (d_debug){
    printf("smallest: %f, t: [%f]\nv0: [%f,%f,%f]\nv1: [%f,%f,%f]\np0: [%f,%f,%f]\np1: [%f,%f,%f]\n",point[3],t_holder,v0[0],v0[1],v0[2],v1[0],v1[1],p1[2],p0[0],p0[1],p0[2],p1[0],p1[1],p1[2]);
  } // for debugging

}

// ============================================ //
// ============ All Host Functions ============ //
// ============================================ //

//
// parses comma delemeted string
//
void parse_comma_delem(string str, short flag, int m_p, int m_c, int c){
  istringstream ss(str);
  string token;
  vector<string> v;
  while(getline(ss, token, ',')) {
    if (debug) cout << token << endl;
    v.push_back(token);
  }
  switch (flag)
  {
    case 1: // matches
      if (gpu_acc){ // we must do mem differently for the GPU accellerated guy
        // based on the data types of our file format
        // do the match locations
        if (debug) cout << "ATTEMPTING TO RELLOCATE 4 ELEMENTS IN VECTOR LENGTH: " << v.size() << endl;
        match_points[m_p]   = stof(v[2]);
        match_points[m_p+1] = stof(v[3]);
        match_points[m_p+2] = stof(v[4]);
        match_points[m_p+3] = stof(v[5]);
        // do the colors, alright chars i think
        // TODO make colors work, need to implement a method from 3 bytes -> 1 byte of chars
        match_colors[m_c]   = stoi(v[6]);
        match_colors[m_c+1] = stoi(v[7]);
        match_colors[m_c+2] = stoi(v[8]);
      } else {
        matches.push_back(v);
      }
      break;
    case 2: // cameras
      if (gpu_acc){
        camera_data[c]   = stof(v[1]);
        camera_data[c+1] = stof(v[2]);
        camera_data[c+2] = stof(v[3]);
        camera_data[c+3] = stof(v[4]);
        camera_data[c+4] = stof(v[5]);
        camera_data[c+5] = stof(v[6]);
      } else {
        cameras.push_back(v);
      }
      break;
    default:
      break;
  }
}

//
// loads matches from a match.txt file
//
void load_matches(){
  ifstream infile(matches_path);
  string line;
  short c = 0;
  bool first = 1;
  int index_p = 0;
  int index_c  = 0;
  while (getline(infile, line)){
      istringstream iss(line);
      if (debug) cout << line << endl;
      if (first){
        first = 0;
        if (gpu_acc){
          // we need to allocate the memory for the number of GPU matches
          // the 4 floating point locations of matches, 4 * (float) * size
          if (debug) cout << "DYNAMICALLY ALLOCATING ON HOST... " << endl << "read: " << stoi(line) << ", genrating: " << (4*stoi(line)) << endl;
          MATCH_POINTS_SIZE = 4*(stoi(line)+1);
          match_points      = new (nothrow) float[MATCH_POINTS_SIZE];
          // now for the clors, 3 * (unsigned char) * size
          // TODO potentially average the colors instead of just picking the first image?
          if (debug) cout << "DYNAMICALLY ALLOCATING ON HOST... " << endl << "read: " << stoi(line) << ", genrating: " << (3*stoi(line)) << endl;
          MATCH_COLORS_SIZE = 3*(stoi(line)+1);
          match_colors      = new (nothrow) int[MATCH_COLORS_SIZE];
        }
      } else{
        parse_comma_delem(line, 1, index_p, index_c, -1);
        index_p += 4;
        index_c += 3;
        c++;
      }
  }
  match_count = c-1;
  if (verbose or debug) cout << "Loaded: " << match_count << " matches" << endl;
}

//
// loads cameras from a camera.txt file
//
void load_cameras(){
  ifstream infile(cameras_path);
  string line;
  unsigned short c = 0;
  unsigned int index = 0;
  // TODO this needs to be generalized past just a 2-view allocation.
  // this would be similar to how the matches dynamically allocates
  // 6 (float) camera properties per camera
  CAMERA_DATA_SIZE = 2 * 6;
  camera_data      = new (nothrow) float[CAMERA_DATA_SIZE];
  while (getline(infile, line)){
      istringstream iss(line);
      if (debug) cout << line << endl;
      parse_comma_delem(line, 2, -1, -1, index);
      index+=6;
      c++;
  }
  camera_count = c;
  if (verbose or debug) cout << "Loaded: " << camera_count << " cameras" << endl;
}

//
// This is used for linear approximation
// TODO develop a better way to do linear approximation
//
float euclid_dist(float p1[3], float p2[3]){
  return sqrt(((p2[0] - p1[0])*(p2[0] - p1[0])) + ((p2[1] - p1[1])*(p2[1] - p1[1])) + ((p2[2] - p1[2])*(p2[2] - p1[2])));
}

//
// This is used to rotate the projection in the z axis
//
vector<float> rotate_projection_x(float x, float y, float z, float r){
  vector<float> v;
  // count clockwize rotatoin around x axis
  float x_n = x;
  float y_n = cos(r)*y + -1*sin(r)*z;
  float z_n = sin(r)*y + cos(r)*z;
  v.push_back(x_n);
  v.push_back(y_n);
  v.push_back(z_n);
  return v;
}

//
// TODO make this
// This is used to rotate the projection in the Y axis
//
vector<float> rotate_projection_y(float x, float y, float z, float r){
  vector<float> v;
  // count clockwize rotatoin around x axis
  // float x_n = cos(r)*x + -1*sin(r)*y;
  // float y_n = sin(r)*x + cos(r)*y;
  // float z_n = z;
  // v.push_back(x_n);
  // v.push_back(y_n);
  // v.push_back(z_n);
  return v;
}

//
// This is used to rotate the projection in the z axis
//
vector<float> rotate_projection_z(float x, float y, float z, float r){
  vector<float> v;
  // count clockwize rotatoin around z axis
  float x_n = cos(r)*x + -1*sin(r)*y;
  float y_n = sin(r)*x + cos(r)*y;
  float z_n = z;
  v.push_back(x_n);
  v.push_back(y_n);
  v.push_back(z_n);
  return v;
}

//
// This is to perform a dot product w the GPU
// This uses CUDA with CUBLAS
//
int dot_product(float *x, float *y, int length, float &val){
  cublasStatus_t stat; // CUBLAS functions status
  cublasHandle_t handle; // CUBLAS context

  //int j;

  float* d_x;  // d_x - x on the  device
  float* d_y;  // d_y - y on the device

  CudaSafeCall(cudaMalloc ((void **)&d_x ,length*sizeof (*x)));     // device memory  alloc  for x
  CudaSafeCall(cudaMalloc ((void **)&d_y ,length*sizeof (*y)));     // device memory  alloc  for y

  stat = cublasCreate (& handle );   //  initialize  CUBLAS  context
  CudaCheckError();

  stat = cublasSetVector(length,sizeof (*x),x,1,d_x ,1);// cp x->d_x
  CudaCheckError();

  stat = cublasSetVector(length,sizeof (*y),y,1,d_y ,1);// cp y->d_y
  CudaCheckError();

  float  result;

  // dot  product  of two  vectors d_x ,d_y:
  // d_x [0]* d_y [0]+...+ d_x[n-1]* d_y[n-1]

  stat = cublasSdot(handle,length,d_x,1,d_y,1,&result);
  CudaCheckError();
  //cout << "dot result: " << result << endl;

  val = result;

  CudaSafeCall(cudaFree(d_x));                             // free  device  memory
  CudaSafeCall(cudaFree(d_y));                             // free  device  memory
  cublasDestroy(handle );               //  destroy  CUBLAS  context
  CudaCheckError();

  return EXIT_SUCCESS;
}

//
// This is used to get the angle
// if flag == 0 // find angle on x
// if flag == 1 // find angle on y
// if flag == 2 // find angle on z
// v is the input vector
// we assume we compare the the vector: [1.0, 0.0, 0.0]
//
float get_angle(float cam[6], int flag){
  // TODO add other methods of the same name
  // to make this work for a more general case
  float w[3] = {1.0,0.0,0.0};
  float v[3] = {cam[3],cam[4],cam[5]};
  //float test[3] = {4.0,3.0,2.0};
  // find the dot product
  //float dot_v_w = v[0]*w[0] + v[1]*w[1] + v[2]*w[2];
  float dot_v_w;
  //dot_product(test,test,3,temp);
  dot_product(v,w,3,dot_v_w);
  //cout << "got: " << temp << endl;
  // calculate magnitude for v
  float v_mag;
  //float v_mag = sqrt(v[0]*v[0] + v[1]*v[1] + v[2]*v[2]);
  dot_product(v,v,3,v_mag);
  // make the fraction:
  //float fract = (dot_v_w)/(v_mag);
  float fract = (dot_v_w)/(sqrt(v_mag));
  // find the angle
  float angle = acos(fract);
  // check to see if outside of second quad
  if (cam[4] < 0.0) angle = 2 * PI - angle;
  if (angle >max_angle) max_angle = angle;
  if (angle < min_angle) min_angle = angle;
  return angle;
}

//
// loads cameras from a camera.txt file
// this assumes the camera is constrained to a line
// this is currently a 2-view system
//
void two_view_reproject_pan(){
  cout << "depricated, do not use" << endl;
}

void cross_product_cpu(float a[3], float b[3], float (&crossed)[3]) {
  crossed[0] = a[1]*b[2] - a[2]*b[1];
  crossed[1] = a[2]*b[0] - a[0]*b[2];
  crossed[2] = a[0]*b[1] - a[1]*b[0];
}

void add_cpu(float a[3], float b[3], float(&added)[3]) {
  added[0] = a[0] + b[0];
  added[1] = a[1] + b[1];
  added[2] = a[2] + b[2];
}


void sub_cpu(float a[3], float b[3], float(&subtracted)[3]) {
  subtracted[0] = a[0] - b[0];
  subtracted[1] = a[1] - b[1];
  subtracted[2] = a[2] - b[2];
}

float dot_product_cpu(float a[3], float b[3]) {
  return a[0]*b[0] + a[1]*b[1] + a[2]*b[2];
}

//
// loads cameras from a camera.txt file
// this assumes that the camera is constrained to a plane
//
void two_view_reproject_cpu(){
  // get the data that we want to compute
  cout << "2-view trianulating... " << endl;
  int length = matches.size();
  if (simple) length = 5000; // limit the number of points to 5k
  for(int i = 0; i < length; i++){
    int   image1     = stoi(matches[i][0].substr(0,4));
    int   image2     = stoi(matches[i][1].substr(0,4));
    float camera1[6] = {stof(cameras[image1-1][1]),stof(cameras[image1-1][2]),stof(cameras[image1-1][3]),stof(cameras[image1-1][4]),stof(cameras[image1-1][5]),stof(cameras[image1-1][6])};
    float camera2[6] = {stof(cameras[image2-1][1]),stof(cameras[image2-1][2]),stof(cameras[image2-1][3]),stof(cameras[image2-1][4]),stof(cameras[image2-1][5]),stof(cameras[image2-1][6])};
    // scale the projection's coordinates
    float x1 = dpix * (stof(matches[i][2]) - res/2.0);
    float y1 = dpix * (-1*stof(matches[i][3]) + res/2.0);
    float x2 = dpix * (stof(matches[i][4]) - res/2.0);
    float y2 = dpix * (-1*stof(matches[i][5]) + res/2.0);

    // NOTE: FROM HERE ON THERE ARE THINGS for the rotation

    // rotate the coords to be correct
    if (debug && 0){ // just set to not do this for now
      cout << "camera1 unit vector: " << camera1[3] << "," << camera1[4] << "," << camera1[5] << endl;
      cout << "rotation1: " << get_angle(camera1, 0) << endl;
      cout << "camera2 unit vector: " << camera2[3] << "," << camera2[4] << "," << camera2[5] << endl;
      cout << "rotation2: " << get_angle(camera2, 0) << endl;
    }

    // get the needed rotation
    float r1 = get_angle(camera1, 0);
    float r2 = get_angle(camera2, 0);

    if (debug) cout << "camera1 angle: " << r1 << ", camera2 angle: " << r2 << endl;

    // for some reason it was not in the right plane?
    vector<float> k1 = rotate_projection_x(x1,y1,0.0,PI/2);
    vector<float> k2 = rotate_projection_x(x2,y2,0.0,PI/2);

    vector<float> kp1 = rotate_projection_z(k1[0],k1[1],k1[2],r1 + PI/2.0); // + PI/2.0
    vector<float> kp2 = rotate_projection_z(k2[0],k2[1],k2[2],r2 + PI/2.0); // + PI/2.0

    // adjust the kp's location
    kp1[0] = camera1[0] - (kp1[0] + (camera1[3] * foc));
    kp1[1] = camera1[1] - (kp1[1] + (camera1[4] * foc));

    kp2[0] = camera2[0] - (kp2[0] + (camera2[3] * foc));
    kp2[1] = camera2[1] - (kp2[1] + (camera2[4] * foc));

    if (debug) cout << "kp1: " << "[" << kp1[0] << "," << kp1[1] << "," << kp1[2] << "]" << endl;
    if (debug) cout << "kp2: " << "[" << kp2[0] << "," << kp2[1] << "," << kp2[2] << "]" << endl;

    float points1[6] = {camera1[0],camera1[1],camera1[2],kp1[0],kp1[1],kp1[2]};
    float points2[6] = {camera2[0],camera2[1],camera2[2],kp2[0],kp2[1],kp2[2]};
    int   rgb[3]     = {stoi(matches[i][6]),stoi(matches[i][7]),stoi(matches[i][8])};

    // this is just for storing the projections for a ply file later
    vector<float> r32;
    r32.push_back(points2[3]);
    r32.push_back(points2[4]);
    r32.push_back(points2[5]);
    matchesr3.push_back(r32);
    vector<float> r31;
    r31.push_back(points1[3]);
    r31.push_back(points1[4]);
    r31.push_back(points1[5]);
    matchesr3.push_back(r31);
    // find the vectors
    float v1[3]    = {points1[3] - points1[0],points1[4] - points1[1],points1[5] - points1[2]};
    float v2[3]    = {points2[3] - points2[0],points2[4] - points2[1],points2[5] - points2[2]};
    // prepare for the linear approximation
    float smallest = numeric_limits<float>::max();
    float j_holder = 0.0;
    float p1[3]; //= {0.0,0.0,0.0};
    float p2[3]; //= {0.0,0.0,0.0};
    float point[4];
    int asdf_counter = 0;
    // Which method will find the 3D point...
    if(line_intersection) {
      // The algebra way
      //===============//
      // https://en.wikipedia.org/wiki/Skew_lines#Nearest_Points
      float temp[3]        = {0, 0, 0};
      float solution1[3]   = {0, 0, 0};
      float solution2[3]   = {0, 0, 0};
      // starting points
      p1[0] = points1[3];
      p1[1] = points1[4];
      p1[2] = points1[5];
      p2[0] = points2[3];
      p2[1] = points2[4];
      p2[2] = points2[5];
      // cross products
      float n0[3] = {0, 0, 0};
      float n1[3] = {0, 0, 0};
      float n2[3] = {0, 0, 0};
      cross_product_cpu(v1, v2, n0);
      cross_product_cpu(v2, n0, n1);
      // build the fraction
      sub_cpu(p2, p1, temp);
      float numer1 = dot_product_cpu(temp, n1);
      float denom1 = dot_product_cpu(v1, n1);
      float fract1 = numer1/denom1;
      temp[0] = fract1*v1[0];
      temp[1] = fract1*v1[1];
      temp[2] = fract1*v1[2];
      add_cpu(p1, temp, solution1);

      // repeat to find second intersection point
      cross_product_cpu(v2, v1, temp);
      cross_product_cpu(v1, temp, n2);
      sub_cpu(p1, p2, temp);
      float numer2 = dot_product_cpu(temp, n2);
      float denom2 = dot_product_cpu(v2, n2);
      float fract2 = numer2/denom2;
      temp[0] = fract2*v2[0];
      temp[1] = fract2*v2[1];
      temp[2] = fract2*v2[2];
      add_cpu(p2, temp, solution2);

      // get the midpoint of the two intersection points
      point[0] = (solution1[0] + solution2[0])/2;
      point[1] = (solution1[1] + solution2[1])/2;
      point[2] = (solution1[2] + solution2[2])/2;

      cout << "We in here boys" << endl;
	//////////////////////////////////************************************************************
    }
    else {
      // The barbarian way
      //=================//
      for (float j = 0.0; j < 8000.0; j += 0.001){
	p1[0] = points1[3] + v1[0]*j; // points1[0]
	p1[1] = points1[4] + v1[1]*j;
	p1[2] = points1[5] + v1[2]*j;
	p2[0] = points2[3] + v2[0]*j;
	p2[1] = points2[4] + v2[1]*j;
	p2[2] = points2[5] + v2[2]*j;
	// cout << j << endl;
	float dist = euclid_dist(p1,p2);
	//cout << dist << endl;
	if (dist <= smallest){
	  smallest = dist;
	  point[0] = (p1[0]+p2[0])/2.0;
	  point[1] = (p1[1]+p2[1])/2.0;
	  point[2] = (p1[2]+p2[2])/2.0;
	  point[3] = smallest;
	  j_holder = j;
	} else break;
	if (debug && asdf_counter >= 30 && 0){ // don't do for now
	  asdf_counter = 0;
	  vector<float> v_t;
	  vector<int>   c_t;
	  vector<float> v_i;
	  vector<int>   c_i;
	  // TODO
	  // do something besides scaling the hell out of this rn
	  v_t.push_back(p1[0]);
	  v_t.push_back(p1[1]);
	  v_t.push_back(p1[2]);
	  c_t.push_back(255);
	  c_t.push_back(105);
	  c_t.push_back(180);
	  points.push_back(v_t);
	  colors.push_back(c_t);
	  v_i.push_back(p2[0]);
	  v_i.push_back(p2[1]);
	  v_i.push_back(p2[2]);
	  c_i.push_back(180);
	  c_i.push_back(105);
	  c_i.push_back(255);
	  points.push_back(v_i);
	  colors.push_back(c_i);
	}
	asdf_counter++;
      }
    } // end brute force method
    if (debug) cout << "smallest: " << smallest << ", j: [" << j_holder << "]" << endl;
    // print v bc wtf
    if (debug){
      cout << "v1: [" << v1[0] << "," << v1[1] << "," << v1[2] << "]" << endl;
      cout << "v2: [" << v2[0] << "," << v2[1] << "," << v2[2] << "]" << endl;
    }
    // store the result if it sasifies the boundary conditions
    // TODO uncomment this after you test to see how far those points go
    //    if (point[3] < 0.5){
    vector<float> v;
    vector<int>   c;
    v.push_back(point[0]);
    v.push_back(point[1]);
    v.push_back(point[2]);
    c.push_back(rgb[0]);
    c.push_back(rgb[1]);
    c.push_back(rgb[2]);
    if (debug) cout << p1[0] << "," << p1[1] << "," << p1[2] << endl;
    if (debug) cout << point[0] << "," << point[1] << "," << point[2] << endl;
    points.push_back(v);
    colors.push_back(c);
    //}
    if (verbose) cout << (((((float)i))/((float)length)) * 100.0) << " */*" << endl;
  }
  cout << "Generated: " << points.size() << " valid points" << endl;
}

//
// A GPU accellerated version of the original CPU one
// TODO: Optimize with a Newton's Method, should be decently simple...
// for now this will use the dumbass iterative solution in the CPU method
//
void two_view_reproject_gpu(){
  if (verbose) cout << "Allocating memory on GPU... " << endl;

  // get ready for moving bytes
  const int MATCH_POINTS_BYTES = MATCH_POINTS_SIZE * sizeof(float);
  const int CAMERA_DATA_BYTES  = CAMERA_DATA_SIZE * sizeof(float);

  const int POINT_CLOUD_SIZE   = 3*(MATCH_POINTS_SIZE/4);
  const int POINT_CLOUD_BYTES  = POINT_CLOUD_SIZE * sizeof(float);

  // create pointers on the device
  float *d_in_m;
  float *d_in_c;
  float *d_out_p;

  // allocate the space on the GPU
  cudaMalloc((void **) &d_in_m, MATCH_POINTS_BYTES);
  cudaMalloc((void **) &d_in_c, CAMERA_DATA_BYTES);
  cudaMalloc((void **) &d_out_p, POINT_CLOUD_BYTES); // this was already a global pointer

  // allocate memory for the point cloud on the CPU
  point_cloud = new (nothrow) float[POINT_CLOUD_SIZE];
  for(int i = 0; i < POINT_CLOUD_SIZE; ++i){
    point_cloud[i] = 0.0f;
  }

  // transfer the memory to the GPU
  cudaMemcpy(d_in_m, match_points, MATCH_POINTS_BYTES, cudaMemcpyHostToDevice);
  cudaMemcpy(d_in_c, camera_data, CAMERA_DATA_BYTES, cudaMemcpyHostToDevice);
  cudaMemcpy(d_out_p, point_cloud, POINT_CLOUD_BYTES, cudaMemcpyHostToDevice);

  // calculate the block & thread count here
  dim3 THREAD_COUNT = {512,1,1};
  dim3 BLOCK_COUNT  = {((POINT_CLOUD_SIZE/3)/THREAD_COUNT.x),1,1};
  if (!BLOCK_COUNT.x){ // a very small point cloud
    BLOCK_COUNT.x = 1;
    THREAD_COUNT.x = POINT_CLOUD_SIZE/3;
  }
  if (debug) cout << "THREAD COUNT: " << THREAD_COUNT.x << endl << "BLOCK COUNT: " << BLOCK_COUNT.x << endl << "Point Cloud Size: " << POINT_CLOUD_SIZE << endl;
  if (verbose) cout << "Reprojecting..." << endl;
  two_view_reproject<<<BLOCK_COUNT,THREAD_COUNT>>>(d_in_m,d_in_c,d_out_p);
  CudaCheckError();

  cudaMemcpy(point_cloud, d_out_p, POINT_CLOUD_BYTES, cudaMemcpyDeviceToHost);

  cudaFree(d_in_m);
  cudaFree(d_in_c);
  cudaFree(d_out_p);

}

void save_ply(){
  if (gpu_acc){
    int point_cloud_size = (MATCH_POINTS_SIZE/4);
    ofstream outputFile1("output.ply");
    outputFile1 << "ply\nformat ascii 1.0\nelement vertex ";
    outputFile1 << point_cloud_size << "\n";
    outputFile1 << "property float x\nproperty float y\nproperty float z\nproperty uchar red\nproperty uchar green\nproperty uchar blue\n";
    outputFile1 << "end_header\n";
    for(int i = 0; i < point_cloud_size*3; i+=3){
      outputFile1 << point_cloud[i] << " " << point_cloud[i+1] << " " << point_cloud[i+2] << " " << match_colors[i] << " " << match_colors[i+1] << " " << match_colors[i+2] << "\n";
      //<< match_colors[i] << " " << match_colors[i+1] << " " << match_colors[i+2] << "\n";
    }
    if (debug){
      cout << "POINT CLOUD DEBUG:" << endl;
      for (int i =0; i < point_cloud_size*3; i++){
        if (!(i%3)) cout << endl;
        cout << point_cloud[i] << "\t\t\t";
      }
      cout << endl;
    }
    if (debug){
      ofstream outputFile2("cameras.ply");
      outputFile2 << "ply\nformat ascii 1.0\nelement vertex ";
      outputFile2 << 2 << "\n";
      outputFile2 << "property float x\nproperty float y\nproperty float z\nproperty uchar red\nproperty uchar green\nproperty uchar blue\n";
      outputFile2 << "end_header\n";
      for(int i = 0; i < 12; i+=6){
        outputFile2 << camera_data[i] << " " << camera_data[i+1] << " " << camera_data[i+2] << " 255 0 0\n";
      }
      cout << "CAMERA DATA DEBUG:" << endl;
      for (int i = 0; i < 12; i++){
        if (!(i%6)) cout << endl;
        cout << camera_data[i] << "\t";
      }
      cout << endl;
    }
  } else {
    ofstream outputFile1("output.ply");
    outputFile1 << "ply\nformat ascii 1.0\nelement vertex ";
    outputFile1 << points.size() << "\n";
    outputFile1 << "property float x\nproperty float y\nproperty float z\nproperty uchar red\nproperty uchar green\nproperty uchar blue\n";
    outputFile1 << "end_header\n";
    for(int i = 0; i < points.size(); i++){
      outputFile1 << points[i][0] << " " << points[i][1] << " " << points[i][2] << " " << colors[i][0] << " " << colors[i][1] << " " << colors[i][2] << "\n";
    }
    if (debug){
      ofstream outputFile2("cameras.ply");
      outputFile2 << "ply\nformat ascii 1.0\nelement vertex ";
      outputFile2 << cameras.size() << "\n";
      outputFile2 << "property float x\nproperty float y\nproperty float z\nproperty uchar red\nproperty uchar green\nproperty uchar blue\n";
      outputFile2 << "end_header\n";
      for(int i = 0; i < cameras.size(); i++){
        outputFile2 << cameras[i][1] << " " << cameras[i][2] << " " << cameras[i][3] << " 255 0 0\n";
      }
      ofstream outputFile3("matches.ply");
      outputFile3 << "ply\nformat ascii 1.0\nelement vertex ";
      outputFile3 << matchesr3.size() << "\n";
      outputFile3 << "property float x\nproperty float y\nproperty float z\nproperty uchar red\nproperty uchar green\nproperty uchar blue\n";
      outputFile3 << "end_header\n";
      int counter = 0;
      int b = 0;
      for(int i = 0; i < matchesr3.size(); i++){
        outputFile3 << matchesr3[i][0] << " " << matchesr3[i][1] << " " << matchesr3[i][2] << " 0 255 " << b << "\n";
        if (counter%2) b += 25;
        if (b > 255) b = 0;
        counter++;
        //cout << "";
      }
    }
  }
}

//
// This is the main method
//
int main(int argc, char* argv[]){
  cout << "*===================* REPROJECTION *===================*" << endl;
  if (argc < 3){
    cout << "not enough arguments ... " << endl;
    cout << "USAGE: " << endl;
    cout << "./reprojection.x path/to/cameras.txt path/to/matches.txt" << endl;
    cout << "*======================================================*" << endl;
    return 0; // end it all. it will be so serene.
  }

  else {
    cout << "*                                                      *" << endl;
    cout << "*                     ~ UGA SSRL ~                     *" << endl;
    cout << "*        Multiview Onboard Computational Imager        *" << endl;
    cout << "*                                                      *" << endl;
  }
  cout << "*======================================================*" << endl;

  if (debug) printDeviceProperties();

  cameras_path = argv[1];
  matches_path = argv[2];
  if (argc == 4) gpu_acc = stoi(argv[3]);

  load_matches();
  load_cameras();
  if (gpu_acc) two_view_reproject_gpu();
  else two_view_reproject_cpu();
  save_ply();

  if (verbose) cout << "done!\nresults saved to output.ply" << endl;
  if (debug) cout << "max angle: " << max_angle << " | min angle: " << min_angle << endl;

  return 0;
}
