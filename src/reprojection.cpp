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
#include <vector>
// my boiz @ nvidia
#include <cuda_runtime.h>
//#include "cublas.h"
#include "cublas_v2.h"
//#include "cusolver.h"
// to remove eventually
#include <time.h>

// alsp remove eventually
#define N 8192
#define  BILLION  1000000000L;

//Just to print all properties on TX2 pertinent to cuda development
using namespace std;


void printDeviceProperties() {
  cout<<"---------------START OF DEVICE PROPERTIES---------------"<<endl;

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
    printf("  -Total Global Memory (bytes): %d\n", prop.totalGlobalMem);
    printf("  -Total Const Memory (bytes): %d\n", prop.totalConstMem);
    printf("  -Max pitch allowed for memcpy in regions allocated by cudaMallocPitch() (bytes): %d\n\n", prop.memPitch);
    printf("  -Shared Memory per block (bytes): %d\n", prop.sharedMemPerBlock);
    printf("  -Max number of threads per block: %d\n",prop.maxThreadsPerBlock);
    printf("  -Max number of blocks: %dx%dx%d\n",prop.maxGridSize[0], prop.maxGridSize[1], prop.maxGridSize[2]);
    printf("  -32bit Registers per block: %d\n", prop.regsPerBlock);
    printf("  -Threads per warp (bytes): %d\n\n", prop.warpSize);
    printf("  -Total number of Multiprocessors: %d\n",prop.multiProcessorCount);
    printf("  -Shared Memory per Multiprocessor (bytes): %d\n",prop.sharedMemPerMultiprocessor);
    printf("  -32bit Registers per Multiprocessor: %d\n\n", prop.regsPerMultiprocessor);
    printf("  -Number of asynchronous engines: %d\n", prop.asyncEngineCount);
    printf("  -Texture alignment requirement (bytes): %d\n  -Texture base addresses that are aligned to "
    "textureAlignment bytes do not need an offset applied to texture fetches.\n", prop.textureAlignment);
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
        printf("\n -The device can map host memory into the CUDA address space for use with\n cudaHostAlloc() or cudaHostGetDevicePointer().\n");
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
    cout<<"----------------END OF DEVICE PROPERTIES----------------"<<endl;

}


// == GLOBAL VARIABLES == //
bool           verbose = 1;
bool           debug   = 0;
bool           simple  = 0;

string cameras_path;
string matches_path;

int    to_pan;

unsigned short match_count;
unsigned short camera_count;

// TODO (some of) this stuff should be set by camera calibration

// This was for the test cases only
unsigned int   res  = 1024;
float          foc  = 0.035;
float          fov  = 0.8575553107; // 49.1343 degrees  // 0.785398163397; // 45 degrees
float          PI   = 3.1415926535;
float          dpix = (foc*tan(fov/2))/(res/2); //float          dpix = 0.00002831538; //(foc*tan(fov/2))/(res/2)

// Test 2 with slightly higher res
//unsigned int   res  = 2000;
//float          dpix = 0.00002831538; //(foc*tan(fov/2))/(res/2)
//float          foc  = 0.035;
//float          fov  = 0.8575553107; // 49.1343 degrees  // 0.785398163397; // 45 degrees
//float          PI   = 3.1415926535;

// this is for the blender sim of mnt everest
//unsigned int   res  = 4208;
//float          foc  = 0.18288;
//float          fov  = 0.174533; // 10 degrees
//float          PI   = 3.1415926535;
//float          dpix = (foc*tan(fov/2))/(res/2);

// for debugging
float          max_angle = -1000.0;
float          min_angle =  1000.0;

vector< vector<string> > matches;
vector< vector<string> > cameras;
vector< vector<string> > projections;
vector< vector<float> >  points;
vector< vector<float> >  matchesr3;
vector< vector<int> >    colors;
// ====================== //

//
// parses comma delemeted string
//
void parse_comma_delem(string str, unsigned short flag){
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
      matches.push_back(v);
      break;
    case 2: // cameras
      cameras.push_back(v);
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
  unsigned short c = 0;
  bool first = 1;
  while (getline(infile, line))
  {
      istringstream iss(line);
      if (debug) cout << line << endl;
      if (first)
      {
        first = 0;
      } else
      {
        parse_comma_delem(line, 1);
        c++;
      }
  }
  match_count = c-1;
  if (verbose or debug) cout << "Loaded: " << match_count << " matches" << endl;
}

//
// loads cameras from a camera.txt file
//
void load_cameras()
{
  ifstream infile(cameras_path);
  string line;
  unsigned short c = 0;
  while (getline(infile, line))
  {
      istringstream iss(line);
      if (debug) cout << line << endl;
      parse_comma_delem(line, 2);
      c++;
  }
  camera_count = c;
  if (verbose or debug) cout << "Loaded: " << camera_count << " cameras" << endl;
}

//
// This is used for linear approximation
// TODO develop a better way to do linear approximation
//
float euclid_dist(float p1[3], float p2[3])
{
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

int vector_scale(float *x, int xdimension, int ydimension, float scalevalue)
{
    //
    //CUBLAS - scale the projection's coordinates
    //
    //allocate system memory to then transfer to gpu memory
    //initialize content then set vector.
    //scale vector stat=cublasSscal(handle,n,&al,d x,1);
    //get vector
    //cudafree
    //cublas destroy handle
    //free host mallocd

    int n = xdimension;
    cudaError_t cudaStat ; // cudaMalloc status
    cublasStatus_t stat ; // CUBLAS functions status
    cublasHandle_t handle ; // CUBLAS context

    // on the device
    float * d_x; // d_x - x on the device
    cudaStat = cudaMalloc (( void **)& d_x ,n* sizeof (*x)); // device
    // memory alloc for x
    stat = cublasCreate (& handle ); // initialize CUBLAS context
    stat = cublasSetVector (n, sizeof (*x) ,x ,1 ,d_x ,1); // cp x- >d_x
    float al =scalevalue; // al =2
    // scale the vector d_x by the scalar al: d_x = al*d_x
    stat=cublasSscal(handle,n,&al,d_x,1);
    stat = cublasGetVector (n, sizeof ( float ) ,d_x ,1 ,x ,1); // cp d_x - >x

    cudaFree (d_x); // free device memory
    cublasDestroy (handle); // destroy CUBLAS context
    free (x); // free host memory
    return EXIT_SUCCESS ;
}

//
// This is to perform a dot product w the GPU
// This uses CUDA with CUBLAS
//
int dot_product(float *x, float *y, int length, float &val){

  cudaError_t    cudaStat; // cudaMalloc status
  cublasStatus_t stat; // CUBLAS functions status
  cublasHandle_t handle; // CUBLAS context

  int j;

  float* d_x;  // d_x - x on the  device
  float* d_y;  // d_y - y on the device

  cudaStat = cudaMalloc ((void **)&d_x ,length*sizeof (*x));     // device memory  alloc  for x
  cudaStat = cudaMalloc ((void **)&d_y ,length*sizeof (*y));     // device memory  alloc  for y

  stat = cublasCreate (& handle );   //  initialize  CUBLAS  context
  stat = cublasSetVector(length,sizeof (*x),x,1,d_x ,1);// cp x->d_x
  stat = cublasSetVector(length,sizeof (*y),y,1,d_y ,1);// cp y->d_y

  float  result;

  // dot  product  of two  vectors d_x ,d_y:
  // d_x [0]* d_y [0]+...+ d_x[n-1]* d_y[n-1]

  stat = cublasSdot(handle,length,d_x,1,d_y,1,&result);

  //cout << "dot result: " << result << endl;

  val = result;

  cudaFree(d_x);                             // free  device  memory
  cudaFree(d_y);                             // free  device  memory
  cublasDestroy(handle );               //  destroy  CUBLAS  context

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
  float test[3] = {4.0,3.0,2.0};
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
  // get the data that we want to compute
  cout << "2-view triangulating... " << endl;
  int length = matches.size();
  if (simple) length = 5000; // limit the number of points to 5k
  for(int i = 0; i < length; i++){
    int   image1     = stoi(matches[i][0].substr(0,4));
    int   image2     = stoi(matches[i][1].substr(0,4));
    float camera1[6] = {stof(cameras[image1-1][1]),stof(cameras[image1-1][2]),stof(cameras[image1-1][3]),stof(cameras[image1-1][4]),stof(cameras[image1-1][5]),stof(cameras[image1-1][6])};
    float camera2[6] = {stof(cameras[image2-1][1]),stof(cameras[image2-1][2]),stof(cameras[image2-1][3]),stof(cameras[image2-1][4]),stof(cameras[image2-1][5]),stof(cameras[image2-1][6])};

    // scale the projection's coordinates
    float x1 = dpix * (stof(matches[i][2]) - res/2.0);
    float y1 = dpix * (stof(matches[i][3]) - res/2.0);
    float x2 = dpix * (stof(matches[i][4]) - res/2.0);
    float y2 = dpix * (stof(matches[i][5]) - res/2.0);

    // NOTE FROM HERE ON THERE ARE THINGS for the rotation

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

    float angle = 0.0;

    // for some reason it was not in the right plane?
    vector<float> k1 = rotate_projection_x(x1,y1,0.0,angle);
    vector<float> k2 = rotate_projection_x(x2,y2,0.0,angle);

    vector<float> kp1 = rotate_projection_z(k1[0],k1[1],k1[2],r1 + angle);
    vector<float> kp2 = rotate_projection_z(k2[0],k2[1],k2[2],r2 + angle);

    // adjust the kp's location
    kp1[0] = camera1[0] - (kp1[0] + (camera1[3] * foc));
    kp1[1] = camera1[1] - (kp1[1] + (camera1[4] * foc));

    kp2[0] = camera2[0] - (kp2[0] + (camera2[3] * foc));
    kp2[1] = camera2[1] - (kp2[1] + (camera2[4] * foc));

    // NOTE This is the pan-view way to do this

    float points1[6] = {camera1[0],camera1[1],camera1[2],x1 + camera1[0],y1 + camera1[1],foc + camera1[2]};
    float points2[6] = {camera2[0],camera2[1],camera2[2],x2 + camera2[0],y2 + camera2[1],foc + camera2[2]};
    int   rgb[3]     = {stoi(matches[i][6]),stoi(matches[i][7]),stoi(matches[i][8])};

    // END NOTE

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
    float v1[3]      = {points1[3] - points1[0],points1[4] - points1[1],points1[5] - points1[2]};
    float v2[3]      = {points2[3] - points2[0],points2[4] - points2[1],points2[5] - points2[2]};
    // prepare for the linear approximation
    float smallest = numeric_limits<float>::max();
    float p1[3] = {0.0,0.0,0.0};
    float p2[3] = {0.0,0.0,0.0};
    //for (float i = 0.0; i < 800.0; i += 0.0001){
    for (float i = 0.0; i < 1000.0; i += 0.0000001){ // for testing more quickly
      // get the points on the lines
      p1[0]  = points1[0] + v1[0]*i;
      p1[1]  = points1[1] + v1[1]*i;
      p1[2]  = points1[2] + v1[2]*i;
      p2[0]  = points2[0] + v2[0]*i;
      p2[1]  = points2[1] + v2[1]*i;
      p2[2]  = points2[2] + v2[2]*i;
      float dist = euclid_dist(p1,p2);
      if (dist < smallest) smallest = dist;
      else break;
    }
    cout << endl;
    // store the result if it sasifies the boundary conditions
    // TODO uncomment this after you test to see how far those points go
    //if (p1[2] > 1.0 && p1[2] < 3.0){
    vector<float> v;
    vector<int>   c;
    v.push_back(p1[0]);
    v.push_back(p1[1]);
    v.push_back(p1[2]);
    c.push_back(rgb[0]);
    c.push_back(rgb[1]);
    c.push_back(rgb[2]);
    if (debug) cout << p1[0] << "," << p1[1] << "," << p1[2] << endl;
    points.push_back(v);
    colors.push_back(c);
    //}
    if (verbose) cout << (((((float)i))/((float)length)) * 100.0) << " \%" << endl;
  }
  cout << "Generated: " << points.size() << " valid points" << endl;
}

//
// loads cameras from a camera.txt file
// this assumes that the camera is constrained to a plane
// this is currently a 2-view system
//
void two_view_reproject_plane(){
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
    if (verbose) cout << (((((float)i))/((float)length)) * 100.0) << " \%" << endl;
  }
  cout << "Generated: " << points.size() << " valid points" << endl;
}

void save_ply()
{
  ofstream outputFile1("output.ply");
  outputFile1 << "ply\nformat ascii 1.0\nelement vertex ";
  outputFile1 << points.size() << "\n";
  outputFile1 << "property float x\nproperty float y\nproperty float z\nproperty uchar red\nproperty uchar green\nproperty uchar blue\n";
  outputFile1 << "end_header\n";
  for(int i = 0; i < points.size(); i++){
    outputFile1 << points[i][0] << " " << points[i][1] << " " << points[i][2] << " " << colors[i][0] << " " << colors[i][1] << " " << colors[i][2] << "\n";
  }
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

//
// This is the main method
//
int main(int argc, char* argv[])
{
  cout << "*===================* REPROJECTION *===================*" << endl;
  if (argc < 4){
    cout << "not enough arguments ... " << endl;
    cout << "USAGE: " << endl;
    cout << "./reprojection.x path/to/cameras.txt path/to/matches.txt 1/0" << endl;
    cout << "the last arg is a 1 if panning and a 0 if plane constrained" << endl;
    cout << "*======================================================*" << endl;
    return 0; // end it all. it will be so serene.
  }

  else
  {
    cout << "*                                                      *" << endl;
    cout << "*                     ~ UGA SSRL ~                     *" << endl;
    cout << "*        Multiview Onboard Computational Imager        *" << endl;
    cout << "*                                                      *" << endl;
  }
  cout << "*======================================================*" << endl;

  cameras_path = argv[1];
  matches_path = argv[2];
  to_pan       = atoi(argv[3]);


  load_matches();
  load_cameras();
  if (to_pan) two_view_reproject_pan();
  else two_view_reproject_plane();
  save_ply();

  if (verbose) cout << "done!\nresults saved to output.ply" << endl;
  if (debug) cout << "max angle: " << max_angle << " | min angle: " << min_angle << endl;
  printDeviceProperties();

  return 0;
}
