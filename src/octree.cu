#include "octree.cuh"

using namespace std;

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


//pretty much just a binary search in each dimension performed by threads
__global__ void getKeys(float3* points, float3* centers, int* keys, float3 c, float W, int N, int D){

  int tx = threadIdx.x;
  int bx = blockIdx.x;
  int globalID = bx * blockDim.x + tx;
  if(globalID < N){
    float x = points[globalID].x;
    float y = points[globalID].y;
    float z = points[globalID].z;
    float leftx = c.x-W/2.0f, rightx = c.x + W/2.0f;
    float lefty = c.y-W/2.0f, righty = c.y + W/2.0f;
    float leftz = c.z-W/2.0f, rightz = c.z + W/2.0f;
    int key = 0;
    int depth = 1;
    while(depth <= D){
      if(x < c.x){
        key <<= 1;
        rightx = c.x;
        c.x = (leftx + rightx)/2.0f;
      }
      else{
        key = (key << 1) + 1;
        leftx = c.x;
        c.x = (leftx + rightx)/2.0f;
      }
      if(y < c.y){
        key <<= 1;
        righty = c.y;
        c.y = (lefty + righty)/2.0f;
      }
      else{
        key = (key << 1) + 1;
        lefty = c.y;
        c.y = (lefty + righty)/2.0f;
      }
      if(z < c.z){
        key <<= 1;
        rightz = c.z;
        c.z = (leftz + rightz)/2.0f;
      }
      else{
        key = (key << 1) + 1;
        leftz = c.z;
        c.z = (leftz + rightz)/2.0f;
      }
      depth++;
    }
    keys[globalID] = key;
    centers[globalID].x = c.x;
    centers[globalID].y = c.y;
    centers[globalID].z = c.z;
  }
}

Octree::Octree(){

}

void Octree::parsePLY(string pathToFile){
  cout<<pathToFile + "'s data to be transfered to an empty octree."<<endl;
	ifstream plystream(pathToFile);
	string currentLine;
  vector<float3> points;
  vector<float3> normals;
	if (plystream.is_open()) {
		while (getline(plystream, currentLine)) {
      stringstream getMyFloats = stringstream(currentLine);
      float value = 0.0;
      int index = 0;
      float3 point;
      float3 normal;
      bool lineIsDone = false;
      while(getMyFloats >> value){
        switch(index){
          case 0:
            point.x = value;
            break;
          case 1:
            point.y = value;
            break;
          case 2:
            point.z = value;
            break;
          case 3:
            normal.x = value;
            break;
          case 4:
            normal.y = value;
            break;
          case 5:
            normal.z = value;
            break;
          default:
            lineIsDone = true;
            points.push_back(point);
            normals.push_back(normal);
            break;
        }
        if(lineIsDone) break;
        ++index;
      }
		}
    this->points = new float3[points.size()];
    this->normals = new float3[normals.size()];
    this->numPoints = (int) points.size();
    for(int i = 0; i < points.size(); ++i){
      this->points[i] = points[i];
      this->normals[i] = normals[i];
      cout<<points[i].x<<" "<<points[i].y<<" "<<points[i].z<<" "<<normals[i].x<<" "<<normals[i].y<<" "<<normals[i].z<<endl;
    }
    cout<<pathToFile + "'s data has been transfered to an initialized octree."<<endl;
	}
	else{
    cout << "Unable to open: " + pathToFile<< endl;
    exit(1);
  }
}

Octree::Octree(string pathToFile, int depth){
  this->parsePLY(pathToFile);
  this->depth = depth;
}
void Octree::findMinMax(){
  this->min = this->points[0];
  this->max = this->points[0];
  float3 currentPoint;
  for(int i = 0; i < numPoints; ++i){
    currentPoint = this->points[i];
    if(currentPoint.x < this->min.x){
      this->min.x = currentPoint.x;
    }
    if(currentPoint.x > this->max.x){
      this->max.x = currentPoint.x;
    }
    if(currentPoint.y < this->min.y){
      this->min.y = currentPoint.y;
    }
    if(currentPoint.y > this->max.y){
      this->max.y = currentPoint.y;
    }
    if(currentPoint.z < this->min.z){
      this->min.z = currentPoint.z;
    }
    if(currentPoint.z > this->max.z){
      this->max.z = currentPoint.z;
    }
  }
  this->center.x = (this->max.x - this->min.x)/2;
  this->center.y = (this->max.y - this->min.y)/2;
  this->center.z = (this->max.z - this->min.z)/2;

}

void Octree::allocateDeviceVariables(){
  CudaSafeCall(cudaMalloc((void**)&this->pointsDevice, this->numPoints * sizeof(float3)));
  CudaSafeCall(cudaMalloc((void**)&this->centersDevice, this->numPoints * sizeof(float3)));
  CudaSafeCall(cudaMalloc((void**)&this->keysDevice, this->numPoints * sizeof(int)));
  CudaSafeCall(cudaMalloc((void**)&this->normalsDevice, this->numPoints * sizeof(int)));

}

void Octree::executeKeyRetrieval(){


  CudaSafeCall(cudaMemcpy(this->pointsDevice, this->points, this->numPoints * sizeof(float3), cudaMemcpyHostToDevice));
  CudaSafeCall(cudaMemcpy(this->centersDevice, this->centers, this->numPoints * sizeof(float3), cudaMemcpyHostToDevice));
  CudaSafeCall(cudaMemcpy(this->keysDevice, this->keys, this->numPoints * sizeof(int), cudaMemcpyHostToDevice));

  getKeys<<<1,1>>>(this->pointsDevice, this->centersDevice, this->keysDevice, this->center, this->width, this->numPoints, this->depth);
  CudaCheckError();
  CudaSafeCall(cudaMemcpy(this->centers, this->centersDevice, this->numPoints * sizeof(float3), cudaMemcpyDeviceToHost));
  CudaSafeCall(cudaMemcpy(this->keys, this->keysDevice, this->numPoints * sizeof(int), cudaMemcpyDeviceToHost));

}

void Octree::cudaFreeMemory(){
  CudaSafeCall(cudaFree(this->keysDevice));
  CudaSafeCall(cudaFree(this->centersDevice));
  CudaSafeCall(cudaFree(this->pointsDevice));
}
