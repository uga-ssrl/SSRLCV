#include "FeatureFactory.cuh"

/*
HOST METHODS
*/
//Base feature factory


ssrlcv::FeatureFactory::FeatureFactory(){

}



ssrlcv::SIFT_FeatureFactory::SIFT_FeatureFactory(bool dense, unsigned int maxOrientations, float orientationThreshold) :
dense(dense), maxOrientations(maxOrientations), orientationThreshold(orientationThreshold){
  this->orientationContribWidth = 1.5;
  this->descriptorContribWidth = 6.0;
}

void ssrlcv::SIFT_FeatureFactory::setOrientationContribWidth(float orientationContribWidth){
  this->orientationContribWidth = orientationContribWidth;
}

void ssrlcv::SIFT_FeatureFactory::setDescriptorContribWidth(float descriptorContribWidth){
  this->descriptorContribWidth = descriptorContribWidth;
}



//binDepth currently not being used
ssrlcv::Unity<ssrlcv::Feature<ssrlcv::SIFT_Descriptor>>* ssrlcv::SIFT_FeatureFactory::generateFeatures(ssrlcv::Image* image){
  Unity<Feature<SIFT_Descriptor>>* features = nullptr;

  if(image->pixels->fore == cpu){
    image->pixels->transferMemoryTo(gpu);
  }
  if(image->colorDepth != 1){
    convertToBW(image->pixels,image->colorDepth);
    image->colorDepth = 1;
  }

  if(this->dense){
    dim3 grid = {1,1,1};
    dim3 block = {1,1,1};

    clock_t timer = clock();

    Unity<int2>* gradients = generatePixelGradients(image);
    Unity<float2>* keyPoints = getLocationsWithinBorder(image->descriptor.size, {12.0f,12.0f});

    printf("\nDense SIFT prep done in %f seconds.\n\n",((float) clock() -  timer)/CLOCKS_PER_SEC);

    image->pixels->clear(gpu);//may want to nullify pixels for space

    features = this->createFeatures(image->descriptor.size,1,sqrtf(2),gradients,keyPoints);

    delete gradients;
    delete keyPoints;
  }
  else{

  }


  return features;
}

ssrlcv::Unity<ssrlcv::Feature<ssrlcv::SIFT_Descriptor>>* ssrlcv::SIFT_FeatureFactory::createFeatures(uint2 imageSize, float pixelWidth, float sigma, Unity<int2>* gradients, Unity<float2>* keyPoints){

  clock_t timer = clock();

  int* thetaNumbers_device = nullptr;
  CudaSafeCall(cudaMalloc((void**)&thetaNumbers_device,keyPoints->numElements*this->maxOrientations*sizeof(int)));
  float* thetas_device = nullptr;
  CudaSafeCall(cudaMalloc((void**)&thetas_device,keyPoints->numElements*this->maxOrientations*sizeof(float)));
  float* thetas_host = new float[keyPoints->numElements*this->maxOrientations];
  for(int i = 0; i < keyPoints->numElements*this->maxOrientations; ++i){
    thetas_host[i] = -1.0f;
  }
  CudaSafeCall(cudaMemcpy(thetas_device,thetas_host,keyPoints->numElements*this->maxOrientations*sizeof(float),cudaMemcpyHostToDevice));

  dim3 grid = {1,1,1};
  dim3 block = {1,1,1};
  if(6*sigma*pixelWidth*this->orientationContribWidth < 32){
    block = {6*sigma*pixelWidth*this->orientationContribWidth,6*sigma*pixelWidth*this->orientationContribWidth,1};
  }
  else{
    block = {32,32,1};
  }
  getGrid(keyPoints->numElements,grid);

  computeThetas<<<grid,block>>>(keyPoints->numElements,imageSize,sigma,pixelWidth,6*sigma*pixelWidth*this->orientationContribWidth,
    keyPoints->device, gradients->device, thetaNumbers_device, this->maxOrientations, this->orientationThreshold,thetas_device);
  cudaDeviceSynchronize();
  CudaCheckError();

  printf("compute thetas done in %f seconds.\n\n",((float) clock() -  timer)/CLOCKS_PER_SEC);
  timer = clock();

  thrust::device_ptr<int> tN(thetaNumbers_device);
  thrust::inclusive_scan(tN, tN + keyPoints->numElements*this->maxOrientations, tN);

  // float* thetas_host = new float[keyPoints->numElements*this->maxOrientations];
  // CudaSafeCall(cudaMemcpy(thetas_host,thetas_device,keyPoints->numElements*this->maxOrientations*sizeof(float),cudaMemcpyDeviceToHost));
  // for(int i = 0; i < keyPoints->numElements*this->maxOrientations; ++i){
  //   printf("%d - %f\n",i/this->maxOrientations,thetas_host[i]);
  // }


  thrust::device_ptr<float> t(thetas_device);
  thrust::remove(t, t + keyPoints->numElements*this->maxOrientations, -1.0f);
  int numFeatures = 0;
  CudaSafeCall(cudaMemcpy(&numFeatures,thetaNumbers_device + (keyPoints->numElements - 1), sizeof(int), cudaMemcpyDeviceToHost));

  grid = {1,1,1};
  block = {4,4,8};
  getGrid(numFeatures,grid);

  Feature<SIFT_Descriptor>* features_device = nullptr;
  CudaSafeCall(cudaMalloc((void**)&features_device,numFeatures*sizeof(Feature<SIFT_Descriptor>)));
  fillDescriptors<<<grid,block>>>(numFeatures,imageSize,features_device,sigma,pixelWidth,this->descriptorContribWidth,2*sigma*this->descriptorContribWidth,
    thetas_device,thetaNumbers_device,keyPoints->device,gradients->device);
  cudaDeviceSynchronize();
  CudaCheckError();

  CudaSafeCall(cudaFree(thetas_device));
  CudaSafeCall(cudaFree(thetaNumbers_device));

  printf("fill descriptors done in %f seconds.\n\n",((float) clock() -  timer)/CLOCKS_PER_SEC);

  return new Unity<Feature<SIFT_Descriptor>>(features_device,numFeatures,gpu);
}



/*
CUDA implementations
*/

__constant__ float ssrlcv::pi = 3.141592653589793238462643383279502884197;

/*
DEVICE METHODS
*/
__device__ __forceinline__ unsigned long ssrlcv::getGlobalIdx_2D_1D(){
  unsigned long blockId = blockIdx.y * gridDim.x + blockIdx.x;
  unsigned long threadId = blockId * blockDim.x + threadIdx.x;
  return threadId;
}
__device__ __forceinline__ float ssrlcv::getMagnitude(const int2 &vector){
  return sqrtf(dotProduct(vector, vector));
}
__device__ __forceinline__ float ssrlcv::getTheta(const int2 &vector){
  float theta = atan2f((float)vector.y, (float)vector.x) + pi;
  return fmod(theta,2.0f*pi);
}
__device__ __forceinline__ float ssrlcv::getTheta(const float2 &vector){
  float theta = atan2f(vector.y, vector.x) + pi;
  return fmod(theta,2.0f*pi);
}
__device__ __forceinline__ float ssrlcv::getTheta(const float2 &vector, const float &offset){
  float theta = (atan2f(vector.y, vector.x) + pi) - offset;
  return fmod(theta + 2.0f*pi,2.0f*pi);
}
__device__ void ssrlcv::trickleSwap(float2 compareWValue, float2* arr, int index, const int &length){
  for(int i = index; i < length; ++i){
    if(compareWValue.x > arr[i].x){
      float2 temp = arr[i];
      arr[i] = compareWValue;
      if((temp.x == 0.0f && temp.y == 0.0f)|| index + 1 == length) return;
      return trickleSwap(temp, arr, index + 1, length);
    }
  }
}
__device__ __forceinline__ long4 ssrlcv::getOrientationContributers(const long2 &loc, const uint2 &imageSize){
  long4 orientationContributers;
  long pixelIndex = loc.y*imageSize.x + loc.x;
  orientationContributers.x = (loc.x == imageSize.x - 1) ? -1 : pixelIndex + 1;
  orientationContributers.y = (loc.x == 0) ? -1 : pixelIndex - 1;
  orientationContributers.z = (loc.y == imageSize.y - 1) ? -1 : (loc.y + 1)*imageSize.x + loc.x;
  orientationContributers.w = (loc.y == 0) ? -1 : (loc.y - 1)*imageSize.x + loc.x;
  return orientationContributers;
}
__device__ __forceinline__ int ssrlcv::floatToOrderedInt(float floatVal){
 int intVal = __float_as_int( floatVal );
 return (intVal >= 0 ) ? intVal : intVal ^ 0x7FFFFFFF;
}
__device__ __forceinline__ float ssrlcv::orderedIntToFloat(int intVal){
 return __int_as_float( (intVal >= 0) ? intVal : intVal ^ 0x7FFFFFFF);
}
__device__ __forceinline__ float ssrlcv::atomicMinFloat (float * addr, float value) {
  float old;
  old = (value >= 0) ? __int_as_float(atomicMin((int *)addr, __float_as_int(value))) :
    __uint_as_float(atomicMax((unsigned int *)addr, __float_as_uint(value)));
  return old;
}
__device__ __forceinline__ float ssrlcv::atomicMaxFloat (float * addr, float value) {
  float old;
  old = (value >= 0) ? __int_as_float(atomicMax((int *)addr, __float_as_int(value))) :
    __uint_as_float(atomicMin((unsigned int *)addr, __float_as_uint(value)));
  return old;
}
__device__ __forceinline__ float ssrlcv::modulus(const float &x, const float &y){
    float z = x;
    int n;
    if(z < 0){
        n = (int)((-z)/y)+1;
        z += n*y;
    }
    n = (int)(z/y);
    z -= n*y;
    return z;
}
__device__ __forceinline__ float2 ssrlcv::rotateAboutPoint(const int2 &loc, const float &theta, const float2 &origin){
  float2 rotatedPoint = {(float) loc.x, (float) loc.y};
  rotatedPoint = rotatedPoint - origin;
  float2 temp = rotatedPoint;

  rotatedPoint.x = (temp.x*cosf(theta)) - (temp.y*sinf(theta)) + origin.x;
  rotatedPoint.y = (temp.x*sinf(theta)) + (temp.y*cosf(theta)) + origin.y;

  return rotatedPoint;
}
/*
KERNELS
*/


//will make a hog of certain size based on sigma
__global__ void ssrlcv::computeThetas(unsigned long numKeyPoints, uint2 imageSize, float sigma, float pixelWidth, int contributerWindowWidth, float2* keyPointLocations, int2* gradients,
int* thetaNumbers, unsigned int maxOrientations, float orientationThreshold, float* thetas){

  int blockId = blockIdx.y* gridDim.x+ blockIdx.x;

  if(blockId < numKeyPoints){
    float2 keyPoint = keyPointLocations[blockId];
    __shared__ float orientationHist[36];
    __shared__ float maxHist;
    maxHist = 0.0f;
    if(threadIdx.y*blockDim.x + threadIdx.x<36) orientationHist[threadIdx.y*blockDim.x + threadIdx.x] = 0.0f;
    __syncthreads();
    int regNumOrient = maxOrientations;
    float pixelWidth_reg = pixelWidth;

    //vector = (x+1) - (x-1), (y+1) - (y-1)
    int windowWidth = contributerWindowWidth;
    int2 gradient = {0,0};
    float temp = 0.0f;
    float2 temp2 = {0.0f,0.0f};
    float lambda_reg = pixelWidth;
    int bin = 0;
    for(int y = (keyPoint.y - windowWidth/2) + threadIdx.y; y <= keyPoint.y + windowWidth/2; y+=blockDim.y){
      for(int x = (keyPoint.x - windowWidth/2) + threadIdx.x; x <= keyPoint.x + windowWidth/2; x+=blockDim.x){
        gradient = gradients[y*imageSize.x + x];
        temp2 = {x*pixelWidth_reg - keyPoint.x,y*pixelWidth_reg - keyPoint.y};
        temp = -sqrtf(dotProduct(temp2,temp2));
        temp /= (2*lambda_reg*lambda_reg*sigma*sigma);
        bin = (int) roundf(36.0f*getTheta(gradient)/(2.0f*pi));
        atomicAdd(&orientationHist[bin],expf(temp)*sqrtf(dotProduct(gradient,gradient)));
      }
    }
    __syncthreads();
    for(int i = 0; i < 6; ++i){
      for(int b = threadIdx.y*blockDim.x + threadIdx.x; b < 36; b += blockDim.x*blockDim.y){
        temp = orientationHist[b];
        if(b > 0){
          temp += orientationHist[b-1];
        }
        else if(b < 35){
          temp += orientationHist[b+1];;
        }
        temp /= 3;
        orientationHist[b] = temp;
      }
    }
    temp = 0.0f;

    __syncthreads();
    for(int i = threadIdx.y*blockDim.x + threadIdx.x; i < 36; i+= blockDim.x*blockDim.y){
      if(maxHist < orientationHist[i]) atomicMaxFloat(&maxHist,orientationHist[i]);
    }
    if(threadIdx.x != 0 || threadIdx.y != 0) return;
    maxHist *= orientationThreshold;//% of max orientation value

    float2* bestMagWThetas = new float2[regNumOrient]();
    float2 tempMagWTheta = {0.0f,0.0f};
    int numThetas = 0;
    for(int b = 0; b < 36; ++b){
      if(orientationHist[b] < maxHist) continue;
      else if(b > 0 && orientationHist[b] < orientationHist[b-1]) continue;
      else if(b < 35 && orientationHist[b] < orientationHist[b+1]) continue;
      tempMagWTheta.x = orientationHist[b];

      if(tempMagWTheta.x < bestMagWThetas[regNumOrient-1].x) continue;
      if(b == 0){
        tempMagWTheta.y = -orientationHist[1]/(-2.0f*orientationHist[0]+orientationHist[1]);
      }
      else if(b == 35){
        tempMagWTheta.y = (orientationHist[34])/(orientationHist[34]-(2.0f*orientationHist[35]));
      }
      else{
        tempMagWTheta.y = (orientationHist[b-1]-orientationHist[b+1])/(orientationHist[b-1]-(2.0f*orientationHist[b])+orientationHist[b+1]);
      }
      tempMagWTheta.y *= (pi/36.0f);
      tempMagWTheta.y += (b*pi/18.0f);

      trickleSwap(tempMagWTheta, bestMagWThetas, 0, regNumOrient);
      ++numThetas;
    }
    thetaNumbers[blockId*regNumOrient] = (numThetas > regNumOrient) ? regNumOrient : numThetas;
    for(int t = 0; t < regNumOrient; ++t){
      if(bestMagWThetas[t].y == 0.0f) break;
      thetas[blockId*regNumOrient + t] = bestMagWThetas[t].y;
    }
    delete[] bestMagWThetas;
  }
}


//fix hog grid stuff with actual formula
__global__ void ssrlcv::fillDescriptors(unsigned long numFeatures, uint2 imageSize, Feature<SIFT_Descriptor>* features, float sigma, float pixelWidth,
  float lambda, int contributerWindowWidth, float* thetas, int* keyPointAddresses, float2* keyPointLocations, int2* gradients){

  unsigned long blockId = blockIdx.y* gridDim.x+ blockIdx.x;
  if(blockId < numFeatures){
    __shared__ ssrlcv::Feature<SIFT_Descriptor> feature;
    __shared__ float norm;
    __shared__ float bin_descriptors[4][4][8];
    bin_descriptors[threadIdx.x][threadIdx.y][threadIdx.z] = 0.0f;
    if(threadIdx.x == 0 && threadIdx.y == 0 && threadIdx.z == 0){
      feature = ssrlcv::Feature<SIFT_Descriptor>(keyPointLocations[keyPointAddresses[blockId]]);
      feature.descriptor.theta = thetas[blockId];
      feature.descriptor.sigma = sigma;
      norm = 0.0f;
    }
    __syncthreads();
    int windowWidth = contributerWindowWidth;

    /*
    FIRST DEFINE HOG GRID
    (x,y) = [(-8.5,-8.5),(8.5,8.5)]
      x' = xcos(theta) - ysin(theta) + feature.x
      y' = ycos(theta) + xsin(theta) + feature.y

    */

    float2 descriptorGridPoint = {0.0f,0.0f};

    int imageWidth = imageSize.x;
    int pixelWidth_reg = pixelWidth;
    float lambda_reg = lambda;
    int2 contribLoc = {0,0};
    float2 gradient = {0,0};
    float temp = 0.0f;
    float2 temp2 = {0.0f,0.0f};

    for(int y = (feature.loc.y-windowWidth/2) + threadIdx.y; y <= feature.loc.y + windowWidth/2; y+=blockDim.y){
      for(int x = (feature.loc.x-windowWidth/2) + threadIdx.x; x <= feature.loc.x + windowWidth/2; x+=blockDim.x){
        contribLoc.x = (((x*pixelWidth_reg - feature.loc.x)*cosf(feature.descriptor.theta)) + ((y*pixelWidth_reg - feature.loc.y)*sinf(feature.descriptor.theta)))/feature.descriptor.sigma;
        contribLoc.y = ((-(x*pixelWidth_reg - feature.loc.x)*sinf(feature.descriptor.theta)) + ((y*pixelWidth_reg - feature.loc.y)*cosf(feature.descriptor.theta)))/feature.descriptor.sigma;

        gradient = {
          gradients[(contribLoc.y + (int)feature.loc.y)*imageWidth + (contribLoc.x + (int)feature.loc.x)].x,
          gradients[(contribLoc.y + (int)feature.loc.y)*imageWidth + (contribLoc.x + (int)feature.loc.x)].y
        };
        //calculate expow
        temp2 = {contribLoc.x*pixelWidth_reg - feature.loc.x,contribLoc.y*pixelWidth_reg - feature.loc.y};
        temp = -sqrtf(dotProduct(temp2,temp2));
        temp /= (2*lambda_reg*lambda_reg*feature.descriptor.sigma*feature.descriptor.sigma);
        descriptorGridPoint.x = sqrtf(dotProduct(gradient,gradient))*expf(temp);
        descriptorGridPoint.y = getTheta(gradient,feature.descriptor.theta);
        for(int nx = 0; nx < 4; ++nx){
          for(int ny = 0; ny < 4; ++ny){
            int histx = (nx - 2.5)*0.5*lambda_reg;
            int histy = (ny - 2.5)*0.5*lambda_reg;
            if(histx - x - feature.loc.x <= 2.5 && histy - y - feature.loc.y <= 2.5 && (threadIdx.z+1)*45.0f*(pi/180.0f) > descriptorGridPoint.y){
              atomicAdd(&bin_descriptors[nx][ny][threadIdx.z],((1-(2/lambda_reg)*(histx - x - feature.loc.x))*
                (1-(2/lambda_reg)*(histy - y - feature.loc.y))*
                (1-(2/pi)*fmod(descriptorGridPoint.y-feature.descriptor.theta+2.0f*pi, 2.0f*pi)))*
                descriptorGridPoint.x);
              break;
            }
          }
        }
      }
    }
    /*
    NORMALIZE
    */
    __syncthreads();
    atomicAdd(&norm, bin_descriptors[threadIdx.x][threadIdx.y][threadIdx.z]*bin_descriptors[threadIdx.x][threadIdx.y][threadIdx.z]);
    __syncthreads();
    if(threadIdx.x == 0 && threadIdx.y == 0 && threadIdx.z == 0) norm = sqrtf(norm);
    __syncthreads();
    if(bin_descriptors[threadIdx.x][threadIdx.y][threadIdx.z] < 0.2*norm){
      temp = bin_descriptors[threadIdx.x][threadIdx.y][threadIdx.z];
      if(512*temp/norm < 255){
        feature.descriptor.values[(threadIdx.y*4 + threadIdx.x)*8 + threadIdx.z] = (unsigned char) (512*temp/norm);
      }
    }
    else{
      feature.descriptor.values[(threadIdx.y*4 + threadIdx.x)*8 + threadIdx.z]  = 102;//512*0.2*norm/norm
    }
    __syncthreads();
    if(threadIdx.x == 0 && threadIdx.y == 0 && threadIdx.z == 0) features[blockId] = feature;
  }
}
