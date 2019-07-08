// #include "FeatureFactory.cuh"
//
// __constant__ float matchTreshold = 0.1;
// __constant__ float pi = 3.1415926535897932384626433832795028841971693993751058209749445923078164062;
// __constant__ int2 immediateNeighbors[9] = {
//   {-1, -1},
//   {-1, 0},
//   {-1, 1},
//   {0, -1},
//   {0, 0},
//   {0, 1},
//   {1, -1},
//   {1, 0},
//   {1, 1}
// };
//
// /*
// DEVICE METHODS
// */
// __device__ __forceinline__ unsigned long getGlobalIdx_2D_1D(){
//   unsigned long blockId = blockIdx.y * gridDim.x + blockIdx.x;
//   unsigned long threadId = blockId * blockDim.x + threadIdx.x;
//   return threadId;
// }
// __device__ __forceinline__ unsigned char rgbToBW(const uchar3 &color){
//   return (color.x/4) + (color.y/2) + (color.z/4);
// }
// __device__ __forceinline__ float getMagnitude(const int2 &vector){
//   return sqrtf(dotProduct(vector, vector));
// }
// __device__ __forceinline__ float getTheta(const int2 &vector){
//   float theta = atan2f((float)vector.y, (float)vector.x) + pi;
//   return fmod(theta,2.0f*pi);
// }
// __device__ __forceinline__ float getTheta(const float2 &vector){
//   float theta = atan2f(vector.y, vector.x) + pi;
//   return fmod(theta,2.0f*pi);
// }
// __device__ __forceinline__ float getTheta(const float2 &vector, const float &offset){
//   float theta = (atan2f(vector.y, vector.x) + pi) - offset;
//   return fmod(theta + 2.0f*pi,2.0f*pi);
// }
// __device__ void trickleSwap(const float2 &compareWValue, float2* &arr, int index, const int &length){
//   for(int i = index; i < length; ++i){
//     if(compareWValue.x > arr[i].x){
//       float2 temp = arr[i];
//       arr[i] = compareWValue;
//       if((temp.x == 0.0f && temp.y == 0.0f)|| index + 1 == length) return;
//       return trickleSwap(temp, arr, index + 1, length);
//     }
//   }
// }
// __device__ __forceinline__ int4 getOrientationContributers(const int2 &loc, const int2 &imageSize){
//   int4 orientationContributers;
//   long pixelIndex = loc.y*imageSize.x + loc.x;
//   orientationContributers.x = (loc.x == imageSize.x - 1) ? -1 : pixelIndex + 1;
//   orientationContributers.y = (loc.x == 0) ? -1 : pixelIndex - 1;
//   orientationContributers.z = (loc.y == imageSize.y - 1) ? -1 : (loc.y + 1)*imageSize.x + loc.x;
//   orientationContributers.w = (loc.y == 0) ? -1 : (loc.y - 1)*imageSize.x + loc.x;
//   return orientationContributers;
// }
// __device__ __forceinline__ int floatToOrderedInt(float floatVal){
//  int intVal = __float_as_int( floatVal );
//  return (intVal >= 0 ) ? intVal : intVal ^ 0x7FFFFFFF;
// }
// __device__ __forceinline__ float orderedIntToFloat(int intVal){
//  return __int_as_float( (intVal >= 0) ? intVal : intVal ^ 0x7FFFFFFF);
// }
// __device__ __forceinline__ float atomicMinFloat (float * addr, float value) {
//   float old;
//   old = (value >= 0) ? __int_as_float(atomicMin((int *)addr, __float_as_int(value))) :
//     __uint_as_float(atomicMax((unsigned int *)addr, __float_as_uint(value)));
//   return old;
// }
// __device__ __forceinline__ float atomicMaxFloat (float * addr, float value) {
//   float old;
//   old = (value >= 0) ? __int_as_float(atomicMax((int *)addr, __float_as_int(value))) :
//     __uint_as_float(atomicMin((unsigned int *)addr, __float_as_uint(value)));
//   return old;
// }
// __device__ __forceinline__ float modulus(const float &x, const float &y){
//     float z = x;
//     int n;
//     if(z < 0){
//         n = (int)((-z)/y)+1;
//         z += n*y;
//     }
//     n = (int)(z/y);
//     z -= n*y;
//     return z;
// }
// __device__ __forceinline__ float2 rotateAboutPoint(const int2 &loc, const float &theta, const float2 &origin){
//   float2 rotatedPoint = {(float) loc.x, (float) loc.y};
//   rotatedPoint = rotatedPoint - origin;
//   float2 temp = rotatedPoint;
//
//   rotatedPoint.x = (temp.x*cosf(theta)) - (temp.y*sinf(theta)) + origin.x;
//   rotatedPoint.y = (temp.x*sinf(theta)) + (temp.y*cosf(theta)) + origin.y;
//
//   return rotatedPoint;
// }
// /*
// KERNELS
// */
// __global__ void initFeatureArrayNoZeros(unsigned int totalFeatures, ssrlcv::Image_Descriptor image, ssrlcv::SIFT_Feature* features, int* numFeatureExtractor, unsigned char* pixels){
//   ssrlcv::Image_Descriptor parentRef = image;
//   int2 locationInParent = {(int)blockIdx.y,(int)blockIdx.x};
//   bool real = ((locationInParent.x - 12) >= 0 && (locationInParent.y - 12) >= 0) && ((locationInParent + 12) < parentRef.size && pixels[blockIdx.y*gridDim.x + blockIdx.x] != 0);
//   features[blockIdx.y*gridDim.x + blockIdx.x] = ssrlcv::SIFT_Feature(locationInParent, parentRef.id, real);
//   numFeatureExtractor[blockIdx.y*gridDim.x + blockIdx.x] = real;
// }
// __global__ void initFeatureArray(unsigned int totalFeatures, ssrlcv::Image_Descriptor image, ssrlcv::SIFT_Feature* features, int* numFeatureExtractor){
//   ssrlcv::Image_Descriptor parentRef = image;
//   int2 locationInParent = {(int)blockIdx.y,(int)blockIdx.x};
//   bool real = ((locationInParent.x - 12) >= 0 && (locationInParent.y - 12) >= 0) && ((locationInParent + 12) < parentRef.size);
//   features[blockIdx.y*gridDim.x + blockIdx.x] = ssrlcv::SIFT_Feature(locationInParent, parentRef.id, real);
//   numFeatureExtractor[blockIdx.y*gridDim.x + blockIdx.x] = real;
// }
// __global__ void computeThetas(unsigned int totalFeatures, ssrlcv::Image_Descriptor image, int numOrientations, unsigned char* pixels, ssrlcv::SIFT_Feature* features, ssrlcv::SIFT_Descriptor* descriptors){
//   unsigned long blockId = blockIdx.y * gridDim.x + blockIdx.x;
//   if(blockId < totalFeatures){
//     ssrlcv::SIFT_Feature feature = features[blockId];
//     __shared__ int2 orientationVectors[9];
//     ssrlcv::Image_Descriptor parentRef = image;
//     int regNumOrient = numOrientations;
//
//     //vector = (x+1) - (x-1), (y+1) - (y-1)
//     int4 orientationContributers = getOrientationContributers(feature.loc + immediateNeighbors[threadIdx.x], parentRef.size);
//     orientationVectors[threadIdx.x].x = ((int)pixels[orientationContributers.x]) - ((int)pixels[orientationContributers.y]);
//     orientationVectors[threadIdx.x].y = ((int)pixels[orientationContributers.z]) - ((int)pixels[orientationContributers.w]);
//
//     __syncthreads();
//
//
//     if(threadIdx.x != 0) return;
//     float2* bestMagWThetas = new float2[numOrientations];
//     for(int i = 0; i < numOrientations; ++i) bestMagWThetas[i] = {0.0f,0.0f};
//     float2 tempMagWTheta = {0.0f,0.0f};
//     int2 currentOrientationVector = {0,0};
//     int2 currentLoc = feature.loc + immediateNeighbors[threadIdx.x];
//     for(int i = 0; i < 9; ++i){
//       currentOrientationVector = orientationVectors[i];
//       tempMagWTheta = {getMagnitude(currentOrientationVector), getTheta(currentOrientationVector)};
//       trickleSwap(tempMagWTheta, bestMagWThetas, 0, regNumOrient);
//     }
//     for(int i = 0; i < regNumOrient; ++i){
//       descriptors[blockId*regNumOrient + i] = ssrlcv::SIFT_Descriptor(bestMagWThetas[i].y);
//     }
//     features[blockId].descriptorIndex = blockId*regNumOrient;
//     delete[] bestMagWThetas;
//   }
// }
// __global__ void fillSIFTDescriptorsDensly(unsigned int totalFeatures, ssrlcv::Image_Descriptor image, int numOrientations, unsigned char* pixels, ssrlcv::SIFT_Feature* features, ssrlcv::SIFT_Descriptor* descriptors){
//   unsigned long blockId = blockIdx.y * gridDim.x + blockIdx.x;
//   int featureIndex = blockId/numOrientations;
//   if(blockId < totalFeatures*numOrientations){
//
//     ssrlcv::Feature feature = features[featureIndex];
//     __shared__ float2 descriptorGrid[18][18];
//     __shared__ float localMax;
//     __shared__ float localMin;
//     descriptorGrid[threadIdx.x][threadIdx.y] = {0.0f,0.0f};
//     localMax = 0.0f;
//     localMin = FLT_MAX;
//     __syncthreads();
//     /*
//     FIRST DEFINE HOG GRID
//     (x,y) = [(-8.5,-8.5),(8.5,8.5)]
//       x' = xcos(theta) - ysin(theta) + feature.x
//       y' = ycos(theta) + xsin(theta) + feature.y
//
//     */
//     ssrlcv::Image_Descriptor parentRef = image;
//     float theta = descriptors[blockId].theta;
//
//     float2 descriptorGridPoint = {0.0f, 0.0f};
//     descriptorGridPoint.x = (((threadIdx.x - 8.5f)*cosf(theta)) - ((threadIdx.y - 8.5f)*sinf(theta))) + feature.loc.x;
//     descriptorGridPoint.y = (((threadIdx.x - 8.5f)*sinf(theta)) + ((threadIdx.y - 8.5f)*cosf(theta))) + feature.loc.y;
//     float2 pixValueLoc = {((threadIdx.x - 8.5f) + feature.loc.x), ((threadIdx.y - 8.5f) + feature.loc.y)};
//
//     float newValue = 0.0f;
//     ulong4 pixContributers;
//     pixContributers.x = (((int)(pixValueLoc.y - 0.5f))*parentRef.size.x) + ((int)(pixValueLoc.x - 0.5f));
//     pixContributers.y = (((int)(pixValueLoc.y - 0.5f))*parentRef.size.x) + ((int)(pixValueLoc.x + 0.5f));
//     pixContributers.z = (((int)(pixValueLoc.y + 0.5f))*parentRef.size.x) + ((int)(pixValueLoc.x - 0.5f));
//     pixContributers.w = (((int)(pixValueLoc.y + 0.5f))*parentRef.size.x) + ((int)(pixValueLoc.x + 0.5f));
//     newValue += pixels[pixContributers.x];
//     newValue += pixels[pixContributers.y];
//     newValue += pixels[pixContributers.z];
//     newValue += pixels[pixContributers.w];
//     newValue /= 4.0f;
//
//     descriptorGrid[threadIdx.x][threadIdx.y].x = newValue;
//     __syncthreads();
//
//     float2 grad = {0.0f, 0.0f};//magnitude, orientation
//     if(threadIdx.x > 0 && threadIdx.x < 17 && threadIdx.y > 0 && threadIdx.y < 17){
//       float2 vector = {descriptorGrid[threadIdx.x + 1][threadIdx.y].x -
//         descriptorGrid[threadIdx.x - 1][threadIdx.y].x,
//         descriptorGrid[threadIdx.x][threadIdx.y + 1].x -
//         descriptorGrid[threadIdx.x][threadIdx.y - 1].x};
//
//       //expf stuff is the gaussian weighting function
//       float expPow = -sqrtf(dotProduct(descriptorGridPoint - feature.loc,descriptorGridPoint - feature.loc))/16.0f;
//       grad.x = sqrtf(dotProduct(vector, vector))*expf(expPow);
//       grad.y = getTheta(vector,theta);
//     }
//     __syncthreads();
//     /*
//     NOW CREATE HOG AND GET DESCRIPTOR
//     */
//     descriptorGrid[threadIdx.x][threadIdx.y] = grad;
//     if(threadIdx.x >= 4 || threadIdx.y >= 4) return;
//     int2 gradDomain = {((int) threadIdx.x*4) + 1, ((int) threadIdx.x + 1)*4};
//     int2 gradRange = {((int) threadIdx.y*4) + 1, ((int) threadIdx.y + 1)*4};
//
//     float bin_descriptors[8] = {0.0f};
//     float rad45 = 45.0f*(pi/180.0f);
//     for(int x = gradDomain.x; x <= gradDomain.y; ++x){
//       for(int y = gradRange.x; y <= gradRange.y; ++y){
//         for(int o = 1; o < 9; ++o){
//           if(o*rad45 > descriptorGrid[x][y].y){
//             bin_descriptors[o - 1] += descriptorGrid[x][y].x;
//             break;
//           }
//         }
//       }
//     }
//     __syncthreads();
//     /*
//     NORMALIZE
//     */
//     for(int d = 0; d < 8; ++d){
//       atomicMinFloat(&localMin, bin_descriptors[d]);
//       atomicMaxFloat(&localMax, bin_descriptors[d]);
//     }
//     __syncthreads();
//     for(int d = 0; d < 8; ++d){
//       descriptors[blockId].descriptor[(threadIdx.y*4 + threadIdx.x)*8 + d] =
//         __float2int_rn(255*(bin_descriptors[d]-localMin)/(localMax-localMin));
//     }
//   }
// }
//
// /*
// START OF HOST METHODS
// */
// //Base feature factory
//
//
// ssrlcv::FeatureFactory::FeatureFactory(){
//   this->image = nullptr;
//   this->allowZeros = true;
// }
// void ssrlcv::FeatureFactory::setImage(Image* image){
//   this->image = image;
//   if(this->image->numDescriptorsPerFeature == 0){
//     this->image->numDescriptorsPerFeature = 1;
//   }
// }
// ssrlcv::SIFT_FeatureFactory::SIFT_FeatureFactory(){
//   this->allowZeros = true;
//   this->image = nullptr;
//   this->numOrientations = 0;
// }
// ssrlcv::SIFT_FeatureFactory::SIFT_FeatureFactory(bool allowZeros){
//   this->allowZeros = true;
//   this->numOrientations = 0;
//   this->image = nullptr;
// }
//
// ssrlcv::SIFT_FeatureFactory::SIFT_FeatureFactory(int numOrientations){
//   this->numOrientations = numOrientations;
//   this->image = nullptr;
// }
// ssrlcv::SIFT_FeatureFactory::SIFT_FeatureFactory(bool allowZeros, int numOrientations){
//   this->allowZeros = allowZeros;
//   this->numOrientations = numOrientations;
// }
// void ssrlcv::SIFT_FeatureFactory::setZeroAllowance(bool allowZeros){
//   this->allowZeros = allowZeros;
// }
// void ssrlcv::SIFT_FeatureFactory::setNumOrientations(int numOrientations){
//   this->numOrientations = numOrientations;
//   if(this->image != nullptr) this->image->numDescriptorsPerFeature = numOrientations;
// }
//
// void ssrlcv::SIFT_FeatureFactory::generateDescriptorsDensly(ssrlcv::SIFT_Feature* features_device, ssrlcv::SIFT_Descriptor* descriptors_device){
//   dim3 grid = {1,1,1};
//   dim3 block = {9,1,1};
//   getGrid(this->image->numFeatures, grid);
//   std::cout<<"computing thetas for feature descriptors..."<<std::endl;
//   clock_t timer = clock();
//   computeThetas<<<grid, block>>>(this->image->numFeatures, this->image->descriptor,
//     this->image->numDescriptorsPerFeature, this->image->pixels_device,
//     features_device, descriptors_device);
//   cudaDeviceSynchronize();
//   CudaCheckError();
//   printf("done in %f seconds.\n\n",((float) clock() -  timer)/CLOCKS_PER_SEC);
//
//   block = {18,18,1};
//   getGrid(this->image->numFeatures*this->numOrientations, grid);
//   std::cout<<"generating feature descriptors..."<<std::endl;
//   timer = clock();
//   fillSIFTDescriptorsDensly<<<grid,block>>>(this->image->numFeatures, this->image->descriptor,
//     this->image->numDescriptorsPerFeature, this->image->pixels_device,
//     features_device, descriptors_device);
//   cudaDeviceSynchronize();
//   CudaCheckError();
//   printf("done in %f seconds.\n\n",((float) clock() -  timer)/CLOCKS_PER_SEC);
// }
// void ssrlcv::SIFT_FeatureFactory::generateFeaturesDensly(){
//   std::cout<<"generating features"<<std::endl;
//   this->image->feature_size = sizeof(ssrlcv::SIFT_Feature);
//   this->image->featureDescriptor_size = sizeof(ssrlcv::SIFT_Descriptor);
//   if(this->image->totalPixels == 0){
//     std::cout<<"ERROR must have pixels for generating features"<<std::endl;
//     exit(-1);
//   }
//   if(this->image->colorDepth != 1){
//     this->image->convertToBW();
//   }
//   if(this->image->numDescriptorsPerFeature == 0){
//     this->image->numDescriptorsPerFeature = 1;
//   }
//   if(this->image->arrayStates[0] == cpu){
//     this->image->setPixelState(gpu);
//     std::cout<<"NOTE pixels are now on GPU"<<std::endl;
//   }
//
//
//   ssrlcv::SIFT_Feature* features_device_temp = nullptr;
//   ssrlcv::SIFT_Feature* features_device = nullptr;
//   ssrlcv::SIFT_Descriptor* descriptors_device = nullptr;
//   CudaSafeCall(cudaMalloc((void**)&features_device_temp, this->image->totalPixels*sizeof(ssrlcv::SIFT_Feature)));
//
//   int* numFeatureExtractor;
//   CudaSafeCall(cudaMalloc((void**)&numFeatureExtractor, this->image->totalPixels*sizeof(int)));
//   this->image->numFeatures = this->image->totalPixels;
//
//   dim3 grid = {(unsigned int)this->image->descriptor.size.x,(unsigned int)this->image->descriptor.size.y,1};
//   dim3 block = {1,1,1};
//   std::cout<<"initializing DSIFT feature array with "<<this->image->numFeatures<<" features..."<<std::endl;
//   clock_t timer = clock();
//   if(this->allowZeros){
//     initFeatureArray<<<grid, block>>>(this->image->numFeatures, this->image->descriptor, features_device_temp, numFeatureExtractor);
//   }
//   else{
//     initFeatureArrayNoZeros<<<grid, block>>>(this->image->numFeatures, this->image->descriptor, features_device_temp, numFeatureExtractor, this->image->pixels_device);
//   }
//   cudaDeviceSynchronize();
//   CudaCheckError();
//
//   thrust::device_ptr<int> sum(numFeatureExtractor);
//   thrust::inclusive_scan(sum, sum + this->image->numFeatures, sum);
//   unsigned long beforeCompaction = this->image->numFeatures;
//   CudaSafeCall(cudaMemcpy(&(this->image->numFeatures),numFeatureExtractor + (beforeCompaction - 1), sizeof(int), cudaMemcpyDeviceToHost));
//   CudaSafeCall(cudaFree(numFeatureExtractor));
//   printf("numFeatures after eliminating ambiguity = %d\n",this->image->numFeatures);
//
//   CudaSafeCall(cudaMalloc((void**)&features_device, this->image->numFeatures*sizeof(ssrlcv::SIFT_Feature)));
//
//   thrust::device_ptr<ssrlcv::SIFT_Feature> arrayToCompact(features_device_temp);
//   thrust::device_ptr<ssrlcv::SIFT_Feature> arrayOut(features_device);
//   thrust::copy_if(arrayToCompact, arrayToCompact + beforeCompaction, arrayOut, feature_is_inbounds());
//   CudaCheckError();
//   CudaSafeCall(cudaFree(features_device_temp));
//
//
//   CudaSafeCall(cudaMalloc((void**)&descriptors_device, this->image->numFeatures*this->numOrientations*sizeof(ssrlcv::SIFT_Descriptor)));
//
//   printf("done in %f seconds.\n\n",((float) clock() -  timer)/CLOCKS_PER_SEC);
//
//   this->generateDescriptorsDensly(features_device, descriptors_device);
//
//   if(this->image->arrayStates[1] == both || this->image->arrayStates[1] == cpu){
//     this->image->features = new ssrlcv::SIFT_Feature[this->image->numFeatures];
//     CudaSafeCall(cudaMemcpy(this->image->features, features_device, this->image->numFeatures*sizeof(ssrlcv::SIFT_Feature), cudaMemcpyDeviceToHost));
//   }
//   else{
//     this->image->features_device = features_device;
//   }
//   if(this->image->arrayStates[1] == cpu){
//     CudaSafeCall(cudaFree(features_device));
//   }
//   if(this->image->arrayStates[2] == both || this->image->arrayStates[2] == cpu){
//     this->image->featureDescriptors = new ssrlcv::SIFT_Descriptor[this->image->numFeatures];
//     CudaSafeCall(cudaMemcpy(this->image->featureDescriptors, descriptors_device, this->image->numFeatures*this->numOrientations*sizeof(ssrlcv::SIFT_Descriptor), cudaMemcpyDeviceToHost));
//   }
//   else{
//     this->image->featureDescriptors_device = descriptors_device;
//   }
//   if(this->image->arrayStates[2] == cpu){
//     CudaSafeCall(cudaFree(descriptors_device));
//   }
// }
