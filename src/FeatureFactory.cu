#include "FeatureFactory.cuh"

/*
HOST METHODS
*/
//Base feature factory


ssrlcv::FeatureFactory::FeatureFactory(){

}
ssrlcv::FeatureFactory::~FeatureFactory(){

}

ssrlcv::FeatureFactory::ScaleSpace::ScaleSpace(){
    this->depth = {0,0};
    this->octaves = nullptr;

}
ssrlcv::FeatureFactory::ScaleSpace::ScaleSpace(Image* image, int startingOctave, uint2 depth, float initialSigma, float2 sigmaMultiplier, int2 kernelSize) : 
depth(depth){ 

    printf("creating scalespace with depth {%d,%d}\n",this->depth.x,this->depth.y);

    Unity<unsigned char>* pixels = new Unity<unsigned char>(nullptr,image->pixels->numElements, gpu);
    uint2 imageSize = image->size;
    uint2 scalar = {2,2};
    MemoryState origin = image->pixels->state;
    if(origin == cpu || image->pixels->fore == cpu) image->pixels->transferMemoryTo(gpu);
    CudaSafeCall(cudaMemcpy(pixels->device, image->pixels->device, pixels->numElements*sizeof(unsigned char),cudaMemcpyDeviceToDevice));
    
    float pixelWidth = 1.0f;

    for(int i = startingOctave; i < 0; ++i){
        pixels->setData(upsample(imageSize,image->colorDepth,pixels)->device,pixels->numElements*4,gpu);   
        imageSize = imageSize*scalar;
        pixelWidth /= 2.0f;
    }
    for(int i = 0; i < startingOctave; ++i){
        pixels->setData(bin(imageSize,image->colorDepth,pixels)->device,pixels->numElements*4,gpu);   
        imageSize = imageSize/scalar;
        pixelWidth *= 2.0f;
    }   
    float* sigmas = new float[this->depth.y]();
    sigmas[0] = initialSigma;
    for(int i = 1; i < this->depth.y; ++i){
        sigmas[i] = sigmas[i-1]*sigmaMultiplier.y;
    }
    
    this->octaves = new Octave*[this->depth.x]();
    for(int i = 0; i < this->depth.x; ++i){
        this->octaves[i] = new Octave(this->depth.y,kernelSize,sigmas,pixels,imageSize,image->colorDepth,pixelWidth);

        if(i + 1 + startingOctave == 0){
            delete pixels;
            pixels = new Unity<unsigned char>(nullptr,image->pixels->numElements,gpu);
            CudaSafeCall(cudaMemcpy(pixels->device, image->pixels->device, pixels->numElements*sizeof(unsigned char),cudaMemcpyDeviceToDevice));
            if(origin == cpu) image->pixels->setMemoryState(cpu);
        }
        else if(i + 1 < this->depth.x){
            pixels->setData(bin(imageSize,image->colorDepth,pixels)->device,pixels->numElements/4,gpu);
        }
        else break;
        for(int b = 0; b < this->depth.y; ++b){
            sigmas[b]*=sigmaMultiplier.x;    
        }
        imageSize = imageSize/scalar;
        pixelWidth *= 2.0f;
    }
    delete pixels;
    delete[] sigmas;

}
ssrlcv::FeatureFactory::ScaleSpace::~ScaleSpace(){
    if(this->octaves != nullptr){
        for(int i = 0; i < this->depth.x; ++i){
            delete this->octaves[i];
        }
        delete[] this->octaves;
    }
 
}

ssrlcv::FeatureFactory::ScaleSpace::Octave::Octave(){
    this->numBlurs = 0;
    this->blurs = nullptr;
}
ssrlcv::FeatureFactory::ScaleSpace::Octave::Octave(unsigned int numBlurs, int2 kernelSize, float* sigmas, Unity<unsigned char>* pixels, uint2 size, unsigned int colorDepth, float pixelWidth) : 
numBlurs(numBlurs){
    printf("creating octave with %d blurs of size {%d,%d}\n",numBlurs,size.x,size.y);
    MemoryState origin = pixels->state;
    if(origin == cpu || pixels->fore == cpu) pixels->transferMemoryTo(gpu);
    this->blurs = new Blur*[this->numBlurs]();
    for(int i = 0; i < this->numBlurs; ++i){
        this->blurs[i] = new Blur(sigmas[i],kernelSize,pixels,size,colorDepth,pixelWidth);
    }
    if(origin == cpu) pixels->setMemoryState(cpu);
}
ssrlcv::FeatureFactory::ScaleSpace::Octave::~Octave(){
    if(this->blurs != nullptr){
        for(int i = 0; i < this->numBlurs; ++i){
            delete this->blurs[i];
        }
        delete[] this->blurs;
    } 
}

ssrlcv::FeatureFactory::ScaleSpace::Octave::Blur::Blur(){
    this->sigma = 0.0f;
    this->pixels = nullptr;
    this->colorDepth = 0;
    this->size = {0,0};
}
ssrlcv::FeatureFactory::ScaleSpace::Octave::Blur::Blur(float sigma, int2 kernelSize, Unity<unsigned char>* pixels, uint2 size, unsigned int colorDepth, float pixelWidth) : 
sigma(sigma),size(size),colorDepth(colorDepth){
    MemoryState origin = pixels->state;
    if(origin == cpu || pixels->fore == cpu) pixels->transferMemoryTo(gpu);
    kernelSize.x = ceil((float)kernelSize.x*this->sigma/pixelWidth);
    kernelSize.y = ceil((float)kernelSize.y*this->sigma/pixelWidth);
    if(kernelSize.x%2 == 0)kernelSize.x++;
    if(kernelSize.y%2 == 0)kernelSize.y++;
    float* gaussian = new float[kernelSize.y*kernelSize.x]();
    for(int y = -kernelSize.y/2, i = 0; y <= kernelSize.y/2; ++y){
        for(int x = -kernelSize.x/2; x <= kernelSize.x/2; ++x){
            gaussian[i++] = expf(-(((x*x) + (y*y))*0.5f/this->sigma/this->sigma))/(2.0f*PI*this->sigma*this->sigma);
        }
    }
    this->pixels = convolve(this->size,pixels,this->colorDepth,kernelSize,gaussian,true);
    if(origin == cpu) pixels->setMemoryState(cpu);
}
ssrlcv::FeatureFactory::ScaleSpace::Octave::Blur::~Blur(){
    if(this->pixels != nullptr) delete this->pixels;
}


ssrlcv::FeatureFactory::DOG* ssrlcv::FeatureFactory::generateDOG(ScaleSpace* scaleSpace){
    DOG* dog = new DOG();
    Unity<float>* pixelsUpper = nullptr;
    Unity<float>* pixelsLower = nullptr;
    MemoryState origin[2];
    dim3 grid = {1,1,1};
    dim3 block = {1,1,1};
    dog->depth = {scaleSpace->depth.x,scaleSpace->depth.y - 1};
    dog->octaves = new ScaleSpace::Octave*[dog->depth.x]();
    for(int o = 0; o < dog->depth.x; o++){
        dog->octaves[o] = new ScaleSpace::Octave();
        dog->octaves[o]->blurs = new ScaleSpace::Octave::Blur*[dog->depth.y]();
        pixelsLower = scaleSpace->octaves[o]->blurs[0]->pixels;
        getFlatGridBlock(pixelsLower->numElements,grid,block);
        for(int b = 0; b < dog->depth.y; ++b){
            dog->octaves[o]->blurs[b] = new ScaleSpace::Octave::Blur();
            dog->octaves[o]->blurs[b]->pixels = new Unity<float>(nullptr,pixelsLower->numElements,gpu);
            pixelsUpper = scaleSpace->octaves[o]->blurs[b+1]->pixels;
            origin[0] = pixelsLower->state;
            origin[1] = pixelsUpper->state;
            if(origin[0] == cpu) pixelsLower->transferMemoryTo(gpu);
            if(origin[1] == cpu) pixelsUpper->transferMemoryTo(gpu);
            subtractImages<<<grid,block>>>(pixelsLower->numElements,pixelsUpper->device,pixelsLower->device,dog->octaves[o]->blurs[b]->pixels->device);
            cudaDeviceSynchronize();
            CudaCheckError();
            if(origin[0] == cpu) pixelsLower->setMemoryState(cpu);
            pixelsLower = pixelsUpper;
        }
    }
    return dog;
}
ssrlcv::Unity<ssrlcv::FeatureFactory::ScaleSpace::SSKeyPoint>* ssrlcv::FeatureFactory::findExtrema(DOG* dog){
    Unity<float>* pixelsUpper = nullptr;
    Unity<float>* pixelsMiddle = nullptr;
    Unity<float>* pixelsLower = nullptr;
    dim3 grid2D = {1,1,1};
    dim3 block2D = {3,3,3};
    dim3 grid = {1,1,1};
    dim3 block = {1,1,1};
    MemoryState origin[3];
    int* maximaAddresses = nullptr;
    int totalMaxima = 0;
    ScaleSpace::SSKeyPoint** maxima2D = new ScaleSpace::SSKeyPoint*[dog->depth.x*(dog->depth.y - 2)];
    int* numMaxima = new int[dog->depth.x*(dog->depth.y-2)]();
    int maximaAtDepth = 0;
    unsigned int colorDepth = dog->octaves[0]->blurs[0]->colorDepth;
    for(int o = 0,kpd = 0; o < dog->depth.x; o++){
        pixelsLower = dog->octaves[o]->blurs[0]->pixels;
        getFlatGridBlock(pixelsLower->numElements,grid,block);
        getGrid(pixelsLower->numElements,grid2D);
        if(maximaAddresses != nullptr) CudaSafeCall(cudaFree(maximaAddresses));
        CudaSafeCall(cudaMalloc((void**)&maximaAddresses,pixelsLower->numElements*sizeof(int)));
        for(int b = 1; b < dog->depth.y - 1; ++b,++kpd){
            pixelsMiddle = dog->octaves[o]->blurs[b]->pixels;
            pixelsUpper = dog->octaves[o]->blurs[b+1]->pixels;
            origin[0] = pixelsLower->state;
            origin[1] = pixelsUpper->state;
            origin[2] = pixelsUpper->state;
            if(origin[0] == cpu) pixelsLower->transferMemoryTo(gpu);
            if(origin[1] == cpu) pixelsMiddle->transferMemoryTo(gpu);
            if(origin[2] == cpu) pixelsUpper->transferMemoryTo(gpu);
            findMaxima<<<grid2D,block2D,colorDepth*sizeof(float)>>>(dog->octaves[o]->blurs[b]->size, colorDepth,pixelsUpper->device,pixelsMiddle->device,pixelsLower->device,maximaAddresses);
            cudaDeviceSynchronize();
            CudaCheckError();

            thrust::device_ptr<int> addr(maximaAddresses);

            thrust::device_ptr<int> new_end = thrust::remove(addr, addr + pixelsLower->numElements,-1);
            cudaDeviceSynchronize();
            CudaCheckError();
            maximaAtDepth = new_end - addr;

            totalMaxima += maximaAtDepth;
            numMaxima[kpd+1] = totalMaxima;

            CudaSafeCall(cudaMalloc((void**)&maxima2D[kpd],maximaAtDepth*sizeof(ScaleSpace::SSKeyPoint)));

            fillMaxima<<<grid,block>>>(maximaAtDepth,dog->octaves[o]->blurs[b]->size,dog->octaves[o]->blurs[b]->colorDepth,{o,b},dog->octaves[o]->blurs[b]->sigma,maximaAddresses,pixelsMiddle->device,maxima2D[kpd]);
            cudaDeviceSynchronize();
            CudaCheckError();

            if(origin[0] == cpu) pixelsLower->setMemoryState(cpu);
            pixelsLower = pixelsMiddle;
            pixelsMiddle = pixelsUpper;
        }
    }
    CudaSafeCall(cudaFree(maximaAddresses));
    cudaDeviceSynchronize();
    Unity<ScaleSpace::SSKeyPoint>* maxima = new Unity<ScaleSpace::SSKeyPoint>(nullptr,totalMaxima,gpu);
    for(int i = 0; i < dog->depth.x*(dog->depth.y-2); ++i){
        if(i == dog->depth.x*(dog->depth.y-2)-1){
            CudaSafeCall(cudaMemcpy(maxima->device + numMaxima[i],maxima2D[i],(totalMaxima-numMaxima[i])*sizeof(ScaleSpace::SSKeyPoint),cudaMemcpyDeviceToDevice));
        }
        else if(numMaxima[i+1] - numMaxima[i] != 0){
            CudaSafeCall(cudaMemcpy(maxima->device + numMaxima[i],maxima2D[i],(numMaxima[i+1]-numMaxima[i])*sizeof(ScaleSpace::SSKeyPoint),cudaMemcpyDeviceToDevice));
        }
        CudaSafeCall(cudaFree(maxima2D[i]));
    }  
    delete[] maxima2D;
    delete numMaxima;
    std::cout<<"found "<<totalMaxima<<" keypoints in initial scaleSpace search"<<std::endl;
    return maxima;
}
void ssrlcv::FeatureFactory::refineSubPixel(DOG* dog, Unity<ScaleSpace::SSKeyPoint>* extrema){

}
void ssrlcv::FeatureFactory::removeNoise(DOG* dog, Unity<ScaleSpace::SSKeyPoint>* extrema, float noiseThreshold){

}
void ssrlcv::FeatureFactory::removeEdges(DOG* dog, Unity<ScaleSpace::SSKeyPoint>* extrema, float edgeThreshold){

}


ssrlcv::Unity<float3>* ssrlcv::FeatureFactory::findKeyPoints(Image* image, int startingOctave, uint2 scaleSpaceDim, float initialSigma, float2 sigmaMultiplier, 
int2 kernelSize, float noiseThreshold, float edgeThreshold, bool subPixel){
    ScaleSpace* scaleSpace = new ScaleSpace(image,startingOctave,scaleSpaceDim,initialSigma,sigmaMultiplier,kernelSize);
    std::cout<<"creating dog from scalespace"<<std::endl;
    DOG* dog = generateDOG(scaleSpace);
    std::cout<<"creating scale space keypoints"<<std::endl;
    delete scaleSpace;
    std::cout<<"creating scale space keypoints"<<std::endl;
    Unity<ScaleSpace::SSKeyPoint>* scaleSpaceKeyPoints = findExtrema(dog);
    if(subPixel) refineSubPixel(dog,scaleSpaceKeyPoints);
    removeNoise(dog,scaleSpaceKeyPoints,noiseThreshold);
    removeEdges(dog,scaleSpaceKeyPoints,edgeThreshold);    
    if(scaleSpaceKeyPoints->state == cpu) scaleSpaceKeyPoints->transferMemoryTo(gpu);

    Unity<float3>* keyPoints = new Unity<float3>(nullptr,scaleSpaceKeyPoints->numElements,gpu);

    dim3 grid = {1,1,1};
    dim3 block = {1,1,1};
    getFlatGridBlock(scaleSpaceKeyPoints->numElements,grid,block);
    convertSSKPToLKP<<<grid,block>>>(scaleSpaceKeyPoints->numElements,keyPoints->device,scaleSpaceKeyPoints->device);
    cudaDeviceSynchronize();
    CudaCheckError();
    delete dog;
    delete scaleSpaceKeyPoints;
    return keyPoints;
} 




__device__ __forceinline__ float ssrlcv::atomicMinFloat (float * addr, float value){
  float old;
  old = (value >= 0) ? __int_as_float(atomicMin((int *)addr, __float_as_int(value))) :
    __uint_as_float(atomicMax((unsigned int *)addr, __float_as_uint(value)));
  return old;
}
__device__ __forceinline__ float ssrlcv::atomicMaxFloat (float * addr, float value){
  float old;
  old = (value >= 0) ? __int_as_float(atomicMax((int *)addr, __float_as_int(value))) :
    __uint_as_float(atomicMin((unsigned int *)addr, __float_as_uint(value)));
  return old;
}

__global__ void ssrlcv::subtractImages(unsigned int numPixels, float* pixelsUpper, float* pixelsLower, float* pixelsOut){
    unsigned int globalID = (blockIdx.y* gridDim.x+ blockIdx.x)*blockDim.x + threadIdx.x;
    if(globalID < numPixels) pixelsOut[globalID] = pixelsUpper[globalID] - pixelsLower[globalID];
}

//optimize
__global__ void ssrlcv::findMaxima(uint2 imageSize, unsigned int colorDepth, float* pixelsUpper, float* pixelsMiddle, float* pixelsLower, int* maxima){
    int blockId = blockIdx.y* gridDim.x+ blockIdx.x;
    int x = (blockId%(imageSize.x*colorDepth));
    int y = (blockId/(imageSize.x*colorDepth));
    if(x > 0 && y > 0 && x < imageSize.x - 1 && y < imageSize.y - 1){
        x += (threadIdx.x - 1);
        y += (threadIdx.y - 1);
        extern __shared__ float maximumValue[];
        for(int i = 0; i < colorDepth; ++i) maximumValue[i] = 0.0f;
        __syncthreads();
        float* value = new float[colorDepth];
        for(int i = 0; i < colorDepth; ++i){
            if(threadIdx.z == 0){
                value[i] = pixelsLower[y*(imageSize.x*colorDepth) + x*colorDepth + i];
            }
            else if(threadIdx.z == 1){
                value[i] = pixelsMiddle[y*(imageSize.x*colorDepth) + x*colorDepth + i];
            }
            else{
                value[i] = pixelsUpper[y*(imageSize.x*colorDepth) + x*colorDepth + i];
            }
            atomicMaxFloat(&maximumValue[i],value[i]);
        }
        __syncthreads();
        if(threadIdx.x == 1 && threadIdx.y == 1 && threadIdx.z == 1){
            for(int i = 0; i < colorDepth; ++i){
                if(maximumValue[i] == value[i]){
                    maxima[blockId] = blockId;
                    delete[] value;
                    return;
                }
            }
            maxima[blockId] = -1;
        }
        delete[] value;
    }
}
__global__ void ssrlcv::fillMaxima(int numKeyPoints, uint2 imageSize, unsigned int colorDepth, int2 ssLoc, float sigma, int* maximaAddresses, float* pixels, FeatureFactory::ScaleSpace::SSKeyPoint* scaleSpaceKP){
    int globalID = (blockIdx.y* gridDim.x+ blockIdx.x)*blockDim.x + threadIdx.x;
    if(globalID < numKeyPoints){
        int index = maximaAddresses[globalID];
        float2 loc = {(float)(globalID%(imageSize.x*colorDepth)),(float)(globalID/(imageSize.x*colorDepth))};
        FeatureFactory::ScaleSpace::SSKeyPoint sskp = {ssLoc.x,ssLoc.y,loc,pixels[index],sigma};
        scaleSpaceKP[globalID] = sskp;
    }
}
__global__ void ssrlcv::refineToSubPixel(uint2 imageSize, unsigned int colorDepth, float* pixelsUpper, float* pixelsMiddle, float* pixelsLower, FeatureFactory::ScaleSpace::SSKeyPoint* scaleSpaceKP){

}
__global__ void ssrlcv::flagNoise(uint2 imageSize, unsigned int colorDepth, FeatureFactory::ScaleSpace::SSKeyPoint* scaleSpaceKP, float threshold){

}
__global__ void ssrlcv::flagEdges(uint2 imageSize, unsigned int colorDepth, FeatureFactory::ScaleSpace::SSKeyPoint* scaleSpaceKP, float threshold){

}




__global__ void ssrlcv::convertSSKPToLKP(unsigned int numKeyPoints, float3* localizedKeyPoints, ssrlcv::FeatureFactory::ScaleSpace::SSKeyPoint* scaleSpaceKP){
    unsigned int globalID = (blockIdx.y* gridDim.x+ blockIdx.x)*blockDim.x + threadIdx.x;
    if(globalID < numKeyPoints){
        FeatureFactory::ScaleSpace::SSKeyPoint kp = scaleSpaceKP[globalID];
        localizedKeyPoints[globalID] = {kp.loc.x,kp.loc.y,kp.sigma};
    }
}
