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
    DOG* dog = nullptr;
    Unity<float>* pixelsUpper = nullptr;
    Unity<float>* pixelsLower = nullptr;
    MemoryState origin[2];
    dim3 grid = {1,1,1};
    dim3 block = {1,1,1};
    dog->depth = {scaleSpace->depth.x,scaleSpace->depth.y - 1};
    dog->octaves = new ScaleSpace::Octave*[dog->depth.x];
    for(int o = 0; o < dog->depth.x; o++){
        dog->octaves[o]->blurs = new ScaleSpace::Octave::Blur*[dog->depth.y];
        pixelsLower = scaleSpace->octaves[o]->blurs[0]->pixels;
        getFlatGridBlock(pixelsLower->numElements,grid,block);
        for(int b = 0; b < dog->depth.y; ++b){
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
    
}
void ssrlcv::FeatureFactory::refineSubPixel(DOG* dog, Unity<ScaleSpace::SSKeyPoint>* extrema){

}
void ssrlcv::FeatureFactory::removeNoise(DOG* dog, Unity<ScaleSpace::SSKeyPoint>* extrema, float noiseThreshold){

}
void ssrlcv::FeatureFactory::removeEdges(DOG* dog, Unity<ScaleSpace::SSKeyPoint>* extrema){

}


ssrlcv::Unity<float3>* ssrlcv::FeatureFactory::findKeyPoints(Image* image, int startingOctave, uint2 scaleSpaceDim, float initialSigma, float2 sigmaMultiplier, int2 kernelSize,float noiseThreshold, bool subPixel){
    ScaleSpace* scaleSpace = new ScaleSpace(image,startingOctave,scaleSpaceDim,initialSigma,sigmaMultiplier,kernelSize);
    DOG* dog = generateDOG(scaleSpace);
    delete scaleSpace;

    Unity<ScaleSpace::SSKeyPoint>* scaleSpaceKeyPoints = findExtrema(dog);
    if(subPixel) refineSubPixel(dog,scaleSpaceKeyPoints);
    removeNoise(dog,scaleSpaceKeyPoints,noiseThreshold);
    removeEdges(dog,scaleSpaceKeyPoints);    
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






__global__ void ssrlcv::subtractImages(unsigned int numPixels, float* pixelsUpper, float* pixelsLower, float* pixelsOut){
    unsigned int globalID = (blockIdx.y* gridDim.x+ blockIdx.x)*blockDim.x + threadIdx.x;
    if(globalID < numPixels) pixelsOut[globalID] = pixelsUpper[globalID] - pixelsLower[globalID];
}

__global__ void ssrlcv::convertSSKPToLKP(unsigned int numKeyPoints, float3* localizedKeyPoints, ssrlcv::FeatureFactory::ScaleSpace::SSKeyPoint* scaleSpaceKP){
    unsigned int globalID = (blockIdx.y* gridDim.x+ blockIdx.x)*blockDim.x + threadIdx.x;
    if(globalID < numKeyPoints){
        FeatureFactory::ScaleSpace::SSKeyPoint kp = scaleSpaceKP[globalID];
        localizedKeyPoints[globalID] = {kp.loc.x,kp.loc.y,kp.sigma};
    }
}
