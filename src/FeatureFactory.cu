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
    this->parentOctave = -1;
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