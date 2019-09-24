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
    if(image->colorDepth != 1){
        convertToBW(pixels,image->colorDepth);
    }
    float pixelWidth = 1.0f;

    for(int i = startingOctave; i < 0; ++i){
        pixels->setData(upsample(imageSize,1,pixels)->device,pixels->numElements*4,gpu);   
        imageSize = imageSize*scalar;
        pixelWidth /= 2.0f;
    }
    for(int i = 0; i < startingOctave; ++i){
        pixels->setData(bin(imageSize,1,pixels)->device,pixels->numElements/4,gpu);   
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
        this->octaves[i] = new Octave(i,this->depth.y,kernelSize,sigmas,pixels,imageSize,pixelWidth);
        if(i + 1 < this->depth.x){
            pixels->setData(bin(imageSize,1,pixels)->device,pixels->numElements/4,gpu);
            imageSize = imageSize/scalar;
            pixelWidth *= 2.0f;
            for(int b = 0; b < this->depth.y; ++b){
                sigmas[b]*=sigmaMultiplier.x;    
            }
        }
        cudaDeviceSynchronize();
        CudaCheckError();
    }
    delete pixels;
    delete[] sigmas;

}
void ssrlcv::FeatureFactory::ScaleSpace::convertToDOG(){
    Unity<float>* pixelsUpper = nullptr;
    Unity<float>* pixelsLower = nullptr;
    MemoryState origin[2];
    dim3 grid = {1,1,1};
    dim3 block = {1,1,1};
    uint2 dogDepth = {this->depth.x,this->depth.y - 1};
    Octave** dogOctaves = new Octave*[dogDepth.x]();
    for(int o = 0; o < dogDepth.x; o++){
        dogOctaves[o] = new Octave();
        dogOctaves[o]->blurs = new Octave::Blur*[dogDepth.y]();
        dogOctaves[o]->numBlurs = dogDepth.y;
        dogOctaves[o]->pixelWidth = this->octaves[o]->pixelWidth;
        pixelsLower = this->octaves[o]->blurs[0]->pixels;
        getFlatGridBlock(pixelsLower->numElements,grid,block);
        for(int b = 0; b < dogDepth.y; ++b){
            dogOctaves[o]->blurs[b] = new Octave::Blur();
            dogOctaves[o]->id = o;
            dogOctaves[o]->blurs[b]->size = this->octaves[o]->blurs[0]->size;
            dogOctaves[o]->blurs[b]->sigma = (this->octaves[o]->pixelWidth/this->octaves[0]->pixelWidth)*this->octaves[0]->blurs[0]->sigma*pow(2,b/3);//TODO check these sigmas
            dogOctaves[o]->blurs[b]->pixels = new Unity<float>(nullptr,pixelsLower->numElements,gpu);
            pixelsUpper = this->octaves[o]->blurs[b+1]->pixels;
            origin[0] = pixelsLower->state;
            origin[1] = pixelsUpper->state;
            if(origin[0] == cpu) pixelsLower->transferMemoryTo(gpu);
            if(origin[1] == cpu) pixelsUpper->transferMemoryTo(gpu);
            subtractImages<<<grid,block>>>(pixelsLower->numElements,pixelsUpper->device,pixelsLower->device,dogOctaves[o]->blurs[b]->pixels->device);
            cudaDeviceSynchronize();
            CudaCheckError();
            if(origin[0] == cpu) pixelsLower->setMemoryState(cpu);
            pixelsLower = pixelsUpper;
        }
    }
    for(int i = 0; i < this->depth.x; ++i){
        delete this->octaves[i];
    }
    delete[] this->octaves;
    this->depth = dogDepth;
    this->octaves = dogOctaves;
}

ssrlcv::FeatureFactory::ScaleSpace::~ScaleSpace(){
    if(this->octaves != nullptr){
        for(int i = 0; i < this->depth.x; ++i){
            delete this->octaves[i];
        }
        delete[] this->octaves;
    }
 
}
void ssrlcv::FeatureFactory::ScaleSpace::dumpData(std::string filePath){
    for(int o = 0; o < this->depth.x; ++o){
        for(int b = 0; b < this->depth.y; ++b){
            Unity<unsigned char>* writable = convertImageToChar(this->octaves[o]->blurs[b]->pixels);
            writable->transferMemoryTo(cpu);
            std::string currentFile = filePath + std::to_string(o) + "_" + std::to_string(b) + ".png";
            writePNG(currentFile.c_str(), writable->host, 1, this->octaves[o]->blurs[b]->size.x, this->octaves[o]->blurs[b]->size.y);
        }
    }
}
void ssrlcv::FeatureFactory::ScaleSpace::findKeyPoints(float noiseThreshold, float edgeThreshold, bool subpixel){
    if(this->depth.y < 4){
        std::cerr<<"findKeyPoints should be done on a dog scale space - this is either not a dog or the number of blurs is insufficient"<<std::endl;
        exit(-1);
    }
    for(int i = 0; i < this->depth.x; ++i){
        this->octaves[i]->searchForExtrema();
        if(this->octaves[i]->extrema->numElements > 0){
            this->octaves[i]->removeNoise(noiseThreshold*0.8);
            if(subpixel){
                this->octaves[i]->refineExtremaLocation();
                this->octaves[i]->removeNoise(noiseThreshold);
            }
            this->octaves[i]->removeEdges(edgeThreshold);
        }  
        std::cout<<"found "<<this->octaves[i]->extrema->numElements<<" keypoints in octave search"<<std::endl; 
    }
}
ssrlcv::Unity<ssrlcv::FeatureFactory::ScaleSpace::SSKeyPoint>* ssrlcv::FeatureFactory::ScaleSpace::getAllKeyPoints(MemoryState destination){
    unsigned int totalKeyPoints = 0;
    if(destination != cpu || destination != gpu){
        std::cerr<<"in getAllKeyPoints, destination must be cpu or gpu"<<std::endl;
        exit(-1);
    }
    MemoryState* origin = new MemoryState[this->depth.x];
    for(int i = 0; i < this->depth.x; ++i){
        origin[i] = this->octaves[i]->extrema->state;
        if(origin[i] != destination) this->octaves[i]->extrema->transferMemoryTo(destination);
        totalKeyPoints += this->octaves[i]->extrema->numElements;
    }
    if(totalKeyPoints == 0){
        std::cerr<<"scale space has no keyPoints generated within its octaves"<<std::endl;
        exit(0);
    }
    Unity<SSKeyPoint>* aggregatedKeyPoints = new Unity<SSKeyPoint>(nullptr,totalKeyPoints,destination);
    int currentIndex = 0;
    for(int i = 0; i < this->depth.x; ++i){
        if(destination == cpu){
            std::memcpy(aggregatedKeyPoints->host + currentIndex, this->octaves[i]->extrema->host, this->octaves[i]->extrema->numElements*sizeof(SSKeyPoint));
        }
        else{
            CudaSafeCall(cudaMemcpy(aggregatedKeyPoints->device + currentIndex, this->octaves[i]->extrema->device, this->octaves[i]->extrema->numElements*sizeof(SSKeyPoint),cudaMemcpyDeviceToDevice));
        }
        currentIndex += this->octaves[i]->extrema->numElements;
        if(origin[i] != destination) this->octaves[i]->extrema->setMemoryState(origin[i]);
    }
    return aggregatedKeyPoints;
}

ssrlcv::FeatureFactory::ScaleSpace::Octave::Octave(){
    this->numBlurs = 0;
    this->blurs = nullptr;
    this->pixelWidth = 0.0f;
    this->extrema = nullptr;
    this->extremaBlurIndices = nullptr;
    this->id = -1;
}
ssrlcv::FeatureFactory::ScaleSpace::Octave::Octave(int id, unsigned int numBlurs, int2 kernelSize, float* sigmas, Unity<unsigned char>* pixels, uint2 size, float pixelWidth) : 
numBlurs(numBlurs),pixelWidth(pixelWidth),id(id){
    this->extrema = nullptr;
    this->extremaBlurIndices = nullptr;
    printf("creating octave with %d blurs of size {%d,%d}\n",numBlurs,size.x,size.y);
    MemoryState origin = pixels->state;

    if(origin == cpu || pixels->fore == cpu) pixels->transferMemoryTo(gpu);
    Unity<float>* blurable = convertImageToFlt(pixels);
    if(origin == cpu) pixels->setMemoryState(cpu);

    this->blurs = new Blur*[this->numBlurs]();

    for(int i = 0; i < this->numBlurs; ++i){
        this->blurs[i] = new Blur(sigmas[i],kernelSize,blurable,size,pixelWidth);
    }
    delete blurable;
}
ssrlcv::FeatureFactory::ScaleSpace::Octave::~Octave(){
    if(this->blurs != nullptr){
        for(int i = 0; i < this->numBlurs; ++i){
            delete this->blurs[i];
        }
        delete[] this->blurs;
    } 
}
void ssrlcv::FeatureFactory::ScaleSpace::Octave::searchForExtrema(){
    Unity<float>* pixelsUpper = nullptr;
    Unity<float>* pixelsMiddle = nullptr;
    Unity<float>* pixelsLower = nullptr;
    dim3 grid2D = {1,1,1};
    dim3 block2D = {3,3,3};
    dim3 grid = {1,1,1};
    dim3 block = {1,1,1};
    MemoryState origin[3];
    int* extremaAddresses = nullptr;
    int totalExtrema = 0;
    SSKeyPoint** extrema2D = new SSKeyPoint*[this->numBlurs - 2];

    this->extremaBlurIndices = new int[this->numBlurs - 2]();
    int extremaAtDepth = 0;

    pixelsLower = this->blurs[0]->pixels;
    getGrid(pixelsLower->numElements,grid2D);
    int* temp = new int[pixelsLower->numElements];
    for(int i = 0; i < pixelsLower->numElements; ++i){
        temp[i] = -1;
    }
    CudaSafeCall(cudaMalloc((void**)&extremaAddresses,pixelsLower->numElements*sizeof(int)));
    for(int b = 1; b < this->numBlurs - 1; ++b){
        CudaSafeCall(cudaMemcpy(extremaAddresses,temp,pixelsLower->numElements*sizeof(int),cudaMemcpyHostToDevice));
        pixelsMiddle = this->blurs[b]->pixels;
        pixelsUpper = this->blurs[b+1]->pixels;
        origin[0] = pixelsLower->state;
        origin[1] = pixelsMiddle->state;
        origin[2] = pixelsUpper->state;
        if(origin[0] == cpu) pixelsLower->transferMemoryTo(gpu);
        if(origin[1] == cpu) pixelsMiddle->transferMemoryTo(gpu);
        if(origin[2] == cpu) pixelsUpper->transferMemoryTo(gpu);
        findExtrema<<<grid2D,block2D>>>(this->blurs[b]->size,pixelsUpper->device,pixelsMiddle->device,pixelsLower->device,extremaAddresses);
        cudaDeviceSynchronize();
        CudaCheckError();

        thrust::device_ptr<int> addr(extremaAddresses);

        thrust::device_ptr<int> new_end = thrust::remove(addr, addr + pixelsLower->numElements,-1);
        cudaDeviceSynchronize();
        CudaCheckError();
        extremaAtDepth = new_end - addr;

        this->extremaBlurIndices[b-1] = totalExtrema;
        totalExtrema += extremaAtDepth;

        if(extremaAtDepth != 0){
            CudaSafeCall(cudaMalloc((void**)&extrema2D[b-1],extremaAtDepth*sizeof(ScaleSpace::SSKeyPoint)));
            grid = {1,1,1}; block = {1,1,1};
            getFlatGridBlock(extremaAtDepth,grid,block);
            fillExtrema<<<grid,block>>>(extremaAtDepth,this->blurs[b]->size,this->pixelWidth,{this->id,b},extremaAddresses,pixelsMiddle->device,extrema2D[b-1]);
            CudaCheckError();
        }
        else{
            extrema2D[b-1] = nullptr;
        }
        
        pixelsLower->fore = gpu;
        if(origin[0] == cpu) pixelsLower->setMemoryState(cpu);
        pixelsLower = pixelsMiddle;
        pixelsMiddle = pixelsUpper;
    }
    delete[] temp;
    CudaSafeCall(cudaFree(extremaAddresses));
    this->extrema = new Unity<ScaleSpace::SSKeyPoint>(nullptr,totalExtrema,gpu);
    for(int i = 0; i < this->numBlurs - 2; ++i){
        if(extrema2D[i] == nullptr) continue;
        if(i == this->numBlurs - 3){
            CudaSafeCall(cudaMemcpy(this->extrema->device + this->extremaBlurIndices[i],extrema2D[i],(totalExtrema-this->extremaBlurIndices[i])*sizeof(ScaleSpace::SSKeyPoint),cudaMemcpyDeviceToDevice));
        }
        else if(this->extremaBlurIndices[i+1] - this->extremaBlurIndices[i] != 0){
            CudaSafeCall(cudaMemcpy(this->extrema->device + this->extremaBlurIndices[i],extrema2D[i],(this->extremaBlurIndices[i+1]-this->extremaBlurIndices[i])*sizeof(ScaleSpace::SSKeyPoint),cudaMemcpyDeviceToDevice));
        }
        CudaSafeCall(cudaFree(extrema2D[i]));
    }  
    delete[] extrema2D;
}
void ssrlcv::FeatureFactory::ScaleSpace::Octave::discardExtrema(){
    MemoryState origin = this->extrema->state;
    if(origin == cpu || this->extrema->fore == cpu) this->extrema->transferMemoryTo(gpu);
    SSKeyPoint** temp = new SSKeyPoint*[this->numBlurs - 2];
    int* numExtrema = new int[this->numBlurs - 2];
    int numExtremaAtBlur = 0;
    for(int i = 0; i < this->numBlurs - 2; ++i){
        if(i < this->numBlurs - 3){
            numExtremaAtBlur = this->extremaBlurIndices[i+1] - this->extremaBlurIndices[i];
        }
        else{
            numExtremaAtBlur = this->extrema->numElements - this->extremaBlurIndices[i];
        }
        numExtrema[i] = numExtremaAtBlur;
        if(numExtremaAtBlur == 0){
            temp[i] = nullptr;
            continue;
        }
        CudaSafeCall(cudaMalloc((void**)&temp[i],numExtremaAtBlur*sizeof(SSKeyPoint)));
        CudaSafeCall(cudaMemcpy(temp[i],this->extrema->device + this->extremaBlurIndices[i],numExtremaAtBlur*sizeof(SSKeyPoint),cudaMemcpyDeviceToDevice));
    }
    this->extrema->clear(this->extrema->state);
    int totalKept = 0;
    for(int i = 0; i < this->numBlurs - 2; ++i){
        numExtremaAtBlur = 0;
        if(temp[i] != nullptr){
            thrust::device_ptr<ScaleSpace::SSKeyPoint> kp(temp[i]);
            thrust::device_ptr<ScaleSpace::SSKeyPoint> new_end = thrust::remove_if(kp,kp+numExtrema[i],ScaleSpace::discard());
            cudaDeviceSynchronize();
            CudaCheckError();
            numExtremaAtBlur = new_end - kp;
        }
        this->extremaBlurIndices[i] = totalKept;
        totalKept += numExtremaAtBlur;
    }
    delete[] numExtrema;

    if(totalKept != 0){
        this->extrema->setData(nullptr,totalKept,gpu);
        for(int i = 0; i < this->numBlurs - 2; ++i){
            if(temp[i] == nullptr) continue;
            if(i == this->numBlurs - 3){
                CudaSafeCall(cudaMemcpy(this->extrema->device + this->extremaBlurIndices[i],temp[i],(totalKept-this->extremaBlurIndices[i])*sizeof(SSKeyPoint),cudaMemcpyDeviceToDevice));
            }
            else{
                CudaSafeCall(cudaMemcpy(this->extrema->device + this->extremaBlurIndices[i],temp[i],(this->extremaBlurIndices[i+1]-this->extremaBlurIndices[i])*sizeof(ScaleSpace::SSKeyPoint),cudaMemcpyDeviceToDevice));
            }
            CudaSafeCall(cudaFree(temp[i]));
        }
        if(origin == cpu) this->extrema->setMemoryState(cpu);
    }
    delete[] temp;
}
void ssrlcv::FeatureFactory::ScaleSpace::Octave::refineExtremaLocation(){

}
void ssrlcv::FeatureFactory::ScaleSpace::Octave::removeNoise(float noiseThreshold){
    MemoryState origin = this->extrema->state;
    if(origin == cpu || this->extrema->fore == cpu) this->extrema->transferMemoryTo(gpu);
    dim3 grid = {1,1,1};
    dim3 block = {1,1,1};
    getFlatGridBlock(this->extrema->numElements,grid,block);
    flagNoise<<<grid,block>>>(this->extrema->numElements,this->extrema->device,noiseThreshold);
    cudaDeviceSynchronize();
    CudaCheckError();
    this->extrema->fore = gpu;
    this->discardExtrema();
    if(origin == cpu) this->extrema->setMemoryState(cpu);
}
void ssrlcv::FeatureFactory::ScaleSpace::Octave::removeEdges(float edgeThreshold){
    MemoryState origin = this->extrema->state;
    if(origin == cpu || this->extrema->fore == cpu) this->extrema->transferMemoryTo(gpu);
    dim3 grid = {1,1,1};
    dim3 block = {1,1,1};
    int numExtremaAtBlur = 0;
    MemoryState pixelOrigin;
    for(int i = 0; i < this->numBlurs - 2; ++i){
        grid = {1,1,1};
        block = {1,1,1};
        if(i < this->numBlurs - 3){
            numExtremaAtBlur = this->extremaBlurIndices[i+1] - this->extremaBlurIndices[i];
        }   
        else{
            numExtremaAtBlur = this->extrema->numElements - this->extremaBlurIndices[i];
        }
        if(numExtremaAtBlur == 0) continue;
        pixelOrigin = this->blurs[i+1]->pixels->state;
        if(pixelOrigin == cpu || this->blurs[i+1]->pixels->fore == cpu) this->blurs[i+1]->pixels->transferMemoryTo(gpu);
        getFlatGridBlock(numExtremaAtBlur,grid,block);
        flagEdges<<<grid,block>>>(numExtremaAtBlur, this->extremaBlurIndices[i], this->blurs[0]->size,this->extrema->device,this->blurs[i+1]->pixels->device,edgeThreshold);
        cudaDeviceSynchronize();
        CudaCheckError();
        if(pixelOrigin == cpu){
            this->blurs[i+1]->pixels->setMemoryState(cpu);
        }
    }
 
    this->extrema->fore = gpu;
    this->discardExtrema();
    if(origin == cpu) this->extrema->setMemoryState(cpu);
}



ssrlcv::FeatureFactory::ScaleSpace::Octave::Blur::Blur(){
    this->sigma = 0.0f;
    this->pixels = nullptr;
    this->size = {0,0};
}
ssrlcv::FeatureFactory::ScaleSpace::Octave::Blur::Blur(float sigma, int2 kernelSize, Unity<float>* blurable, uint2 size, float pixelWidth) : 
sigma(sigma),size(size){
    MemoryState origin = blurable->state;
    if(origin == cpu || blurable->fore == cpu) blurable->transferMemoryTo(gpu);
    kernelSize.x = ceil((float)kernelSize.x*this->sigma/pixelWidth);
    kernelSize.y = ceil((float)kernelSize.y*this->sigma/pixelWidth);
    if(kernelSize.x%2 == 0)kernelSize.x++;
    if(kernelSize.y%2 == 0)kernelSize.y++;
    float* gaussian = new float[kernelSize.y*kernelSize.x]();
    for(int y = -kernelSize.y/2, i = 0; y <= kernelSize.y/2; ++y){
        for(int x = -kernelSize.x/2; x <= kernelSize.x/2; ++x){
            gaussian[i++] = expf(-(((x*x) + (y*y))/2.0f/this->sigma/this->sigma))/2.0f/PI/this->sigma/this->sigma;
        }
    }
    blurable->setData(convolve(this->size,blurable,1,kernelSize,gaussian,true)->device,blurable->numElements,gpu);
    blurable->fore = gpu;
    this->pixels = new Unity<float>(nullptr,blurable->numElements,gpu);
    CudaSafeCall(cudaMemcpy(this->pixels->device,blurable->device,blurable->numElements*sizeof(float),cudaMemcpyDeviceToDevice));
    if(origin == cpu) blurable->setMemoryState(cpu);
}
ssrlcv::FeatureFactory::ScaleSpace::Octave::Blur::~Blur(){
    if(this->pixels != nullptr) delete this->pixels;
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
__device__ __forceinline__ float ssrlcv::edgeness(const float (&hessian)[2][2]){
    float e = trace(hessian);
    return e*e/determinant(hessian);    
}

__global__ void ssrlcv::subtractImages(unsigned int numPixels, float* pixelsUpper, float* pixelsLower, float* pixelsOut){
    unsigned int globalID = (blockIdx.y* gridDim.x+ blockIdx.x)*blockDim.x + threadIdx.x;
    if(globalID < numPixels) pixelsOut[globalID] = pixelsUpper[globalID] - pixelsLower[globalID];
}

__global__ void ssrlcv::findExtrema(uint2 imageSize, float* pixelsUpper, float* pixelsMiddle, float* pixelsLower, int* extrema){
    int blockId = blockIdx.y* gridDim.x+ blockIdx.x;
    int x = blockId%imageSize.x;
    int y = blockId/imageSize.x;
    if(x > 0 && y > 0 && x < imageSize.x - 1 && y < imageSize.y - 1){
        float value = 0.0f;
        x += (((int)threadIdx.x) - 1);
        y += (((int)threadIdx.y) - 1);
        __shared__ float maximumValue;
        __shared__ float minimumValue;
        minimumValue = FLT_MAX;
        maximumValue = -FLT_MAX;
        __syncthreads();
        if(threadIdx.z == 0){
            value = pixelsLower[y*imageSize.x + x];
        }
        else if(threadIdx.z == 1){
            value = pixelsMiddle[y*imageSize.x + x];
        }
        else{
            value = pixelsUpper[y*imageSize.x + x];
        }
        atomicMaxFloat(&maximumValue,value);
        atomicMinFloat(&minimumValue,value);
        __syncthreads();
        if(threadIdx.x == 1 && threadIdx.y == 1 && threadIdx.z == 1){
            if(maximumValue == value || minimumValue == value){
                extrema[blockId] = blockId;
            }
            else{
                extrema[blockId] = -1;
            }
        }
        else return;
    }
}

__global__ void ssrlcv::fillExtrema(int numKeyPoints, uint2 imageSize, float pixelWidth, int2 ssLoc, int* extremaAddresses, float* pixels, FeatureFactory::ScaleSpace::SSKeyPoint* scaleSpaceKP){
    int globalID = (blockIdx.y* gridDim.x+ blockIdx.x)*blockDim.x + threadIdx.x;
    if(globalID < numKeyPoints){
        int index = extremaAddresses[globalID];
        float2 loc = {(float)(index%imageSize.x),(float)(index/imageSize.x)};
        scaleSpaceKP[globalID] = {ssLoc.x,ssLoc.y,loc,{loc.x*pixelWidth,loc.y*pixelWidth},pixels[index],0.0f,false};
    }
}

//currently not able to go up and reevaluate at another scale
__global__ void ssrlcv::refineLocation(unsigned int numKeyPoints, uint2 imageSize, float sigmaMin, float pixelWidthRatio, float pixelWidth, float* pixelsUpper, float* pixelsMiddle, float* pixelsLower, FeatureFactory::ScaleSpace::SSKeyPoint* scaleSpaceKP){
    int globalID = (blockIdx.y* gridDim.x+ blockIdx.x)*blockDim.x + threadIdx.x;
    if(globalID < numKeyPoints){
        FeatureFactory::ScaleSpace::SSKeyPoint kp = scaleSpaceKP[globalID];
        int2 loc = {(int)kp.loc.x,(int)kp.loc.y};
        float hessian[3][3] = {0.0f};
        float hessian_inv[3][3] = {0.0f};
        float gradient[3] = {0.0f};
        float temp[3] = {0.0f};
        float offset[3] = {0.0f};
        for(int attempt = 0; attempt < 5; ++attempt){
            gradient[0] =  pixelsMiddle[loc.y*imageSize.x + loc.x + 1] - pixelsMiddle[loc.y*imageSize.x + loc.x - 1];
            gradient[1] =  pixelsMiddle[(loc.y+1)*imageSize.x + loc.x] - pixelsMiddle[(loc.y-1)*imageSize.x + loc.x];
            gradient[2] =  pixelsUpper[loc.y*imageSize.x + loc.x] - pixelsLower[loc.y*imageSize.x + loc.x];
            hessian[0][0] = gradient[0] - 2*pixelsMiddle[loc.y*imageSize.x + loc.x];
            hessian[0][1] = (pixelsMiddle[(loc.y+1)*imageSize.x + loc.x + 1] - 
                pixelsMiddle[(loc.y-1)*imageSize.x + loc.x + 1] - 
                pixelsMiddle[(loc.y+1)*imageSize.x + loc.x - 1] + 
                pixelsMiddle[(loc.y-1)*imageSize.x + loc.x - 1])/4.0f;
            hessian[0][2] = (pixelsUpper[loc.y*imageSize.x + loc.x + 1] - 
                pixelsLower[loc.y*imageSize.x + loc.x + 1] - 
                pixelsUpper[loc.y*imageSize.x + loc.x - 1] + 
                pixelsLower[loc.y*imageSize.x + loc.x - 1])/4.0f;
            hessian[1][0] = hessian[0][1];
            hessian[1][1] = gradient[1] - 2*pixelsMiddle[loc.y*imageSize.x + loc.x];
            hessian[1][2] = (pixelsUpper[(loc.y+1)*imageSize.x + loc.x] - 
                pixelsLower[(loc.y+1)*imageSize.x + loc.x] - 
                pixelsUpper[(loc.y-1)*imageSize.x + loc.x] + 
                pixelsLower[(loc.y-1)*imageSize.x + loc.x])/4.0f;
            hessian[2][0] = hessian[0][2];
            hessian[2][1] = hessian[1][2];
            hessian[2][2] = gradient[2] - 2*pixelsMiddle[loc.y*imageSize.x + loc.x];
            for(int r = 0; r < 3; ++r){
                for(int c = 0; c < 3; ++c){
                    hessian[r][c] *= -1.0f;
                }
            }
            inverse(hessian,hessian_inv);
            multiply(hessian_inv,gradient,offset);
            multiply(gradient, hessian, temp);
            if(offset[0] < 0.6f && offset[1] < 0.6f && offset[2] < 0.6f){
                kp.intensity = pixelsMiddle[loc.y*imageSize.x + loc.x] - (0.5f*dotProduct(temp,gradient));
                kp.abs_loc = {pixelWidth*(offset[0]+loc.x),pixelWidth*(offset[1]+loc.y)};
                kp.sigma = pixelWidthRatio*sigmaMin*powf(2,(offset[2]+kp.blur)/3);
                break;
            }
            else if(attempt == 4){
                kp.discard = true;
            }
            else{
                loc.x += (int)roundf(offset[0]);
                loc.y += (int)roundf(offset[1]);
                //need to be able to switch depths
            }
        }
        scaleSpaceKP[globalID] = kp;
    }
}
__global__ void ssrlcv::flagNoise(unsigned int numKeyPoints, FeatureFactory::ScaleSpace::SSKeyPoint* scaleSpaceKP, float threshold){
    unsigned int globalID = (blockIdx.y* gridDim.x+ blockIdx.x)*blockDim.x + threadIdx.x;
    if(globalID < numKeyPoints){
        scaleSpaceKP[globalID].discard = abs(scaleSpaceKP[globalID].intensity) < threshold;
    }
}
__global__ void ssrlcv::flagEdges(unsigned int numKeyPoints, unsigned int startingIndex, uint2 imageSize, FeatureFactory::ScaleSpace::SSKeyPoint* scaleSpaceKP, float* pixels, float threshold){
    unsigned int globalID = (blockIdx.y* gridDim.x+ blockIdx.x)*blockDim.x + threadIdx.x;
    if(globalID < numKeyPoints){
        globalID += startingIndex;
        int2 loc = {(int)scaleSpaceKP[globalID].loc.x,(int)scaleSpaceKP[globalID].loc.y};
        float hessian[2][2] = {0.0f};
        hessian[0][0] = -2.0f*pixels[loc.y*imageSize.x + loc.x];
        hessian[1][1] = hessian[0][0] + pixels[(loc.y + 1)*imageSize.x + loc.x] + pixels[(loc.y - 1)*imageSize.x + loc.x];
        hessian[0][0] += pixels[loc.y*imageSize.x + loc.x + 1] + pixels[loc.y*imageSize.x + loc.x - 1];
        hessian[0][1] = (
            pixels[(loc.y + 1)*imageSize.x + loc.x + 1] - pixels[(loc.y - 1)*imageSize.x + loc.x + 1] -
            pixels[(loc.y + 1)*imageSize.x + loc.x - 1] + pixels[(loc.y - 1)*imageSize.x + loc.x - 1]
        );
        hessian[1][0] = hessian[0][1];
        scaleSpaceKP[globalID].discard = edgeness(hessian) > threshold;
    }
}

