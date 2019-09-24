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
        this->octaves[i] = new Octave(this->depth.y,kernelSize,sigmas,pixels,imageSize,pixelWidth);
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


ssrlcv::FeatureFactory::ScaleSpace::Octave::Octave(){
    this->numBlurs = 0;
    this->blurs = nullptr;
    this->pixelWidth = 0.0f;
}
ssrlcv::FeatureFactory::ScaleSpace::Octave::Octave(unsigned int numBlurs, int2 kernelSize, float* sigmas, Unity<unsigned char>* pixels, uint2 size, float pixelWidth) : 
numBlurs(numBlurs),pixelWidth(pixelWidth){
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
            dog->octaves[o]->blurs[b]->size = scaleSpace->octaves[o]->blurs[0]->size;
            dog->octaves[o]->blurs[b]->sigma = 0.0f;//NOTE
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
    for(int o = 0,kpd = 0; o < dog->depth.x; o++){
        pixelsLower = dog->octaves[o]->blurs[0]->pixels;
        getGrid(pixelsLower->numElements,grid2D);
        int* temp = new int[pixelsLower->numElements];
        for(int i = 0; i < pixelsLower->numElements; ++i){
            temp[i] = -1;
        }
        if(maximaAddresses != nullptr) CudaSafeCall(cudaFree(maximaAddresses));
        CudaSafeCall(cudaMalloc((void**)&maximaAddresses,pixelsLower->numElements*sizeof(int)));
        for(int b = 1; b < dog->depth.y - 1; ++b,++kpd){
            CudaSafeCall(cudaMemcpy(maximaAddresses,temp,pixelsLower->numElements*sizeof(int),cudaMemcpyHostToDevice));
            pixelsMiddle = dog->octaves[o]->blurs[b]->pixels;
            pixelsUpper = dog->octaves[o]->blurs[b+1]->pixels;
            origin[0] = pixelsLower->state;
            origin[1] = pixelsUpper->state;
            origin[2] = pixelsUpper->state;
            if(origin[0] == cpu) pixelsLower->transferMemoryTo(gpu);
            if(origin[1] == cpu) pixelsMiddle->transferMemoryTo(gpu);
            if(origin[2] == cpu) pixelsUpper->transferMemoryTo(gpu);
            findMaxima<<<grid2D,block2D>>>(dog->octaves[o]->blurs[b]->size,pixelsUpper->device,pixelsMiddle->device,pixelsLower->device,maximaAddresses);
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
            std::cout<<maximaAtDepth<<std::endl;
            if(maximaAtDepth != 0){
                grid = {1,1,1}; block = {1,1,1};
                getFlatGridBlock(maximaAtDepth,grid,block);
                fillMaxima<<<grid,block>>>(maximaAtDepth,dog->octaves[o]->blurs[b]->size,dog->octaves[o]->pixelWidth,{o,b},maximaAddresses,pixelsMiddle->device,maxima2D[kpd]);
                CudaCheckError();
            }

            if(origin[0] == cpu) pixelsLower->setMemoryState(cpu);
            pixelsLower = pixelsMiddle;
            pixelsMiddle = pixelsUpper;
        }
    }
    CudaSafeCall(cudaFree(maximaAddresses));
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
    std::string dump = "./out/scalespace";
    scaleSpace->dumpData(dump);
    std::cout<<"creating dog from scalespace"<<std::endl;
    DOG* dog = generateDOG(scaleSpace);
    dump = "./out/dog";
    dog->dumpData(dump);
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

__global__ void ssrlcv::findMaxima(uint2 imageSize, float* pixelsUpper, float* pixelsMiddle, float* pixelsLower, int* maxima){
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
                maxima[blockId] = blockId;
            }
            else{
                maxima[blockId] = -1;
            }
        }
        else return;
    }
}

__global__ void ssrlcv::fillMaxima(int numKeyPoints, uint2 imageSize, float pixelWidth, int2 ssLoc, int* maximaAddresses, float* pixels, FeatureFactory::ScaleSpace::SSKeyPoint* scaleSpaceKP){
    int globalID = (blockIdx.y* gridDim.x+ blockIdx.x)*blockDim.x + threadIdx.x;
    if(globalID < numKeyPoints){
        int index = maximaAddresses[globalID];
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
__global__ void ssrlcv::flagNoise(uint2 imageSize, FeatureFactory::ScaleSpace::SSKeyPoint* scaleSpaceKP, float threshold){

}
__global__ void ssrlcv::flagEdges(uint2 imageSize, FeatureFactory::ScaleSpace::SSKeyPoint* scaleSpaceKP, float threshold){

}




__global__ void ssrlcv::convertSSKPToLKP(unsigned int numKeyPoints, float3* localizedKeyPoints, ssrlcv::FeatureFactory::ScaleSpace::SSKeyPoint* scaleSpaceKP){
    unsigned int globalID = (blockIdx.y* gridDim.x+ blockIdx.x)*blockDim.x + threadIdx.x;
    if(globalID < numKeyPoints){
        FeatureFactory::ScaleSpace::SSKeyPoint kp = scaleSpaceKP[globalID];
        localizedKeyPoints[globalID] = {kp.loc.x,kp.loc.y,kp.sigma};
        //printf("%f,%f\n",kp.loc.x,kp.loc.y);//should be in world coordinate frame TODO
    }
}
