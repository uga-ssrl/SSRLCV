#include "FeatureFactory.cuh"



ssrlcv::FeatureFactory::ScaleSpace::Octave::Blur::Blur(){
    this->sigma = 0.0f;
    this->pixels = nullptr;
    this->gradients = nullptr;
    this->size = {0,0};
}
ssrlcv::FeatureFactory::ScaleSpace::Octave::Blur::Blur(float sigma, int2 kernelSize, Unity<float>* pixels, uint2 size, float pixelWidth) : 
sigma(sigma),size(size){
    MemoryState origin = pixels->getMemoryState();
    if(origin != gpu) pixels->setMemoryState(gpu);
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
    pixels->setData(convolve(this->size,pixels,kernelSize,gaussian,true)->device,pixels->size(),gpu);

    this->pixels = new Unity<float>(nullptr,pixels->size(),gpu);
    CudaSafeCall(cudaMemcpy(this->pixels->device,pixels->device,pixels->size()*sizeof(float),cudaMemcpyDeviceToDevice));
    
    if(origin != gpu){
        pixels->setMemoryState(origin);
        this->pixels->setMemoryState(origin);
    } 
    this->gradients = nullptr;
}
void ssrlcv::FeatureFactory::ScaleSpace::Octave::Blur::computeGradients(){
    MemoryState origin = this->pixels->getMemoryState();
    if(origin != gpu) this->pixels->setMemoryState(gpu);
    this->gradients = generatePixelGradients(this->size, this->pixels);
    if(origin != gpu){
        this->pixels->setMemoryState(origin);
        this->gradients->setMemoryState(origin);
    } 
}
ssrlcv::FeatureFactory::ScaleSpace::Octave::Blur::~Blur(){
    if(this->pixels != nullptr) delete this->pixels;
    if(this->gradients != nullptr) delete this->gradients;
}

ssrlcv::FeatureFactory::ScaleSpace::Octave::Octave(){
    this->numBlurs = 0;
    this->blurs = nullptr;
    this->pixelWidth = 0.0f;
    this->extrema = nullptr;
    this->extremaBlurIndices = nullptr;
    this->id = -1;
}
ssrlcv::FeatureFactory::ScaleSpace::Octave::Octave(int id, unsigned int numBlurs, int2 kernelSize, float* sigmas, Unity<float>* pixels, uint2 size, float pixelWidth, int keepPixelsAfterBlur) : 
numBlurs(numBlurs),pixelWidth(pixelWidth),id(id){
    this->extrema = nullptr;
    this->extremaBlurIndices = nullptr;
    printf("creating octave[%d] with %d blurs of size {%d,%d}\n",this->id,this->numBlurs,size.x,size.y);
    MemoryState origin = pixels->getMemoryState();
    if(origin != gpu) pixels->setMemoryState(gpu);

    this->blurs = new Blur*[this->numBlurs]();

    for(int i = 0; i < keepPixelsAfterBlur; ++i){
        this->blurs[i] = new Blur(sigmas[i],kernelSize,pixels,size,pixelWidth);
    }
    Unity<float>* blurable = new Unity<float>(nullptr,pixels->size(),gpu);
    CudaSafeCall(cudaMemcpy(blurable->device,pixels->device,pixels->size()*sizeof(float),cudaMemcpyDeviceToDevice));
    if(origin != gpu) pixels->setMemoryState(origin);
    for(int i = keepPixelsAfterBlur; i < numBlurs; ++i){
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
    if(this->extrema != nullptr) delete this->extrema;
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
    SSKeyPoint** extrema2D = new SSKeyPoint*[this->numBlurs - 2]();

    this->extremaBlurIndices = new int[this->numBlurs]();
    this->extremaBlurIndices[0] = 0;
    int extremaAtDepth = 0;

    pixelsLower = this->blurs[0]->pixels;
    getGrid(pixelsLower->size(),grid2D);
    int* temp = new int[pixelsLower->size()];
    for(int i = 0; i < pixelsLower->size(); ++i){
        temp[i] = -1;
    }
    CudaSafeCall(cudaMalloc((void**)&extremaAddresses,pixelsLower->size()*sizeof(int)));
    for(int b = 1; b < this->numBlurs - 1; ++b){
        CudaSafeCall(cudaMemcpy(extremaAddresses,temp,pixelsLower->size()*sizeof(int),cudaMemcpyHostToDevice));
        pixelsMiddle = this->blurs[b]->pixels;
        pixelsUpper = this->blurs[b+1]->pixels;
        origin[0] = pixelsLower->getMemoryState();
        origin[1] = pixelsMiddle->getMemoryState();
        origin[2] = pixelsUpper->getMemoryState();
        if(origin[0] != gpu) pixelsLower->setMemoryState(gpu);
        if(origin[1] != gpu) pixelsMiddle->setMemoryState(gpu);
        if(origin[2] != gpu) pixelsUpper->setMemoryState(gpu);
        findExtrema<<<grid2D,block2D>>>(this->blurs[b]->size,pixelsUpper->device,pixelsMiddle->device,pixelsLower->device,extremaAddresses);
        cudaDeviceSynchronize();
        CudaCheckError();

        thrust::device_ptr<int> addr(extremaAddresses);

        thrust::device_ptr<int> new_end = thrust::remove(addr, addr + pixelsLower->size(),-1);
        cudaDeviceSynchronize();
        CudaCheckError();
        extremaAtDepth = new_end - addr;

        this->extremaBlurIndices[b] = totalExtrema;
        totalExtrema += extremaAtDepth;

        if(extremaAtDepth != 0){
            CudaSafeCall(cudaMalloc((void**)&extrema2D[b-1],extremaAtDepth*sizeof(ScaleSpace::SSKeyPoint)));
            grid = {1,1,1}; block = {1,1,1};
            getFlatGridBlock(extremaAtDepth,grid,block,fillExtrema);
            fillExtrema<<<grid,block>>>(extremaAtDepth,this->blurs[b]->size,this->pixelWidth,{this->id,b},this->blurs[b]->sigma,extremaAddresses,pixelsMiddle->device,extrema2D[b-1]);
            CudaCheckError();
        }
        else{
            extrema2D[b-1] = nullptr;
        }
        
        if(origin[0] != gpu) pixelsLower->setMemoryState(origin[0]);
        pixelsLower = pixelsMiddle;
        pixelsMiddle = pixelsUpper;
    }
    if(origin[1] != gpu) pixelsMiddle->setMemoryState(origin[1]);
    if(origin[2] != gpu) pixelsUpper->setMemoryState(origin[2]);
    delete[] temp;
    CudaSafeCall(cudaFree(extremaAddresses));
    if(totalExtrema != 0){
        this->extrema = new Unity<ScaleSpace::SSKeyPoint>(nullptr,totalExtrema,gpu);
        this->extremaBlurIndices[this->numBlurs - 1] = this->extrema->size();
        for(int i = 1; i < this->numBlurs - 1 && totalExtrema != 0; ++i){
            if(extrema2D[i-1] == nullptr) continue;
            if(this->extremaBlurIndices[i+1] - this->extremaBlurIndices[i] != 0){
                CudaSafeCall(cudaMemcpy(this->extrema->device + this->extremaBlurIndices[i],extrema2D[i-1],(this->extremaBlurIndices[i+1]-this->extremaBlurIndices[i])*sizeof(ScaleSpace::SSKeyPoint),cudaMemcpyDeviceToDevice));
            }
            CudaSafeCall(cudaFree(extrema2D[i-1]));
        }  
    }
    
    delete[] extrema2D;
}
void ssrlcv::FeatureFactory::ScaleSpace::Octave::discardExtrema(){
    if(this->extrema == nullptr) return;
    MemoryState origin = this->extrema->getMemoryState();
    if(origin != gpu) this->extrema->setMemoryState(gpu);
    SSKeyPoint** temp = new SSKeyPoint*[this->numBlurs];
    int* numExtrema = new int[this->numBlurs];
    int numExtremaAtBlur = 0;
    for(int i = 0; i < this->numBlurs; ++i){
        if(i < this->numBlurs - 1){
            numExtremaAtBlur = this->extremaBlurIndices[i+1] - this->extremaBlurIndices[i];
        }
        else{
            numExtremaAtBlur = this->extrema->size() - this->extremaBlurIndices[i];
        }
        numExtrema[i] = numExtremaAtBlur;
        if(numExtremaAtBlur == 0){
            temp[i] = nullptr;
            continue;
        }
        CudaSafeCall(cudaMalloc((void**)&temp[i],numExtremaAtBlur*sizeof(SSKeyPoint)));
        CudaSafeCall(cudaMemcpy(temp[i],this->extrema->device + this->extremaBlurIndices[i],numExtremaAtBlur*sizeof(SSKeyPoint),cudaMemcpyDeviceToDevice));
    }
    int totalKept = 0;

    for(int i = 0; i < this->numBlurs; ++i){
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
        for(int i = 0; i < this->numBlurs; ++i){
            if(temp[i] == nullptr) continue;
            if(i == this->numBlurs - 1){
                CudaSafeCall(cudaMemcpy(this->extrema->device + this->extremaBlurIndices[i],temp[i],(totalKept-this->extremaBlurIndices[i])*sizeof(SSKeyPoint),cudaMemcpyDeviceToDevice));
            }
            else{
                CudaSafeCall(cudaMemcpy(this->extrema->device + this->extremaBlurIndices[i],temp[i],(this->extremaBlurIndices[i+1]-this->extremaBlurIndices[i])*sizeof(ScaleSpace::SSKeyPoint),cudaMemcpyDeviceToDevice));
            }
            CudaSafeCall(cudaFree(temp[i]));
        }
        if(origin != gpu) this->extrema->setMemoryState(origin);
    }
    else{ 
        delete this->extrema;
        this->extrema = nullptr;
        delete[] this->extremaBlurIndices;
        this->extremaBlurIndices = nullptr;
    }
    delete[] temp;
}

void ssrlcv::FeatureFactory::ScaleSpace::Octave::refineExtremaLocation(){
    if(this->extrema == nullptr) return;
    MemoryState origin = this->extrema->getMemoryState();
    if(origin != gpu) this->extrema->setMemoryState(gpu);
    MemoryState* pixelsOrigin = new MemoryState[this->numBlurs];
    for(int i = 0; i < this->numBlurs; ++i){
        pixelsOrigin[i] = this->blurs[i]->pixels->getMemoryState();
        if(pixelsOrigin[i] != gpu) this->blurs[i]->pixels->setMemoryState(gpu);
    } 

    /*
    1. refine location
    2. discard extrema
    3. resort extrema
    */
    float** allPixels_device = nullptr;
    CudaSafeCall(cudaMalloc((void**)&allPixels_device,this->numBlurs*sizeof(float*)));
    for(int i = 0; i < this->numBlurs; ++i){
        CudaSafeCall(cudaMemcpy(allPixels_device + i,&this->blurs[i]->pixels->device,sizeof(float*),cudaMemcpyHostToDevice));
    }

    dim3 grid = {1,1,1};
    dim3 block = {1,1,1};
    getFlatGridBlock(this->extrema->size(),grid,block,refineLocation);
    refineLocation<<<grid,block>>>(this->extrema->size(), this->blurs[0]->size, this->blurs[0]->sigma, this->blurs[1]->sigma/this->blurs[0]->sigma, this->numBlurs, allPixels_device, this->extrema->device);
    cudaDeviceSynchronize();
    CudaCheckError();
    
    CudaSafeCall(cudaFree(allPixels_device));

    this->discardExtrema();
    if(this->extrema == nullptr) return;

    thrust::device_ptr<SSKeyPoint> kp(this->extrema->device);
    thrust::stable_sort(kp, kp + this->extrema->size());
    this->extrema->transferMemoryTo(cpu);
    this->extremaBlurIndices[0] = 0;
    this->extremaBlurIndices[1] = 0;
    for(int i = 1,blur = 2; i < this->extrema->size() && blur < this->numBlurs - 1; ++i){
        if(this->extrema->host[i-1] < this->extrema->host[i]){
            this->extremaBlurIndices[blur++] = i; 
        } 
    }
    this->extremaBlurIndices[this->numBlurs - 1] = this->extrema->size();
    this->extrema->setFore(cpu);//ensuring that Unity knows where most up to date memory is
    if(origin != this->extrema->getMemoryState()) this->extrema->setMemoryState(origin);
    for(int i = 0; i < this->numBlurs; ++i){
        if(pixelsOrigin[i] != gpu) this->blurs[i]->pixels->setMemoryState(pixelsOrigin[i]);
    }
    delete[] pixelsOrigin;
}
void ssrlcv::FeatureFactory::ScaleSpace::Octave::removeNoise(float noiseThreshold){
    if(this->extrema == nullptr) return;
    MemoryState origin = this->extrema->getMemoryState();
    if(origin != gpu) this->extrema->setMemoryState(gpu);
    dim3 grid = {1,1,1};
    dim3 block = {1,1,1};
    getFlatGridBlock(this->extrema->size(),grid,block,flagNoise);
    flagNoise<<<grid,block>>>(this->extrema->size(),this->extrema->device,noiseThreshold);
    cudaDeviceSynchronize();
    CudaCheckError();
    this->discardExtrema();
    if(this->extrema != nullptr && origin != gpu) this->extrema->setMemoryState(origin);
}
void ssrlcv::FeatureFactory::ScaleSpace::Octave::removeEdges(float edgeThreshold){
    if(this->extrema == nullptr) return;
    MemoryState origin = this->extrema->getMemoryState();
    if(origin != gpu) this->extrema->setMemoryState(gpu);
    dim3 grid = {1,1,1};
    dim3 block = {1,1,1};
    int numExtremaAtBlur = 0;
    MemoryState pixelOrigin;
    for(int i = 0; i < this->numBlurs; ++i){
        grid = {1,1,1};
        block = {1,1,1};
        if(i < this->numBlurs - 1){
            numExtremaAtBlur = this->extremaBlurIndices[i+1] - this->extremaBlurIndices[i];
        }   
        else{
            numExtremaAtBlur = this->extrema->size() - this->extremaBlurIndices[i];
        }
        if(numExtremaAtBlur == 0) continue;
        pixelOrigin = this->blurs[i+1]->pixels->getMemoryState();
        if(pixelOrigin != gpu) this->blurs[i]->pixels->setMemoryState(gpu);
        
        getFlatGridBlock(numExtremaAtBlur,grid,block,flagEdges);
        flagEdges<<<grid,block>>>(numExtremaAtBlur, this->extremaBlurIndices[i], this->blurs[0]->size,this->extrema->device,this->blurs[i]->pixels->device,edgeThreshold);
        cudaDeviceSynchronize();
        CudaCheckError();

        if(pixelOrigin != gpu) this->blurs[i]->pixels->setMemoryState(pixelOrigin);
    }
    this->discardExtrema();
    if(this->extrema != nullptr && origin != gpu) this->extrema->setMemoryState(origin);
}
void ssrlcv::FeatureFactory::ScaleSpace::Octave::removeBorder(float2 border){
    if(this->extrema == nullptr) return;
    MemoryState origin = this->extrema->getMemoryState();
    if(origin != gpu) this->extrema->setMemoryState(gpu);
    dim3 grid = {1,1,1};
    dim3 block = {1,1,1};
    int numExtremaAtBlur = 0;
    
    getFlatGridBlock(numExtremaAtBlur,grid,block,flagBorder);
    flagBorder<<<grid,block>>>(numExtremaAtBlur, this->blurs[0]->size,this->extrema->device,border);
    cudaDeviceSynchronize();
    CudaCheckError();
       
    this->discardExtrema();
    if(this->extrema != nullptr && origin != gpu) this->extrema->setMemoryState(origin);
}

void ssrlcv::FeatureFactory::ScaleSpace::Octave::normalize(){
    for(int i = 0; i < this->numBlurs; ++i){
        normalizeImage(this->blurs[i]->pixels);//letting normalizeImage take care of memory here
    }
}

ssrlcv::FeatureFactory::ScaleSpace::ScaleSpace(){
    this->depth = {0,0};
    this->octaves = nullptr;
    this->isDOG = false;
}
ssrlcv::FeatureFactory::ScaleSpace::ScaleSpace(Image* image, int startingOctave, uint2 depth, float initialSigma, float2 sigmaMultiplier, int2 kernelSize, bool makeDOG) : 
depth(depth), isDOG(makeDOG){ 

    int numResize = (int)powf(2, startingOctave+depth.x);
    if(image->size.x/numResize == 0 || image->size.y/numResize == 0){
        logger.err<<"This image is too small to make a ScaleSpace of the specified depth"<<"\n";
        exit(-1);
    }

    printf("creating scalespace with depth {%d,%d}\n",this->depth.x,this->depth.y);
    Unity<float>* pixels = nullptr;
    MemoryState origin = image->pixels->getMemoryState();
    if(origin != gpu) image->pixels->setMemoryState(gpu);
    if(image->colorDepth != 1){
        Unity<unsigned char>* charPixels = new Unity<unsigned char>(nullptr,image->pixels->size(),gpu);
        CudaSafeCall(cudaMemcpy(charPixels->device, image->pixels->device, pixels->size()*sizeof(unsigned char),cudaMemcpyDeviceToDevice));
        convertToBW(charPixels,image->colorDepth);
        pixels = convertImageToFlt(charPixels);
        delete charPixels;
    }
    else{
        pixels = convertImageToFlt(image->pixels);
    }


    if(origin != gpu) image->pixels->setMemoryState(origin);
    uint2 imageSize = image->size;
    uint2 scalar = {2,2};
    
    bool canBinEarly = imageSize.x%2 == 0 && imageSize.y%2 == 0;
    if(canBinEarly) makeBinnable(imageSize,pixels,startingOctave+depth.x);
    float pixelWidth = 1.0f;
    for(int i = startingOctave; i < 0; ++i){
        pixels->setData(upsample(imageSize,pixels)->device,pixels->size()*4,gpu);  
        imageSize = imageSize*2;
        pixelWidth /= 2.0f;
        if(i == startingOctave && !canBinEarly){
            makeBinnable(imageSize,pixels,depth.x-i);
        } 
    }
    for(int i = 0; i < startingOctave; ++i){
        pixels->setData(bin(imageSize,pixels)->device,pixels->size()/4,gpu);   
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
        std::cout<<"\t";
        this->octaves[i] = new Octave(i,this->depth.y,kernelSize,sigmas,pixels,imageSize,pixelWidth,this->depth.y - 2);
        if(i + 1 < this->depth.x){
            pixels->setData(bin(imageSize,pixels)->device,pixels->size()/4,gpu);
            imageSize = imageSize/scalar;
            pixelWidth *= 2.0f;
            for(int b = 0; b < this->depth.y; ++b){
                sigmas[b]*=sigmaMultiplier.x;   
            }
        }
        this->octaves[i]->normalize();
    }
    delete pixels;
    delete[] sigmas;
    if(this->isDOG) this->convertToDOG();
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
        getFlatGridBlock(pixelsLower->size(),grid,block,subtractImages);
        for(int b = 0; b < dogDepth.y; ++b){
            dogOctaves[o]->blurs[b] = new Octave::Blur();
            dogOctaves[o]->id = o;
            dogOctaves[o]->blurs[b]->size = this->octaves[o]->blurs[0]->size;
            dogOctaves[o]->blurs[b]->sigma = this->octaves[o]->blurs[b]->sigma;
            //dogOctaves[o]->blurs[b]->sigma = this->octaves[o]->blurs[0]*powf(this->octaves[o]->blurs[1]->sigma/this->octaves[o]->blurs[0]->sigma,(float)b + 0.5f);
            dogOctaves[o]->blurs[b]->pixels = new Unity<float>(nullptr,pixelsLower->size(),gpu);
            pixelsUpper = this->octaves[o]->blurs[b+1]->pixels;
            origin[0] = pixelsLower->getMemoryState();
            origin[1] = pixelsUpper->getMemoryState();
            if(origin[0] != gpu) pixelsLower->setMemoryState(gpu);
            if(origin[1] != gpu) pixelsUpper->setMemoryState(gpu);
            subtractImages<<<grid,block>>>(pixelsLower->size(),pixelsUpper->device,pixelsLower->device,dogOctaves[o]->blurs[b]->pixels->device);
            cudaDeviceSynchronize();
            CudaCheckError();            
            if(origin[0] != gpu) pixelsLower->setMemoryState(origin[0]);
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
bool ssrlcv::FeatureFactory::ScaleSpace::checkIfDOG(){
    return this->isDOG;
}
void ssrlcv::FeatureFactory::ScaleSpace::dumpData(std::string filePath){
    MemoryState origin;
    for(int o = 0; o < this->depth.x; ++o){
        for(int b = 0; b < this->depth.y; ++b){
            origin = this->octaves[o]->blurs[b]->pixels->getMemoryState();
            if(origin != gpu) this->octaves[o]->blurs[b]->pixels->setMemoryState(gpu);
            Unity<unsigned char>* writable = convertImageToChar(this->octaves[o]->blurs[b]->pixels);
            writable->setMemoryState(cpu);
            if(origin != gpu) this->octaves[o]->blurs[b]->pixels->setMemoryState(origin);
            std::string currentFile = filePath + std::to_string(o) + "_" + std::to_string(b) + ".png";
            writePNG(currentFile.c_str(), writable->host, 1, this->octaves[o]->blurs[b]->size.x, this->octaves[o]->blurs[b]->size.y);
            delete writable;
        }
    }
}
void ssrlcv::FeatureFactory::ScaleSpace::findKeyPoints(float noiseThreshold, float edgeThreshold, bool subpixel){
    std::cout<<"looking for keypoints..."<<"\n";
    if(this->depth.y < 4){
        logger.err<<"findKeyPoints should be done on a dog scale space - this is either not a dog or the number of blurs is insufficient"<<"\n";
        exit(-1);
    }
    int temp = 0;
    Unity<SSKeyPoint>* currentExtrema = nullptr;
    for(int i = 0; i < this->depth.x; ++i){
        if(i != 0 && this->octaves[i]->extrema != nullptr) std::cout<<"\n";
        this->octaves[i]->searchForExtrema();
        this->octaves[i]->normalize();
        currentExtrema = this->octaves[i]->extrema;
        if(currentExtrema == nullptr) continue;
        temp = currentExtrema->size();
        if(currentExtrema == nullptr) continue;
        std::cout<<"\tkeypoints in octave["<<i<<"] = ";
        if(currentExtrema == nullptr){
            std::cout<<0;
            continue;
        }
        std::cout<<temp;
        this->octaves[i]->removeNoise(noiseThreshold*0.8);
        std::cout<<"-"<<temp - this->octaves[i]->extrema->size();
        if(currentExtrema == nullptr) continue;
        if(subpixel){
            temp = currentExtrema->size();
            this->octaves[i]->refineExtremaLocation();
            if(currentExtrema == nullptr) continue;
            std::cout<<"-"<<temp - currentExtrema->size();
            temp = currentExtrema->size();
            this->octaves[i]->removeNoise(noiseThreshold);
            if(currentExtrema == nullptr) continue;
            std::cout<<"-"<<temp - currentExtrema->size();
        }
        temp = currentExtrema->size();
        this->octaves[i]->removeEdges(edgeThreshold);
        if(currentExtrema == nullptr) continue;
        std::cout<<"-"<<temp - currentExtrema->size();
        std::cout<<"="<<currentExtrema->size()<<"\n";
    }
    for(int i = 0; i < this->depth.x; ++i){
        if(this->octaves[i]->extrema == nullptr){
            delete[] this->octaves[i]->extremaBlurIndices;
            this->octaves[i]->extremaBlurIndices = nullptr;
        }
    }
}
ssrlcv::Unity<ssrlcv::FeatureFactory::ScaleSpace::SSKeyPoint>* ssrlcv::FeatureFactory::ScaleSpace::getAllKeyPoints(MemoryState destination){
    unsigned int totalKeyPoints = 0;
    MemoryState* origin = new MemoryState[this->depth.x];
    bool keepThenTransfer = destination == both; 
    for(int i = 0; i < this->depth.x; ++i){
        origin[i] = this->octaves[i]->extrema->getMemoryState();
        if(!keepThenTransfer &&  origin[i] != both && origin[i] != destination) this->octaves[i]->extrema->transferMemoryTo(destination);
        else if(keepThenTransfer && origin[i] == both && this->octaves[i]->extrema->getFore() == cpu) this->octaves[i]->extrema->transferMemoryTo(gpu);
        totalKeyPoints += this->octaves[i]->extrema->size();
    }
    if(totalKeyPoints == 0){
        logger.err<<"scale space has no keyPoints generated within its octaves"<<"\n";
        exit(0);
    }
    Unity<SSKeyPoint>* aggregatedKeyPoints = new Unity<SSKeyPoint>(nullptr,totalKeyPoints,keepThenTransfer ? gpu : destination);
    int currentIndex = 0;
    for(int i = 0; i < this->depth.x; ++i){
        if(destination == cpu && !keepThenTransfer){
            std::memcpy(aggregatedKeyPoints->host + currentIndex, this->octaves[i]->extrema->host, this->octaves[i]->extrema->size()*sizeof(SSKeyPoint));
        }
        else{
            CudaSafeCall(cudaMemcpy(aggregatedKeyPoints->device + currentIndex, this->octaves[i]->extrema->device, this->octaves[i]->extrema->size()*sizeof(SSKeyPoint),cudaMemcpyDeviceToDevice));
        }
        currentIndex += this->octaves[i]->extrema->size();
        if(origin[i] != this->octaves[i]->extrema->getMemoryState()) this->octaves[i]->extrema->setMemoryState(origin[i]);
    }
    if(keepThenTransfer) aggregatedKeyPoints->transferMemoryTo(cpu);
    return aggregatedKeyPoints;
}

void ssrlcv::FeatureFactory::ScaleSpace::computeKeyPointOrientations(float orientationThreshold, unsigned int maxOrientations, float contributerWindowWidth, bool keepGradients){
    std::cout<<"computing keypoint orientations..."<<"\n";
    ScaleSpace::Octave* currentOctave = nullptr;
    ScaleSpace::Octave::Blur* currentBlur = nullptr;
    int* thetaAddresses_device = nullptr;
    float* thetas_device = nullptr;
    dim3 grid = {1,1,1};
    dim3 block = {1,1,1};
    unsigned long numKeyPointsAtBlur = 0;
    MemoryState origin;
    unsigned int numOrientedKeyPoints = 0;
    unsigned int totalKeyPoints = 0;
    ScaleSpace::SSKeyPoint** orientedKeyPoints2D = nullptr;
    unsigned int keyPointIndex = 0;
    bool gradientsExisted = false;
    for(int o = 0; o < this->depth.x; ++o){
        currentOctave = this->octaves[o];
        if(currentOctave->extrema == nullptr) continue;
        totalKeyPoints = 0;
        orientedKeyPoints2D = new ScaleSpace::SSKeyPoint*[this->depth.y];
        origin = currentOctave->extrema->getMemoryState();
        if(origin == cpu || currentOctave->extrema->getFore() == cpu){
            currentOctave->extrema->setMemoryState(gpu);
        }
        for(int b = 0; b < this->depth.y; ++b){
            orientedKeyPoints2D[b] = nullptr;
            currentBlur = currentOctave->blurs[b];
            if(b + 1 != this->depth.y){
                numKeyPointsAtBlur = currentOctave->extremaBlurIndices[b + 1] - currentOctave->extremaBlurIndices[b];
            }
            else{
                numKeyPointsAtBlur = currentOctave->extrema->size() - currentOctave->extremaBlurIndices[b];
            }
            if(numKeyPointsAtBlur == 0){
                currentOctave->extremaBlurIndices[b] = totalKeyPoints;
                continue;
            } 
            keyPointIndex = currentOctave->extremaBlurIndices[b];
            grid = {1,1,1};
            block = {1,1,1}; 
            getFlatGridBlock(numKeyPointsAtBlur, grid, block,computeThetas);
                        
            CudaSafeCall(cudaMalloc((void**)&thetas_device, numKeyPointsAtBlur*maxOrientations*sizeof(float)));
            CudaSafeCall(cudaMalloc((void**)&thetaAddresses_device, numKeyPointsAtBlur*maxOrientations*sizeof(int)));

            gradientsExisted = currentBlur->gradients != nullptr;
            if(!gradientsExisted) currentBlur->computeGradients();
            if(currentBlur->gradients->getMemoryState() != gpu) currentBlur->gradients->setMemoryState(gpu);
            
            computeThetas<<<grid,block>>>(numKeyPointsAtBlur,keyPointIndex,currentBlur->size, currentOctave->pixelWidth,
                contributerWindowWidth,currentOctave->extrema->device, currentBlur->gradients->device, thetaAddresses_device, maxOrientations, orientationThreshold, thetas_device);
            cudaDeviceSynchronize();
            CudaCheckError();

            if(!keepGradients && !gradientsExisted){
                delete currentBlur->gradients;
                currentBlur->gradients = nullptr;
            } 

            thrust::device_ptr<float> t(thetas_device);
            thrust::device_ptr<float> new_end = thrust::remove(t, t + (numKeyPointsAtBlur*maxOrientations), -FLT_MAX);
            thrust::device_ptr<int> tN(thetaAddresses_device);
            thrust::device_ptr<int> end = thrust::remove(tN, tN + (numKeyPointsAtBlur*maxOrientations), -1);
            numOrientedKeyPoints = end - tN;
            
            currentOctave->extremaBlurIndices[b] = totalKeyPoints;
            totalKeyPoints += numOrientedKeyPoints;

            if(numOrientedKeyPoints != 0){
                grid = {1,1,1};
                block = {1,1,1};
                getFlatGridBlock(numOrientedKeyPoints,grid,block,expandKeyPoints);
                CudaSafeCall(cudaMalloc((void**)&orientedKeyPoints2D[b],numOrientedKeyPoints*sizeof(ScaleSpace::SSKeyPoint)));
                expandKeyPoints<<<grid,block>>>(numOrientedKeyPoints, currentOctave->extrema->device, orientedKeyPoints2D[b], thetaAddresses_device, thetas_device);
                cudaDeviceSynchronize();
                CudaCheckError();
            }

            CudaSafeCall(cudaFree(thetas_device));
            CudaSafeCall(cudaFree(thetaAddresses_device));
        }
        printf("\tafter computing theta for each keyPoint octave[%d] has %d keyPoints\n",o,totalKeyPoints);
        if(totalKeyPoints != 0){
            currentOctave->extrema->setData(nullptr,totalKeyPoints,gpu);
            for(int i = 0; i < currentOctave->numBlurs; ++i){
                if(orientedKeyPoints2D[i] == nullptr) continue;
                if(i == currentOctave->numBlurs - 1 && totalKeyPoints-currentOctave->extremaBlurIndices[i] != 0){
                    CudaSafeCall(cudaMemcpy(currentOctave->extrema->device + currentOctave->extremaBlurIndices[i],orientedKeyPoints2D[i],(totalKeyPoints-currentOctave->extremaBlurIndices[i])*sizeof(ScaleSpace::SSKeyPoint),cudaMemcpyDeviceToDevice));
                }
                else if(i != currentOctave->numBlurs - 1 && currentOctave->extremaBlurIndices[i+1]-currentOctave->extremaBlurIndices[i] != 0){
                    CudaSafeCall(cudaMemcpy(currentOctave->extrema->device + currentOctave->extremaBlurIndices[i],orientedKeyPoints2D[i],(currentOctave->extremaBlurIndices[i+1]-currentOctave->extremaBlurIndices[i])*sizeof(ScaleSpace::SSKeyPoint),cudaMemcpyDeviceToDevice));
                }
                CudaSafeCall(cudaFree(orientedKeyPoints2D[i]));
            }  
            if(origin == cpu) currentOctave->extrema->setMemoryState(cpu);
        }
        else{
            delete currentOctave->extrema;
            currentOctave->extrema = nullptr;
        } 
        delete[] orientedKeyPoints2D;
    }
}

ssrlcv::FeatureFactory::FeatureFactory(float orientationContribWidth, float descriptorContribWidth):
orientationContribWidth(orientationContribWidth), descriptorContribWidth(descriptorContribWidth)
{}


ssrlcv::Unity<ssrlcv::Feature<ssrlcv::Window_3x3>>* ssrlcv::FeatureFactory::generate3x3Windows(Image* image){
    MemoryState origin = image->pixels->getMemoryState();
    if(origin == cpu || image->pixels->getFore() == cpu){
        image->pixels->setMemoryState(gpu);
    }
    dim3 grid = {1,1,1};
    dim3 block = {3,3,1};//most devices should be capable of this
    unsigned int numWindows = (image->size.x-2)*(image->size.y-2);
    getGrid(numWindows,grid);
    checkDims(grid,block);
    Unity<Feature<Window_3x3>>* windows = new Unity<Feature<Window_3x3>>(nullptr,numWindows,gpu);
    fillWindows<<<grid,block>>>(image->size,image->id,image->pixels->device,windows->device);
    cudaDeviceSynchronize();
    CudaCheckError();
    if(origin == cpu){
        image->pixels->setMemoryState(cpu);
        windows->setMemoryState(cpu);
    }    
    return windows;
}
ssrlcv::Unity<ssrlcv::Feature<ssrlcv::Window_9x9>>* ssrlcv::FeatureFactory::generate9x9Windows(Image* image){
    MemoryState origin = image->pixels->getMemoryState();
    if(origin == cpu || image->pixels->getFore() == cpu){
        image->pixels->setMemoryState(gpu);
    }
    dim3 grid = {1,1,1};
    dim3 block = {9,9,1};//some devices may not be capable of this
    unsigned int numWindows = (image->size.x-8)*(image->size.y-8);
    getGrid(numWindows,grid);
    checkDims(grid,block);
    Unity<Feature<Window_9x9>>* windows = new Unity<Feature<Window_9x9>>(nullptr,numWindows,gpu);
    fillWindows<<<grid,block>>>(image->size,image->id,image->pixels->device,windows->device);
    cudaDeviceSynchronize();
    CudaCheckError();
    if(origin == cpu){
        image->pixels->setMemoryState(cpu);
        windows->setMemoryState(cpu);
    }
    return windows;
}
ssrlcv::Unity<ssrlcv::Feature<ssrlcv::Window_15x15>>* ssrlcv::FeatureFactory::generate15x15Windows(Image* image){
    MemoryState origin = image->pixels->getMemoryState();
    if(origin == cpu || image->pixels->getFore() == cpu){
        image->pixels->setMemoryState(gpu);
    }
    dim3 grid = {1,1,1};
    dim3 block = {15,15,1};//some devices may not be capable of this
    unsigned int numWindows = (image->size.x-14)*(image->size.y-14);
    getGrid(numWindows,grid);
    checkDims(grid,block);
    Unity<Feature<Window_15x15>>* windows = new Unity<Feature<Window_15x15>>(nullptr,numWindows,gpu);
    fillWindows<<<grid,block>>>(image->size,image->id,image->pixels->device,windows->device);
    cudaDeviceSynchronize();
    CudaCheckError();
    if(origin == cpu){
        image->pixels->setMemoryState(cpu);
        windows->setMemoryState(cpu);
    }
    return windows;
}
ssrlcv::Unity<ssrlcv::Feature<ssrlcv::Window_25x25>>* ssrlcv::FeatureFactory::generate25x25Windows(Image* image){
    MemoryState origin = image->pixels->getMemoryState();
    if(origin == cpu || image->pixels->getFore() == cpu){
        image->pixels->setMemoryState(gpu);
    }
    dim3 grid = {1,1,1};
    dim3 block = {25,25,1};//some devices may not be capable of this
    unsigned int numWindows = (image->size.x-24)*(image->size.y-24);
    getGrid(numWindows,grid);
    checkDims(grid,block);
    Unity<Feature<Window_25x25>>* windows = new Unity<Feature<Window_25x25>>(nullptr,numWindows,gpu);
    fillWindows<<<grid,block>>>(image->size,image->id,image->pixels->device,windows->device);
    cudaDeviceSynchronize();
    CudaCheckError();
    if(origin == cpu){
        image->pixels->setMemoryState(cpu);
        windows->setMemoryState(cpu);
    }    
    return windows;
}
ssrlcv::Unity<ssrlcv::Feature<ssrlcv::Window_31x31>>* ssrlcv::FeatureFactory::generate31x31Windows(Image* image){
    MemoryState origin = image->pixels->getMemoryState();
    if(origin == cpu || image->pixels->getFore() == cpu){
        image->pixels->setMemoryState(gpu);
    }
    dim3 grid = {1,1,1};
    dim3 block = {31,31,1};//some devices will not be capable of this
    unsigned int numWindows = (image->size.x-30)*(image->size.y-30);
    getGrid(numWindows,grid);
    checkDims(grid,block);
    Unity<Feature<Window_31x31>>* windows = new Unity<Feature<Window_31x31>>(nullptr,numWindows,gpu);
    fillWindows<<<grid,block>>>(image->size,image->id,image->pixels->device,windows->device);
    cudaDeviceSynchronize();
    CudaCheckError();
    if(origin == cpu){
        image->pixels->setMemoryState(cpu);
        windows->setMemoryState(cpu);
    }
    return windows;
}


ssrlcv::FeatureFactory::~FeatureFactory(){

}

__constant__ float ssrlcv::pi = 3.1415927;

/*
const long double PI = 3.141592653589793238L;
const double PI = 3.141592653589793;
const float PI = 3.1415927;
*/

__device__ __forceinline__ float ssrlcv::getMagnitude(const int2 &vector){
  return sqrtf((float)dotProduct(vector, vector));
}
__device__ __forceinline__ float ssrlcv::getMagnitude(const float2 &vector){
  return sqrtf(dotProduct(vector, vector));
}
__device__ void ssrlcv::trickleSwap(const float2 &compareWValue, float2* arr, const int &index, const int &length){
  for(int i = index; i < length; ++i){
    if(compareWValue.x > arr[i].x){
      float2 temp = arr[i];
      arr[i] = compareWValue;
      if((temp.x == 0.0f && temp.y == 0.0f)|| index + 1 == length) return;
      return trickleSwap(temp, arr, index + 1, length);
    }
  }
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

__global__ void ssrlcv::fillWindows(uint2 size, int parent, unsigned char* pixels, Feature<Window_3x3>* windows){
    unsigned long blockId = blockIdx.y* gridDim.x+ blockIdx.x;
    uint2 loc = {(unsigned int)(blockId%(size.x-2) + 1),(unsigned int)(blockId/(size.x-2) + 1)};
    if(loc.x < size.x - 1&& loc.y < size.y - 1){
        Feature<Window_3x3> window = Feature<Window_3x3>();
        window.descriptor.values[threadIdx.x][threadIdx.y] = pixels[(loc.y+threadIdx.y-1)*size.x + (loc.x+threadIdx.x-1)];
        window.loc = {(float)loc.x,(float)loc.y};
        window.parent = parent;
        windows[blockId] = window;
    }
}
__global__ void ssrlcv::fillWindows(uint2 size, int parent, unsigned char* pixels, Feature<Window_9x9>* windows){
    unsigned long blockId = blockIdx.y*gridDim.x + blockIdx.x;
    uint2 loc = {(unsigned int)((blockId%(size.x-8)) + 4),(unsigned int)((blockId/(size.x-8)) + 4)};
    if(loc.x < size.x - 4 && loc.y < size.y - 4){
        Feature<Window_9x9> window = Feature<Window_9x9>();
        window.descriptor.values[threadIdx.x][threadIdx.y] = pixels[(loc.y+threadIdx.y-4)*size.x + (loc.x+threadIdx.x-4)];
        window.loc = {(float)loc.x,(float)loc.y};
        window.parent = parent;
        windows[blockId] = window;
    }
}
__global__ void ssrlcv::fillWindows(uint2 size, int parent, unsigned char* pixels, Feature<Window_15x15>* windows){
    unsigned long blockId = blockIdx.y* gridDim.x+ blockIdx.x;
    uint2 loc = {(unsigned int)(blockId%(size.x-14) + 7),(unsigned int)(blockId/(size.x-14) + 7)};
    if(loc.x < size.x - 7 && loc.y < size.y - 7){
        Feature<Window_15x15> window = Feature<Window_15x15>();
        window.descriptor.values[threadIdx.x][threadIdx.y] = pixels[(loc.y+threadIdx.y-7)*size.x + (loc.x+threadIdx.x-7)];
        window.loc = {(float)loc.x,(float)loc.y};
        window.parent = parent;
        windows[blockId] = window;
    }
}
__global__ void ssrlcv::fillWindows(uint2 size, int parent, unsigned char* pixels, Feature<Window_25x25>* windows){
    unsigned long blockId = blockIdx.y* gridDim.x+ blockIdx.x;
    uint2 loc = {(unsigned int)(blockId%(size.x-24) + 12),(unsigned int)(blockId/(size.x-24) + 12)};
    if(loc.x < size.x - 12 && loc.y < size.y - 12){
        Feature<Window_25x25> window = Feature<Window_25x25>();
        window.descriptor.values[threadIdx.x][threadIdx.y] = pixels[(loc.y+threadIdx.y-12)*size.x + (loc.x+threadIdx.x-12)];
        window.loc = {(float)loc.x,(float)loc.y};
        window.parent = parent;
        windows[blockId] = window;
    }
}
__global__ void ssrlcv::fillWindows(uint2 size, int parent, unsigned char* pixels, Feature<Window_31x31>* windows){
    unsigned long blockId = blockIdx.y* gridDim.x+ blockIdx.x;
    uint2 loc = {(unsigned int)(blockId%(size.x-30) + 15),(unsigned int)(blockId/(size.x-30) + 15)};
    if(loc.x < size.x - 15 && loc.y < size.y - 15){
        Feature<Window_31x31> window = Feature<Window_31x31>();
        window.descriptor.values[threadIdx.x][threadIdx.y] = pixels[(loc.y+threadIdx.y-15)*size.x + (loc.x+threadIdx.x-15)];
        window.loc = {(float)loc.x,(float)loc.y};
        window.parent = parent;
        windows[blockId] = window;
    }
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
__global__ void ssrlcv::fillExtrema(int numKeyPoints, uint2 imageSize, float pixelWidth, int2 ssLoc, float sigma, int* extremaAddresses, float* pixels, FeatureFactory::ScaleSpace::SSKeyPoint* scaleSpaceKP){
    int globalID = (blockIdx.y* gridDim.x+ blockIdx.x)*blockDim.x + threadIdx.x;
    if(globalID < numKeyPoints){
        int index = extremaAddresses[globalID];
        float2 loc = {(float)(index%imageSize.x),(float)(index/imageSize.x)};
        scaleSpaceKP[globalID] = {ssLoc.x,ssLoc.y,loc,pixels[index],sigma,-1.0f,false};
    }
}

__global__ void ssrlcv::refineLocation(unsigned int numKeyPoints, uint2 imageSize, float sigmaMin, float blurSigmaMultiplier, unsigned int numBlurs, float** pixels, FeatureFactory::ScaleSpace::SSKeyPoint* scaleSpaceKP){
    int globalID = (blockIdx.y* gridDim.x+ blockIdx.x)*blockDim.x + threadIdx.x;
    if(globalID < numKeyPoints){
        FeatureFactory::ScaleSpace::SSKeyPoint kp = scaleSpaceKP[globalID];
        int2 loc = {(int)roundf(kp.loc.x),(int)roundf(kp.loc.y)};
        float hessian[3][3] = {0.0f};
        float hessian_inv[3][3] = {0.0f};
        float gradient[3] = {0.0f};
        float temp[3] = {0.0f};
        float offset[3] = {0.0f};
        float* pixelsLower = pixels[kp.blur - 1];
        float* pixelsMiddle = pixels[kp.blur];
        float* pixelsUpper = pixels[kp.blur + 1];
 
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
            if(abs(offset[0]) <= 0.5f && abs(offset[1]) <= 0.5f && abs(offset[2]) <= 0.5f){ 
                kp.loc = {(float)loc.x + offset[0],(float)loc.y + offset[1]};
                loc = {(int)roundf(kp.loc.x),(int)roundf(kp.loc.y)};
                kp.discard = (loc.x <= 0 || loc.y <= 0 || loc.x >= imageSize.x - 1 || loc.y >= imageSize.y - 1);
                if(kp.discard) break;//to prevent more operations
                kp.intensity = pixelsMiddle[loc.y*imageSize.x + loc.x] - (0.5f*dotProduct(temp,gradient));
                kp.sigma = sigmaMin*powf(blurSigmaMultiplier,((float)kp.blur + offset[2]));
                if(abs(offset[2]) > 0.5) kp.blur += (offset[2] > 0) ? 1 : -1;
                break;
            }
            else if(attempt == 4){
                kp.discard = true;
                break;
            }
            else{
                if(abs(offset[0]) > 0.5) loc.x += (offset[0] > 0) ? 1 : -1;
                if(abs(offset[1]) > 0.5) loc.y += (offset[1] > 0) ? 1 : -1;
                kp.loc = {(float)loc.x,(float)loc.y};
                if(abs(offset[2]) > 0.5) kp.blur += (offset[2] > 0) ? 1 : -1;
                if(kp.blur >= numBlurs - 1||kp.blur <= 0||loc.x <= 0||
                    loc.y <= 0||loc.x >= imageSize.x - 1||loc.y >= imageSize.y - 1){//cannot traverse blurs anymore
                    kp.discard = true;
                    break;
                }
                pixelsLower = pixels[kp.blur - 1];
                pixelsMiddle = pixels[kp.blur];
                pixelsUpper = pixels[kp.blur + 1];
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
        int2 loc = {(int)roundf(scaleSpaceKP[globalID].loc.x),(int)roundf(scaleSpaceKP[globalID].loc.y)};
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

__global__ void ssrlcv::flagBorder(unsigned int numKeyPoints, uint2 imageSize, FeatureFactory::ScaleSpace::SSKeyPoint* scaleSpaceKP, float2 border){
    unsigned int globalID = (blockIdx.y* gridDim.x+ blockIdx.x)*blockDim.x + threadIdx.x;
    if(globalID < numKeyPoints){
        FeatureFactory::ScaleSpace::SSKeyPoint kp = scaleSpaceKP[globalID];
        if(kp.loc.x < border.x || kp.loc.y < border.y || kp.loc.x >= (float)imageSize.x - border.x || kp.loc.y >= (float)imageSize.y - border.y){
            scaleSpaceKP[globalID].discard = true;
        }
    }

}


__global__ void ssrlcv::computeThetas(unsigned long numKeyPoints, unsigned int keyPointIndex, uint2 imageSize, float pixelWidth, 
float lambda, FeatureFactory::ScaleSpace::SSKeyPoint* keyPoints, float2* gradients, 
int* thetaNumbers, unsigned int maxOrientations, float orientationThreshold, float* thetas){
   unsigned int globalID = (blockIdx.y* gridDim.x+ blockIdx.x)*blockDim.x + threadIdx.x;
    if(globalID < numKeyPoints){
        FeatureFactory::ScaleSpace::SSKeyPoint kp = keyPoints[globalID+keyPointIndex];
        float2 keyPoint = kp.loc;
        float windowWidth = ceil(kp.sigma*3.0f*lambda/pixelWidth);
        int regNumOrient = maxOrientations;

        float2 min = {(keyPoint.x - windowWidth),(keyPoint.y - windowWidth)};
        float2 max = {(keyPoint.x + windowWidth),(keyPoint.y + windowWidth)};

        if(min.x < 0.0f || min.y < 0.0f || max.x >= imageSize.x - 1 || max.y >= imageSize.y - 1){
            for(int i = 0; i < regNumOrient; ++i){
                thetaNumbers[globalID*regNumOrient + i] = -1;
                thetas[globalID*regNumOrient + i] = -FLT_MAX;
            }
            return;
        }
        float orientationHist[36] = {0.0f};
        float maxHist = 0.0f;
        float2 gradient = {0.0f,0.0f};
        float2 temp2 = {0.0f,0.0f};
        unsigned int imageWidth = imageSize.x;
        float weight = 2.0f*lambda*lambda*kp.sigma*kp.sigma;
        float angle = 0.0f;
        float rad10 = pi/18.0f;
        for(float y = min.y; y <= max.y; y+=1.0f){
            for(float x = min.x; x <= max.x; x+=1.0f){
                gradient = gradients[llroundf(y)*imageWidth + llroundf(x)];//may want to do interpolation here
                temp2 = {x - keyPoint.x,y - keyPoint.y};
                angle = fmodf(atan2f(gradient.y,gradient.x) + (2.0f*pi),2.0f*pi);//atan2f returns between -pi to pi
                orientationHist[(int)floor(angle/rad10)] += getMagnitude(gradient)*expf(-((temp2.x*temp2.x)+(temp2.y*temp2.y))/weight);//(/pi/weight);
            }
        }
        //apparently has negligable impact
        // float3 convHelper = {orientationHist[35],orientationHist[0],orientationHist[1]};
        // for(int i = 0; i < 6; ++i){
        //  temp2.x = orientationHist[0];//need to hold on to this for id = 35 conv
        //  for(int id = 1; id < 36; ++id){
        //    orientationHist[id] = (convHelper.x+convHelper.y+convHelper.z)/3.0f;
        //    convHelper.x = convHelper.y;
        //    convHelper.y = convHelper.z;
        //    convHelper.z = (id < 35) ? orientationHist[id+1] : temp2.x;
        //    if(i == 5){
        //      if(orientationHist[id] > maxHist){
        //        maxHist = orientationHist[id];
        //      }
        //    }
        //  }
        // }
        for(int i = 0; i < 36; ++i){
          if(orientationHist[i] > maxHist) maxHist = orientationHist[i];
        }
        maxHist *= orientationThreshold;//% of max orientation value

        float2* bestMagWThetas = new float2[regNumOrient]();
        float2 tempMagWTheta = {0.0f,0.0f};
        for(int b = 0; b < 36; ++b){
            if(orientationHist[b] < maxHist ||
            (b > 0 && orientationHist[b] < orientationHist[b-1]) ||
            (b < 35 && orientationHist[b] < orientationHist[b+1]) ||
            (b == 0 && orientationHist[b] < orientationHist[35]) || 
            (b == 35 && orientationHist[b] < orientationHist[0]) ||
            (orientationHist[b] < bestMagWThetas[regNumOrient-1].x)){
                continue;
            } 


            tempMagWTheta.x = orientationHist[b];

            if(b == 0){
              tempMagWTheta.y = (orientationHist[35]-orientationHist[1])/(orientationHist[35]-(2.0f*orientationHist[0])+orientationHist[1]);
            }
            else if(b == 35){
              tempMagWTheta.y = (orientationHist[34]-orientationHist[0])/(orientationHist[34]-(2.0f*orientationHist[35])+orientationHist[0]);
            }
            else{
              tempMagWTheta.y = (orientationHist[b-1]-orientationHist[b+1])/(orientationHist[b-1]-(2.0f*orientationHist[b])+orientationHist[b+1]);
            }

            tempMagWTheta.y *= (pi/36.0f);
            tempMagWTheta.y += (b*rad10);
            tempMagWTheta.y = fmodf(tempMagWTheta.y + (2.0f*pi),2.0f*pi);

            for(int i = 0; i < regNumOrient; ++i){
              if(tempMagWTheta.x > bestMagWThetas[i].x){
                for(int ii = i; ii < regNumOrient; ++ii){
                  temp2 = bestMagWThetas[ii];
                  bestMagWThetas[ii] = tempMagWTheta;
                  tempMagWTheta = temp2;
                }
              }
            }
        }
        for(int i = 0; i < regNumOrient; ++i){
            if(bestMagWThetas[i].x == 0.0f){
                thetaNumbers[globalID*regNumOrient + i] = -1;
                thetas[globalID*regNumOrient + i] = -FLT_MAX;
            }
            else{
                thetaNumbers[globalID*regNumOrient + i] = globalID + keyPointIndex;
                thetas[globalID*regNumOrient + i] = bestMagWThetas[i].y;
            }
        }
        delete[] bestMagWThetas;
    } 
}

__global__ void ssrlcv::expandKeyPoints(unsigned int numKeyPoints, FeatureFactory::ScaleSpace::SSKeyPoint* keyPointsIn, FeatureFactory::ScaleSpace::SSKeyPoint* keyPointsOut, int* thetaAddresses, float* thetas){
    unsigned int globalID = (blockIdx.y* gridDim.x+ blockIdx.x)*blockDim.x + threadIdx.x;
    if(globalID < numKeyPoints){
        FeatureFactory::ScaleSpace::SSKeyPoint kp = keyPointsIn[thetaAddresses[globalID]];
        kp.theta = thetas[globalID];
        keyPointsOut[globalID] = kp;
    }

}