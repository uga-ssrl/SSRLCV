#include "pngUtilities.h"
#include "common_includes.h"

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
  err = cudaDeviceSynchronize();
  if (cudaSuccess != err) {
    fprintf(stderr, "cudaCheckError() with sync failed at %s:%i : %s\n",
    file, line, cudaGetErrorString(err));
    exit(-1);
  }
#endif

  return;
}

__device__ int4 minI = {std::numeric_limits<int>::max(),
  std::numeric_limits<int>::max(),
  std::numeric_limits<int>::max(),
  std::numeric_limits<int>::max()};
__device__ int4 maxI = {std::numeric_limits<int>::min(),
  std::numeric_limits<int>::min(),
  std::numeric_limits<int>::min(),
  std::numeric_limits<int>::min()};

__device__ __forceinline__ int floatToOrderedInt(float floatVal);
__device__ __forceinline__ float orderedIntToFloat(int intVal);
__device__ __forceinline__ uchar1 normalize(float multiplier, float1 &min, float1 &max, float1 &value);
__device__ __forceinline__ uchar2 normalize(float multiplier, float2 &min, float2 &max, float2 &value);
__device__ __forceinline__ uchar3 normalize(float multiplier, float3 &min, float3 &max, float3 &value);
__device__ __forceinline__ uchar4 normalize(float multiplier, float4 &min, float4 &max, float4 &value);

__global__ void applyPSF(int width, float1* subpixelArray, int psfSideLength, float1* psf);
__global__ void applyPSF(int width, float2* subpixelArray, int psfSideLength, float2* psf);
__global__ void applyPSF(int width, float3* subpixelArray, int psfSideLength, float3* psf);
__global__ void applyPSF(int width, float4* subpixelArray, int psfSideLength, float4* psf);

__global__ void binImage(int width, int degreeOfBinning, float1* pixelArray, uchar1* output);
__global__ void binImage(int width, int degreeOfBinning, float2* pixelArray, uchar2* output);
__global__ void binImage(int width, int degreeOfBinning, float3* pixelArray, uchar3* output);
__global__ void binImage(int width, int degreeOfBinning, float4* pixelArray, uchar4* output);

int main(int argc, char *argv[]){
  clock_t totalTimer = clock();
  try{
    std::stringstream bin;
    std::stringstream side;
    int degreeOfBinning = 0;
    int psfSideLength = 0;
    bool correctNumberOfArguments = false;
    if(argc == 5){
      correctNumberOfArguments = true;
      side << argv[3];
      bin << argv[4];
    }
    if(correctNumberOfArguments && (bin >> degreeOfBinning) && (side >> psfSideLength)){
      const char* filePath = argv[1];
      std::string originalFileName = filePath;
      std::string psfFile = argv[2];
      if(originalFileName.substr(originalFileName.length() - 3, 3) != "png"){
        std::cout<<"IMAGE FILE IS NOT A PNG"<<std::endl;
        exit(-1);
      }
      if(psfFile.substr(psfFile.length() - 3, 3) != "csv"){
        std::cout<<"PSF FILE IS NOT A CSV"<<std::endl;
        exit(-1);
      }
      if(psfSideLength % 2 == 0){
        std::cout<<"PSF must be a square matrix with odd side lengths"<<std::endl;
        exit(-1);
      }
      std::cout<<psfFile<<"="<<psfSideLength<<"x"<<psfSideLength<<std::endl;
      int height = 0;
      int width = 0;
      png_byte color_type;
      png_byte bit_depth;
      unsigned char** row_pointers;
      //fills row_pointers and updates height and width
      readPNG(filePath, row_pointers, height, width, color_type, bit_depth);

      int wAfterPSF = (width - psfSideLength - 1);
      int hAfterPSF = (height - psfSideLength - 1);
      if(wAfterPSF%degreeOfBinning != 0 || hAfterPSF%degreeOfBinning != 0){
        printf("ERROR...wAfterPSF = %d, hAfterPSF = %d -> need to be multiples of sqrt(#subpixel/pixel) = %d\n",wAfterPSF, hAfterPSF, degreeOfBinning);
        exit(-1);
      }
      int wAfterBinning = wAfterPSF/degreeOfBinning;
      int hAfterBinning = hAfterPSF/degreeOfBinning;
      dim3 gridPSF = {(unsigned int) hAfterPSF, (unsigned int) wAfterPSF, 1};
      dim3 blockPSF = {(unsigned int) psfSideLength, (unsigned int) psfSideLength, 1};
      dim3 gridBin = {(unsigned int) hAfterBinning, (unsigned int) wAfterBinning, 1};
      dim3 blockBin = {(unsigned int) degreeOfBinning, (unsigned int) degreeOfBinning, 1};

      if(color_type == PNG_COLOR_TYPE_GRAY){
        std::cout<<"PNG_COLOR_TYPE_GRAY"<<std::endl;
        float1* bwPSF = new float1[psfSideLength*psfSideLength];
        float1* bwPSFDevice;
        float1* bwImageDevice;
        uchar1* bwImageChar;
        uchar1* bwImageCharDevice;
        float1* bwImage = new float1[width*height];

        readPSF(psfFile, bwPSF, psfSideLength);
        getPixelArray(bwImage, row_pointers, width, height);
        //CUDA PROCESSING
        CudaSafeCall(cudaMalloc((void**)&bwPSFDevice, psfSideLength*psfSideLength*sizeof(float1)));
        CudaSafeCall(cudaMalloc((void**)&bwImageDevice, width*height*sizeof(float1)));
        CudaSafeCall(cudaMemcpy(bwPSFDevice, bwPSF, psfSideLength*psfSideLength*sizeof(float1), cudaMemcpyHostToDevice));
        CudaSafeCall(cudaMemcpy(bwImageDevice, bwImage, width*height*sizeof(float1), cudaMemcpyHostToDevice));
        delete[] bwImage;
        delete[] bwPSF;

        applyPSF<<<gridPSF, blockPSF>>>(width, bwImageDevice, psfSideLength, bwPSFDevice);
        cudaDeviceSynchronize();
        CudaCheckError();
        CudaSafeCall(cudaFree(bwPSFDevice));
        bwImageChar = new uchar1[wAfterBinning*hAfterBinning];
        CudaSafeCall(cudaMalloc((void**)&bwImageCharDevice, wAfterBinning*hAfterBinning*sizeof(uchar1)));
        binImage<<<gridBin, blockBin>>>(wAfterPSF, degreeOfBinning, bwImageDevice, bwImageCharDevice);
        CudaCheckError();
        CudaSafeCall(cudaMemcpy(bwImageChar, bwImageCharDevice, wAfterBinning*hAfterBinning*sizeof(uchar1), cudaMemcpyDeviceToHost));
        CudaSafeCall(cudaFree(bwImageCharDevice));
        CudaSafeCall(cudaFree(bwImageDevice));

        for(int row = 0; row < hAfterBinning; ++row){
          for(int col = 0; col < wAfterBinning; ++col){
            row_pointers[row][col] = bwImageChar[row*wAfterBinning + col].x;
          }
        }
        delete[] bwImageChar;
      }
      else if(color_type == PNG_COLOR_TYPE_GRAY_ALPHA){
        std::cout<<"PNG_COLOR_TYPE_GRAY_ALPHA"<<std::endl;
        float2* bwaPSF = new float2[psfSideLength*psfSideLength];;
        float2* bwaPSFDevice;
        float2* bwaImageDevice;
        uchar2* bwaImageChar;
        uchar2* bwaImageCharDevice;
        float2* bwaImage = new float2[width*height];

        readPSF(psfFile, bwaPSF, psfSideLength);
        getPixelArray(bwaImage, row_pointers, width, height);
        //CUDA PROCESSING
        CudaSafeCall(cudaMalloc((void**)&bwaPSFDevice, psfSideLength*psfSideLength*sizeof(float2)));
        CudaSafeCall(cudaMalloc((void**)&bwaImageDevice, width*height*sizeof(float2)));
        CudaSafeCall(cudaMemcpy(bwaPSFDevice, bwaPSF, psfSideLength*psfSideLength*sizeof(float2), cudaMemcpyHostToDevice));
        CudaSafeCall(cudaMemcpy(bwaImageDevice, bwaImage, width*height*sizeof(float2), cudaMemcpyHostToDevice));
        delete[] bwaPSF;
        delete[] bwaImage;

        applyPSF<<<gridPSF, blockPSF>>>(width, bwaImageDevice, psfSideLength, bwaPSFDevice);
        cudaDeviceSynchronize();
        CudaCheckError();
        CudaSafeCall(cudaFree(bwaPSFDevice));
        bwaImageChar = new uchar2[wAfterBinning*hAfterBinning];
        CudaSafeCall(cudaMalloc((void**)&bwaImageCharDevice, wAfterBinning*hAfterBinning*sizeof(uchar2)));
        binImage<<<gridBin, blockBin>>>(wAfterPSF, degreeOfBinning, bwaImageDevice, bwaImageCharDevice);
        CudaCheckError();
        CudaSafeCall(cudaMemcpy(bwaImageChar, bwaImageCharDevice, wAfterBinning*hAfterBinning*sizeof(uchar2), cudaMemcpyDeviceToHost));
        CudaSafeCall(cudaFree(bwaImageCharDevice));
        CudaSafeCall(cudaFree(bwaImageDevice));

        for(int row = 0; row < hAfterBinning; ++row){
          for(int col = 0; col < wAfterBinning; ++col){
            row_pointers[row][col*2] = bwaImageChar[row*wAfterBinning + col].x;
            row_pointers[row][col*2 + 1] = bwaImageChar[row*wAfterBinning + col].y;
          }
        }
        delete[] bwaImageChar;
      }
      else if(color_type == PNG_COLOR_TYPE_RGB){
        std::cout<<"PNG_COLOR_TYPE_RGB"<<std::endl;
        float3* rgbPSF = new float3[psfSideLength*psfSideLength];;
        float3* rgbPSFDevice;
        float3* rgbImageDevice;
        uchar3* rgbImageChar;
        uchar3* rgbImageCharDevice;
        float3* rgbImage = new float3[width*height];

        readPSF(psfFile, rgbPSF, psfSideLength);
        getPixelArray(rgbImage, row_pointers, width, height);
        //CUDA PROCESSING
        CudaSafeCall(cudaMalloc((void**)&rgbPSFDevice, psfSideLength*psfSideLength*sizeof(float3)));
        CudaSafeCall(cudaMalloc((void**)&rgbImageDevice, width*height*sizeof(float3)));
        CudaSafeCall(cudaMemcpy(rgbPSFDevice, rgbPSF, psfSideLength*psfSideLength*sizeof(float3), cudaMemcpyHostToDevice));
        CudaSafeCall(cudaMemcpy(rgbImageDevice, rgbImage, width*height*sizeof(float3), cudaMemcpyHostToDevice));
        delete[] rgbPSF;
        delete[] rgbImage;

        applyPSF<<<gridPSF, blockPSF>>>(width, rgbImageDevice, psfSideLength, rgbPSFDevice);
        cudaDeviceSynchronize();
        CudaCheckError();
        CudaSafeCall(cudaFree(rgbPSFDevice));
        rgbImageChar = new uchar3[wAfterBinning*hAfterBinning];
        CudaSafeCall(cudaMalloc((void**)&rgbImageCharDevice, wAfterBinning*hAfterBinning*sizeof(uchar3)));
        binImage<<<gridBin, blockBin>>>(wAfterPSF, degreeOfBinning, rgbImageDevice, rgbImageCharDevice);
        CudaCheckError();
        CudaSafeCall(cudaMemcpy(rgbImageChar, rgbImageCharDevice, wAfterBinning*hAfterBinning*sizeof(uchar3), cudaMemcpyDeviceToHost));
        CudaSafeCall(cudaFree(rgbImageCharDevice));
        CudaSafeCall(cudaFree(rgbImageDevice));

        for(int row = 0; row < hAfterBinning; ++row){
          for(int col = 0; col < wAfterBinning; ++col){
            row_pointers[row][col*3] = rgbImageChar[row*wAfterBinning + col].x;
            row_pointers[row][col*3 + 1] = rgbImageChar[row*wAfterBinning + col].y;
            row_pointers[row][col*3 + 2] = rgbImageChar[row*wAfterBinning + col].z;
          }
        }
        delete[] rgbImageChar;
      }
      else if(color_type == PNG_COLOR_TYPE_RGB_ALPHA){
        std::cout<<"PNG_COLOR_TYPE_RGB_ALPHA"<<std::endl;
        float4* rgbaPSF = new float4[psfSideLength*psfSideLength];;
        float4* rgbaPSFDevice;
        float4* rgbaImageDevice;
        uchar4* rgbaImageChar;
        uchar4* rgbaImageCharDevice;
        float4* rgbaImage = new float4[width*height];

        readPSF(psfFile, rgbaPSF, psfSideLength);
        getPixelArray(rgbaImage, row_pointers, width, height);
        //CUDA PROCESSING
        CudaSafeCall(cudaMalloc((void**)&rgbaPSFDevice, psfSideLength*psfSideLength*sizeof(float4)));
        CudaSafeCall(cudaMalloc((void**)&rgbaImageDevice, width*height*sizeof(float4)));
        CudaSafeCall(cudaMemcpy(rgbaPSFDevice, rgbaPSF, psfSideLength*psfSideLength*sizeof(float4), cudaMemcpyHostToDevice));
        CudaSafeCall(cudaMemcpy(rgbaImageDevice, rgbaImage, width*height*sizeof(float4), cudaMemcpyHostToDevice));
        delete[] rgbaPSF;
        delete[] rgbaImage;

        applyPSF<<<gridPSF, blockPSF>>>(width, rgbaImageDevice, psfSideLength, rgbaPSFDevice);
        cudaDeviceSynchronize();
        CudaCheckError();
        CudaSafeCall(cudaFree(rgbaPSFDevice));
        rgbaImageChar = new uchar4[wAfterBinning*hAfterBinning];
        CudaSafeCall(cudaMalloc((void**)&rgbaImageCharDevice, wAfterBinning*hAfterBinning*sizeof(uchar4)));
        binImage<<<gridBin, blockBin>>>(wAfterPSF, degreeOfBinning, rgbaImageDevice, rgbaImageCharDevice);
        CudaCheckError();
        CudaSafeCall(cudaMemcpy(rgbaImageChar, rgbaImageCharDevice, wAfterBinning*hAfterBinning*sizeof(uchar4), cudaMemcpyDeviceToHost));
        CudaSafeCall(cudaFree(rgbaImageCharDevice));
        CudaSafeCall(cudaFree(rgbaImageDevice));
        std::cout<<"CUDA PROCESSING COMPLETED"<<std::endl;

        for(int row = 0; row < hAfterBinning; ++row){
          for(int col = 0; col < wAfterBinning; ++col){
            row_pointers[row][col*4] = rgbaImageChar[row*wAfterBinning + col].x;
            row_pointers[row][col*4 + 1] = rgbaImageChar[row*wAfterBinning + col].y;
            row_pointers[row][col*4 + 2] = rgbaImageChar[row*wAfterBinning + col].z;
            row_pointers[row][col*4 + 3] = rgbaImageChar[row*wAfterBinning + col].w;
          }
        }
        delete[] rgbaImageChar;
      }

      std::string newFileName = originalFileName.substr(0, originalFileName.length() - 4) + "_psfApplied.png";
      writePNG(newFileName.c_str(), row_pointers, wAfterBinning, hAfterBinning, color_type, bit_depth);
      for (int row = 0; row < height; ++row){
        delete[] row_pointers[row];
      }
      delete[] row_pointers;
      totalTimer = clock() - totalTimer;
      printf("\nTOTAL TIME = %f seconds.\n\n",((float) totalTimer)/CLOCKS_PER_SEC);

      return 0;
    }
    else{
      std::cout<<"Usage = ./bin/exe <png> <psf.csv> <psfSideLength> <binning value>"<<std::endl;
      exit(1);
    }
  }
  catch (const std::exception &e){
      std::cerr << "Caught exception: " << e.what() << '\n';
      std::exit(1);
  }
  catch (...){
      std::cerr << "Caught unknown exception\n";
      std::exit(1);
  }

}

__device__ __forceinline__ int floatToOrderedInt(float floatVal){
 int intVal = __float_as_int( floatVal );
 return (intVal >= 0 ) ? intVal : intVal ^ 0x7FFFFFFF;
}
__device__ __forceinline__ float orderedIntToFloat(int intVal){
 return __int_as_float( (intVal >= 0) ? intVal : intVal ^ 0x7FFFFFFF);
}
__device__ __forceinline__ uchar1 normalize(float multiplier, float1 &min, float1 &max, float1 &value){
  float1 result = {0.0f};
  result.x = value.x - min.x / (max.x - min.x);
  result.x *= multiplier;
  return {(unsigned char)result.x};
}
__device__ __forceinline__ uchar2 normalize(float multiplier, float2 &min, float2 &max, float2 &value){
  float2 result = {0.0f, 0.0f};
  result.x = value.x - min.x / (max.x - min.x);
  result.x *= multiplier;
  result.y = value.y - min.y / (max.y- min.y);
  result.y *= multiplier;
  return {(unsigned char)result.x,(unsigned char)result.y};
}
__device__ __forceinline__ uchar3 normalize(float multiplier, float3 &min, float3 &max, float3 &value){
  float3 result = {0.0f, 0.0f, 0.0f};
  result.x = value.x - min.x / (max.x - min.x);
  result.x *= multiplier;
  result.y = value.y - min.y / (max.y - min.y);
  result.y *= multiplier;
  result.z = value.z - min.z / (max.z - min.z);
  result.z *= multiplier;
  return {(unsigned char)result.x,(unsigned char)result.y,(unsigned char)result.z};
}
__device__ __forceinline__ uchar4 normalize(float multiplier, float4 &min, float4 &max, float4 &value){
  float4 result = {0.0f, 0.0f, 0.0f, 0.0f};
  result.x = value.x - min.x / (max.x - min.x);
  result.x *= multiplier;
  result.y = value.y - min.y / (max.y - min.y);
  result.y *= multiplier;
  result.z = value.z - min.z / (max.z - min.z);
  result.z *= multiplier;
  result.w = value.w - min.w / (max.w - min.w);
  result.w *= multiplier;
  return {(unsigned char)result.x,(unsigned char)result.y,(unsigned char)result.z,(unsigned char)result.w};
}

__global__ void applyPSF(int width, float1* subpixelArray, int psfSideLength, float1* psf){
  int bx = blockIdx.x;
  int by = blockIdx.y;
  int regW = width;
  int regPSFL = psfSideLength;
  int x = threadIdx.x;
  int y = threadIdx.y;
  __shared__ float1 result;
  result = {0.0f};
  float1 regSubPixel;
  float1 regPSFValue;
  int1 subPixelI;
  regSubPixel = subpixelArray[(y+by)*regW + (x+bx)];
  regPSFValue = psf[y*regPSFL + x];
  subPixelI = {floatToOrderedInt(regSubPixel.x)};
  atomicMax(&maxI.x, subPixelI.x);
  atomicMin(&minI.x, subPixelI.x);
  atomicAdd(&result.x, regSubPixel.x*regPSFValue.x);
  __syncthreads();
  if(x == 0 && y == 0){
    subpixelArray[by*(regW-regPSFL-1) + bx] = {result.x/(psfSideLength*psfSideLength)};
  }
}
__global__ void applyPSF(int width, float2* subpixelArray, int psfSideLength, float2* psf){
  int bx = blockIdx.x;
  int by = blockIdx.y;
  int regW = width;
  int regPSFL = psfSideLength;
  int x = threadIdx.x;
  int y = threadIdx.y;
  __shared__ float2 result;
  result = {0.0f,0.0f};
  float2 regSubPixel;
  float2 regPSFValue;
  int2 subPixelI;
  regSubPixel = subpixelArray[(y+by)*regW + (x+bx)];
  regPSFValue = psf[y*regPSFL + x];
  subPixelI = {floatToOrderedInt(regSubPixel.x),
    floatToOrderedInt(regSubPixel.y)};
  atomicMax(&maxI.x, subPixelI.x);
  atomicMax(&maxI.y, subPixelI.y);
  atomicMin(&minI.x, subPixelI.x);
  atomicMin(&minI.y, subPixelI.y);
  atomicAdd(&result.x, regSubPixel.x*regPSFValue.x);
  atomicAdd(&result.y, regSubPixel.y*regPSFValue.y);
  __syncthreads();
  if(x == 0 && y == 0){
    subpixelArray[by*(regW-regPSFL-1) + bx] = {result.x/(psfSideLength*psfSideLength),
      result.y/(psfSideLength*psfSideLength)};
  }
}
__global__ void applyPSF(int width, float3* subpixelArray, int psfSideLength, float3* psf){
  int bx = blockIdx.x;
  int by = blockIdx.y;
  int regW = width;
  int regPSFL = psfSideLength;
  int x = threadIdx.x;
  int y = threadIdx.y;
  __shared__ float3 result;
  result = {0.0f,0.0f,0.0f};
  float3 regSubPixel;
  float3 regPSFValue;
  int3 subPixelI;
  regSubPixel = subpixelArray[(y+by)*regW + (x+bx)];
  regPSFValue = psf[y*regPSFL + x];
  subPixelI = {floatToOrderedInt(regSubPixel.x),
    floatToOrderedInt(regSubPixel.y),
    floatToOrderedInt(regSubPixel.z)};
  atomicMax(&maxI.x, subPixelI.x);
  atomicMax(&maxI.y, subPixelI.y);
  atomicMax(&maxI.z, subPixelI.z);
  atomicMin(&minI.x, subPixelI.x);
  atomicMin(&minI.y, subPixelI.y);
  atomicMin(&minI.z, subPixelI.z);
  atomicAdd(&result.x, regSubPixel.x*regPSFValue.x);
  atomicAdd(&result.y, regSubPixel.y*regPSFValue.y);
  atomicAdd(&result.z, regSubPixel.z*regPSFValue.z);
  __syncthreads();
  if(x == 0 && y == 0){
    subpixelArray[by*(regW-regPSFL-1) + bx] = {result.x/(psfSideLength*psfSideLength),
      result.y/(psfSideLength*psfSideLength), result.z/(psfSideLength*psfSideLength)};
  }
}
__global__ void applyPSF(int width, float4* subpixelArray, int psfSideLength, float4* psf){
  int bx = blockIdx.x;
  int by = blockIdx.y;
  int regW = width;
  int regPSFL = psfSideLength;
  int x = threadIdx.x;
  int y = threadIdx.y;
  __shared__ float4 result;
  result = {0.0f,0.0f,0.0f,0.0f};
  float4 regSubPixel;
  float4 regPSFValue;
  int4 subPixelI;
  regSubPixel = subpixelArray[(y+by)*regW + (x+bx)];
  regPSFValue = psf[y*regPSFL + x];
  subPixelI = {floatToOrderedInt(regSubPixel.x),
    floatToOrderedInt(regSubPixel.y),
    floatToOrderedInt(regSubPixel.z),
    floatToOrderedInt(regSubPixel.w)};
  atomicMax(&maxI.x, subPixelI.x);
  atomicMax(&maxI.y, subPixelI.y);
  atomicMax(&maxI.z, subPixelI.z);
  atomicMax(&maxI.w, subPixelI.w);
  atomicMin(&minI.x, subPixelI.x);
  atomicMin(&minI.y, subPixelI.y);
  atomicMin(&minI.z, subPixelI.z);
  atomicMin(&minI.w, subPixelI.w);
  atomicAdd(&result.x, regSubPixel.x*regPSFValue.x);
  atomicAdd(&result.y, regSubPixel.y*regPSFValue.y);
  atomicAdd(&result.z, regSubPixel.z*regPSFValue.z);
  atomicAdd(&result.w, regSubPixel.w*regPSFValue.w);
  __syncthreads();
  if(x == 0 && y == 0){
    subpixelArray[by*(regW-regPSFL-1) + bx] = {result.x/(psfSideLength*psfSideLength),
      result.y/(psfSideLength*psfSideLength), result.z/(psfSideLength*psfSideLength),
      result.w/(psfSideLength*psfSideLength)};
  }
}

__global__ void binImage(int width, int degreeOfBinning, float1* pixelArray, uchar1* output){
  int bx = blockIdx.x;
  int by = blockIdx.y;
  int regW = width;
  int regBin = degreeOfBinning;
  int x = threadIdx.x;
  int y = threadIdx.y;
  float1 regMin = {orderedIntToFloat(minI.x)};
  float1 regMax = {orderedIntToFloat(maxI.x)};
  __shared__ float1 result;
  result = {0.0f};
  float1 regSubPixel;
  regSubPixel = pixelArray[(by*regBin*regW) + (y*regW) + (bx*regBin) + x];
  atomicAdd(&result.x, regSubPixel.x);
  __syncthreads();
  if(x == 0 && y == 0){
    result.x /= (degreeOfBinning*degreeOfBinning);
    output[by*(regW/degreeOfBinning) + bx] = normalize(255.0f, regMin, regMax, result);
  }
}
__global__ void binImage(int width, int degreeOfBinning, float2* pixelArray, uchar2* output){
  int bx = blockIdx.x;
  int by = blockIdx.y;
  int regW = width;
  int regBin = degreeOfBinning;
  int x = threadIdx.x;
  int y = threadIdx.y;
  float2 regMin = {orderedIntToFloat(minI.x),
    orderedIntToFloat(minI.y)};
  float2 regMax = {orderedIntToFloat(maxI.x),
    orderedIntToFloat(minI.y)};
  __shared__ float2 result;
  result = {0.0f,0.0f};
  float2 regSubPixel;
  regSubPixel = pixelArray[(by*regBin*regW) + (y*regW) + (bx*regBin) + x];
  atomicAdd(&result.x, regSubPixel.x);
  atomicAdd(&result.y, regSubPixel.y);
  __syncthreads();
  if(x == 0 && y == 0){
    result.x /= (degreeOfBinning*degreeOfBinning);
    result.y /= (degreeOfBinning*degreeOfBinning);
    output[by*(regW/degreeOfBinning) + bx] = normalize(255.0f, regMin, regMax, result);
  }
}
__global__ void binImage(int width, int degreeOfBinning, float3* pixelArray, uchar3* output){
  int bx = blockIdx.x;
  int by = blockIdx.y;
  int regW = width;
  int regBin = degreeOfBinning;
  int x = threadIdx.x;
  int y = threadIdx.y;
  float3 regMin = {orderedIntToFloat(minI.x),
    orderedIntToFloat(minI.y),
    orderedIntToFloat(minI.z)};
  float3 regMax = {orderedIntToFloat(maxI.x),
    orderedIntToFloat(maxI.y),
    orderedIntToFloat(maxI.z)};
  __shared__ float3 result;
  result = {0.0f,0.0f,0.0f};
  float3 regSubPixel;
  regSubPixel = pixelArray[(by*regBin*regW) + (y*regW) + (bx*regBin) + x];
  atomicAdd(&result.x, regSubPixel.x);
  atomicAdd(&result.y, regSubPixel.y);
  atomicAdd(&result.z, regSubPixel.z);
  __syncthreads();
  if(x == 0 && y == 0){
    result.x /= (degreeOfBinning*degreeOfBinning);
    result.y /= (degreeOfBinning*degreeOfBinning);
    result.z /= (degreeOfBinning*degreeOfBinning);
    output[by*(regW/degreeOfBinning) + bx] = normalize(255.0f, regMin, regMax, result);
  }
}
__global__ void binImage(int width, int degreeOfBinning, float4* pixelArray, uchar4* output){
  int bx = blockIdx.x;
  int by = blockIdx.y;
  int regW = width;
  int regBin = degreeOfBinning;
  int x = threadIdx.x;
  int y = threadIdx.y;
  float4 regMin = {orderedIntToFloat(minI.x),
    orderedIntToFloat(minI.y),
    orderedIntToFloat(minI.z),
    orderedIntToFloat(minI.w)};
  float4 regMax = {orderedIntToFloat(maxI.x),
    orderedIntToFloat(maxI.y),
    orderedIntToFloat(maxI.z),
    orderedIntToFloat(maxI.w)};
  __shared__ float4 result;
  result = {0.0f,0.0f,0.0f,0.0f};
  float4 regSubPixel;
  regSubPixel = pixelArray[(by*regBin*regW) + (y*regW) + (bx*regBin) + x];
  atomicAdd(&result.x, regSubPixel.x);
  atomicAdd(&result.y, regSubPixel.y);
  atomicAdd(&result.z, regSubPixel.z);
  atomicAdd(&result.w, regSubPixel.w);
  __syncthreads();
  if(x == 0 && y == 0){
    result.x /= (degreeOfBinning*degreeOfBinning);
    result.y /= (degreeOfBinning*degreeOfBinning);
    result.z /= (degreeOfBinning*degreeOfBinning);
    result.w /= (degreeOfBinning*degreeOfBinning);
    output[by*(regW/degreeOfBinning) + bx] = normalize(255.0f, regMin, regMax, result);
  }
}
