#include <iostream>
#include "cuda_runtime_api.h"

using namespace std;

int main(int argc, char ** argv) { 
    int count;
    cudaGetDeviceCount(&count); 
    if(count == 0) { 
        std::cerr << "Could not find a CUDA device";
        return 1;
    }
    if(count != 1) { 
        std::cerr << "Warning: Expected exactly one CUDA device, got " << count; 
    }

    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, 0); 

    std::cout << prop.major << prop.minor;
    return 0;
}