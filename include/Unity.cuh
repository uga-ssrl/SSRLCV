#ifndef UNITY_CUH
#define UNITY_CUH

#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <cuda.h>

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
  // err = cudaDeviceSynchronize();
  if (cudaSuccess != err) {
    fprintf(stderr, "cudaCheckError() with sync failed at %s:%i : %s\n",
    file, line, cudaGetErrorString(err));
    exit(-1);
  }
#endif

  return;
}

namespace ssrlcv{

  typedef enum MemoryState{
    null = 0,
    cpu = 1,
    gpu = 2,
    both = 3,
    pinned = 4
  } MemoryState;

  namespace{
    struct IllegalUnityTransition{
      std::string msg;
      IllegalUnityTransition(){
        msg = "Illegal Unity memory transfer";
      }
      IllegalUnityTransition(std::string msg) : msg("Illegal Unity memory transfer: " + msg){}
      virtual const char* what() const throw(){
        return msg.c_str();
      }
    };

    struct NullUnityException{
      std::string msg;
      NullUnityException(){
        msg = "Illegal attempt to use null set Unity";
      }
      NullUnityException(std::string msg) : msg("Illegal attempt to use null set Unity: " + msg){}
      virtual const char* what() const throw(){
        return msg.c_str();
      }
    };

    inline std::string memoryStateToString(MemoryState state){
      switch(state){
        case null:
          return "null";
        case cpu:
          return "cpu";
        case gpu:
          return "gpu";
        case both:
          return "both cpu & gpu";
        default:
          std::cerr<<"ERROR: unknown MemoryState when calling memoryStateToString()"<<std::endl;
          exit(-1);
      }
    }
  }

  template<typename T>
  class Unity{

  public:
    MemoryState fore;//can be used to keep track of recently updated memory
    MemoryState state;

    T* device;
    T* host;
    unsigned long numElements;

    Unity();
    Unity(T* data, unsigned long numElements, MemoryState state);
    ~Unity();

    void clear(MemoryState state = both);//hard clear
    void transferMemoryTo(MemoryState state);//soft set - no deletes
    void setData(T* data, unsigned long numElements, MemoryState state);//hard set
  };

  template<typename T>
  Unity<T>::Unity(){
    this->host = nullptr;
    this->device = nullptr;
    this->state = null;
    this->numElements = 0;
  }
  template<typename T>
  Unity<T>::Unity(T* data, unsigned long numElements, MemoryState state){
    this->host = nullptr;
    this->device = nullptr;
    this->state = state;
    this->fore = state;
    this->numElements = numElements;
    if(state == cpu) this->host = data;
    else if(state == gpu) this->device = data;
    else{
      throw IllegalUnityTransition("cannot instantiate memory on device and host with only one pointer");
    }
  }
  template<typename T>
  Unity<T>::~Unity(){
    this->clear();
  }

  template<typename T>
  void Unity<T>::clear(MemoryState state){
    if(state == null){
      std::cerr<<"WARNING: Unity<T>::clear(ssrlcv::null) does nothing"<<std::endl;
      return;
    }
    else if(state != both && this->state != both && this->state != state){
      std::cerr<<"WARNING: Attempt to clear null memory in location "
      <<memoryStateToString(state)<<"...action prevented"<<std::endl;
      return;
    }
    switch(this->state){
      case null:
        std::cerr<<"WARNING: Attempt to clear null (empty) Unity...action prevented"<<std::endl;
        break;
      case cpu:
        if(this->host != nullptr){
          delete[] this->host;
        }
        break;
      case gpu:
        if(this->device != nullptr){
          CudaSafeCall(cudaFree(this->device));
        }
        break;
      case both:
        if(state != both){
          if(state == cpu && this->host != nullptr){
            delete[] this->host;
            this->host = nullptr;
            this->state = gpu;
            this->fore = gpu;
          }
          else if(state == gpu && this->device != nullptr){
            CudaSafeCall(cudaFree(this->device));
            this->device = nullptr;
            this->state = cpu;
            this->fore = cpu;
          }
          return;
        }
        else{
          if(host != nullptr){
            delete[] this->host;
          }
          if(device != nullptr){
            CudaSafeCall(cudaFree(this->device));
          }
        }
        break;
      default:
        throw IllegalUnityTransition("unkown memory state");
    }
    this->host = nullptr;
    this->device = nullptr;
    this->state = null;
    this->fore = null;
    this->numElements = 0;
  }
  template<typename T>
  void Unity<T>::transferMemoryTo(MemoryState state){
    if(this->state == null || sizeof(T)*this->numElements == 0){
      throw NullUnityException("thrown in Unity<T>::transferMemoryTo()");
    }
    else if(state == null){
      throw IllegalUnityTransition("Cannot transfer unity memory to null");
    }
    else if(this->state == state){
      std::cerr<<"WARNING: transfering memory to same location does nothing: "<<memoryStateToString(state)<<std::endl;
      return;
    }
    else if(state == both){
      if(this->fore == cpu){
        if(this->device == nullptr){
          CudaSafeCall(cudaMalloc((void**)&this->device, sizeof(T)*this->numElements));
        }
        CudaSafeCall(cudaMemcpy(this->device,this->host, sizeof(T)*this->numElements, cudaMemcpyHostToDevice));
      }
      else if(this->fore == gpu){
        if(this->host == nullptr){
          this->host = new T[this->numElements];
        }
        CudaSafeCall(cudaMemcpy(this->host, this->device, sizeof(T)*this->numElements, cudaMemcpyDeviceToHost));
      }
    }
    else{
      if(this->fore == state){
        std::cerr<<"WARNING: most updated memory location is being overwritten: "<<
        "fore = "<<memoryStateToString(this->fore)<<std::endl;
      }
      if(state == gpu){
        if(this->device == nullptr){
          CudaSafeCall(cudaMalloc((void**)&this->device, sizeof(T)*this->numElements));
        }
        CudaSafeCall(cudaMemcpy(this->device, this->host, sizeof(T)*this->numElements, cudaMemcpyHostToDevice));
      }
      else if(state == cpu){
        if(this->host == nullptr){
          this->host = new T[this->numElements];
        }
        CudaSafeCall(cudaMemcpy(this->host, this->device, sizeof(T)*this->numElements, cudaMemcpyDeviceToHost));
      }
      else{
        throw IllegalUnityTransition("unkown memory state");
      }
    }
    this->state = both;
    this->fore = both;
  }
  template<typename T>
  void Unity<T>::setData(T* data, unsigned long numElements, MemoryState state){
    this->clear();
    this->state = state;
    this->fore = state;
    this->numElements = numElements;
    if(state == cpu) this->host = data;
    else if(state == gpu) this->device = data;
    else{
      throw IllegalUnityTransition("cannot instantiate memory on device and host with only one pointer");
    }
  }
}

#endif /*UNITY_CUH*/
