/** \file   Unity.cuh
 * \author  Jackson Parker
 * \date    1 Sep 2019
 * \brief   File containing Unity class and CUDA error handlers
 */

#ifndef UNITY_CUH
#define UNITY_CUH

#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <cuda.h>

#define CUDA_ERROR_CHECK
#define CudaSafeCall( err ) __cudaSafeCall( err, __FILE__, __LINE__ )
#define CudaCheckError()    __cudaCheckError( __FILE__, __LINE__ )

/**
 * \brief CudaSafeCall called as a function wrapper will
 * identify CUDA errors associated with internal call
 */
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
/**
 * \brief CudaCheckError() called after CUDA kernel execution will
 * identify CUDA errors occuring during the kernel. Uncommenting
 * err = cudaDeviceSynchronize(); on line 55 will allow for more
 * careful checking.
 */
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

  /**
  * \brief Internal way of representing pointer location and
  * CUDA memory type.
  */
  typedef enum MemoryState{
    null = 0,
    cpu = 1,
    gpu = 2,
    both = 3,
    pinned = 4
  } MemoryState;

  namespace{
    /**
    * \brief base unity exception.
    */
    struct UnityException{
      std::string msg;
      UnityException(){
        msg = "Unknown Unity Exception";
      }
      UnityException(std::string msg) : msg("Unity Exception: " + msg){}
      virtual const char* what() const throw(){
        return msg.c_str();
      }
    };
    /**
    * \brief Custom exception called when attempting to transition
    * Unity memory to state that would cause segfault.
    */
    struct IllegalUnityTransition : public UnityException{
      std::string msg;
      IllegalUnityTransition(){
        msg = "Illegal Unity memory transfer";
      }
      IllegalUnityTransition(std::string msg) : msg("Illegal Unity memory transfer: " + msg){}
      virtual const char* what() const throw(){
        return msg.c_str();
      }
    };
    /**
    * \brief Custom exception used when an action would lead to
    * segfault due to nullptr.
    */
    struct NullUnityException : public UnityException{
      std::string msg;
      NullUnityException(){
        msg = "Illegal attempt to use null set Unity";
      }
      NullUnityException(std::string msg) : msg("Illegal attempt to use null set Unity: " + msg){}
      virtual const char* what() const throw(){
        return msg.c_str();
      }
    };

    /**
    * \brief returns a string based on the MemoryState type it is fed.
    */
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


  /**
  * \class Unity
  * \brief SSRLCV CUDA memory handler class.
  * \todo implement pinned and unified memory methods for this class
  */
  template<typename T>
  class Unity{

  public:
    /**
    * used to keep track of most up to date memory
    * \warning user must keep track of this on their own to be useful
    */
    MemoryState fore;
    MemoryState state;/**\brief current state of data*/

    T* device;/**\brief pointer to device memory (gpu)*/
    T* host;/**brief pointer to host memory (cpu)*/
    unsigned long numElements;/**\brief number of elements in *device or *host*/

    /**
    * \brief default constructor
    */
    Unity();
    /**
    * \brief constructor
    * \tparam T datatype of data
    * \param data pointer to dynamically allocated array on host or device
    * \param numElements number of elements inside data pointer
    * \param state MemoryState of data
    */
    Unity(T* data, unsigned long numElements, MemoryState state);
    /**
    * \brief destructor
    */
    ~Unity();

    /**
    * This method will clear memory in a specified location.
    * \param MemoryState - location to clear
    */
    void clear(MemoryState state = both);
    /**
    * This method will transfer memory to a specified location.
    * \param MemoryState - location to transfer too
    * \warning this method will not delete memory in previous location
    */
    void transferMemoryTo(MemoryState state);
    /**
    * This method will set the memory state to a specified location.
    * \param MemoryState - location to set memory to
    * \warning this is a hard set and will delete memory in previous location
    */
    void setMemoryState(MemoryState state);
    /**
    * This method will clear data and set to parameterized data.
    * \param data - must be of previous type
    * \param numElements - size of new data
    * \param MemoryState - location of new data
    */
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
          this->state = null;
          this->fore = null;
        }
        break;
      case gpu:
        if(this->device != nullptr){
          CudaSafeCall(cudaFree(this->device));
          this->state = null;
          this->fore = null;
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
  void Unity<T>::setMemoryState(MemoryState state){
    if(state == this->state){
      std::cerr<<"WARNING: hard setting of memory state to same memory state does nothing: "<<memoryStateToString(this->state)<<std::endl;
    }
    else if(this->state == null){
      throw NullUnityException("Cannot setMemoryState of a null Unity");
    }
    else if(state == null) this->clear();
    else if(this->state == both){
      if(cpu == this->fore){
        this->transferMemoryTo(gpu);
      }
      else if(gpu == this->fore){
        this->transferMemoryTo(cpu);
      }
      this->fore = both;
    }
    else{
      this->transferMemoryTo(state);
      if(state == cpu){
        this->clear(gpu);
        this->fore = cpu;
      }
      else{
        this->clear(cpu);
        this->fore = gpu;
      }
    }
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
