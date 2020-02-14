/** 
* \file   Unity.cuh
* \author  Jackson Parker
* \date    1 Sep 2019
* \brief   File containing Unity class and CUDA error handlers
* \details This file contains the Unity header only CUDA memory handler
* as well as the CUDA error checkers like the CudaSafeCall() function wrapper 
* and the CudaCheckError() function.
*/

#ifndef UNITY_CUH
#define UNITY_CUH

#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <cuda.h>
#include <stdio.h>
#include <string>
#include <cstring>
#include <iostream> 

#define CUDA_ERROR_CHECK
#define CudaSafeCall( err ) __cudaSafeCall( err, __FILE__, __LINE__ )
#define CudaCheckError()    __cudaCheckError( __FILE__, __LINE__ )
/**
* \defgroup error_util
* \{
*/

/**
 * \brief CUDA error checking function wrapper. 
 * \details This should be used as a function wrapper for CUDA error checking 
 * on cudaMemcpy, cudaFree, and cudaMalloc calls.
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
 * \brief Error checker function after kernel calls.
 * \details Calling this function after kernel calls will 
 * allow for CUDA error checking on kernels with the error line 
 * likely coming from the next thread fence 
 * (cuda memory transfers or cudaDeviceSynchronize()).
 * \note Uncommenting err = cudaDeviceSynchronize(); 
 * on line 55 in Unity.cuh (and doing make clean;make) 
 * will allow for more careful checking. (this will slow things down)
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
/**
* \}
*/
namespace ssrlcv{
  /**
  * \defgroup unity
  * \{
  */

  /**
  * \brief Internal way of representing pointer location and
  * CUDA memory type.
  * \note only types null,cpu,gpu,both are supported right now. 
  */
  typedef enum MemoryState{
    null = 0,///< device = host = nullptr
    cpu = 1,///< host != nullptr, device = nullptr
    gpu = 2,///< device != nullptr, host = nullptr
    both = 3,///< device != nullptr, host != nullptr
    pinned = 4,///< not supported yet
    unified = 5///< not supported yet
  } MemoryState;

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
        return "both";
      case pinned:
        return "pinned";
      case unified:
        return "unified";
      default:
        std::clog<<"ERROR: unknown MemoryState when calling memoryStateToString()"<<std::endl;
        exit(-1);
    }
  }

  /**
  * \brief base unity exception.
  * \ingroup error_util
  */
  struct UnityException : std::exception{
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
  * \brief Exception thrown when attempting an illegal memory transition. 
  * \details This exception is primarily used to avoid segfaults. It is thrown 
  * when attempting to transfer to an unknown or impossible memory state.
  * \ingroup error_util
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
  * \brief Exception thrown with operation on null memory.
  * \details This exception is primarily thrown when trying to operate on a null Unity 
  * or when trying to setMemory to null. (clear() should be used for that) 
  * \ingroup error_util
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
  * \class Unity
  * \brief This data structure is designed to hold an array of any data 
  * type utilized in cuda processing. 
  * \todo implement pinned and unified memory methods for this class
  */
  template<typename T>
  class Unity{
  private:
    /**
    * \brief Current MemoryState
    * \details This is a private class member that is used to keep track of the 
    * MemoryState of Unity. Allowing users to change this variable would 
    * result in exceptions when attempting to transfer memory.
    * \note Can be retrieved in Unity<T>::getMemoryState()
    */
    MemoryState state;
    /**
    * \brief Location of most up to date memory
    * \details This class member is used  to ensure that most updated memory 
    * is handled properly when state = both. To ensure this, user functions 
    * should use setFore(MemoryState) after updating data held in device or host with the 
    * corresponding MemoryState. 
    * \note Users do not need to pay attention to this when state != both
    */
    MemoryState fore;

    unsigned long numElements;///< \brief number of elements in *device or *host

  public:
    
    T* device;///< \brief pointer to device memory (gpu)
    T* host;///< \brief pointer to host memory (cpu)

    /**
    * \brief Default contructor 
    * \details This constructor will just set everything to null and numElements to 0.
    */
    Unity();

    /**
    * \brief Primary constructor
    * \tparam T datatype of data
    * \param data pointer to dynamically allocated array on host or device
    * \param numElements number of elements inside data pointer
    * \param state MemoryState of data
    */
    Unity(T* data, unsigned long numElements, MemoryState state);

    /**
    * \brief Copy constructor (this becomes a copy of argument)
    * \param copy Unity<T>* to be copied
    */
    Unity(Unity<T>* copy);

    /**
    * \brief Destructor
    * \details This destructor just calls clear(both). 
    * \see Unity<T>::clear
    */
    ~Unity();
    /**
    * \brief Get the number of elements in the unity
    * \details Used so that users cannot alter numElements and cause
    * memory access issues.
    */
    unsigned long size();
    /**
    * \brief Resize Unity and keep data under specified index.
    * \details This method will remove data that is positioned after past the number 
    * of elements specified for resize. This will keep this->fore and this->state the same.
    * \param resizeLength - numElements to keep
    */
    void resize(unsigned long resizeLength);
    /**
    * \brief Clear memory in a specified location.
    * \details This method deletes memory in a specific location. Default argument is 
    * both, which would effectively delete the contents of Unity and set this->numElments to 0.
    * \param state - location to clear (default = clearing both this->device and this->host)
    */
    void clear(MemoryState state = both);
    /**
    * \brief Convert elements in a specific location to T().
    * \details This method will call the default constructor of T() for data 
    * in a specific location. Default argument is both which would do so for both cpu and 
    * gpu.
    * \param state - location to zero out (default = zero out both this->device and this->host)
    * \warning if this->state == both and location to be zero'd out is not also both, zero'd out location 
    * becomes this->fore
    */
    void zeroOut(MemoryState state = both);
    /**
    * \brief Get current MemoryState.
    * \details This method should be used within user functions to ensure that data 
    * is located in the proper place before operating on either device or host. 
    * \returns current memory state
    */
    MemoryState getMemoryState();
    /**
    * \brief Ensure that memory is in a specified MemoryState.
    * \details When this method is called with an argument of cpu or gpu, the 
    * other memory location will be cleared. This method will ensure that the 
    * data stored at the MemoryState fore is located in the specified state. 
    * \param state - MemoryState location to set memory to 
    * \warning This method can overwrite recently updated memory if user is not 
    * setting fore properly when this->state = both. 
    */
    void setMemoryState(MemoryState state);
    /**
    * \brief Get MemoryState of recently updated data when state = both.
    * \details This method should be used by user functions to check where data was 
    * last updated. 
    * \returns MemoryState of data with most recent updates.
    */
    MemoryState getFore();
    /**
    * \brief Set MemoryState after updating memory in a location when state = both.
    * \details This method should be used by user functions to ensure that Unity 
    * transfer and utility functions will handle recently updated memory properly.
    * \param state - MemoryState of recently updated data (both is an illegal argument)
    * \warning If this is not set properly, there is a risk of recent changes to data 
    * in device or host transferMemoryTo or setMemoryStateTo.
    */
    void setFore(MemoryState state);
    /**
    * \brief This method will transfer memory to a specified location.
    * \details This is similar to the Unity<T>::setMemoryState function, but is a "soft" version.
    * In other words, this function will not delete memory in either device or host 
    * regardless of the transfer and will result in state = both.
    * \param state - MemoryState location to transfer too
    * \warning this can overwrite most up to date memory if user is not setting fore 
    * after updates to a location when this->state = both
    */
    void transferMemoryTo(MemoryState state);
    /**
    * \brief Reset Unity based on input data.
    * \details This method will erase all data in Unity and replace it with 
    * data provided to this method. This does the same as the primary constructor. 
    * \param data - must be of previous type (can use nullptr for blank array of length numElements)
    * \param numElements - size of new data
    * \param state - location of new data (must be cpu or gpu)
    */
    void setData(T* data, unsigned long numElements, MemoryState state);
    /**
    * \brief Return a copy of this Unity<T>* with a specified memory state.
    * \details This method will return a copy of this Unity with a new or the same 
    * memory location. Due to this returning a Unity, destination can be both. This 
    * will leave the current Unity untouches as in this->fore and this->state remain 
    * the same and the copied Unity will have this->fore and this->state = destination.
    * \param destination - location of copied data (host,device,both)
    * \returns copy of data located in this
    */
    Unity<T>* copy(MemoryState destination);

    /**
    * \brief Print information about the Unity.
    */
    void printInfo();
  };

  template<typename T>
  Unity<T>::Unity(){
    this->host = nullptr;
    this->device = nullptr;
    this->state = null;
    this->fore = null;
    this->numElements = 0;
  }
  template<typename T>
  Unity<T>::Unity(T* data, unsigned long numElements, MemoryState state){
    this->host = nullptr;
    this->device = nullptr;
    this->state = null;
    this->fore = null;
    this->setData(data, numElements, state);
  }
  template<typename T>
  Unity<T>::Unity(Unity<T>* copy){
    if(copy == nullptr || copy->state == null){
      throw NullUnityException("attempt to use Unity<T> copy constructor with a null unity");
    }
    if(this->state <= 3){
      this = copy->copy(copy->state);
    }
    else{
      throw IllegalUnityTransition("attempt to use Unity<T> copy constructor with unsupported memory state (supported states = both, cpu & gpu");
    }
  }
  template<typename T>
  Unity<T>::~Unity(){
    this->clear();
  }
  template<typename T>
  unsigned long Unity<T>::size(){
    return this->numElements;
  }
  template<typename T> 
  void Unity<T>::resize(unsigned long resizeLength){
    if(this->state == null){
      throw NullUnityException("cannot resize and empty Unity");
    }
    else if(this->numElements <= resizeLength || resizeLength == 0U){
      throw IllegalUnityTransition("in Unity<T>::resize resizeLength must be > 0 and < this->numElements");
    }
    else if(this->state <= 3){
      if(this->state == cpu || this->state == both){
        T* replacement = new T[resizeLength]();
        std::memcpy(replacement,this->host,resizeLength*sizeof(T));
        delete[] this->host;
        this->host = replacement;
      }
      if(this->state == gpu || this->state == both){
        T* replacement  = nullptr;
        CudaSafeCall(cudaMalloc((void**)&replacement,resizeLength*sizeof(T)));
        CudaSafeCall(cudaMemcpy(replacement,this->device,resizeLength*sizeof(T),cudaMemcpyDeviceToDevice));
        CudaSafeCall(cudaFree(this->device));
        this->device = replacement;
      }
      this->numElements = resizeLength;
    }
    else{
      std::string error = "please implement resize for newly supported MemoryState = ";
      error += memoryStateToString(this->state);
      throw UnityException(error);
    }
  }
  template<typename T>
  void Unity<T>::clear(MemoryState state){
    if(state == null){
      std::clog<<"WARNING: Unity<T>::clear(ssrlcv::null) does nothing"<<std::endl;
      return;
    }
    else if(state != both && this->state != both && this->state != state){
      std::clog<<"WARNING: Attempt to clear null memory in location "
      <<memoryStateToString(state)<<"...action prevented"<<std::endl;
      return;
    }
    else if(this->state == null){
      std::clog<<"WARNING: Attempt to clear null (empty) Unity...action prevented"<<std::endl;
      return;
    }
    else if(this->state <= 3){//currently supported types
      if(state == cpu || (state == both && this->host != nullptr)){
        delete[] this->host;
        this->host = nullptr;
      }
      if(state == gpu || (state == both && this->device != nullptr)){
        CudaSafeCall(cudaFree(this->device));
        this->device = nullptr;
      }
      this->fore = (state == both) ? null : (state == cpu) ? gpu : cpu;
      this->state = (state == both) ? null : (state == cpu) ? gpu : cpu;
    }
    else{
      throw IllegalUnityTransition("unknown memory state in clear() (supported states = both, cpu & gpu)");
    }
  }
  template<typename T>
  void Unity<T>::zeroOut(MemoryState state){
    if(state == null){
      throw NullUnityException("cannot zero out an empty unity with state null");
    }
    else if(state <= 3){
      if (state != both && this->state != both && state != this->state){
        std::string error = "cannot zero out ";
        error += (this->state == cpu) ? "device because state == cpu" : "this->host because state == gpu";
        throw IllegalUnityTransition(error);
      }
      else{
        if(state == cpu || (state == both && this->host != nullptr)){
          delete[] this->host;
          this->host = new T[this->numElements]();
        }
        if(state == gpu || (state == both && this->device != nullptr)){
          T* zerod = (state == both && this->host != nullptr) ? this->host : new T[this->numElements]();
          CudaSafeCall(cudaMemcpy(this->device,zerod,this->numElements*sizeof(T),cudaMemcpyHostToDevice));
          if(state != both || this->host == nullptr) delete[] zerod;
        }
        this->fore = state;
      }
    }
    else{
      throw IllegalUnityTransition("unknown memory state in zeroOut() (supported states = both, cpu & gpu)");
    }
  }
  template<typename T>
  MemoryState Unity<T>::getMemoryState(){
    return this->state;
  }
  template<typename T>
  void Unity<T>::setMemoryState(MemoryState state){
    if(state == this->state){
      std::clog<<"WARNING: hard setting of memory state to same memory state does nothing: "<<memoryStateToString(this->state)<<std::endl;
      return;
    }
    else if(this->state == null){
      throw NullUnityException("Cannot setMemoryState of a null Unity");
    }
    else if(state == null) this->clear();
    else if(state == both){
      if(cpu == this->fore) this->transferMemoryTo(gpu);
      else if(gpu == this->fore) this->transferMemoryTo(cpu);
    }
    else{
      if(this->fore != state) this->transferMemoryTo(state);
      if(state == cpu) this->clear(gpu);
      else if(state == gpu) this->clear(cpu);
    }
  }
  template<typename T>
  void Unity<T>::setData(T* data, unsigned long numElements, MemoryState state){
    if(state == null){
      throw NullUnityException("cannot use null as state of T* data in Unity<T>::setData");
    }
    else if(numElements == 0){
      throw IllegalUnityTransition("cannot fill Unity with T* data, numElements = 0");
    }
    else if(data != nullptr && (data == this->host || data == this->device)){
      throw UnityException("cannot use Unity<T>::setData where T* data is this->host or this->device");
    }
    if(this->state != null) this->clear();
    this->numElements = numElements;
    this->state = state;
    this->fore = state;
    if(data == nullptr){
      if(state == null || state > 3){//greater than three means pinned or unified
        throw IllegalUnityTransition("attempt to instantiate unkown MemoryState fron nullptr (supported states = both, cpu & gpu)");
      }
      if(state == cpu || state == both){
        this->host = new T[numElements]();
      }
      if(state == gpu || state == both){
        CudaSafeCall(cudaMalloc((void**)&this->device,this->numElements*sizeof(T)));
        this->zeroOut(gpu);
      }
    }
    else if(state <= 2){
      if(state == cpu) this->host = data;
      else if(state == gpu) this->device = data;
    }
    else{
      if(state == both){
        throw IllegalUnityTransition("cannot use both as state of T* data, not enough information to use data pointer");
      }
      else{
        std::string error = "currently no support for Unity<T>::setData with T* data at MemoryState = ";
        error += memoryStateToString(state);
        throw IllegalUnityTransition(error);
      }
    }
  }
  template<typename T> 
  MemoryState Unity<T>::getFore(){
    return this->fore;
  }
  template<typename T>
  void Unity<T>::setFore(MemoryState state){
    if(this->state == null){
      throw NullUnityException("attempt to Unity<T>::setFore(MemoryState state) when this->state == null");
    }
    else if(this->fore == state){
      std::clog<<"WARNING: Unity<T>::setFore(MemoryState state) when state == this->fore does nothing"<<std::endl;
      return;
    }
    else if(state == both){
      std::clog<<"ERROR: cannot set fore to both manually: \n\tuse setMemoryState(both) or transferMemoryTo((this->fore == gpu) ? cpu : gpu)"<<std::endl;
      exit(-1);
    }
    else if(this->state != both && this->state != state){
      if(this->state == cpu){
        throw IllegalUnityTransition("attempt to Unity<T>::setFore(MemoryState state) to gpu when this->device == nullptr");
      }
      else{
        throw IllegalUnityTransition("attempt to Unity<T>::setFore(MemoryState state) to cpu when this->host == nullptr");
      }
    }
    this->fore = state;
  }
  template<typename T>
  void Unity<T>::transferMemoryTo(MemoryState state){
    if(this->state == null || sizeof(T)*this->numElements == 0){
      throw NullUnityException("thrown in Unity<T>::transferMemoryTo()");
    }
    else if(state == null){
      throw IllegalUnityTransition("Cannot transfer unity memory to null");
    }
    if(state <= 3){
      if(this->fore == state){
        std::clog<<"WARNING: transfering memory to location of fore does nothing: "<<memoryStateToString(state)<<std::endl;
        return;
      }
      else{
        if(this->state != both){
          if(this->state == cpu && this->device == nullptr){
            CudaSafeCall(cudaMalloc((void**)&this->device,this->numElements*sizeof(T)));
          }
          else if(this->state == gpu && this->host == nullptr){
            this->host = new T[this->numElements]();
          }
          this->state = both;
        }
        if(this->fore == cpu){
          CudaSafeCall(cudaMemcpy(this->device,this->host,this->numElements*sizeof(T),cudaMemcpyHostToDevice));
        }
        else if(this->fore == gpu){
          CudaSafeCall(cudaMemcpy(this->host,this->device,this->numElements*sizeof(T),cudaMemcpyDeviceToHost));
        }
      } 
      this->fore = both;
    }
    else{
      throw IllegalUnityTransition("unsupported memory destination in Unity<T>::transferMemoryTo (supported states = both, cpu & gpu)");
    }
  }
  template<typename T> 
  Unity<T>* Unity<T>::copy(MemoryState destination){
    Unity<T>* copied = nullptr;
    if(destination == null){
      throw NullUnityException("cannot use null as destination for Unity<T>::copy()");
    }
    else if(destination <= 3){//this would be unsupported currently 
      copied = new Unity<T>(nullptr,this->numElements,destination);
      if(this->fore == both || this->fore == destination){
        if(destination == cpu || destination == both){
          std::memcpy(copied->host,this->host,this->numElements*sizeof(T));
        }
        if(destination == gpu || destination == both){
          CudaSafeCall(cudaMemcpy(copied->device,this->device,this->numElements*sizeof(T),cudaMemcpyDeviceToDevice));
        } 
      }
      else{
        if(this->fore == cpu){//cpu to gpu
          CudaSafeCall(cudaMemcpy(copied->device,this->host,this->numElements*sizeof(T),cudaMemcpyHostToDevice));
          copied->setFore(cpu);
        }
        else if(this->fore == gpu){//gpu to cpu
          CudaSafeCall(cudaMemcpy(copied->host,this->device,this->numElements*sizeof(T),cudaMemcpyDeviceToHost));
          copied->setFore(gpu);
        }
        if(destination == both) copied->transferMemoryTo(both);
      }    
    }
    else{
      throw IllegalUnityTransition("unsupported memory destination in Unity<T>::copy (supported states = both, cpu & gpu)");
    }
    return copied;
  }
  template<typename T> 
  void Unity<T>::printInfo(){
    std::cout<<"numElements = "<<this->numElements;
    std::cout<<" state = "<<memoryStateToString(this->state);
    std::cout<<" fore = "<<memoryStateToString(this->fore)<<std::endl;
  }

  /**
  * \}
  */
}


#endif /*UNITY_CUH*/
