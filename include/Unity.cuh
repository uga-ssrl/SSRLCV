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

#include <cuda.h>
#include <thrust/device_ptr.h>
#include <thrust/sort.h>
#include <thrust/remove.h>
#include <thrust/copy.h>
#include <stdio.h>
#include <cstdio>
#include <string>
#include <cstring>
#include <iostream> 
#include <fstream>
#include <typeinfo>
#include "Logger.hpp"

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
    unified = 4,///< device == host
    nc = 10///< utilized for default parameters and stands for no change (fore preservation)
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
      case unified:
        return "unified";
      case nc:
        return "no change (this should only be used to help with data manipulation methods)";
      default:
        logger.err<<"ERROR: unknown MemoryState when calling memoryStateToString()\n";
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
  * \brief Exception thrown checkpoint file io error.
  * \details 
  * \ingroup error_util
  */
  struct CheckpointException : public UnityException{
    std::string msg;
    CheckpointException(){
      msg = "Error in writing checkpoint";
    }
    CheckpointException(std::string msg) : msg("Checkpoint Error: " + msg){}
    virtual const char* what() const throw(){
      return msg.c_str();
    }
  };
  /**
  * \class Unity
  * \brief This data structure is designed to hold an array of any data 
  * type utilized in cuda processing. 
  * \todo implement pinned and unified memory methods for this class
  * \todo make sure to flesh out docs for new methods
  * \todo add identifier variable of some sort
  * \todo change getMemoryState to state() and getFore to fore() (have more verbose names for variables)
  */
  template<class T>
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


    bool pinned;

    unsigned long numElements;///< \brief number of elements in *device or *host

  public:
    typedef bool (*comp_ptr)(const T& a, const T& b);
    typedef bool (*pred_ptr)(const T& a);

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
    Unity(T* data, unsigned long numElements, MemoryState state, bool pinned = false);

    /**
    * \brief Copy constructor (this becomes an exact copy of the argument)
    * \param copy Unity<T>* to be copied
    */
    Unity(Unity<T>* copy);

    /**
    * \brief Copy constructor with predicate
    * \param copy Unity<T>* to be copied
    * \param predicate - predicate for copy_if
    */
    Unity(Unity<T>* copy,pred_ptr predicate);
    
    /**
    * \brief Checkpoint constructor
    * \param path path to .uty file 
    */
    Unity(std::string path, bool pinned = false);

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
    * both, which would effectively delete the contents of Unity and set this->numElements to 0.
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
    
    //TODO Comment
    bool isPinned();
    void pin();
    void unpin();

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
    void setData(T* data, unsigned long numElements, MemoryState state, bool pinned = false);
    
    /**
    * \brief remove elements of a unity
    * \details This will remove elements of a unity based on a predicate. 
    * for details on how to create a usable device function pointer, see 
    * example.
    * \param predicate - predicate for remove if
    * \param destination - location where data will be after alteration - optional
    * \see example_unity.cu
    * \note nc here means that the fore will default to origin
    */
    void remove(pred_ptr predicate, MemoryState destination = nc);

    /**
    * \brief sort a unity based on overloaded < and > operators
    * \details uses thrust::stable_sort
    * \param greater - indicate if sort is descending instead of ascending - optional 
    * \param destination - location where data will be after alteration - optional
    * \see example_unity.cu
    * \note nc here means that the fore will default to origin
    */
    void sort(bool greater = false, MemoryState destination = nc);
    /**
    * \brief sort a unity based on overloaded < and > operators
    * \details Uses thrust::stable_sort but utilizes a custom comparator 
    * function pointer. For details in creating that function pointer properly see 
    * example.
    * \param comparator - indicate if sort is descending instead of ascending - optional 
    * \param destination - location where data will be after alteration - optional
    * \see example_unity.cu
    * \note nc here means that the fore will default to origin
    */
    void sort(comp_ptr comparator, MemoryState destination = nc);

    /**
    * \brief write this Unity to a file
    * \details Binary file writer for unity. Will write 
    * type details as well as data from cpu in binary format. 
    * \param id 
    * \param dirPath - location to write to - optional 
    */
    void checkpoint(int id, std::string dirPath = "./");

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
    this->pinned = false;
    this->numElements = 0;
  }
  template<typename T>
  Unity<T>::Unity(T* data, unsigned long numElements, MemoryState state, bool pinned){
    this->host = nullptr;
    this->device = nullptr;
    this->state = null;
    this->fore = null;
    this->pinned = false;
    this->setData(data, numElements, state, pinned);
  }
  template<typename T>
  Unity<T>::Unity(Unity<T>* copy){
    this->host = nullptr;
    this->device = nullptr;
    this->state = null;
    this->fore = null;
    this->pinned = false;
    this->numElements = 0;
    if(copy->getMemoryState() == null || copy->size() == 0){
      throw NullUnityException("cannot copy a null Unity<T>");
    }
    this->setData(nullptr,copy->size(),copy->getMemoryState());
    this->fore = copy->getFore();
    this->state = copy->getMemoryState();
    if(this->state == cpu || this->state == both){
      std::memcpy(this->host,copy->host,this->numElements*sizeof(T));
    }
    if(this->state == gpu || this->state == both){
      CudaSafeCall(cudaMemcpy(this->device,copy->device,this->numElements*sizeof(T),cudaMemcpyDeviceToDevice));
    } 
  }
  template<typename T>
  Unity<T>::Unity(Unity<T>* copy,bool (*predicate)(const T&)){
    this->host = nullptr;
    this->device = nullptr;
    this->state = null;
    this->fore = null;
    this->pinned = false;
    this->numElements = 0;
    if(copy->getMemoryState() == null || copy->size() == 0){
      throw NullUnityException("cannot copy_if a null Unity<T>");
    }
    thrust::device_ptr<T> in_ptr;
    T* tmp_device = nullptr;
    if(copy->getFore() != cpu){
      in_ptr = thrust::device_ptr<T>(copy->device);
    }
    else{
      CudaSafeCall(cudaMalloc((void**)&tmp_device,copy->size()*sizeof(T)));
      CudaSafeCall(cudaMemcpy(tmp_device,copy->host,copy->size()*sizeof(T),cudaMemcpyHostToDevice));
      in_ptr = thrust::device_ptr<T>(tmp_device);
    }
    this->setData(nullptr,copy->size(),gpu);
    thrust::device_ptr<T> out_ptr(this->device);
    thrust::device_ptr<T> new_end = thrust::copy_if(in_ptr,in_ptr+copy->size(),out_ptr,predicate);
    CudaCheckError();
    unsigned long compressedSize = new_end - out_ptr;
    this->resize(compressedSize);
    if(tmp_device != nullptr) CudaSafeCall(cudaFree(tmp_device));
    if(this->state != copy->getMemoryState()) this->setMemoryState(copy->getMemoryState());
  }
  template<typename T>
  Unity<T>::Unity(std::string path, bool pinned){
    this->host = nullptr;
    this->device = nullptr;
    this->state = null;
    this->fore = null;
    this->pinned = false;
    this->numElements = 0;

    std::ifstream cp(path.c_str(), std::ifstream::binary);
    char eol;

    if(cp.is_open()){
      const std::type_info& t_info = typeid(T);

      std::string name_str;
      getline(cp,name_str);
      if(name_str != std::string(t_info.name())){
        throw CheckpointException("names of type T do not match up in Unity checkpoint reader");
      }
      
      size_t hash_code;
      cp.read((char*)&hash_code,sizeof(size_t));
      cp.read(&eol,1);
        
      if(hash_code != t_info.hash_code()){
        throw CheckpointException("hash_codes of type T do not match up in Unity checkpoint reader");
      }

      MemoryState origin;
      cp.read((char*)&origin,sizeof(MemoryState));
      cp.read((char*)&this->numElements,sizeof(unsigned long));
      cp.read(&eol,1);

      if(origin == null){
        throw CheckpointException("read origin in Unity checkpoint header shows null");
      }
      this->pinned = pinned;
      if(!pinned) this->host = new T[this->numElements]();
      else CudaSafeCall(cudaMallocHost((void**)&this->host,this->numElements*sizeof(T)));
      this->state = cpu;
      this->fore = cpu;
      for(unsigned long i = 0; i < this->numElements; ++i){
        cp.read((char*)&this->host[i],sizeof(T));
      }
      if(this->state != origin) this->setMemoryState(origin);

      cp.close();
      if(cp.good()){
        std::cout<<"Unity created from checkpoint "<<path<<"\n";
      }
      else{
        path = "could not successfully read checkpoint " + path;
        throw CheckpointException(path);
      }
    }
    else{
      path = "cannot open for read: " + path;
      throw CheckpointException(path);
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
    else if(resizeLength == 0U){
      this->clear();
    }
    else if(this->state <= 3){
      unsigned long toCopy = ((resizeLength > this->numElements) ? this->numElements: resizeLength);
      if(this->state == cpu || this->state == both){
        T* replacement = nullptr;
        if(!this->pinned){
          replacement = new T[resizeLength]();
          std::memcpy(replacement,this->host,toCopy*sizeof(T));
          delete[] this->host;
        }
        else{
          CudaSafeCall(cudaMallocHost((void**)replacement,resizeLength*sizeof(T)));
          std::memcpy(replacement,this->host,toCopy*sizeof(T));
          CudaSafeCall(cudaFreeHost(this->host));
        }
         this->host = replacement;
      }
      //TODO look at this for optimization
      if(this->state == gpu || this->state == both){
        T* replacement  = nullptr;
        CudaSafeCall(cudaMalloc((void**)&replacement,resizeLength*sizeof(T)));
        if(resizeLength > this->numElements){
          T* replacement_host = new T[resizeLength]();
          CudaSafeCall(cudaMemcpy(replacement,replacement_host,resizeLength*sizeof(T),cudaMemcpyHostToDevice));
          delete[] replacement_host;
        }
        CudaSafeCall(cudaMemcpy(replacement,this->device,toCopy*sizeof(T),cudaMemcpyDeviceToDevice));
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
      logger.warn<<"WARNING: Unity<T>::clear(ssrlcv::null) does nothing\n";
      return;
    }
    else if(state != both && this->state != both && this->state != state){
      logger.warn<<"WARNING: Attempt to clear null memory in location "
      <<memoryStateToString(state)<<"...action prevented\n";
      return;
    }
    else if(this->state == null){
      logger.warn<<"WARNING: Attempt to clear null (empty) Unity...action prevented\n";
      return;
    }
    else if(this->state <= 3){//currently supported types
      if(state == cpu || (state == both && this->host != nullptr)){
        if(!this->pinned) delete[] this->host;
        else CudaSafeCall(cudaFreeHost(this->host));
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
          T* zerod = new T[this->numElements]();
          if(!this->pinned){
            delete[] this->host;
            this->host = zerod;
          } 
          else{
            CudaSafeCall(cudaFreeHost(this->host));
            CudaSafeCall(cudaMallocHost((void**)&this->host,this->numElements*sizeof(T)));
            std::memcpy(this->host,zerod,this->numElements*sizeof(T));
            delete[] zerod;
          } 
        }
        if(state == gpu || (state == both && this->device != nullptr)){
          T* zerod = (state == both && this->host != nullptr) ? this->host : new T[this->numElements]();
          CudaSafeCall(cudaMemcpy(this->device,zerod,this->numElements*sizeof(T),cudaMemcpyHostToDevice));
          if(this->host == nullptr) delete[] zerod;
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
      logger.warn<<"WARNING: hard setting of memory state to same memory state does nothing: "<<memoryStateToString(this->state)<<"\n";
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
  bool Unity<T>::isPinned(){
    return this->pinned;
  }




  //TODO test and add checks for 0 and null
  template<typename T>
  void Unity<T>::pin(){
    if(this->pinned){
      logger.warn<<"WARNING: attempt to pin already pinned Unity<T> does nothing\n";
      return;
    }
    //determine what to do if it is unified
    this->pinned = true;
    if(this->state != gpu){
      T* pinned_host = nullptr;
      CudaSafeCall(cudaMallocHost((void**)&pinned_host,this->numElements*sizeof(T)));
      std::memcpy(pinned_host,this->host,this->numElements*sizeof(T));
      delete[] this->host;
      this->host = pinned_host;
    }
  }
  template<typename T>
  void Unity<T>::unpin(){
    if(!this->pinned){
      logger.warn<<"WARNING: attempt to unpin nonpinned Unity<T> does nothing\n";
      return;
    }
    //determine what to do if it is unified
    this->pinned = false;
    if(this->state != gpu){
      T* pageable_host = new T[this->numElements]();
      memcpy(pageable_host,this->host,this->numElements*sizeof(T));
      CudaSafeCall(cudaFreeHost(this->host));
      this->host = pageable_host;
    }
  }





  template<typename T>
  void Unity<T>::setData(T* data, unsigned long numElements, MemoryState state, bool pinned){
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
    this->pinned = pinned;
    if(data == nullptr){
      if(state == null || state > 3){//greater than three means pinned or unified
        throw IllegalUnityTransition("attempt to instantiate unkown MemoryState fron nullptr (supported states = both, cpu & gpu)");
      }
      if(state == cpu || state == both){
        if(!this->pinned) this->host = new T[numElements]();
        else{
          CudaSafeCall(cudaMallocHost((void**)&this->host,this->numElements*sizeof(T)));
          this->zeroOut(cpu);
        } 
      }
      if(state == gpu || state == both){
        CudaSafeCall(cudaMalloc((void**)&this->device,this->numElements*sizeof(T)));
        this->zeroOut(gpu);
      }
    }
    else if(state <= 2){
      if(state == cpu) this->host = data;//TODO warn about using pinned here
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
      logger.warn<<"WARNING: Unity<T>::setFore(MemoryState state) when state == this->fore does nothing\n";
      return;
    }
    else if(state == both){
      logger.warn<<"ERROR: cannot set fore to both manually: \n\tuse setMemoryState(both) or transferMemoryTo((this->fore == gpu) ? cpu : gpu)\n";
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
        logger.warn<<"WARNING: transfering memory to location of fore does nothing: "<<memoryStateToString(state)<<"\n";
        return;
      }
      else{
        if(this->state != both){
          if(this->state == cpu && this->device == nullptr){
            CudaSafeCall(cudaMalloc((void**)&this->device,this->numElements*sizeof(T)));
          }
          else if(this->state == gpu && this->host == nullptr){
            if(!this->pinned) this->host = new T[this->numElements]();
            else CudaSafeCall(cudaMallocHost((void**)&this->host,this->numElements*sizeof(T)));
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
  void Unity<T>::remove(bool (*predicate)(const T&),MemoryState destination){
    if(this->state == null || this->numElements == 0){
      throw NullUnityException("cannot remove anything from an already null Unity<T>");
    }
    if(destination == nc) destination = this->state;
    if(this->state == null){
      throw UnityException("cannot perform sort on an empty Unity<T>");
    }
    else if(this->fore == cpu){
      this->transferMemoryTo(gpu);
    }
    thrust::device_ptr<T> data_ptr(this->device);
    thrust::device_ptr<T> new_end = thrust::remove_if(data_ptr,data_ptr+this->numElements,predicate);
    CudaCheckError();
    this->fore = gpu;
    unsigned long compressedSize = new_end - data_ptr;
    if(compressedSize == 0){
      logger.warn<<"Unity<T>::remove(bool(*validate)(const T&)) led to all elements being removed (data cleared)\n";
      this->clear();
      return;
    }
    else if(compressedSize != this->numElements){
      this->resize(compressedSize);
    }
    if(destination != this->state) this->setMemoryState(destination);
  }
  template<typename T>
  void Unity<T>::sort(bool greater, MemoryState destination){
    if(this->state == null || this->numElements == 0){
      throw NullUnityException("cannot sort a null Unity<T>");
    }
    MemoryState origin = (destination == nc) ? this->state : destination;
    if(origin == null){
      throw UnityException("cannot perform sort on an empty Unity<T>");
    }
    if(this->fore == cpu){
      this->transferMemoryTo(gpu);
    } 
    thrust::device_ptr<T> data_ptr(this->device);
    if(greater){
      thrust::stable_sort(data_ptr,data_ptr+this->numElements,thrust::greater<T>());
    }
    else{
      thrust::stable_sort(data_ptr,data_ptr+this->numElements,thrust::less<T>());
    }
    CudaCheckError();
    this->fore = gpu;
    if(origin != this->state) this->setMemoryState(origin);
  }
  template<typename T>
  void Unity<T>::sort(comp_ptr comparator, MemoryState destination){
    if(this->state == null || this->numElements == 0){
      throw NullUnityException("cannot sort a null Unity<T>");
    }
    MemoryState origin = (destination == nc) ? this->state : destination;
    if(origin == null){
      throw UnityException("cannot perform sort on an empty Unity<T>");
    }
    if(this->fore == cpu){
      this->transferMemoryTo(gpu);
    } 
    thrust::device_ptr<T> data_ptr(this->device);
    thrust::stable_sort(data_ptr,data_ptr+this->numElements,comparator);
    CudaCheckError();
    this->fore = gpu;
    if(origin != this->state) this->setMemoryState(origin);
  }
  template<typename T>
  void Unity<T>::checkpoint(int id, std::string dirPath){
    if(this->state == null){
      throw NullUnityException("cannot write a checkpoint with a null Unity<T>");
    }

    const std::type_info& ti = typeid(T);
    size_t hash_code = ti.hash_code();
    const char* name = ti.name();

    char eol = '\n';

    std::string pathToFile = dirPath + std::to_string(id) + "_";
    pathToFile += name;
    pathToFile += ".uty";
    std::ofstream cp(pathToFile.c_str(), std::ofstream::binary);

    MemoryState origin = this->state;
    if(this->fore == gpu) this->transferMemoryTo(cpu);
    
    if(cp.is_open()){
      //header
      cp.write(name,strlen(name));
      cp.write(&eol,sizeof(char));
      cp.write((char*)&hash_code,sizeof(size_t));
      cp.write(&eol,sizeof(char));
      cp.write((char*)&origin,sizeof(MemoryState));
      cp.write((char*)&this->numElements,sizeof(unsigned long));
      cp.write(&eol,sizeof(char));
      //body

      for(unsigned long i = 0; i < this->numElements; ++i){
        cp.write((char*)&this->host[i],sizeof(T));
      }
      cp.close();
      if(cp.good()){
        std::cout<<"checkpoint "<<pathToFile<<" successfully written\n";
      }
      else{
        pathToFile = "could not write Unity<T> checkpoint: " + pathToFile;
        throw CheckpointException(pathToFile);
      }
      if(this->state != origin) this->setMemoryState(origin);
    }
    else{
      pathToFile = "could not open for writing: " + pathToFile;
      throw CheckpointException(pathToFile);
    }
  }
  template<typename T> 
  void Unity<T>::printInfo(){
    std::cout<<"numElements = "<<this->numElements;
    std::cout<<" state = "<<memoryStateToString(this->state);
    if(this->pinned && (this->state == cpu || this->state == both)) std::cout<<" (pinned)";
    std::cout<<" fore = "<<memoryStateToString(this->fore);
    std::cout<<" type = "<<typeid(T).name()<<"\n";
  }

  /**
  * \}
  */
}


#endif /*UNITY_CUH*/