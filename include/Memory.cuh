/** 
* \file   Unity.cuh
* \author  Eric Miller
* \date    22 August 2022
* \brief   File containing Unity class and CUDA error handlers
* \details This file contains the Unity header only CUDA memory handler
* as well as the CUDA error checkers like the CudaSafeCall() function wrapper 
* and the CudaCheckError() function.
*/

#ifndef MEMORY_CUH
#define MEMORY_CUH

#include <stdio.h>
#include <unistd.h>
#include <cuda.h>
#include <type_traits>

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
      logger.err.printf("cudaSafeCall() failed at %s:%i : %s",
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
    logger.err.printf("cudaCheckError() failed at %s:%i : %s",
    file, line, cudaGetErrorString(err));
    exit(-1);
  }

  // More careful checking. However, this will affect performance.
  // Comment away if needed.
  // err = cudaDeviceSynchronize();
  if (cudaSuccess != err) {
    logger.err.printf("cudaCheckError() with sync failed at %s:%i : %s",
    file, line, cudaGetErrorString(err));
    exit(-1);
  }
#endif

  return;
}
/**
* \}
*/

namespace ssrlcv {
  template<class T>
  struct host_pinned_delete {
    #ifdef LOG_MEM
    long size;
    host_pinned_delete(long size)
    {
      this->size = size;
    }
    #endif

    void operator ()( T * ptr)
    { 
      CudaSafeCall(cudaFreeHost((void *)ptr));
      #ifdef LOG_MEM
      logger.mem.logHostPinned(-this->size);
      #endif
    }
  };

  template<class T>
  struct host_unpinned_delete {
    #ifdef LOG_MEM
    long size;
    host_unpinned_delete(long size)
    {
      this->size = size;
    }
    #endif

    void operator ()( T * ptr)
    { 
      delete[] ptr;
      #ifdef LOG_MEM
      logger.mem.logHostUnpinned(-this->size);
      #endif
    }
  };

  template<class T>
  struct device_delete {
    #ifdef LOG_MEM
    long size;
    device_delete(long size)
    {
      this->size = size;
    }
    #endif

    void operator ()( T * ptr)
    { 
      CudaSafeCall(cudaFree(ptr));
      #ifdef LOG_MEM
      logger.mem.logDevice(-this->size);
      #endif
    }
  };

  namespace ptr {

    template <typename T>
    class base {
    protected:
      std::shared_ptr<T> ptr;
    public:
      base() noexcept : ptr(nullptr) { }

      base(std::nullptr_t) noexcept : ptr(nullptr) { }

      base( const base& r ) : ptr(r.ptr) { }

      base<T>& operator=(const base& r) {
        this->ptr = r.ptr;
        return *this;
      }

      T* get() const noexcept {
        return ptr.get();
      }

      T& operator*() const noexcept {
        return ptr.operator*();
      }

      T* operator->() const noexcept {
        return ptr.operator->();
      }

      explicit operator bool() const noexcept {
        return ptr.operator bool();
      }

      operator std::shared_ptr<T>() const noexcept {
        return ptr;
      }

      void clear() {
        ptr.reset();
      }

      virtual void set(long n, bool pinned = false) {}

    };

    template <typename T>
    class device : public base<T> {
    public:
      using base<T>::ptr;
      using base<T>::set;

      void set(long n, bool pinned = false) {
        T *tmp;
        CudaSafeCall(cudaMalloc((void**)&tmp, n * sizeof(T)));
        #ifdef LOG_MEM
        ptr.reset(tmp, device_delete<T>(n * sizeof(T)));
        logger.mem.logDevice(n * sizeof(T));
        #else
        ptr.reset(tmp, device_delete<T>());
        #endif
      }

      device(long n) {
        set(n);
      }

      device() {}

      device( const device& r ) {
        this->ptr = r.ptr;
      }

      device(std::nullptr_t) noexcept { }

      T& operator[]( std::ptrdiff_t idx ) const {
        return ptr.get()[idx];
      }

    };

    template <typename T>
    class host : public base<T> {
    public:
      using base<T>::ptr;
      using base<T>::set;

      void set(long n, bool pinned = false) {
        if (pinned) {
          T *tmp;
          CudaSafeCall(cudaMallocHost((void**)&tmp, n * sizeof(T)));
          #ifdef LOG_MEM
            ptr.reset(tmp, host_pinned_delete<T>(n * sizeof(T)));
            logger.mem.logHostPinned(n * sizeof(T));
          #else
            ptr.reset(tmp, host_pinned_delete<T>());
          #endif
        } else {
          #ifdef LOG_MEM
            ptr.reset(new T[n], host_unpinned_delete<T>(n * sizeof(T)));
            logger.mem.logHostUnpinned(n * sizeof(T));
          #else
            ptr.reset(new T[n], host_unpinned_delete<T>());
          #endif
        }
      }

      host() {}

      host(std::nullptr_t) noexcept { }

      host( const host& r ) {
        this->ptr = r.ptr;
      }

      host(long n, bool pinned = false) {
        set(n, pinned);
      }

      T& operator[]( std::ptrdiff_t idx ) const {
        return ptr.get()[idx];
      }

    };

    template <typename T>
    class value : public base<T> {
    public:
      using base<T>::ptr;

      value(std::nullptr_t) noexcept { }

      value() {}

      value( const value& r ) {
        this->ptr = r.ptr;
      }

      template <typename... Args>
      void construct(Args&&... args) {
        ptr = std::make_shared<T>(std::forward<Args>(args)...);
      }

      template <typename Arg, typename... Args, typename = typename std::enable_if<std::is_constructible<T, Arg, Args ...>::value && (sizeof...(Args) != 0 || !std::is_same<typename std::remove_reference<Arg>::type, value<T>>::value)>::type>
      value(Arg&& arg, Args&&... args) {
        ptr = std::make_shared<T>(arg, std::forward<Args>(args)...);
      }

    };

    template <typename T, typename U>
    bool operator==( const base<T>& lhs, const base<U>& rhs ) noexcept {
      return lhs.get() == rhs.get();
    }

    template <typename T, typename U>
    bool operator!=( const base<T>& lhs, const base<U>& rhs ) noexcept {
      return lhs.get() != rhs.get();
    }

    template <typename T>
    bool operator==( const base<T>& lhs, std::nullptr_t rhs ) noexcept {
      return lhs.get() == nullptr;
    }

    template <typename T>
    bool operator==( std::nullptr_t lhs, const base<T>& rhs ) noexcept {
      return nullptr == rhs.get();
    }

    template <typename T>
    bool operator!=( const base<T>& lhs, std::nullptr_t rhs ) noexcept {
      return lhs.get() != nullptr;
    }

    template <typename T>
    bool operator!=( std::nullptr_t lhs, const base<T>& rhs ) noexcept {
      return nullptr != rhs.get();
    }

  }
}

#endif