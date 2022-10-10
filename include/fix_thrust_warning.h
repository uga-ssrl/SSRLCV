#ifndef FIX_THRUST_WARNING_H
#define FIX_THRUST_WARNING_H

// Fix "cannot call host function from host device function" warnings
namespace thrust{
  namespace detail{
    template <typename T, typename U>
    __host__ __device__ T aligned_reinterpret_cast(U u);
  }
}

#endif