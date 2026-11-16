#ifndef FIX_THRUST_WARNING_H
#define FIX_THRUST_WARNING_H

// Fix "cannot call host function from host device function" warnings
// Only needed for old Thrust (CUDA < 11); newer Thrust uses an inline versioned
// namespace, so redeclaring thrust::detail here makes it ambiguous.
#include <thrust/version.h>
#if THRUST_VERSION < 101000
namespace thrust{
  namespace detail{
    template <typename T, typename U>
    __host__ __device__ T aligned_reinterpret_cast(U u);
  }
}
#endif

#endif
