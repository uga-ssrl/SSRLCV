#ifndef KDTREE_CUH
#define KDTREE_CUH

#include "common_includes.hpp"
#include "Unity.cuh"

namespace ssrlcv{
    template<typename T>
    class KDTree{
    private:
    
    public:
        unsigned int dim;
        unsigned int depth;
        ssrlcv::ptr::value<ssrlcv::Unity<uint8_t>> keyBytes;//depth/8 = number of keySegments per object
        ssrlcv::ptr::value<ssrlcv::Unity<T>> data;

        KDTree();
        KDTree(unsigned int dim, unsigned int depth);
    };
}



#endif /* KDTREE_CUH */
