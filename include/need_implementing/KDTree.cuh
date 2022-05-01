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
        std::shared_ptr<ssrlcv::Unity<uint8_t>> keyBytes;//depth/8 = number of keySegments per object
        std::shared_ptr<ssrlcv::Unity<T>> data;

        KDTree();
        KDTree(unsigned int dim, unsigned int depth);
    };
}



#endif /* KDTREE_CUH */
