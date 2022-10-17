//////////////////////////////////////////////////////////////////////////////////////////
//                                                                                      //    
//                                  License Agreement                                   // 
//                      For Open Source Computer Vision Library                         //
//                                                                                      //    
//              Copyright (C) 2000-2008, Intel Corporation, all rights reserved.        //
//              Copyright (C) 2009, Willow Garage Inc., all rights reserved.            //
//              Copyright (C) 2013, OpenCV Foundation, all rights reserved.             //
//              Copyright (C) 2015, Itseez Inc., all rights reserved.                   //
//              Third party copyrights are property of their respective owners.         //    
//                                                                                      //
//  Redistribution and use in source and binary forms, with or without modification,    //
//  are permitted provided that the following conditions are met:                       //
//                                                                                      //
//   * Redistribution's of source code must retain the above copyright notice,          //
//     this list of conditions and the following disclaimer.                            //    
//                                                                                      //
//   * Redistribution's in binary form must reproduce the above copyright notice,       //
//     this list of conditions and the following disclaimer in the documentation        //
//     and/or other materials provided with the distribution.                           //
//                                                                                      //
//   * The name of the copyright holders may not be used to endorse or promote products //
//     derived from this software without specific prior written permission.            //
//                                                                                      //    
// This software is provided by the copyright holders and contributors "as is" and      //
// any express or implied warranties, including, but not limited to, the implied        //
// warranties of merchantability and fitness for a particular purpose are disclaimed.   //
// In no event shall the Intel Corporation or contributors be liable for any direct,    //
// indirect, incidental, special, exemplary, or consequential damages                   //
// (including, but not limited to, procurement of substitute goods or services;         //
// loss of use, data, or profits; or business interruption) however caused              //
// and on any theory of liability, whether in contract, strict liability,               //
// or tort (including negligence or otherwise) arising in any way out of                //
// the use of this software, even if advised of the possibility of such damage.         //
//                                                                                      //
//////////////////////////////////////////////////////////////////////////////////////////

#pragma once 

#ifndef KDTREE_CUH
#define KDTREE_CUH

#ifdef __CUDACC__
#define CUDA_CALLABLE_MEMBER __host__ __device__
#else
#define CUDA_CALLABLE_MEMBER
#endif 

#include "common_includes.hpp"
#include "Feature.cuh"
#include "Match.cuh"

#include <vector>
#include <algorithm>
#include <cstdio>
#include <iostream>
#include <thrust/host_vector.h>
#include <thrust/device_vector.h>

using namespace std;

const int DIMS = 128; // dimensions
const int MAX_TREE_DEPTH = 32; // upper bound for tree level, equivalent to 4 billion generated features 

namespace ssrlcv {

    /****************
    * KD-TREE CLASS *
    *****************/

    //template <typename T>
    class KDTree {

    public: 
        
        // the node of the search tree
        struct Node {
            Node() : idx(-1), left(-1), right(-1), boundary(0.f) {}
            Node(int _idx, int _left, int _right, float _boundary)
                : idx(_idx), left(_left), right(_right), boundary(_boundary) {}

            // split dimension; >=0 for nodes (dim), < 0 for leaves (index of the point)
            int idx;
            // node indices of the left and the right branches
            int left, right;
            // go to the left if query[node.idx]<=node.boundary, otherwise go to the right
            float boundary;
        };

        // constructors
        KDTree();
        KDTree(ssrlcv::ptr::value<ssrlcv::Unity<ssrlcv::Feature<ssrlcv::SIFT_Descriptor>>> _points);
        KDTree(ssrlcv::ptr::value<ssrlcv::Unity<ssrlcv::Feature<ssrlcv::SIFT_Descriptor>>> _points, thrust::host_vector<int> _labels);

        // builds the search tree
        void build(ssrlcv::ptr::value<ssrlcv::Unity<ssrlcv::Feature<ssrlcv::SIFT_Descriptor>>> _points);
        void build(ssrlcv::ptr::value<ssrlcv::Unity<ssrlcv::Feature<ssrlcv::SIFT_Descriptor>>> _points, thrust::host_vector<int> _labels);

        // return a point with the specified index
        const float2 getPoint(int ptidx, int *label = 0) const;

        // print the kd tree
        void printKDTree();

        thrust::host_vector<Node> nodes; // all the tree nodes
        ssrlcv::ptr::value<ssrlcv::Unity<ssrlcv::Feature<ssrlcv::SIFT_Descriptor>>> points; // all the points 
        thrust::host_vector<int> labels; // the parallel array of labels
        int maxDepth;

    }; // KD-Tree class

    /************************
    * PRIORITY QUEUE STRUCT *
    *************************/

    struct PQueueElem {
        CUDA_CALLABLE_MEMBER PQueueElem() : dist(0), idx(0) {}
        CUDA_CALLABLE_MEMBER PQueueElem(float _dist, int _idx) : dist(_dist), idx(_idx) {}
        float dist; // distance of the query point from the node
        int idx; // current tree position
    };

    /**************************
    * KD-TREE SEARCH FUNCTION *
    ***************************/

    /**
     * \brief finds the k nearest neighbors to a point while looking at emax (at most) leaves. matching is done wihtout a seed image.
     * \param kdtree the KD-Tree to search through
     * \param nodes the nodes of the KD-Tree
     * \param treeFeatures the feature points i.e. the leaf nodes of the KD-Tree 
     * \param queryFeature the query feature point
     * \param emax the max number of leaf nodes to search. a value closer to the total number tree features correleates to a higher accuracy
     * \param absoluteThreshold the maximum distance between two matched points
     * \param k the number of nearest neighbors. by default this value finds the 1 closest tree feature to a given query feature
    */ 
    //template<typename T> 
    __device__ ssrlcv::DMatch findNearest(KDTree* kdtree, typename KDTree::Node* nodes, Feature<ssrlcv::SIFT_Descriptor>* treeFeatures, 
    Feature<ssrlcv::SIFT_Descriptor> queryFeature, unsigned int queryImageID, unsigned int targetImageID, int emax, float absoluteThreshold, int k = 1);
    
    /**
     * \brief finds the k nearest neighbors to a point while looking at emax (at most) leaves. matching is done WITH a seed image.
     * \param kdtree the KD-Tree to search through
     * \param nodes the nodes of the KD-Tree
     * \param treeFeatures the feature points i.e. the leaf nodes of the KD-Tree 
     * \param queryFeature the query feature point
     * \param emax the max number of leaf nodes to search. a value closer to the total number tree features correleates to a higher accuracy
     * \param relativeThreshold the realtive threshold based on closest seed descriptor 
     * \param absoluteThreshold the maximum distance threshold between two matched points
     * \param nearestSeed the seed distance corresponding to the query point 
     * \param k the number of nearest neighbors. by default this value finds the 1 closest tree feature to a given query feature
    */ 
    //template<typename T> 
    __device__ ssrlcv::DMatch findNearest(ssrlcv::KDTree* kdtree, typename KDTree::Node* nodes, ssrlcv::Feature<ssrlcv::SIFT_Descriptor>* treeFeatures, 
    ssrlcv::Feature<ssrlcv::SIFT_Descriptor> queryFeature, unsigned int queryImageID, unsigned int targetImageID, int emax, float relativeThreshold, float absoluteThreshold, float nearestSeed, int k = 1); 

} // namepsace ssrlcv

#endif