#include "KDTree.cuh"

//template class ssrlcv::KDTree<ssrlcv::SIFT_Descriptor>;
//template class ssrlcv::KDTree<ssrlcv::Window_3x3>;
//template class ssrlcv::KDTree<ssrlcv::Window_9x9>;
//template class ssrlcv::KDTree<ssrlcv::Window_15x15>;
//template class ssrlcv::KDTree<ssrlcv::Window_25x25>;
//template class ssrlcv::KDTree<ssrlcv::Window_31x31>;

/******************
* KD TREE METHODS *
*******************/

//template<typename T>
ssrlcv::KDTree::KDTree() {}

//template<typename T>
ssrlcv::KDTree::KDTree(ssrlcv::ptr::value<ssrlcv::Unity<ssrlcv::Feature<ssrlcv::SIFT_Descriptor>>> _points) {
    build(_points);
}

//template<typename T>
ssrlcv::KDTree::KDTree(ssrlcv::ptr::value<ssrlcv::Unity<ssrlcv::Feature<ssrlcv::SIFT_Descriptor>>> _points, thrust::host_vector<int> _labels) {
    build(_points, _labels);
}

struct SubTree {
    SubTree() : first(0), last(0), nodeIdx(0), depth(0) {}
    SubTree(int _first, int _last, int _nodeIdx, int _depth)
        : first(_first), last(_last), nodeIdx(_nodeIdx), depth(_depth) {}
    int first;
    int last;
    int nodeIdx;
    int depth;
};

static float medianPartition(size_t* ofs, int a, int b, const unsigned char* vals) {
    int k, a0 = a, b0 = b;
    int middle = (a + b)/2;
    while( b > a ) {
        int i0 = a, i1 = (a+b)/2, i2 = b;
        float v0 = vals[ofs[i0]], v1 = vals[ofs[i1]], v2 = vals[ofs[i2]];
        int ip = v0 < v1 ? (v1 < v2 ? i1 : v0 < v2 ? i2 : i0) :
                 v0 < v2 ? (v1 == v0 ? i2 : i0): (v1 < v2 ? i2 : i1);
        float pivot = vals[ofs[ip]];
        swap(ofs[ip], ofs[i2]);

        for( i1 = i0, i0--; i1 <= i2; i1++ ) {
            if( vals[ofs[i1]] <= pivot ) {
                i0++; 
                swap(ofs[i0], ofs[i1]);
            }
        } // for
        if( i0 == middle )
            break;
        if( i0 > middle )
            b = i0 - (b == i0);
        else
            a = i0;
    } // while

    float pivot = vals[ofs[middle]];
    for( k = a0; k < middle; k++ ) {
        if( !(vals[ofs[k]] <= pivot) ) {
           logger.err<<"ERROR: median partition unsuccessful"<<"\n"; 
        }
    }
    for( k = b0; k > middle; k-- ) {
       if( !(vals[ofs[k]] >= pivot) ) {
           logger.err<<"ERROR: median partition unsuccessful"<<"\n"; 
        } 
    }

    return vals[ofs[middle]];
} // medianPartition

//template<typename T>
static void computeSums(ssrlcv::Feature<ssrlcv::SIFT_Descriptor>* points, int start, int end, unsigned char *sums) {
   
    int i, j; 
    ssrlcv::Feature<ssrlcv::SIFT_Descriptor> data; 

    // initilize sums array with 0
    for(j = 0; j < DIMS; j++)
        sums[j*2] = sums[j*2+1] = 0;

    // compute the square of each element in the values array 
    for(i = start; i <= end; i++) {
        data = points[i];
        for(j = 0; j < DIMS; j++) {
            double t = data.descriptor.values[j], s = sums[j*2] + t, s2 = sums[j*2+1] + t*t;
            sums[j*2] = s; sums[j*2+1] = s2;
        }
    }
} // computeSums

//template<typename T>
void ssrlcv::KDTree::build(ssrlcv::ptr::value<ssrlcv::Unity<ssrlcv::Feature<ssrlcv::SIFT_Descriptor>>> _points) {
    thrust::host_vector<int> labels;
    build(_points, labels);
} // build

//template<typename T>
void ssrlcv::KDTree::build(ssrlcv::ptr::value<ssrlcv::Unity<ssrlcv::Feature<ssrlcv::SIFT_Descriptor>>> _points, thrust::host_vector<int> _labels) {

    if (_points->size() == 0) {
        logger.err<<"ERROR: number of features in image must be greater than zero"<<"\n";
    }
    
    // initilize nodes of KD Tree
    nodes.clear();
    nodes.shrink_to_fit();
    points = _points;

    int i, j, n = _points->size(), top = 0;
    const unsigned char* data = _points->host->descriptor.values;

    // size of object in memory 
    size_t step = sizeof(ssrlcv::Feature<ssrlcv::SIFT_Descriptor>);

    labels.resize(n); // labels and points array will share same size 
    const int* _labels_data = 0;

    if( !_labels.empty() ) {
        int nlabels = n*DIMS;
        if ( !(nlabels==n) ) {
            logger.err<<"ERROR: labels size must be equal to points size"<<"\n";
        } 
        _labels_data = _labels.data(); 
    }

    // will hold the SIFT_Descriptor values array AND its squares
    unsigned char sumstack[MAX_TREE_DEPTH*2][DIMS*2];
    SubTree stack[MAX_TREE_DEPTH*2]; 

    vector<size_t> _ptofs(n);
    size_t* ptofs = &_ptofs[0];

    for (i = 0; i < n; i++) { 
        ptofs[i] = i*step;
    }

    nodes.push_back(Node());
    computeSums(points->host.get(), 0, n-1, sumstack[top]);
    stack[top++] = SubTree(0, n-1, 0, 0);
    int _maxDepth = 0;

    while (--top >= 0) {
        int first = stack[top].first, last = stack[top].last;
        int depth = stack[top].depth, nidx = stack[top].nodeIdx;
        int count = last - first + 1, dim = -1;
        const unsigned char* sums = sumstack[top]; // points to the first element in uchar array
        double invCount = 1./count, maxVar = -1.;

        if (count == 1) {
            int idx0 = (int)(ptofs[first]/step);
            int idx = idx0; // the dimension
            nodes[nidx].idx = ~idx;
            
            labels[idx] = _labels_data ? _labels_data[idx0] : idx0;
            _maxDepth = std::max(_maxDepth, depth);
            continue;
        }

        // find the dimensionality with the biggest variance
        for ( j = 0; j < DIMS; j++ ) {
            unsigned char m = sums[j*2]*invCount;
            unsigned char varj = sums[j*2+1]*invCount - m*m;
            if ( maxVar < varj ) {
                maxVar = varj;
                dim = j;
            }
        }

        int left = (int)nodes.size(), right = left + 1;
        nodes.push_back(Node());
        nodes.push_back(Node());
        nodes[nidx].idx = dim;
        nodes[nidx].left = left;
        nodes[nidx].right = right;
        nodes[nidx].boundary = medianPartition(ptofs, first, last, data + dim);

        int middle = (first + last)/2;
        unsigned char* lsums = (unsigned char*)sums, *rsums = lsums + DIMS*2;
        computeSums(points->host.get(), middle+1, last, rsums);

        for (j = 0; j < DIMS*2; j++) {
            lsums[j] = sums[j] - rsums[j];
        }
        stack[top++] = SubTree(first, middle, left, depth+1);
        stack[top++] = SubTree(middle+1, last, right, depth+1);
    } // while
    maxDepth = _maxDepth;
} // build

// The below algorithm is from:
// J.S. Beis and D.G. Lowe. Shape Indexing Using Approximate Nearest-Neighbor Search
// in High-Dimensional Spaces. In Proc. IEEE Conf. Comp. Vision Patt. Recog.,
// pages 1000--1006, 1997. https://www.cs.ubc.ca/~lowe/papers/cvpr97.pdf
//template<typename T> 
__device__ ssrlcv::DMatch ssrlcv::findNearest(ssrlcv::KDTree* kdtree, typename KDTree::Node* nodes, ssrlcv::Feature<ssrlcv::SIFT_Descriptor>* featuresTree,
ssrlcv::Feature<ssrlcv::SIFT_Descriptor> feature, unsigned int queryImageID, unsigned int targetImageID, int emax, float absoluteThreshold, int k) {

    ssrlcv::SIFT_Descriptor desc = feature.descriptor;
    const unsigned char *vec = desc.values; // descriptor values[128] from query

    int i, j, ncount = 0, e = 0;
    int qsize = 0;
    const int maxqsize = 1 << 10;

    int idx[2]; // holds the node indices
    float dist[2]; // holds the euclidean distances

    ssrlcv::PQueueElem pqueue[maxqsize]; // priority queue to search the tree

    for (e = 0; e < emax;) {
        float d, alt_d = 0.f; 
        int nidx; // node index
        
        if (e == 0) { nidx = 0; } 
        else {
            // take the next node from the priority queue
            if (qsize == 0) { break; }
            nidx = pqueue[0].idx; // current tree position
            alt_d = pqueue[0].dist; // distance of the query point from the node
            
            if (--qsize > 0) {
                
                // std::swap(pqueue[0], pqueue[qsize]);
                ssrlcv::PQueueElem temp = pqueue[0];
                pqueue[0] = pqueue[qsize];
                pqueue[qsize] = temp; 

                d = pqueue[0].dist;
                for (i = 0;;) {
                    int left = i*2 + 1, right = i*2 + 2;
                    if (left >= qsize)
                        break;
                    if (right < qsize && pqueue[right].dist < pqueue[left].dist)
                        left = right;
                    if (pqueue[left].dist >= d)
                        break;
                    
                    // std::swap(pqueue[i], pqueue[left]);
                    ssrlcv::PQueueElem temp = pqueue[i];
                    pqueue[i] = pqueue[left];
                    pqueue[left] = temp;

                    i = left;
                } // for
            } // if
            if (ncount == k && alt_d > dist[ncount-1]) { continue; }
        } // if-else

        for (;;) {
            if (nidx < 0) 
                break;
                
            const typename KDTree::Node& n = nodes[nidx];

            if (n.idx < 0) { // if it is a leaf node
                i = ~n.idx; 
                const unsigned char* row = featuresTree[i].descriptor.values; // descriptor values[128] from tree

                // euclidean distance
                for (j = 0, d = 0.f; j < DIMS; j++) {
                    float t = vec[j] - row[j];
                    d += t*t;
                }
                dist[ncount] = d;
                //printf("\nthreadIdx[%d] dist[%d] = %f\n", threadIdx.x, ncount, dist[ncount]);
                idx[ncount] = i;
                //printf("\nthreadIdx[%d] idx[%d] = %f\n", threadIdx.x, ncount, idx[ncount]);

                for (i = ncount-1; i >= 0; i--) {
                    if (dist[i] <= d)
                        break;
                    // std::swap(dist[i], dist[i+1]);
                    float dtemp = dist[i];
                    dist[i] = dist[i+1];
                    dist[i+1] = dtemp;
                    // std::swap(idx[i], idx[i+1]);
                    int itemp = idx[i];
                    idx[i] = idx[i+1];
                    idx[i+1] = itemp; 
                } // for
                ncount += ncount < k;
                e++;
                break; 

            } // if

            int alt;
            if (vec[n.idx] <= n.boundary) {
                nidx = n.left;
                alt = n.right;
            } else {
                nidx = n.right;
                alt = n.left;
            }

            d = vec[n.idx] - n.boundary;
            d = d*d + alt_d; // euclidean distance

            // subtree prunning
            if (ncount == k && d > dist[ncount-1])
                continue;
            // add alternative subtree to the priority queue
            pqueue[qsize] = PQueueElem(d, alt);
            for (i = qsize; i > 0;) {
                int parent = (i-1)/2;
                if (parent < 0 || pqueue[parent].dist <= d)
                    break;

                // std::swap(pqueue[i], pqueue[parent]);
                ssrlcv::PQueueElem temp = pqueue[i];
                pqueue[i] = pqueue[parent];
                pqueue[parent] = temp; 

                i = parent;
            } // for
            qsize += qsize+1 < maxqsize;
        } // for
    } // for

    DMatch match;
    match.distance = dist[0]; // smallest distance
    int matchIndex = idx[0]; // index of corresponding leaf node/point

    if (match.distance >= absoluteThreshold) { match.invalid = true; } 
    else {
      match.invalid = false;
      match.keyPoints[0].loc = feature.loc; // img1, query features
      match.keyPoints[1].loc = featuresTree[matchIndex].loc; // img2, kdtree features
      match.keyPoints[0].parentId = queryImageID;  
      match.keyPoints[1].parentId = targetImageID;
    }

    return match;
} // findNearest

// The below algorithm is from:
// J.S. Beis and D.G. Lowe. Shape Indexing Using Approximate Nearest-Neighbor Search
// in High-Dimensional Spaces. In Proc. IEEE Conf. Comp. Vision Patt. Recog.,
// pages 1000--1006, 1997. https://www.cs.ubc.ca/~lowe/papers/cvpr97.pdf
//template<typename T> 
__device__ ssrlcv::DMatch ssrlcv::findNearest(ssrlcv::KDTree* kdtree, typename KDTree::Node* nodes, ssrlcv::Feature<ssrlcv::SIFT_Descriptor>* featuresTree,
ssrlcv::Feature<ssrlcv::SIFT_Descriptor> feature, unsigned int queryImageID, unsigned int targetImageID, int emax, float relativeThreshold, float absoluteThreshold, float nearestSeed, int k) {

    ssrlcv::SIFT_Descriptor desc = feature.descriptor;
    const unsigned char *vec = desc.values; // descriptor values[128] from query

    int i, j, ncount = 0, e = 0;
    int qsize = 0;
    const int maxqsize = 1 << 10;

    int idx[2]; // holds the node indices
    float dist[2]; // holds the euclidean distances

    ssrlcv::PQueueElem pqueue[maxqsize]; // priority queue to search the search

    for (e = 0; e < emax;) {
        float d, alt_d = 0.f; 
        int nidx; // node index
        
        if (e == 0) { nidx = 0; } 
        else {
            // take the next node from the priority queue
            if (qsize == 0) { break; }
            nidx = pqueue[0].idx; // current tree position
            alt_d = pqueue[0].dist; // distance of the query point from the node
            
            if (--qsize > 0) {
                
                // std::swap(pqueue[0], pqueue[qsize]);
                ssrlcv::PQueueElem temp = pqueue[0];
                pqueue[0] = pqueue[qsize];
                pqueue[qsize] = temp; 

                d = pqueue[0].dist;
                for (i = 0;;) {
                    int left = i*2 + 1, right = i*2 + 2;
                    if (left >= qsize)
                        break;
                    if (right < qsize && pqueue[right].dist < pqueue[left].dist)
                        left = right;
                    if (pqueue[left].dist >= d)
                        break;
                    
                    // std::swap(pqueue[i], pqueue[left]);
                    ssrlcv::PQueueElem temp = pqueue[i];
                    pqueue[i] = pqueue[left];
                    pqueue[left] = temp;

                    i = left;
                } // for
            } // if
            if (ncount == k && alt_d > dist[ncount-1]) { continue; }
        } // if-else

        for (;;) {
            if (nidx < 0) 
                break;
                
            const typename KDTree::Node& n = nodes[nidx];

            if (n.idx < 0) { // if it is a leaf node
                i = ~n.idx; 
                const unsigned char* row = featuresTree[i].descriptor.values; // descriptor values[128] from tree

                // euclidean distance
                for (j = 0, d = 0.f; j < DIMS; j++) {
                    float t = vec[j] - row[j];
                    d += t*t;
                }
                dist[ncount] = d;
                //printf("\nthreadIdx[%d] dist[%d] = %f\n", threadIdx.x, ncount, dist[ncount]);
                idx[ncount] = i;
                //printf("\nthreadIdx[%d] idx[%d] = %f\n", threadIdx.x, ncount, idx[ncount]);

                for (i = ncount-1; i >= 0; i--) {
                    if (dist[i] <= d)
                        break;
                    // std::swap(dist[i], dist[i+1]);
                    float dtemp = dist[i];
                    dist[i] = dist[i+1];
                    dist[i+1] = dtemp;
                    // std::swap(idx[i], idx[i+1]);
                    int itemp = idx[i];
                    idx[i] = idx[i+1];
                    idx[i+1] = itemp; 
                } // for
                ncount += ncount < k;
                e++;
                break; 

            } // if

            int alt;
            if (vec[n.idx] <= n.boundary) {
                nidx = n.left;
                alt = n.right;
            } else {
                nidx = n.right;
                alt = n.left;
            }

            d = vec[n.idx] - n.boundary;
            d = d*d + alt_d; // euclidean distance

            // subtree prunning
            if (ncount == k && d > dist[ncount-1])
                continue;
            // add alternative subtree to the priority queue
            pqueue[qsize] = PQueueElem(d, alt);
            for (i = qsize; i > 0;) {
                int parent = (i-1)/2;
                if (parent < 0 || pqueue[parent].dist <= d)
                    break;

                // std::swap(pqueue[i], pqueue[parent]);
                ssrlcv::PQueueElem temp = pqueue[i];
                pqueue[i] = pqueue[parent];
                pqueue[parent] = temp; 

                i = parent;
            } // for
            qsize += qsize+1 < maxqsize;
        } // for
    } // for

    DMatch match;
    match.distance = dist[0]; // smallest distance
    int matchIndex = idx[0]; // index of corresponding leaf node/point

    // printf("match.distance at threadID[%d] = %f\n", threadIdx.x, match.distance);
    // printf("absoluteThreshold at threadID[%d] = %f\n", threadIdx.x, absoluteThreshold);
    // printf("relativeThreshold at threadID[%d] = %f\n", threadIdx.x, relativeThreshold);
    // printf("nearestSeed at threadID[%d] = %f\n", threadIdx.x, nearestSeed);

    if (match.distance >= absoluteThreshold) {
        match.invalid = true; 
    } else { 
      if (match.distance/nearestSeed > relativeThreshold*relativeThreshold) {
        match.invalid = true;
      } else {
        match.invalid = false;
        match.keyPoints[0].loc = feature.loc; // img1, query features
        match.keyPoints[1].loc = featuresTree[matchIndex].loc; // img2, kdtree features
        match.keyPoints[0].parentId = queryImageID;  
        match.keyPoints[1].parentId = targetImageID;
        // printf("\nwith seed image: (%f, %f)\n", match.keyPoints[0].loc.x, match.keyPoints[0].loc.y); 
      } 
    } // if-else

    return match;
} // findNearest

/****************
* DEBUG METHODS *
*****************/

//template<typename T>
const float2 ssrlcv::KDTree::getPoint(int ptidx, int *label) const {
    if ( !((unsigned)ptidx < (unsigned)points->size()) ) {
        logger.err<<"ERROR: point index is out of range"<<"\n";
    } 
    if (label) { *label = labels[ptidx]; }
    return points->host[ptidx].loc;
} // getPoint
