#include "PoseEstimator.cuh"
#include <cublas_v2.h>
#include <cusolverDn.h>
#include <list>

ssrlcv::PoseEstimator::PoseEstimator(ssrlcv::ptr::value<ssrlcv::Image> queryImage, ssrlcv::ptr::value<ssrlcv::Image> targetImage, ssrlcv::ptr::value<ssrlcv::Unity<ssrlcv::KeyPoint>> keyPoints) :
    queryImage(queryImage),
    targetImage(targetImage),
    A(nullptr, keyPoints->size() / 2 * 9, ssrlcv::cpu),
    keyPoints(keyPoints) {}

void ssrlcv::PoseEstimator::estimatePoseRANSAC() {
    this->fillA();

    unsigned long numMatches = this->keyPoints->size() / 2;
    int N = numMatches / 7;

    cusolverDnHandle_t cusolverH = nullptr;
    cusolverStatus_t cusolver_status = CUSOLVER_STATUS_SUCCESS;
    cudaStream_t stream = nullptr;
    gesvdjInfo_t gesvdj_params = nullptr;

    cusolver_status = cusolverDnCreate(&cusolverH);
    assert(CUSOLVER_STATUS_SUCCESS == cusolver_status);

    CudaSafeCall(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));

    cusolver_status = cusolverDnSetStream(cusolverH, stream);
    assert(CUSOLVER_STATUS_SUCCESS == cusolver_status);

    cusolver_status = cusolverDnCreateGesvdjInfo(&gesvdj_params);
    assert(CUSOLVER_STATUS_SUCCESS == cusolver_status);



    

    int m = 7;
    int n = 9;
    int lwork = 0;

    ssrlcv::ptr::device<float> d_S(n * N);
    ssrlcv::ptr::device<float> d_U(m * m * N);
    ssrlcv::ptr::device<float> d_V(n * n * N);
    ssrlcv::ptr::device<int> devInfo(1 * N);

    A->transferMemoryTo(gpu);
    cusolver_status = cusolverDnSgesvdjBatched_bufferSize(
        cusolverH,
        CUSOLVER_EIG_MODE_VECTOR,
        m,
        n,
        A->device.get(),
        m,
        d_S.get(),
        d_U.get(),
        m,
        d_V.get(),
        n,
        &lwork,
        gesvdj_params,
        N
    );
    cudaDeviceSynchronize();
    assert(cusolver_status == CUSOLVER_STATUS_SUCCESS);

    ssrlcv::ptr::device<float> d_work(lwork);

    cusolver_status = cusolverDnSgesvdjBatched(
        cusolverH,
        CUSOLVER_EIG_MODE_VECTOR,
        m,
        n,
        A->device.get(),
        m,
        d_S.get(),
        d_U.get(),
        m,
        d_V.get(),
        n,
        d_work.get(),
        lwork,
        devInfo.get(),
        gesvdj_params,
        N
    );

    cudaDeviceSynchronize();
    assert(cusolver_status == CUSOLVER_STATUS_SUCCESS);

    dim3 grid = {1,1,1};
    dim3 block = {1,1,1};
    void (*fp)(ssrlcv::KeyPoint *, int, float *, unsigned long, FMatrixInliers *) = &computeFMatrixAndInliers;
    getFlatGridBlock(N,grid,block,fp);
    this->keyPoints->transferMemoryTo(gpu);
    ssrlcv::ptr::value<ssrlcv::Unity<ssrlcv::FMatrixInliers>> matricesAndInliers(nullptr, N, gpu);
    computeFMatrixAndInliers<<<grid, block>>>(this->keyPoints->device.get(), this->keyPoints->size(), d_V.get(), N, matricesAndInliers->device.get());
    cudaDeviceSynchronize();
    CudaCheckError();
    this->keyPoints->transferMemoryTo(gpu);
    matricesAndInliers->transferMemoryTo(cpu);

    int best = 0;
    int bestIdx = 0;
    for (int i = 0; i < N; i ++) {
        if (matricesAndInliers->host[i].valid) {
            if (matricesAndInliers->host[i].inliers > best) {
                best = matricesAndInliers->host[i].inliers;
                bestIdx = i;
            }
        }
    }
    printf("%d / %d", best, this->keyPoints->size() / 2);
}

void ssrlcv::PoseEstimator::fillA() {
    const unsigned long length = this->keyPoints->size()/2;
    std::vector<unsigned long> shuffle(length);
    std::iota(shuffle.begin(), shuffle.end(), 0);
    std::random_shuffle(shuffle.begin(), shuffle.end());

    for (int i = 0; i < this->keyPoints->size(); i += 2) {
        float *row = this->A->host.get() + (shuffle[i / 2] * 9);
        float2 loc1 = this->keyPoints->host.get()[i].loc;
        float2 loc2 = this->keyPoints->host.get()[i+1].loc;
        row[0] = loc2.x * loc1.x;
        row[1] = loc2.x * loc1.y;
        row[2] = loc2.x;
        row[3] = loc2.y * loc1.x;
        row[4] = loc2.y * loc1.y;
        row[5] = loc2.y;
        row[6] = loc1.x;
        row[7] = loc1.y;
        row[8] = 1;
    }
}

__global__ void ssrlcv::computeFMatrixAndInliers(ssrlcv::KeyPoint *keyPoints, int numKeyPoints, float *V, unsigned long N, ssrlcv::FMatrixInliers *matricesAndInliers) {
    unsigned long globalID = (blockIdx.y* gridDim.x+ blockIdx.x)*blockDim.x + threadIdx.x;

    if(globalID < N) {

        float *M = V + N;

        float F1[3][3] = {
            {M[7], M[16], M[25]},
            {M[34], M[43], M[52]},
            {M[61], M[70], M[79]}
        };

        float F2[3][3] = {
            {M[8], M[17], M[26]},
            {M[35], M[44], M[53]},
            {M[62], M[71], M[80]}
        };

        // cubic

        // cubed coefficient
        float a = (F1[0][0]*F1[1][1]*F1[2][2] - F1[0][0]*F1[1][1]*F2[2][2] - F1[0][0]*F1[1][2]*F1[2][1] + F1[0][0]*F1[1][2]*F2[2][1] + F1[0][0]*F1[2][1]*F2[1][2] - F1[0][0]*F1[2][2]*F2[1][1] + F1[0][0]*F2[1][1]*F2[2][2] - F1[0][0]*F2[1][2]*F2[2][1] - F1[0][1]*F1[1][0]*F1[2][2] + F1[0][1]*F1[1][0]*F2[2][2] + F1[0][1]*F1[1][2]*F1[2][0] - F1[0][1]*F1[1][2]*F2[2][0] - F1[0][1]*F1[2][0]*F2[1][2] + F1[0][1]*F1[2][2]*F2[1][0] - F1[0][1]*F2[1][0]*F2[2][2] + F1[0][1]*F2[1][2]*F2[2][0] + F1[0][2]*F1[1][0]*F1[2][1] - F1[0][2]*F1[1][0]*F2[2][1] - F1[0][2]*F1[1][1]*F1[2][0] + F1[0][2]*F1[1][1]*F2[2][0] + F1[0][2]*F1[2][0]*F2[1][1] - F1[0][2]*F1[2][1]*F2[1][0] + F1[0][2]*F2[1][0]*F2[2][1] - F1[0][2]*F2[1][1]*F2[2][0] - F1[1][0]*F1[2][1]*F2[0][2] + F1[1][0]*F1[2][2]*F2[0][1] - F1[1][0]*F2[0][1]*F2[2][2] + F1[1][0]*F2[0][2]*F2[2][1] + F1[1][1]*F1[2][0]*F2[0][2] - F1[1][1]*F1[2][2]*F2[0][0] + F1[1][1]*F2[0][0]*F2[2][2] - F1[1][1]*F2[0][2]*F2[2][0] - F1[1][2]*F1[2][0]*F2[0][1] + F1[1][2]*F1[2][1]*F2[0][0] - F1[1][2]*F2[0][0]*F2[2][1] + F1[1][2]*F2[0][1]*F2[2][0] + F1[2][0]*F2[0][1]*F2[1][2] - F1[2][0]*F2[0][2]*F2[1][1] - F1[2][1]*F2[0][0]*F2[1][2] + F1[2][1]*F2[0][2]*F2[1][0] + F1[2][2]*F2[0][0]*F2[1][1] - F1[2][2]*F2[0][1]*F2[1][0] - F2[0][0]*F2[1][1]*F2[2][2] + F2[0][0]*F2[1][2]*F2[2][1] + F2[0][1]*F2[1][0]*F2[2][2] - F2[0][1]*F2[1][2]*F2[2][0] - F2[0][2]*F2[1][0]*F2[2][1] + F2[0][2]*F2[1][1]*F2[2][0]);

        // squared coefficient
        float b = (F1[0][0]*F1[1][1]*F2[2][2] - F1[0][0]*F1[1][2]*F2[2][1] - F1[0][0]*F1[2][1]*F2[1][2] + F1[0][0]*F1[2][2]*F2[1][1] - 2*F1[0][0]*F2[1][1]*F2[2][2] + 2*F1[0][0]*F2[1][2]*F2[2][1] - F1[0][1]*F1[1][0]*F2[2][2] + F1[0][1]*F1[1][2]*F2[2][0] + F1[0][1]*F1[2][0]*F2[1][2] - F1[0][1]*F1[2][2]*F2[1][0] + 2*F1[0][1]*F2[1][0]*F2[2][2] - 2*F1[0][1]*F2[1][2]*F2[2][0] + F1[0][2]*F1[1][0]*F2[2][1] - F1[0][2]*F1[1][1]*F2[2][0] - F1[0][2]*F1[2][0]*F2[1][1] + F1[0][2]*F1[2][1]*F2[1][0] - 2*F1[0][2]*F2[1][0]*F2[2][1] + 2*F1[0][2]*F2[1][1]*F2[2][0] + F1[1][0]*F1[2][1]*F2[0][2] - F1[1][0]*F1[2][2]*F2[0][1] + 2*F1[1][0]*F2[0][1]*F2[2][2] - 2*F1[1][0]*F2[0][2]*F2[2][1] - F1[1][1]*F1[2][0]*F2[0][2] + F1[1][1]*F1[2][2]*F2[0][0] - 2*F1[1][1]*F2[0][0]*F2[2][2] + 2*F1[1][1]*F2[0][2]*F2[2][0] + F1[1][2]*F1[2][0]*F2[0][1] - F1[1][2]*F1[2][1]*F2[0][0] + 2*F1[1][2]*F2[0][0]*F2[2][1] - 2*F1[1][2]*F2[0][1]*F2[2][0] - 2*F1[2][0]*F2[0][1]*F2[1][2] + 2*F1[2][0]*F2[0][2]*F2[1][1] + 2*F1[2][1]*F2[0][0]*F2[1][2] - 2*F1[2][1]*F2[0][2]*F2[1][0] - 2*F1[2][2]*F2[0][0]*F2[1][1] + 2*F1[2][2]*F2[0][1]*F2[1][0] + 3*F2[0][0]*F2[1][1]*F2[2][2] - 3*F2[0][0]*F2[1][2]*F2[2][1] - 3*F2[0][1]*F2[1][0]*F2[2][2] + 3*F2[0][1]*F2[1][2]*F2[2][0] + 3*F2[0][2]*F2[1][0]*F2[2][1] - 3*F2[0][2]*F2[1][1]*F2[2][0]);

        // linear coefficient
        float c = (F1[0][0]*F2[1][1]*F2[2][2] - F1[0][0]*F2[1][2]*F2[2][1] - F1[0][1]*F2[1][0]*F2[2][2] + F1[0][1]*F2[1][2]*F2[2][0] + F1[0][2]*F2[1][0]*F2[2][1] - F1[0][2]*F2[1][1]*F2[2][0] - F1[1][0]*F2[0][1]*F2[2][2] + F1[1][0]*F2[0][2]*F2[2][1] + F1[1][1]*F2[0][0]*F2[2][2] - F1[1][1]*F2[0][2]*F2[2][0] - F1[1][2]*F2[0][0]*F2[2][1] + F1[1][2]*F2[0][1]*F2[2][0] + F1[2][0]*F2[0][1]*F2[1][2] - F1[2][0]*F2[0][2]*F2[1][1] - F1[2][1]*F2[0][0]*F2[1][2] + F1[2][1]*F2[0][2]*F2[1][0] + F1[2][2]*F2[0][0]*F2[1][1] - F1[2][2]*F2[0][1]*F2[1][0] - 3*F2[0][0]*F2[1][1]*F2[2][2] + 3*F2[0][0]*F2[1][2]*F2[2][1] + 3*F2[0][1]*F2[1][0]*F2[2][2] - 3*F2[0][1]*F2[1][2]*F2[2][0] - 3*F2[0][2]*F2[1][0]*F2[2][1] + 3*F2[0][2]*F2[1][1]*F2[2][0]);

        // constant
        float d = (F2[0][0]*F2[1][1]*F2[2][2] - F2[0][0]*F2[1][2]*F2[2][1] - F2[0][1]*F2[1][0]*F2[2][2] + F2[0][1]*F2[1][2]*F2[2][0] + F2[0][2]*F2[1][0]*F2[2][1] - F2[0][2]*F2[1][1]*F2[2][0]);

        float xn = 0, fx, fpx;
        for (int i = 0 ; i < 50; i ++) {
            fx = a * xn * xn * xn + b * xn * xn + c * xn + d;
            fpx = 3 * a * xn * xn + 2 * b * xn + c;
            xn -= fx / fpx;
        }

        if (a * xn * xn * xn + b * xn * xn + c * xn + d < 1.e-5) {
            matricesAndInliers[globalID].valid = true;
            for (int i = 0; i < 3; i ++) {
                for (int j = 0; j < 3; j ++) {
                    matricesAndInliers[globalID].fmatrix[i][j] = xn * F1[i][j] + (1 - xn) * F2[i][j];
                }
            }

            float F[3][3];
            memcpy(F, matricesAndInliers[globalID].fmatrix, 9 * sizeof(float));
            float Ft[3][3];
            transpose(F, Ft);

            float dist;
            float X1[3], X2[3], l[3];
            unsigned long inliers;

            for (int i = 0; i < numKeyPoints; i += 2) {
                X1[0] = keyPoints[i].loc.x;
                X1[1] = keyPoints[i].loc.y;
                X1[2] = 1;
                X2[0] = keyPoints[i+1].loc.x;
                X2[1] = keyPoints[i+1].loc.y;
                X2[2] = 1;

                multiply(F, X1, l);
                dist = (l[0]*X2[0] + l[1]*X2[1] + l[2]*X2[2]) / sqrtf(l[0]*l[0]+l[1]*l[1]);

                multiply(Ft, X2, l);
                dist += (l[0]*X1[0] + l[1]*X1[1] + l[2]*X1[2]) / sqrtf(l[0]*l[0]+l[1]*l[1]);

                if (dist < 1) inliers += 1;
            }

            matricesAndInliers[globalID].inliers = inliers;

        }

        /* float x0 = xn, x1, x2;
        bool threeSols = false;

        if (a * xn * xn * xn + b * xn * xn + c * xn + d > 1.e-5) {
            return;
        } else {
            
            // now quadratic

            float A = a;
            float B = b + a * x0;
            float C = a * c * x0 * x0 + b * c * x0;

            if (B * B - 4 * A * C >= 0) {
                threeSols = true;
                x1 = (-B + sqrtf(B * B - 4 * A * C)) / (2 * A);
                x2 = (-B - sqrtf(B * B - 4 * A * C)) / (2 * A);
                if(globalID == 0) {
                    printf("%f %f %f\n", x0, x1, x2); 
                }
            }

        } */



    }

}