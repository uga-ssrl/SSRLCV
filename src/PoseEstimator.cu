#include "PoseEstimator.cuh"
#include "PointCloudFactory.cuh"
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

    assert(CUSOLVER_STATUS_SUCCESS == cusolver_status);


    int m = 9;
    int n = 7;
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
    computeFMatrixAndInliers<<<grid, block>>>(this->keyPoints->device.get(), this->keyPoints->size(), d_U.get(), N, matricesAndInliers->device.get());
    cudaDeviceSynchronize();
    CudaCheckError();
    this->keyPoints->transferMemoryTo(cpu);
    matricesAndInliers->transferMemoryTo(cpu);
    A->transferMemoryTo(cpu);

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
    printf("%d\n", bestIdx);
    printf("%d / %d\n", best, this->keyPoints->size() / 2);
    memcpy(this->F, matricesAndInliers->host[bestIdx].fmatrix, 9*sizeof(float));

    /*float X[7][9], Y[9];
    memcpy(&X, A->host.get(), 7*9*sizeof(float));
    float tmp[9][9];
    CudaSafeCall(cudaMemcpy(&tmp[0][0], d_U.get(), 81*sizeof(float), cudaMemcpyDeviceToHost));
    for (int i = 0; i < 9; i ++) {
        Y[i] = tmp[6][i];
    }

    for (int i = 0; i < 7; i ++) {
        float prod = 0;
        for (int j = 0; j < 9; j ++) {
            prod += X[i][j] * Y[j];
        }
        printf("%f\n", prod);
    }
    exit(0); */
    /* float U[81];
    CudaSafeCall(cudaMemcpy(&U, d_U.get(), 81*sizeof(float), cudaMemcpyDeviceToHost));
    for(int i = 0; i < 81; i ++) {
        printf("U[%d][%d] = %f\n", i/9, i%9, U[i]);
    }
    exit(0); */

    //exit(0);
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

void ssrlcv::PoseEstimator::getRotations(bool relative) {
    if (relative) {
        float K[3][3] = {
            {this->queryImage->camera.foc / this->queryImage->camera.dpix.x, 0, this->queryImage->size.x / 2.0f},
            {0, this->queryImage->camera.foc / this->queryImage->camera.dpix.y, this->queryImage->size.y / 2.0f},
            {0, 0, 1}
        }; // camera calibration matrix (same for both)

        float Kt[3][3]; // tranpose calibration matrix
        transpose(K, Kt);

        float E[3][3]; // essential matrix

        float KtF[3][3];
        multiply(Kt, this->F, KtF);
        multiply(KtF, K, E);

        float W[3][3] = {
            {0, -1, 0},
            {1, 0, 0},
            {0, 0, 1}
        };
        float Wt[3][3];
        transpose(W, Wt);

        cusolverDnHandle_t cusolverH = nullptr;
        cusolverStatus_t cusolver_status = CUSOLVER_STATUS_SUCCESS;

        ssrlcv::ptr::device<float> d_A(3*3);
        ssrlcv::ptr::device<float> d_S(3*3);
        ssrlcv::ptr::device<int> devInfo(1);
        ssrlcv::ptr::device<float> d_rwork(2);
        ssrlcv::ptr::device<float> d_U(3*3);
        ssrlcv::ptr::device<float> d_Vt(3*3);
        int lwork = 0;

        CudaSafeCall(cudaMemcpy(d_A.get(), E, 3*3*sizeof(float), cudaMemcpyHostToDevice));

        cusolver_status = cusolverDnCreate(&cusolverH);
        assert(CUSOLVER_STATUS_SUCCESS == cusolver_status);

        cusolver_status = cusolverDnSgesvd_bufferSize(cusolverH, 3, 3, &lwork);
        assert(cusolver_status == CUSOLVER_STATUS_SUCCESS);

        ssrlcv::ptr::device<float> d_work(lwork);
        cusolver_status = cusolverDnSgesvd(cusolverH, 'A', 'A', 3, 3,
        d_A.get(), 3, d_S.get(), d_U.get(), 3, d_Vt.get(), 3, d_work.get(), lwork, d_rwork.get(), devInfo.get());
        cudaDeviceSynchronize();
        assert(cusolver_status == CUSOLVER_STATUS_SUCCESS);

        float U[3][3];
        float Vt[3][3];
        CudaSafeCall(cudaMemcpy(&U[0][0], d_Vt.get(), 9*sizeof(float), cudaMemcpyDeviceToHost));
        CudaSafeCall(cudaMemcpy(&Vt[0][0], d_U.get(), 9*sizeof(float), cudaMemcpyDeviceToHost));

        float UW[3][3], UWt[3][3];
        float R1[3][3], R2[3][3];
        multiply(U, W, UW);
        multiply(U, Wt, UWt);
        multiply(UW, Vt, R1);
        multiply(UWt, Vt, R2);


        unsigned int globalID = 1702;
        unsigned int qIdx = globalID * 2;
        unsigned int tIdx = qIdx + 1;
        float3 queryPnt = {0, 0, 0};
        float3 queryVec = {
            this->queryImage->camera.dpix.x * ((this->keyPoints->host[qIdx].loc.x) - (this->queryImage->camera.size.x / 2.0f)),
            this->queryImage->camera.dpix.y * ((this->keyPoints->host[qIdx].loc.y) - (this->queryImage->camera.size.y / 2.0f)),
            this->queryImage->camera.foc
        }; // identity, since it's relative rotation, so no rotation for query
        normalize(queryVec);
        float3 targetPnt1 = {U[0][2], U[1][2], U[2][2]}; // 3D position in pixel units
        float3 targetPnt2 = -1 * targetPnt1;
        float3 targetVec1 = {
            this->targetImage->camera.dpix.x * ((this->keyPoints->host[tIdx].loc.x) - (this->targetImage->camera.size.x / 2.0f)),
            this->targetImage->camera.dpix.y * ((this->keyPoints->host[tIdx].loc.y) - (this->targetImage->camera.size.y / 2.0f)),
            this->targetImage->camera.foc
        };
        float3 targetVec2 = targetVec1;
        targetVec1 = matrixMulVector(targetVec1, R1);
        targetVec2 = matrixMulVector(targetVec2, R2);
        normalize(targetVec1);
        normalize(targetVec2);

        Bundle::Line queryBundle = {queryVec, queryPnt};
        Bundle::Line targetBundles[4] = {
            {targetVec1, targetPnt1},
            {targetVec1, targetPnt2},
            {targetVec2, targetPnt1},
            {targetVec2, targetPnt2}
        };

        Bundle::Line *L1 = &queryBundle;
        Bundle::Line *L2 = targetBundles;

        // float4 tmp[3], P2[3];
        // tmp[0] = {R[0][0], R[1][0], R[2][0], U[0][2]};
        // tmp[1] = {R[0][1], R[1][1], R[2][1], U[1][2]};
        // tmp[2] = {R[0][2], R[1][2], R[2][2], U[2][2]};
        // float3 tmp2[3];
        // tmp2[0] = {K[0][0], K[0][1], K[0][2]};
        // tmp2[1] = {K[1][0], K[1][1], K[1][2]};
        // tmp2[2] = {K[2][0], K[2][1], K[2][2]};
        // multiply(tmp2, tmp, P2);
        // float4 P1[3];
        // P1[0] = {K[0][0], K[0][1], K[0][2], 0};
        // P1[1] = {K[1][0], K[1][1], K[1][2], 0};
        // P1[2] = {K[2][0], K[2][1], K[2][2], 0};
        
        for (int i = 0; i < 4; i++, L2++) {
            // calculate the normals
            float3 n2 = crossProduct(L2->vec,crossProduct(L1->vec,L2->vec));
            float3 n1 = crossProduct(L1->vec,crossProduct(L1->vec,L2->vec));

            // calculate the numerators
            float numer1 = dotProduct((L2->pnt - L1->pnt),n2);
            float numer2 = dotProduct((L1->pnt - L2->pnt),n1);

            // calculate the denominators
            float denom1 = dotProduct(L1->vec,n2);
            float denom2 = dotProduct(L2->vec,n1);

            // get the S points
            float3 s1 = L1->pnt + (numer1/denom1) * L1->vec;
            float3 s2 = L2->pnt + (numer2/denom2) * L2->vec;
            float3 point = (s1 + s2)/2.0;

            bool ok1 = magnitude(point - (L1->pnt + L1->vec)) < magnitude(point - (L1->pnt));
            bool ok2 = magnitude(point - (L2->pnt + L2->vec)) < magnitude(point - (L2->pnt));
            if (ok1 && ok2) {
                printf("Point position: %f %f %f\n", point.x/ this->queryImage->camera.dpix.x / 1000, point.y/ this->queryImage->camera.dpix.x / 1000, point.z/ this->queryImage->camera.dpix.x / 1000);
            } else {
                printf("Bad position: %f %f %f\n", point.x/ this->queryImage->camera.dpix.x / 1000, point.y/ this->queryImage->camera.dpix.x / 1000, point.z/ this->queryImage->camera.dpix.x / 1000);
            }
        }


        float x_rot = atanf(R1[2][1] / R1[2][2]) * 180 / PI;
        float y_rot = atanf(-R1[2][0] / (R1[2][2]/cosf(x_rot))) * 180 / PI;
        float z_rot = atanf(R1[1][0] / R1[0][0]) * 180 / PI;

        printf("r: %f %f %f\n", x_rot, y_rot, z_rot);

        x_rot = atanf(R2[2][1] / R2[2][2]) * 180 / PI;
        y_rot = atanf(-R2[2][0] / (R2[2][2]/cosf(x_rot))) * 180 / PI;
        z_rot = atanf(R2[1][0] / R2[0][0]) * 180 / PI;

        printf("r: %f %f %f\n", x_rot, y_rot, z_rot);

        printf("t: %f %f %f\n", U[0][2] / this->queryImage->camera.dpix.x / 1000, U[1][2] / this->queryImage->camera.dpix.x / 1000, U[2][2] / this->queryImage->camera.dpix.x / 1000);
        

        //printf("\nE: [%f,%f,%f,%f,%f,%f,%f,%f,%f]\n", E[0][0], E[0][1], E[0][2], E[1][0], E[1][1], E[1][2], E[2][0], E[2][1], E[2][2]);

        //printf("\nKtF: [%f,%f,%f,%f,%f,%f,%f,%f,%f]\n", KtF[0][0], KtF[0][1], KtF[0][2], KtF[1][0], KtF[1][1], KtF[1][2], KtF[2][0], KtF[2][1], KtF[2][2]);

        //printf("\nF: [%f,%f,%f,%f,%f,%f,%f,%f,%f]\n", this->F[0][0], this->F[0][1], this->F[0][2], this->F[1][0], this->F[1][1], this->F[1][2], this->F[2][0], this->F[2][1], this->F[2][2]);

    } else {
        // not yet implemented
    }
}

__global__ void ssrlcv::computeFMatrixAndInliers(ssrlcv::KeyPoint *keyPoints, int numKeyPoints, float *V, unsigned long N, ssrlcv::FMatrixInliers *matricesAndInliers) {
    unsigned long globalID = (blockIdx.y* gridDim.x+ blockIdx.x)*blockDim.x + threadIdx.x;

    if(globalID < N) {

        float *M = V + globalID * 81;

        float F1[3][3];

        float F2[3][3];

        memcpy(F1[0], &M[63], 9 * sizeof(float));
        memcpy(F2[0], &M[72], 9 * sizeof(float));

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

            float dist, denom;
            float3 X1, X2, FX1, FTX2;
            unsigned long inliers;

            for (int i = 0; i < numKeyPoints; i += 2) {
                X1.x = keyPoints[i].loc.x;
                X1.y = keyPoints[i].loc.y;
                X1.z = 1;
                X2.x = keyPoints[i+1].loc.x;
                X2.y = keyPoints[i+1].loc.y;
                X2.z = 1;

                FX1 = {
                    F[0][0] * X1.x + F[0][1] * X1.y + F[0][2] * X1.z,
                    F[1][0] * X1.x + F[1][1] * X1.y + F[1][2] * X1.z,
                    F[2][0] * X1.x + F[2][1] * X1.y + F[2][2] * X1.z
                };

                FTX2 = {
                    F[0][0] * X2.x + F[1][0] * X2.y + F[2][0] * X2.z,
                    F[0][1] * X2.x + F[1][1] * X2.y + F[2][1] * X2.z,
                    F[0][2] * X2.x + F[1][2] * X2.y + F[2][2] * X2.z
                };

                denom = FX1.x * FX1.x;
                denom += FX1.y * FX1.y;
                denom += FTX2.x * FTX2.x;
                denom += FTX2.y * FTX2.y;

                dist = dotProduct(X2, FX1) * dotProduct(X2, FX1) / denom;

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