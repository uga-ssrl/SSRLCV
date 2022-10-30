#include "PoseEstimator.cuh"
#include "PointCloudFactory.cuh"
#include <cublas_v2.h>
#include <cusolverDn.h>
#include <list>

ssrlcv::PoseEstimator::PoseEstimator(ssrlcv::ptr::value<ssrlcv::Image> query, ssrlcv::ptr::value<ssrlcv::Image> target, ssrlcv::ptr::value<ssrlcv::Unity<ssrlcv::Match>> matches) :
    query(query),
    target(target),
    A(nullptr, matches->size() * 9, ssrlcv::cpu),
    matches(matches) {}

ssrlcv::Pose ssrlcv::PoseEstimator::estimatePoseRANSAC() {
    this->fillA();
    printf("Done filling A\n");
    unsigned long numMatches = this->matches->size();
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

    A->setMemoryState(gpu);
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
    void (*fp)(ssrlcv::Match *, int, float *, unsigned long, FMatrixInliers *) = &computeFMatrixAndInliers;
    getFlatGridBlock(N,grid,block,fp);
    this->matches->setMemoryState(gpu);
    ssrlcv::ptr::value<ssrlcv::Unity<ssrlcv::FMatrixInliers>> matricesAndInliers(nullptr, N, gpu);
    computeFMatrixAndInliers<<<grid, block>>>(this->matches->device.get(), this->matches->size(), d_U.get(), N, matricesAndInliers->device.get());
    cudaDeviceSynchronize();
    CudaCheckError();
    matricesAndInliers->setMemoryState(cpu);
    A->setMemoryState(cpu);

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
    printf("%d / %d\n", best, this->matches->size());
    float F[3][3];
    memcpy(F, matricesAndInliers->host[bestIdx].fmatrix, 9*sizeof(float));

    ssrlcv::ptr::device<float> d_F(9);
    CudaSafeCall(cudaMemcpy(d_F.get(), F, 3*3*sizeof(float), cudaMemcpyHostToDevice));
    void (*fp2)(ssrlcv::Match *, int, float *) = &computeOutliers;
    getFlatGridBlock(this->matches->size(),grid,block,fp2);
    computeOutliers<<<grid, block>>>(this->matches->device.get(), this->matches->size(), d_F.get());
    cudaDeviceSynchronize();
    CudaCheckError();
    this->matches->setMemoryState(cpu);

    MatchFactory<ssrlcv::SIFT_Descriptor> mf;
    mf.validateMatches(this->matches);

    return getRelativePose(F);
}

void ssrlcv::PoseEstimator::fillA() {
    const unsigned long length = this->matches->size();
    std::vector<unsigned long> shuffle(length);
    std::iota(shuffle.begin(), shuffle.end(), 0);
    std::random_shuffle(shuffle.begin(), shuffle.end());

    for (int i = 0; i < this->matches->size(); i += 1) {
        float *row = this->A->host.get() + (shuffle[i] * 9);
        float2 loc1 = this->matches->host[i].keyPoints[0].loc;
        float2 loc2 = this->matches->host[i].keyPoints[1].loc;
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

ssrlcv::Pose ssrlcv::PoseEstimator::getRelativePose(const float (&F)[3][3]) {
    float K[3][3] = {
        {this->query->camera.foc / this->query->camera.dpix.x, 0, this->query->size.x / 2.0f},
        {0, this->query->camera.foc / this->query->camera.dpix.y, this->query->size.y / 2.0f},
        {0, 0, 1}
    }; // camera calibration matrix (same for both)

    float Kt[3][3]; // tranpose calibration matrix
    transpose(K, Kt);

    float E[3][3]; // essential matrix

    float KtF[3][3];
    multiply(Kt, F, KtF);
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
    ssrlcv::KeyPoint qKP = this->matches->host[globalID].keyPoints[0];
    ssrlcv::KeyPoint tKP = this->matches->host[globalID].keyPoints[1];
    float3 queryPnt = {0, 0, 0};
    float3 queryVec = {
        this->query->camera.dpix.x * ((qKP.loc.x) - (this->query->camera.size.x / 2.0f)),
        this->query->camera.dpix.y * ((qKP.loc.y) - (this->query->camera.size.y / 2.0f)),
        this->query->camera.foc
    }; // identity, since it's relative rotation, so no rotation for query
    normalize(queryVec);
    float3 targetPnt1 = {U[0][2], U[1][2], U[2][2]}; // 3D position in pixel units
    float3 targetPnt2 = -1 * targetPnt1;
    float3 targetVec1 = {
        this->target->camera.dpix.x * ((tKP.loc.x) - (this->target->camera.size.x / 2.0f)),
        this->target->camera.dpix.y * ((tKP.loc.y) - (this->target->camera.size.y / 2.0f)),
        this->target->camera.foc
    };
    float3 targetVec2 = targetVec1;
    float R1t[3][3], R2t[3][3];
    transpose(R1, R1t);
    transpose(R2, R2t);
    targetVec1 = matrixMulVector(targetVec1, R1t);
    targetVec2 = matrixMulVector(targetVec2, R2t);
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

    int best = 0; // TODO: turn this (starting at global id) into kernel that votes for the best one
    
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
            printf("Point position: %f %f %f\n", point.x/ this->query->camera.dpix.x / 1000, point.y/ this->query->camera.dpix.x / 1000, point.z/ this->query->camera.dpix.x / 1000);
            best = i;
        }
    }

    /*****************************************************************
    Note: using transpose for rotations from camera -> world coords
    *****************************************************************/

    Pose pose;

    if (best == 0 || best == 1) {
        pose.roll = atanf(R1t[2][1] / R1t[2][2]);
        pose.pitch = atanf(-R1t[2][0] / (R1t[2][2]/cosf(pose.roll)));
        pose.yaw = atanf(R1t[1][0] / R1t[0][0]);
    } else {
        pose.roll = atanf(R2t[2][1] / R2t[2][2]);
        pose.pitch = atanf(-R2t[2][0] / (R2t[2][2]/cosf(pose.roll)));
        pose.yaw = atanf(R2t[1][0] / R2t[0][0]);
    }

    printf("r: %f %f %f\n", pose.roll, pose.pitch, pose.yaw);

    if (best == 0 || best == 2) {
        pose.x = U[0][2] / this->query->camera.dpix.x / 1000000;
        pose.y = U[1][2] / this->query->camera.dpix.x / 1000000;
        pose.z = U[2][2] / this->query->camera.dpix.x / 1000000;
    } else {
        pose.x = - U[0][2] / this->query->camera.dpix.x / 1000000;
        pose.y = - U[1][2] / this->query->camera.dpix.x / 1000000;
        pose.z = - U[2][2] / this->query->camera.dpix.x / 1000000;
    }

    printf("t: %f %f %f\n", pose.x, pose.y, pose.z);

    return pose;
}

void ssrlcv::PoseEstimator::LM_optimize(ssrlcv::Pose pose) {
    float lambda = 1e11;

    // TODO: catch assertion failures
    do {
        printf("Pose rotations: %f %f %f\n", pose.roll, pose.pitch, pose.yaw);
        printf("Pose positions: %f %f %f\n", pose.x, pose.y, pose.z);
    }
    while(LM_iteration(&pose, &lambda));
}

bool ssrlcv::PoseEstimator::LM_iteration(ssrlcv::Pose *pose, float *lambda) {
    ssrlcv::ptr::value<ssrlcv::Unity<float>> f(nullptr, 4 * this->matches->size(), gpu); // residuals
    ssrlcv::ptr::value<ssrlcv::Unity<float>> J(nullptr, 6 * 4 * this->matches->size(), gpu); // Jacobian of f
    ssrlcv::ptr::value<ssrlcv::Unity<float>> JTJ(nullptr, 6 * 6, gpu);
    ssrlcv::ptr::value<ssrlcv::Unity<float>> JTf(nullptr, 6, gpu);
    this->matches->setMemoryState(gpu);
    ssrlcv::Pose newPose;

    dim3 grid = {1,1,1};
    dim3 block = {1,1,1};

    void (*residualAndJacobianFP)(ssrlcv::Match *, int, ssrlcv::Pose, ssrlcv::Image::Camera, ssrlcv::Image::Camera, float *, float *) = &computeResidualsAndJacobian;
    getFlatGridBlock(this->matches->size(),grid,block,residualAndJacobianFP);
    computeResidualsAndJacobian<<<grid, block>>>(this->matches->device.get(), this->matches->size(), *pose, this->query->camera, this->target->camera, f->device.get(), J->device.get());
    cudaDeviceSynchronize();
    CudaCheckError();

    ssrlcv::ptr::value<ssrlcv::Unity<float>> cost(nullptr, 1, cpu);
    *(cost->host.get()) = 0;
    cost->setMemoryState(gpu);
    void (*costFP)(ssrlcv::Match *, int, ssrlcv::Pose, ssrlcv::Image::Camera, ssrlcv::Image::Camera, float *) = &computeCost;
    getFlatGridBlock(this->matches->size(),grid,block,costFP);
    computeCost<<<grid, block>>>(this->matches->device.get(), this->matches->size(), *pose, this->query->camera, this->target->camera, cost->device.get());
    cudaDeviceSynchronize();
    CudaCheckError();
    cost->setMemoryState(cpu);
    printf("Starting cost: %f\n", *(cost->host.get()));

    void (*fp2)(float *, unsigned long, float *) = &computeJTJ;
    getFlatGridBlock(this->matches->size() * 4,grid,block,fp2);
    computeJTJ<<<grid, block>>>(J->device.get(), this->matches->size() * 4, JTJ->device.get());
    cudaDeviceSynchronize();
    CudaCheckError();

    ssrlcv::ptr::value<ssrlcv::Unity<float>> newCost(nullptr, 1, cpu);
    *(newCost->host.get()) = *(cost->host.get()) + 100; // just to make sure it starts off greater

    float old_lambda = 0;

    void (*fp3)(float *, float *, unsigned long, float *) = &computeJTf;
    getFlatGridBlock(this->matches->size() * 4,grid,block,fp3);
    computeJTf<<<grid, block>>>(J->device.get(), f->device.get(), this->matches->size() * 4, JTf->device.get());
    cudaDeviceSynchronize();
    CudaCheckError();

    int num_iterations = 0;
    
    while(*(cost->host.get()) <= *(newCost->host.get())) {
        if(num_iterations >= 10) {
            return false;
        }
        num_iterations += 1;

        JTJ->setMemoryState(cpu);
        for(int i = 0; i < 6; i ++) {
            JTJ->host[i + 6 * i] += (*lambda - old_lambda);
        }
        //printf("JTJ[0][0] = %f\n", JTJ->host[0]);
        JTJ->setMemoryState(gpu);

        cusolverDnHandle_t  cusolverH       = NULL;
        cusolverStatus_t    cusolver_status = CUSOLVER_STATUS_SUCCESS;

        ssrlcv::ptr::value<ssrlcv::Unity<float>> S  = ssrlcv::ptr::value<ssrlcv::Unity<float>>(nullptr,6,ssrlcv::gpu);
        ssrlcv::ptr::value<ssrlcv::Unity<float>> UT  = ssrlcv::ptr::value<ssrlcv::Unity<float>>(nullptr,6*6,ssrlcv::gpu); // transpose because column major
        ssrlcv::ptr::value<ssrlcv::Unity<float>> V = ssrlcv::ptr::value<ssrlcv::Unity<float>>(nullptr,6*6,ssrlcv::gpu); // not transpose because column major

        cusolver_status = cusolverDnCreate(&cusolverH);
        assert(CUSOLVER_STATUS_SUCCESS == cusolver_status);

        // cuSOLVER SVD
        int lwork       = 0;
        int *devInfo    = NULL;
        float *d_work   = NULL;
        float *d_rwork  = NULL;
        cusolver_status = cusolverDnDgesvd_bufferSize(cusolverH,6,6,&lwork);
        assert (cusolver_status == CUSOLVER_STATUS_SUCCESS);

        CudaSafeCall(cudaMalloc((void**)&d_work , sizeof(float)*lwork));
        CudaSafeCall(cudaMalloc ((void**)&devInfo, sizeof(int)));

        cusolver_status = cusolverDnSgesvd(cusolverH,'A','A',6,6,JTJ->device.get(),6,S->device.get(),UT->device.get(),6,V->device.get(),6,d_work,lwork,d_rwork,devInfo); // JTJ should technically be column major but it's symmetric so doesn't matter
        cudaDeviceSynchronize();
        assert (cusolver_status == CUSOLVER_STATUS_SUCCESS);

        S->setMemoryState(cpu);
        UT->setMemoryState(cpu);
        V->setMemoryState(cpu);
        JTJ->setMemoryState(cpu);
        JTf->setMemoryState(cpu);

        float JTJ_inv[6][6];
        float VS[6][6];
        float delta[6];
        
        float mult;
        for(int i = 0; i < 6; i ++) {
            for(int j = 0; j < 6; j ++) {
                mult = (S->host[j] > 0.0001) ? 1 / S->host[j] : 0;
                VS[i][j] = V->host[6*i + j] * mult;
            }
        }

        for(int i = 0; i < 6; i ++) {
            for(int j = 0; j < 6; j ++) {
                JTJ_inv[i][j] = 0;
                for (int k = 0; k < 6; k ++) {
                    JTJ_inv[i][j] += VS[i][k] * UT->host[k * 6 + j];
                }
            }
        }

        // delta = - JTJ_inv x JTf

        for(int i = 0; i < 6; i ++) {
            delta[i] = 0;
            for(int j = 0; j < 6; j ++) {
                delta[i] += - JTJ_inv[i][j] * JTf->host[j];
            }
        }

        newPose.roll = pose->roll + delta[0];
        newPose.pitch = pose->pitch + delta[1];
        newPose.yaw = pose->yaw + delta[2];
        newPose.x = pose->x + delta[3];
        newPose.y = pose->y + delta[4];
        newPose.z = pose->z + delta[5];

        //printf("delta[1] = %f\n", delta[1]);

        // now check if new pose is better
        *(newCost->host.get()) = 0;
        newCost->setMemoryState(ssrlcv::gpu);
        getFlatGridBlock(this->matches->size(),grid,block,costFP);
        //printf("Block: %d %d %d\n", block.x, block.y, block.z);
        computeCost<<<grid, block>>>(this->matches->device.get(), this->matches->size(), newPose, this->query->camera, this->target->camera, newCost->device.get());
        cudaDeviceSynchronize();
        CudaCheckError();
        newCost->setMemoryState(ssrlcv::cpu);
        printf("New cost: %f\n", *(newCost->host.get()));

        old_lambda = *lambda;
        *lambda *= 10;
        //printf("Lambda: %f\n", *lambda);
    }

    *lambda /= 100; // really just dividing by 10, but need to account for the multiplication by 10 in the last loop

    // update pose
    pose->roll = newPose.roll;
    pose->pitch = newPose.pitch;
    pose->yaw = newPose.yaw;
    pose->x = newPose.x;
    pose->y = newPose.y;
    pose->z = newPose.z;


    this->matches->setMemoryState(cpu);

    return true;
    
}

__global__ void ssrlcv::computeFMatrixAndInliers(ssrlcv::Match *matches, int numMatches, float *V, unsigned long N, ssrlcv::FMatrixInliers *matricesAndInliers) {
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

            for (int i = 0; i < numMatches; i ++) {
                X1.x = matches[i].keyPoints[0].loc.x;
                X1.y = matches[i].keyPoints[0].loc.y;
                X1.z = 1;
                X2.x = matches[i].keyPoints[1].loc.x;
                X2.y = matches[i].keyPoints[1].loc.y;
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

                if (dist < 0.25) inliers += 1;
            }

            matricesAndInliers[globalID].inliers = inliers;
        }

    }

}

__global__ void ssrlcv::computeOutliers(ssrlcv::Match *matches, int numMatches, float *F) {
    unsigned long globalID = (blockIdx.y* gridDim.x+ blockIdx.x)*blockDim.x + threadIdx.x;

    if(globalID < numMatches) {

        float dist, denom;
        float3 X1, X2, FX1, FTX2;
        
        X1.x = matches[globalID].keyPoints[0].loc.x;
        X1.y = matches[globalID].keyPoints[0].loc.y;
        X1.z = 1;
        X2.x = matches[globalID].keyPoints[1].loc.x;
        X2.y = matches[globalID].keyPoints[1].loc.y;
        X2.z = 1;

        FX1 = {
            F[0] * X1.x + F[1] * X1.y + F[2] * X1.z,
            F[3] * X1.x + F[4] * X1.y + F[5] * X1.z,
            F[6] * X1.x + F[7] * X1.y + F[8] * X1.z
        };

        FTX2 = {
            F[0] * X2.x + F[1] * X2.y + F[2] * X2.z,
            F[3] * X2.x + F[4] * X2.y + F[5] * X2.z,
            F[6] * X2.x + F[7] * X2.y + F[8] * X2.z
        };

        denom = FX1.x * FX1.x;
        denom += FX1.y * FX1.y;
        denom += FTX2.x * FTX2.x;
        denom += FTX2.y * FTX2.y;

        dist = dotProduct(X2, FX1) * dotProduct(X2, FX1) / denom;

        if (dist >= 0.25) {
            matches[globalID].invalid = true;
        }
    }
}

__global__ void ssrlcv::computeResidualsAndJacobian(ssrlcv::Match *matches, int numMatches, ssrlcv::Pose pose, ssrlcv::Image::Camera query, ssrlcv::Image::Camera target, float *residuals, float *jacobian) {
    unsigned long globalID = (blockIdx.y* gridDim.x+ blockIdx.x)*blockDim.x + threadIdx.x;

    if (globalID < numMatches) {
        float2 q_loc = matches[globalID].keyPoints[0].loc;
        float2 t_loc = matches[globalID].keyPoints[1].loc;
        float *r_out = residuals + 4 * globalID;
        float *j_out = jacobian + 6 * 4 * globalID; // Jacobian is 4 rows of 6 per match

        float delta = 1e-5;

        float4 res = getResidual(pose, &query, &target, q_loc, t_loc);
        r_out[0] = res.x;
        r_out[1] = res.y;
        r_out[2] = res.z;
        r_out[3] = res.w;

        float4 left, right;
        float saved; // set back to initial to avoid fp errors

        saved = pose.roll;
        pose.roll += delta;
        right = getResidual(pose, &query, &target, q_loc, t_loc);
        pose.roll -= 2 * delta;
        left = getResidual(pose, &query, &target, q_loc, t_loc);
        pose.roll = saved;
        j_out[0] = (right.x - left.x) / (2 * delta);
        j_out[6] = (right.y - left.y) / (2 * delta);
        j_out[12] = (right.z - left.z) / (2 * delta);
        j_out[18] = (right.w - left.w) / (2 * delta);
        ++j_out;

        saved = pose.pitch;
        pose.pitch += delta;
        right = getResidual(pose, &query, &target, q_loc, t_loc);
        pose.pitch -= 2 * delta;
        left = getResidual(pose, &query, &target, q_loc, t_loc);
        pose.pitch = saved;
        j_out[0] = (right.x - left.x) / (2 * delta);
        j_out[6] = (right.y - left.y) / (2 * delta);
        j_out[12] = (right.z - left.z) / (2 * delta);
        j_out[18] = (right.w - left.w) / (2 * delta);
        ++j_out;
        
        saved = pose.yaw;
        pose.yaw += delta;
        right = getResidual(pose, &query, &target, q_loc, t_loc);
        pose.yaw -= 2 * delta;
        left = getResidual(pose, &query, &target, q_loc, t_loc);
        pose.yaw = saved;
        j_out[0] = (right.x - left.x) / (2 * delta);
        j_out[6] = (right.y - left.y) / (2 * delta);
        j_out[12] = (right.z - left.z) / (2 * delta);
        j_out[18] = (right.w - left.w) / (2 * delta);
        ++j_out;

        saved = pose.x;
        pose.x += delta;
        right = getResidual(pose, &query, &target, q_loc, t_loc);
        pose.x -= 2 * delta;
        left = getResidual(pose, &query, &target, q_loc, t_loc);
        pose.x = saved;
        j_out[0] = (right.x - left.x) / (2 * delta);
        j_out[6] = (right.y - left.y) / (2 * delta);
        j_out[12] = (right.z - left.z) / (2 * delta);
        j_out[18] = (right.w - left.w) / (2 * delta);
        ++j_out;

        saved = pose.y;
        pose.y += delta;
        right = getResidual(pose, &query, &target, q_loc, t_loc);
        pose.y -= 2 * delta;
        left = getResidual(pose, &query, &target, q_loc, t_loc);
        pose.y = saved;
        j_out[0] = (right.x - left.x) / (2 * delta);
        j_out[6] = (right.y - left.y) / (2 * delta);
        j_out[12] = (right.z - left.z) / (2 * delta);
        j_out[18] = (right.w - left.w) / (2 * delta);
        ++j_out;

        saved = pose.z;
        pose.z += delta;
        right = getResidual(pose, &query, &target, q_loc, t_loc);
        pose.z -= 2 * delta;
        left = getResidual(pose, &query, &target, q_loc, t_loc);
        pose.z = saved;
        j_out[0] = (right.x - left.x) / (2 * delta);
        j_out[6] = (right.y - left.y) / (2 * delta);
        j_out[12] = (right.z - left.z) / (2 * delta);
        j_out[18] = (right.w - left.w) / (2 * delta);
        ++j_out;
    }
}

__global__ void ssrlcv::computeCost(ssrlcv::Match *matches, int numMatches, ssrlcv::Pose pose, ssrlcv::Image::Camera query, ssrlcv::Image::Camera target, float *cost) {
    unsigned long globalID = (blockIdx.y* gridDim.x+ blockIdx.x)*blockDim.x + threadIdx.x;
    if (globalID < numMatches) {
        float2 q_loc = matches[globalID].keyPoints[0].loc;
        float2 t_loc = matches[globalID].keyPoints[1].loc;

        float4 res = getResidual(pose, &query, &target, q_loc, t_loc);
        float sum = res.x * res.x + res.y * res.y + res.z * res.z + res.w * res.w;

        atomicAdd(cost, sum);
    }
}

__device__ __host__ float4 ssrlcv::getResidual(ssrlcv::Pose pose, ssrlcv::Image::Camera *query, ssrlcv::Image::Camera *target, float2 q_loc, float2 t_loc) {
    float3 queryPnt = {0, 0, 0};
    float3 queryVec = {
        query->dpix.x * ((q_loc.x) - (query->size.x / 2.0f)),
        query->dpix.y * ((q_loc.y) - (query->size.y / 2.0f)),
        query->foc
    }; // identity, since it's relative rotation, so no rotation for query
    normalize(queryVec);
    float3 targetPnt = {pose.x, pose.y, pose.z}; // 3D position in pixel units
    float3 targetVec = {
        target->dpix.x * ((t_loc.x) - (target->size.x / 2.0f)),
        target->dpix.y * ((t_loc.y) - (target->size.y / 2.0f)),
        target->foc
    };
    targetVec = rotatePoint(targetVec, {pose.roll, pose.pitch, pose.yaw});
    normalize(targetVec);

    Bundle::Line queryBundle = {queryVec, queryPnt};
    Bundle::Line targetBundle = {targetVec, targetPnt};

    Bundle::Line *L1 = &queryBundle;
    Bundle::Line *L2 = &targetBundle;
    
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
    float4 point_homog = {point.x, point.y, point.z, 1};

    ssrlcv::Image::Camera q_mock, t_mock;
    float4 q_P[3], t_P[3];
    q_mock = *query;
    q_mock.cam_pos = {0, 0, 0};
    q_mock.ecef_offset = {0, 0, 0};
    q_mock.cam_rot = {0, 0, 0};
    t_mock = *target;
    t_mock.cam_pos = {pose.x, pose.y, pose.z};
    t_mock.ecef_offset = {0, 0, 0};
    t_mock.cam_rot = {pose.roll, pose.pitch, pose.yaw};
    getProjectionMatrix(q_P, &q_mock);
    getProjectionMatrix(t_P, &t_mock);

    float3 q_loc_hat, t_loc_hat;
    multiply(q_P, point_homog, q_loc_hat);
    multiply(t_P, point_homog, t_loc_hat);

    return {
        q_loc.x - q_loc_hat.x/q_loc_hat.z,
        q_loc.y - q_loc_hat.y/q_loc_hat.z,
        t_loc.x - t_loc_hat.x/t_loc_hat.z,
        t_loc.y - t_loc_hat.y/t_loc_hat.z
    };
}

__global__ void ssrlcv::computeJTJ(float *jacobian, unsigned long rows, float *output) {
    unsigned long globalID = (blockIdx.y* gridDim.x+ blockIdx.x)*blockDim.x + threadIdx.x;

    if (globalID < rows) {
        for (int i = 0; i < 6; i ++) {
            for (int j = 0; j < 6; j ++) {
                float left = (jacobian + globalID * 6)[i];
                float right = (jacobian + globalID * 6)[j];
                atomicAdd(output + i + 6*j, left * right);
            }
        }
    }
}

__global__ void ssrlcv::computeJTf(float *jacobian, float *f, unsigned long rows, float *output) {
    unsigned long globalID = (blockIdx.y* gridDim.x+ blockIdx.x)*blockDim.x + threadIdx.x;

    if (globalID < rows) {
        for (int i = 0; i < 6; i ++) {
            float left = (jacobian + globalID * 6)[i];
            float right = (f + globalID)[0];
            atomicAdd(output + i, left * right);
        }
    }
}