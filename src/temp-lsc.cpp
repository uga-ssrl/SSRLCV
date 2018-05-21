#include <stdlib.h>
#include <stdio.h>
#include <assert.h>

#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <cusolverDn.h>


int main(int argc, char ** argv)
{
	cusolverDnHandle_t	h_cusolver		= NULL;
	cublasHandle_t		h_cublas		= NULL;
	cublasStatus_t		status_cublas		= CUBLAS_STATUS_SUCCESS;
	cusolverStatus_t	status_cusolver		= CUSOLVER_STATUS_SUCCESS;

	// Example vectors
	
	int A_rows = 3;
	int A_cols = 3;
	int A_ld   = 3;
	int B_cols = 1;
	int B_ld   = 3;

	double A[9] = { 1.0, 4.0, 2.0, 2.0, 5.0, 1.0, 3.0, 6.0, 1.0 };// Coefficient Matrix
	double B[3] = { 6.0, 15.0, 4.0 };// Operand
	double XC[3]; // Solution vector

	// Memory in GPU
	double * d_A	= NULL; // Linear memory in GPU
	double * d_tau	= NULL; // ''
	double * d_B 	= NULL;

	// Todo: Find the meaning of the below variables
	int * devInfo	= NULL; // info in gpu
	int info_gpu	= 0;

	double * d_work = NULL;
	int lwork 	= 0;

	assert(cusolverDnCreate(&h_cusolver) == CUSOLVER_STATUS_SUCCESS);
	assert(cublasCreate(&h_cublas) == CUBLAS_STATUS_SUCCESS);

	assert(cudaSuccess == cudaMalloc((void**) &d_A, 	sizeof(double) * A_rows * A_ld));
	assert(cudaSuccess == cudaMalloc((void**) &d_Tau, 	sizeof(double) * A_rows));
	assert(cudaSuccess == cudaMalloc((void**) &d_B,		sizeof(double) * B_cols * B_ld));
	assert(cudaSuccess == cudaMalloc((void**) &devInfo,	sizeof(int)));
	assert(cudaSuccess == cudaMemcpy(d_A, A, sizeof(double) * A_rows * A_cols));
	assert(cudaMemcpy(d_B, B, sizeof(double) * A_rows * B_cols));

	// "Step 3: Query working space of geqrf and ormqr"
	// This is a helper function that determines the size of the buffer necessary to work with
	// "cusolverDn<t>geqrf()"
	// https://docs.nvidia.com/cuda/cusolver/index.html#cuds-lt-t-gt-geqrf
	assert(CUSOLVER_STATUS_SUCCESS == cusolverDnDgeqrf_bufferSize(cusolverH, A_rows, A_cols, d_A, A_ld, &lwork));
	assert(cudaSuccess == cudaMalloc((void **) &d_work, sizeof(double) * lwork));

	// Compute the thing
	// One of these A_rows should probably be A_cols.  The example provided is pretty rubbish
	assert(CUSOLVER_STATUS_SUCCESS == cusolverDnDgeqrf(h_cusolver, CUBLAS_SIDE_LEFT, CUBLAS_OP_T, A_rows, B_cols, A_rows,  d_A, A_ld, d_tau, d_B, B_ld, d_work, lwork, devInfo));
	assert(cudaSuccess == cudaDeviceSynchronize());
	
	// "See if good"
	assert(cudaSuccess == cudaMemcpy(&info_gpu, devInfo, sizeof(int), cudaMemcpyDeviceToHost));
	assert(info_gpu == 0);
	
}

