#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>
#include <cublas_v2.h>

#ifndef NUM_REPS
#define NUM_REPS 10
#endif

void checkCublas(cublasStatus_t stat, const char *msg)
{
    if (stat != CUBLAS_STATUS_SUCCESS)
    {
        fprintf(stderr, "cuBLAS error: %s (code %d)\n", msg, stat);
        exit(EXIT_FAILURE);
    }
}

void checkCuda(cudaError_t err, const char *msg)
{
    if (err != cudaSuccess)
    {
        fprintf(stderr, "CUDA error: %s (%s)\n", msg, cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }
}

int main(int argc, char **argv)
{
    if (argc < 2)
    {
        fprintf(stderr, "Usage: %s <matrix_size>\n", argv[0]);
        return 1;
    }

    cudaFree(0);

    int N = atoi(argv[1]);
    size_t size = N * N * sizeof(double);

    // Host memory
    double *h_A = (double *)malloc(size);
    double *h_B = (double *)malloc(size);
    double *h_C = (double *)malloc(size);
    if (!h_A || !h_B || !h_C)
    {
        fprintf(stderr, "Host malloc failed\n");
        return 1;
    }

    for (int i = 0; i < N * N; ++i)
    {
        h_A[i] = 1.0;
        h_B[i] = 2.0;
    }

    // Device memory
    double *d_A, *d_B, *d_C;
    checkCuda(cudaMalloc(&d_A, size), "cudaMalloc d_A");
    checkCuda(cudaMalloc(&d_B, size), "cudaMalloc d_B");
    checkCuda(cudaMalloc(&d_C, size), "cudaMalloc d_C");

    checkCuda(cudaMemcpy(d_A, h_A, size, cudaMemcpyHostToDevice), "Memcpy A");
    checkCuda(cudaMemcpy(d_B, h_B, size, cudaMemcpyHostToDevice), "Memcpy B");

    // cuBLAS setup
    cublasHandle_t handle;
    checkCublas(cublasCreate(&handle), "create handle");

    double alpha = 1.0, beta = 0.0;

    // Timing
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    float total_ms = 0.0;
    for (int rep = 0; rep < NUM_REPS; ++rep)
    {
        cudaMemset(d_C, 0, size);
        cudaEventRecord(start);

        checkCublas(cublasDgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N,
                                N, N, N, &alpha, d_A, N, d_B, N, &beta, d_C, N),
                    "cublasDgemm");

        cudaEventRecord(stop);
        cudaEventSynchronize(stop);

        float ms;
        cudaEventElapsedTime(&ms, start, stop);
        total_ms += ms;
    }

    float avg_time_ms = total_ms / NUM_REPS;
    printf("cuBLAS: N=%d → Avg time = %.6f ms\n", N, avg_time_ms);

    // Copy result back
    checkCuda(cudaMemcpy(h_C, d_C, size, cudaMemcpyDeviceToHost), "Memcpy C");

    // Print example values
    printf("some of the results: C[0] = %f, C[%d] = %f\n", h_C[0], N * N - 1, h_C[N * N - 1]);

    // Clean up
    cublasDestroy(handle);
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
    free(h_A);
    free(h_B);
    free(h_C);

    return 0;
}
