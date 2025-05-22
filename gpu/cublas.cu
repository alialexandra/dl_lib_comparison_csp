#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>
#include <cublas_v2.h>

#ifndef NUM_REPS
#define NUM_REPS 5
#endif

double gpu_timer(cudaEvent_t start, cudaEvent_t stop)
{
    float ms;
    cudaEventElapsedTime(&ms, start, stop);
    return ms / 1000.0;
}

int main(int argc, char **argv)
{
    if (argc < 2)
    {
        fprintf(stderr, "Usage: %s <matrix_size>\n", argv[0]);
        return 1;
    }

    int N = atoi(argv[1]);
    size_t size = N * N * sizeof(double);

    // Host memory
    double *h_A = (double *)malloc(size);
    double *h_B = (double *)malloc(size);
    double *h_C = (double *)malloc(size); // final result

    for (int i = 0; i < N * N; ++i)
    {
        h_A[i] = 1.0;
        h_B[i] = 2.0;
        h_C[i] = 0.0;
    }

    // Device memory
    double *d_A, *d_B, *d_C, *d_temp;
    cudaMalloc(&d_A, size);
    cudaMalloc(&d_B, size);
    cudaMalloc(&d_C, size);    // will hold B·Aᵀ + A²·B
    cudaMalloc(&d_temp, size); // for intermediate matrices

    checkCuda(cudaMemcpy(d_A, h_A, size, cudaMemcpyHostToDevice), "Memcpy A");
    checkCuda(cudaMemcpy(d_B, h_B, size, cudaMemcpyHostToDevice), "Memcpy B");

    // cuBLAS setup
    cublasHandle_t handle;
    cublasCreate(&handle);
    const double alpha = 1.0, beta = 0.0;

    // Timing
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    double total_time = 0.0;

    for (int rep = 0; rep < NUM_REPS; ++rep)
    {
        cudaMemset(d_C, 0, size);
        cudaMemset(d_temp, 0, size);

        cudaEventRecord(start);

        // 1. d_C = B × Aᵀ
        cublasDgemm(handle, CUBLAS_OP_N, CUBLAS_OP_T,
                    N, N, N, &alpha,
                    d_B, N, d_A, N, &beta,
                    d_C, N);

        // 2. d_temp = A × A
        cublasDgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N,
                    N, N, N, &alpha,
                    d_A, N, d_A, N, &beta,
                    d_temp, N);

        // 3. d_temp = A² × B
        cublasDgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N,
                    N, N, N, &alpha,
                    d_temp, N, d_B, N, &beta,
                    d_temp, N);

        // 4. Final sum: d_C += d_temp
        cublasDaxpy(handle, N * N, &alpha, d_temp, 1, d_C, 1);

        cudaEventRecord(stop);
        cudaEventSynchronize(stop);
        total_time += gpu_timer(start, stop);
    }

    double total = total_time / NUM_REPS;
    double gflops = (6.0 * N * N * N) / (total * 1e9);
    printf("%.6f %.2f\n", total, gflops);

    // Cleanup
    cublasDestroy(handle);
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
    cudaFree(d_temp);
    free(h_A);
    free(h_B);
    free(h_C);

    return 0;
}