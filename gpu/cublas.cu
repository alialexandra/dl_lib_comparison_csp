#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>
#include <cublas_v2.h>

int main(int argc, char **argv)
{
    if (argc < 2)
    {
        fprintf(stderr, "Usage: %s <matrix_size>\n", argv[0]);
        return 1;
    }

    int N = atoi(argv[1]);
    int NUM_REPS = 3;

    size_t size = N * N * sizeof(double);
    double *h_A = (double *)malloc(size);
    double *h_B = (double *)malloc(size);
    double *h_C = (double *)malloc(size);

    for (int i = 0; i < N * N; ++i)
    {
        h_A[i] = 1.0;
        h_B[i] = 2.0;
    }

    double *d_A, *d_B, *d_C;
    cudaMalloc(&d_A, size);
    cudaMalloc(&d_B, size);
    cudaMalloc(&d_C, size);

    cudaMemcpy(d_A, h_A, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, size, cudaMemcpyHostToDevice);

    cublasHandle_t handle;
    cublasCreate(&handle);

    double alpha = 1.0, beta = 0.0;

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    double total = 0.0;
    for (int rep = 0; rep < NUM_REPS; ++rep)
    {
        cudaMemset(d_C, 0, size);
        cudaEventRecord(start);
        cublasDgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N,
                    N, N, N, &alpha, d_A, N, d_B, N, &beta, d_C, N);
        cudaEventRecord(stop);
        cudaEventSynchronize(stop);

        float ms;
        cudaEventElapsedTime(&ms, start, stop);
        total += ms / 1000.0;
    }

    size_t free_mem, total_mem;
    cudaMemGetInfo(&free_mem, &total_mem);

    double avg_time = total / NUM_REPS;
    printf("cuBLAS DGEMM: N=%d → Avg time = %.6f seconds\n", N, avg_time);

    FILE *log = fopen("cublas_results.csv", "a");
    if (log)
    {
        fprintf(log, "%d,%.6f,%zu,%zu\n", N, avg_time, total_mem, free_mem);
        fclose(log);
    }

    cublasDestroy(handle);
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
    free(h_A);
    free(h_B);
    free(h_C);
    return 0;
}
