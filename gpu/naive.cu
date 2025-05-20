#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>
#include <cublas_v2.h>

#ifndef NUM_REPS
#define NUM_REPS 3
#endif

__global__ void matrixMulKernel(const double *A, const double *B, double *C, int n)
{
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    if (row < n && col < n)
    {
        double sum = 0.0;
        for (int k = 0; k < n; ++k)
            sum += A[row * n + k] * B[k * n + col];
        C[row * n + col] = sum;
    }
}

double gpu_timer(cudaEvent_t start, cudaEvent_t stop)
{
    float ms;
    cudaEventElapsedTime(&ms, start, stop);
    return ms / 1000.0; // seconds
}

int main(int argc, char **argv)
{
    if (argc < 2)
    {
        fprintf(stderr, "Usage: %s <matrix_size>\n", argv[0]);
        return 1;
    }

    int N = atoi(argv[1]);
    if (N <= 0)
    {
        fprintf(stderr, "Invalid matrix size: %d\n", N);
        return 1;
    }

    printf("Running matrix multiplication with N = %d\n", N);

    int size = N * N * sizeof(double);
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

    dim3 threadsPerBlock(16, 16);
    dim3 blocksPerGrid((N + 15) / 16, (N + 15) / 16);

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    double total = 0.0;
    for (int rep = 0; rep < NUM_REPS; ++rep)
    {
        cudaMemset(d_C, 0, size);
        cudaEventRecord(start);
        matrixMulKernel<<<blocksPerGrid, threadsPerBlock>>>(d_A, d_B, d_C, N);
        cudaEventRecord(stop);
        cudaEventSynchronize(stop);
        total += gpu_timer(start, stop);
    }

    // Log memory info
    size_t free_mem, total_mem;
    cudaMemGetInfo(&free_mem, &total_mem);

    // Print summary
    double avg_time = total / NUM_REPS;
    printf("Naive GPU: N=%d → Avg time = %.6f seconds\n", N, avg_time);

    // Save results to CSV
    FILE *log = fopen("naive_gpu_results.csv", "a");
    if (log)
    {
        fprintf(log, "%d,%d,%d,%.6f,%zu,%zu\n",
                N, threadsPerBlock.x, blocksPerGrid.x,
                avg_time, total_mem, free_mem);
        fclose(log);
    }

    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
    free(h_A);
    free(h_B);
    free(h_C);
    return 0;
}
