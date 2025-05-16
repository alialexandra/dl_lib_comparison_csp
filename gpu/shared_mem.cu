#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>

#ifndef N
#define N 1024
#endif

#ifndef TILE_SIZE
#define TILE_SIZE 16
#endif

#ifndef NUM_REPS
#define NUM_REPS 3
#endif

__global__ void matrixMulShared(const double *A, const double *B, double *C, int n)
{
    __shared__ double tileA[TILE_SIZE][TILE_SIZE];
    __shared__ double tileB[TILE_SIZE][TILE_SIZE];

    int row = blockIdx.y * TILE_SIZE + threadIdx.y;
    int col = blockIdx.x * TILE_SIZE + threadIdx.x;

    double sum = 0.0;

    for (int tile = 0; tile < n / TILE_SIZE; ++tile)
    {
        tileA[threadIdx.y][threadIdx.x] = A[row * n + tile * TILE_SIZE + threadIdx.x];
        tileB[threadIdx.y][threadIdx.x] = B[(tile * TILE_SIZE + threadIdx.y) * n + col];
        __syncthreads();

        for (int k = 0; k < TILE_SIZE; ++k)
            sum += tileA[threadIdx.y][k] * tileB[k][threadIdx.x];

        __syncthreads();
    }

    C[row * n + col] = sum;
}

double gpu_timer(cudaEvent_t start, cudaEvent_t stop)
{
    float ms;
    cudaEventElapsedTime(&ms, start, stop);
    return ms / 1000.0;
}

int main()
{
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

    dim3 threads(TILE_SIZE, TILE_SIZE);
    dim3 blocks(N / TILE_SIZE, N / TILE_SIZE);

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    double total = 0.0;
    for (int rep = 0; rep < NUM_REPS; ++rep)
    {
        cudaMemset(d_C, 0, size);
        cudaEventRecord(start);
        matrixMulShared<<<blocks, threads>>>(d_A, d_B, d_C, N);
        cudaEventRecord(stop);
        cudaEventSynchronize(stop);
        total += gpu_timer(start, stop);
    }

    printf("GPU Shared Memory: N=%d TILE_SIZE=%d → Avg time = %.6f seconds\n",
           N, TILE_SIZE, total / NUM_REPS);

    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
    free(h_A);
    free(h_B);
    free(h_C);
    return 0;
}
