#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>

#ifndef NUM_REPS
#define NUM_REPS 10
#endif

__global__ void matrixMulShared(const double *A, const double *B, double *C, int n, int tileSize)
{
    extern __shared__ double shared[];
    double *tileA = shared;
    double *tileB = &shared[tileSize * tileSize];

    int row = blockIdx.y * tileSize + threadIdx.y;
    int col = blockIdx.x * tileSize + threadIdx.x;

    double sum = 0.0;

    for (int tile = 0; tile < n / tileSize; ++tile)
    {
        tileA[threadIdx.y * tileSize + threadIdx.x] = A[row * n + tile * tileSize + threadIdx.x];
        tileB[threadIdx.y * tileSize + threadIdx.x] = B[(tile * tileSize + threadIdx.y) * n + col];
        __syncthreads();

        for (int k = 0; k < tileSize; ++k)
            sum += tileA[threadIdx.y * tileSize + k] * tileB[k * tileSize + threadIdx.x];

        __syncthreads();
    }

    C[row * n + col] = sum;
}

double gpu_timer(cudaEvent_t start, cudaEvent_t stop)
{
    float ms;
    cudaEventElapsedTime(&ms, start, stop);
    return ms; // return milliseconds directly
}

int main(int argc, char **argv)
{
    if (argc < 3)
    {
        fprintf(stderr, "Usage: %s <matrix_size> <tile_size>\n", argv[0]);
        return 1;
    }

    int N = atoi(argv[1]);
    int tileSize = atoi(argv[2]);

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

    dim3 threads(tileSize, tileSize);
    dim3 blocks((N + tileSize - 1) / tileSize, (N + tileSize - 1) / tileSize);
    size_t sharedMemSize = 2 * tileSize * tileSize * sizeof(double);

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    double total = 0.0;
    for (int rep = 0; rep < NUM_REPS; ++rep)
    {
        cudaMemset(d_C, 0, size);
        cudaEventRecord(start);
        matrixMulShared<<<blocks, threads, sharedMemSize>>>(d_A, d_B, d_C, N, tileSize);
        cudaEventRecord(stop);
        cudaEventSynchronize(stop);
        total += gpu_timer(start, stop);
    }

    double avg_time = total / NUM_REPS;
    printf("Shared GPU: N=%d TILE_SIZE=%d → Avg time = %.6f ms\n", N, tileSize, avg_time);

    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
    free(h_A);
    free(h_B);
    free(h_C);
    return 0;
}
