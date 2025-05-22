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

    for (int tile = 0; tile < (n + tileSize - 1) / tileSize; ++tile)
    {
        int tiledRowA = row;
        int tiledColA = tile * tileSize + threadIdx.x;

        if (tiledRowA < n && tiledColA < n)
            tileA[threadIdx.y * tileSize + threadIdx.x] = A[tiledRowA * n + tiledColA];
        else
            tileA[threadIdx.y * tileSize + threadIdx.x] = 0.0;

        int tiledRowB = tile * tileSize + threadIdx.y;
        int tiledColB = col;

        if (tiledRowB < n && tiledColB < n)
            tileB[threadIdx.y * tileSize + threadIdx.x] = B[tiledRowB * n + tiledColB];
        else
            tileB[threadIdx.y * tileSize + threadIdx.x] = 0.0;

        __syncthreads();

        for (int k = 0; k < tileSize; ++k)
            sum += tileA[threadIdx.y * tileSize + k] * tileB[k * tileSize + threadIdx.x];

        __syncthreads();
    }

    if (row < n && col < n)
        C[row * n + col] = sum;
}

double gpu_timer(cudaEvent_t start, cudaEvent_t stop)
{
    float ms;
    cudaEventElapsedTime(&ms, start, stop);
    return ms;
}

int main(int argc, char **argv)
{
    if (argc < 2)
    {
        fprintf(stderr, "Usage: %s <matrix_size>\n", argv[0]);
        return 1;
    }

    int N = atoi(argv[1]);
    int tileSize = 16; // Safe and portable

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
    printf("some of the results: C[0] = %f, C[%d] = %f\n", d_C[0], N * N - 1, d_C[N * N - 1]);
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, 0);

    int maxThreadsPerSM = prop.maxThreadsPerMultiProcessor;
    int numSMs = prop.multiProcessorCount;

    int threadsPerBlock = threads.x * threads.y;
    int numBlocks = blocks.x * blocks.y;
    int totalThreads = threadsPerBlock * numBlocks;
    int theoreticalMaxThreads = maxThreadsPerSM * numSMs;

    float occupancy = 100.0f * totalThreads / theoreticalMaxThreads;

    printf("Occupancy ≈ %.2f%%\n", occupancy);
    printf("Shared memory per block: %lu bytes\n", sharedMemSize);

    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
    free(h_A);
    free(h_B);
    free(h_C);
    return 0;
}
