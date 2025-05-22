#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>

#ifndef NUM_REPS
#define NUM_REPS 5
#endif

// Shared memory matrix multiplication kernel: C = A * B
__global__ void sharedMatrixMulKernel(const double *A, const double *B, double *C, int n, int tile_size) {
    extern __shared__ double shared[];
    double* tileA = shared;
    double* tileB = &shared[tile_size * tile_size];

    int row = blockIdx.y * tile_size + threadIdx.y;
    int col = blockIdx.x * tile_size + threadIdx.x;

    double sum = 0.0;
    for (int t = 0; t < n / tile_size; ++t) {
        int tiled_row = row * n + t * tile_size + threadIdx.x;
        int tiled_col = (t * tile_size + threadIdx.y) * n + col;

        tileA[threadIdx.y * tile_size + threadIdx.x] = A[tiled_row];
        tileB[threadIdx.y * tile_size + threadIdx.x] = B[tiled_col];
        __syncthreads();

        for (int k = 0; k < tile_size; ++k)
            sum += tileA[threadIdx.y * tile_size + k] * tileB[k * tile_size + threadIdx.x];
        __syncthreads();
    }

    if (row < n && col < n)
        C[row * n + col] = sum;
}

__global__ void addKernel(const double *X, const double *Y, double *Z, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n)
        Z[idx] = X[idx] + Y[idx];
}

double gpu_timer(cudaEvent_t start, cudaEvent_t stop) {
    float ms;
    cudaEventElapsedTime(&ms, start, stop);
    return ms / 1000.0; // seconds
}

int main(int argc, char **argv) {
    if (argc < 2) {
        fprintf(stderr, "Usage: %s <matrix_size> [tile_size]\n", argv[0]);
        return 1;
    }

    int N = atoi(argv[1]);
    int TILE_SIZE = (argc >= 3) ? atoi(argv[2]) : 16;

    size_t size = N * N * sizeof(double);
    double *h_A = (double *)malloc(size);
    double *h_B = (double *)malloc(size);
    double *h_C = (double *)malloc(size);

    for (int i = 0; i < N * N; ++i) {
        h_A[i] = 1.0;
        h_B[i] = 2.0;
        h_C[i] = 0.0;
    }

    double *d_A, *d_B, *d_C, *d_temp;
    cudaMalloc(&d_A, size);
    cudaMalloc(&d_B, size);
    cudaMalloc(&d_C, size);
    cudaMalloc(&d_temp, size);

    cudaMemcpy(d_A, h_A, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, size, cudaMemcpyHostToDevice);

    dim3 threadsPerBlock(TILE_SIZE, TILE_SIZE);
    dim3 blocksPerGrid((N + TILE_SIZE - 1) / TILE_SIZE, (N + TILE_SIZE - 1) / TILE_SIZE);
    size_t sharedMemSize = 2 * TILE_SIZE * TILE_SIZE * sizeof(double);

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    double total_time = 0.0;

    for (int rep = 0; rep < NUM_REPS; ++rep) {
        cudaMemset(d_C, 0, size);
        cudaMemset(d_temp, 0, size);

        cudaEventRecord(start);

        sharedMatrixMulKernel<<<blocksPerGrid, threadsPerBlock, sharedMemSize>>>(d_B, d_A, d_C, N, TILE_SIZE);      // d_C = B * A^T
        sharedMatrixMulKernel<<<blocksPerGrid, threadsPerBlock, sharedMemSize>>>(d_A, d_A, d_temp, N, TILE_SIZE);   // d_temp = A^2
        sharedMatrixMulKernel<<<blocksPerGrid, threadsPerBlock, sharedMemSize>>>(d_temp, d_B, d_temp, N, TILE_SIZE); // d_temp = A^2 * B

        int threads = TILE_SIZE * TILE_SIZE;
        int blocks = (N * N + threads - 1) / threads;
        addKernel<<<blocks, threads>>>(d_C, d_temp, d_C, N * N);

        cudaEventRecord(stop);
        cudaEventSynchronize(stop);
        total_time += gpu_timer(start, stop);
    }

    double total = total_time / NUM_REPS;
    double gflops = (6.0 * N * N * N) / (total * 1e9);
    printf("%.2f %.2f\n", total, gflops);

    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
    cudaFree(d_temp);
    free(h_A);
    free(h_B);
    free(h_C);

    return 0;
}
