#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>

#ifndef NUM_REPS
<<<<<<< Updated upstream
#define NUM_REPS 10
=======
#define NUM_REPS 5
>>>>>>> Stashed changes
#endif

// Kernel: C = A * B
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

// Move this outside main
__global__ void addKernel(const double *X, const double *Y, double *Z, int n)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n)
        Z[idx] = X[idx] + Y[idx];
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
        fprintf(stderr, "Usage: %s <matrix_size> [block_size]\n", argv[0]);
        return 1;
    }

    int N = atoi(argv[1]);
    int BLOCK_SIZE = (argc >= 3) ? atoi(argv[2]) : 16;

    size_t size = N * N * sizeof(double);
    double *h_A = (double *)malloc(size);
    double *h_B = (double *)malloc(size);
    double *h_C = (double *)malloc(size);

    for (int i = 0; i < N * N; ++i)
    {
        h_A[i] = 1.0;
        h_B[i] = 2.0;
        h_C[i] = 0.0;
    }

    double *d_A, *d_B, *d_C, *d_temp;
    cudaMalloc(&d_A, size);
    cudaMalloc(&d_B, size);
    cudaMalloc(&d_C, size);
    cudaMalloc(&d_temp, size); // for intermediate result

    cudaMemcpy(d_A, h_A, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, size, cudaMemcpyHostToDevice);

    dim3 threadsPerBlock(BLOCK_SIZE, BLOCK_SIZE);
    dim3 blocksPerGrid((N + BLOCK_SIZE - 1) / BLOCK_SIZE, (N + BLOCK_SIZE - 1) / BLOCK_SIZE);

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    double total_time = 0.0;

    for (int rep = 0; rep < NUM_REPS; ++rep)
    {
        cudaMemset(d_C, 0, size);
        cudaMemset(d_temp, 0, size);

        cudaEventRecord(start);

        // Compute Aᵀ (by swapping row/col access)
        matrixMulKernel<<<blocksPerGrid, threadsPerBlock>>>(d_B, d_A, d_C, N);      // d_C = B * Aᵀ
        matrixMulKernel<<<blocksPerGrid, threadsPerBlock>>>(d_A, d_A, d_temp, N);   // d_temp = A²
        matrixMulKernel<<<blocksPerGrid, threadsPerBlock>>>(d_temp, d_B, d_temp, N); // d_temp = A² * B

        // Sum both results into d_C = B*Aᵀ + A²*B
        // Element-wise addition
        int threads = BLOCK_SIZE * BLOCK_SIZE;
        int blocks = (N * N + threads - 1) / threads;
        addKernel<<<blocks, threads>>>(d_C, d_temp, d_C, N * N);

        cudaEventRecord(stop);
        cudaEventSynchronize(stop);
        total_time += gpu_timer(start, stop);
    }


<<<<<<< Updated upstream
    // Print summary
    double avg_time = total / NUM_REPS;

    printf("Naive GPU: N=%d → Avg time = %.6f seconds\n", N, avg_time);
    printf("some of the results: C[0] = %f, C[%d] = %f\n", d_C[0], N * N - 1, d_C[N * N - 1]);
=======
    double total = total_time / NUM_REPS;
    double gflops = (6.0 * N * N * N) / (total * 1e9);
    printf("%.2f %.2f\n", total, gflops);
>>>>>>> Stashed changes

    // Cleanup
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
    cudaFree(d_temp);
    free(h_A);
    free(h_B);
    free(h_C);
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, 0);

    int maxThreadsPerSM = prop.maxThreadsPerMultiProcessor;
    int numSMs = prop.multiProcessorCount;

    int activeThreads = blocksPerGrid.x * blocksPerGrid.y * threadsPerBlock.x * threadsPerBlock.y;
    int theoreticalMaxThreads = maxThreadsPerSM * numSMs;
    float occupancy = 100.0f * activeThreads / theoreticalMaxThreads;

    printf("GPU Config: ThreadsPerBlock=%d, Blocks=%d × %d, SMs=%d, MaxThreads/SM=%d\n",
           threadsPerBlock.x * threadsPerBlock.y, blocksPerGrid.x, blocksPerGrid.y, numSMs, maxThreadsPerSM);
    printf("Active Threads: %d, Theoretical Max: %d → Occupancy ≈ %.2f%%\n",
           activeThreads, theoreticalMaxThreads, occupancy);

    return 0;
}
