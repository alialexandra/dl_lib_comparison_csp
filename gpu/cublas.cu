#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>
#include <cublas_v2.h>

#ifndef NUM_REPS
<<<<<<< Updated upstream
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
=======
#define NUM_REPS 5
#endif

double gpu_timer(cudaEvent_t start, cudaEvent_t stop) {
    float ms;
    cudaEventElapsedTime(&ms, start, stop);
    return ms / 1000.0;
}

int main(int argc, char **argv) {
    if (argc < 2) {
>>>>>>> Stashed changes
        fprintf(stderr, "Usage: %s <matrix_size>\n", argv[0]);
        return 1;
    }

<<<<<<< Updated upstream
    cudaFree(0);

=======
>>>>>>> Stashed changes
    int N = atoi(argv[1]);
    size_t size = N * N * sizeof(double);

    // Host memory
    double *h_A = (double *)malloc(size);
    double *h_B = (double *)malloc(size);
<<<<<<< Updated upstream
    double *h_C = (double *)malloc(size);
    if (!h_A || !h_B || !h_C)
    {
        fprintf(stderr, "Host malloc failed\n");
        return 1;
    }
=======
    double *h_C = (double *)malloc(size);  // final result
>>>>>>> Stashed changes

    for (int i = 0; i < N * N; ++i) {
        h_A[i] = 1.0;
        h_B[i] = 2.0;
        h_C[i] = 0.0;
    }

    // Device memory
<<<<<<< Updated upstream
    double *d_A, *d_B, *d_C;
    checkCuda(cudaMalloc(&d_A, size), "cudaMalloc d_A");
    checkCuda(cudaMalloc(&d_B, size), "cudaMalloc d_B");
    checkCuda(cudaMalloc(&d_C, size), "cudaMalloc d_C");
=======
    double *d_A, *d_B, *d_C, *d_temp;
    cudaMalloc(&d_A, size);
    cudaMalloc(&d_B, size);
    cudaMalloc(&d_C, size);     // will hold B·Aᵀ + A²·B
    cudaMalloc(&d_temp, size);  // for intermediate matrices
>>>>>>> Stashed changes

    checkCuda(cudaMemcpy(d_A, h_A, size, cudaMemcpyHostToDevice), "Memcpy A");
    checkCuda(cudaMemcpy(d_B, h_B, size, cudaMemcpyHostToDevice), "Memcpy B");

    // cuBLAS setup
    cublasHandle_t handle;
<<<<<<< Updated upstream
    checkCublas(cublasCreate(&handle), "create handle");

    double alpha = 1.0, beta = 0.0;
=======
    cublasCreate(&handle);
    const double alpha = 1.0, beta = 0.0;
>>>>>>> Stashed changes

    // Timing
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

<<<<<<< Updated upstream
    float total_ms = 0.0;
    for (int rep = 0; rep < NUM_REPS; ++rep)
    {
=======
    double total_time = 0.0;

    for (int rep = 0; rep < NUM_REPS; ++rep) {
>>>>>>> Stashed changes
        cudaMemset(d_C, 0, size);
        cudaMemset(d_temp, 0, size);

        cudaEventRecord(start);

<<<<<<< Updated upstream
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

    // Print example values
    printf("some of the results: C[0] = %f, C[%d] = %f\n", d_C[0], N * N - 1, d_C[N * N - 1]);

    // Clean up
=======
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
    printf("%.2f %.2f\n", total, gflops);

    // Cleanup
>>>>>>> Stashed changes
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