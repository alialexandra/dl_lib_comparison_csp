#include <stdio.h>
#include <stdlib.h>
#include <sys/time.h>

#ifndef N
#define N 1024
#endif

#ifndef BLOCK_SIZE
#define BLOCK_SIZE 64
#endif

#ifndef NUM_REPS
#define NUM_REPS 3
#endif

void blocked_multiply(double *A, double *B, double *C, int n, int block_size)
{
    for (int bi = 0; bi < n; bi += block_size)
        for (int bj = 0; bj < n; bj += block_size)
            for (int bk = 0; bk < n; bk += block_size)
                for (int i = bi; i < bi + block_size && i < n; ++i)
                    for (int j = bj; j < bj + block_size && j < n; ++j)
                    {
                        double sum = 0.0;
                        for (int k = bk; k < bk + block_size && k < n; ++k)
                            sum += A[i * n + k] * B[k * n + j];
                        C[i * n + j] += sum;
                    }
}

double get_time()
{
    struct timeval tv;
    gettimeofday(&tv, NULL);
    return tv.tv_sec + tv.tv_usec * 1e-6;
}

int main()
{
    int size = N * N;
    double *A = malloc(size * sizeof(double));
    double *B = malloc(size * sizeof(double));
    double *C = calloc(size, sizeof(double));

    for (int i = 0; i < size; ++i)
    {
        A[i] = 1.0;
        B[i] = 2.0;
    }

    double total_time = 0.0;

    for (int rep = 0; rep < NUM_REPS; ++rep)
    {
        double start = get_time();
        blocked_multiply(A, B, C, N, BLOCK_SIZE);
        double end = get_time();
        double elapsed = end - start;
        total_time += elapsed;
        printf("Run %d: %.6f seconds\n", rep + 1, elapsed);
    }

    printf("Average time over %d runs: N=%d BLOCK_SIZE=%d → %.6f seconds\n",
           NUM_REPS, N, BLOCK_SIZE, total_time / NUM_REPS);

    free(A);
    free(B);
    free(C);
    return 0;
}
