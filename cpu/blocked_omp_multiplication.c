#include <stdio.h>
#include <stdlib.h>
#include <sys/time.h>
#include <omp.h>

#ifndef N
#define N 1024
#endif

#ifndef BLOCK_SIZE
#define BLOCK_SIZE 64
#endif

#ifndef NUM_REPS
#define NUM_REPS 3
#endif

#ifndef THREADS
#define THREADS 4
#endif

void blocked_multiply_omp(double *A, double *B, double *C, int n, int block_size)
{
#pragma omp parallel for collapse(2)
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

    int threads = THREADS;
    char *env_threads = getenv("OMP_NUM_THREADS");
    if (env_threads != NULL)
        threads = atoi(env_threads);

    omp_set_num_threads(threads);

    printf("Running with N=%d, BLOCK_SIZE=%d, THREADS=%d (env or compile-time)\n", N, BLOCK_SIZE, threads);

    double total_time = 0.0;

    for (int rep = 0; rep < NUM_REPS; ++rep)
    {
        double start = get_time();
        blocked_multiply_omp(A, B, C, N, BLOCK_SIZE);
        double end = get_time();
        double elapsed = end - start;
        total_time += elapsed;
        printf("Run %d: %.6f seconds\n", rep + 1, elapsed);
    }

    printf("Average time over %d runs: %.6f seconds\n", NUM_REPS, total_time / NUM_REPS);

    free(A);
    free(B);
    free(C);
    return 0;
}
