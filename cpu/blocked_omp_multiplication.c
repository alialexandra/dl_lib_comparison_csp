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

void transpose(double *A, double *A_T, int n)
{
#pragma omp parallel for collapse(2)
    for (int i = 0; i < n; ++i)
        for (int j = 0; j < n; ++j)
            A_T[j * n + i] = A[i * n + j];
}

void matrix_add(double *A, double *B, double *C, int n)
{
#pragma omp parallel for
    for (int i = 0; i < n * n; ++i)
        C[i] = A[i] + B[i];
}

void bmm_blocked_omp(double *A, double *B, double *C, int n, int bs)
{
#pragma omp parallel for collapse(2)
    for (int bi = 0; bi < n; bi += bs)
        for (int bj = 0; bj < n; bj += bs)
            for (int bk = 0; bk < n; bk += bs)
                for (int i = bi; i < bi + bs && i < n; ++i)
                    for (int j = bj; j < bj + bs && j < n; ++j)
                    {
                        double sum = 0.0;
                        for (int k = bk; k < bk + bs && k < n; ++k)
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
    double *C1 = calloc(size, sizeof(double));
    double *C2 = calloc(size, sizeof(double));
    double *C = calloc(size, sizeof(double));

    for (int i = 0; i < size; ++i)
    {
        A[i] = 1.0;
        B[i] = 2.0;
    }

    double total_time = 0.0;

    for (int rep = 0; rep < NUM_REPS; ++rep)
    {
        double *A_T = malloc(size * sizeof(double));
        double *A2 = calloc(size, sizeof(double));

        double start = get_time();

        transpose(A, A_T, N);
        bmm_blocked_omp(B, A_T, C1, N, BLOCK_SIZE);
        bmm_blocked_omp(A, A, A2, N, BLOCK_SIZE);
        bmm_blocked_omp(A2, B, C2, N, BLOCK_SIZE);
        matrix_add(C1, C2, C, N);
        printf("C[0][0] = %f\n", C[0]);

        double end = get_time();
        double elapsed = end - start;
        total_time += elapsed;
        printf("Run %d: %.6f seconds\n", rep + 1, elapsed);

        free(A_T);
        free(A2);
        for (int i = 0; i < size; ++i)
            C1[i] = C2[i] = 0.0;
    }

    printf("Blocked + OMP C = B*Aᵀ + A²*B: N=%d → Avg = %.6f s\n", N, total_time / NUM_REPS);

    free(A);
    free(B);
    free(C1);
    free(C2);
    free(C);
    return 0;
}
