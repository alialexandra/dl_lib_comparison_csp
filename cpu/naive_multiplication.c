#include <stdio.h>
#include <stdlib.h>
#include <sys/time.h>

#ifndef N
#define N 1024
#endif

#ifndef NUM_REPS
#define NUM_REPS 3
#endif

void matrix_multiply(double *A, double *B, double *C, int n)
{
    for (int i = 0; i < n; ++i)
        for (int j = 0; j < n; ++j)
        {
            double sum = 0.0;
            for (int k = 0; k < n; ++k)
                sum += A[i * n + k] * B[k * n + j];
            C[i * n + j] = sum;
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
    double *C = malloc(size * sizeof(double));

    for (int i = 0; i < size; ++i)
    {
        A[i] = 1.0;
        B[i] = 2.0;
    }

    double total_time = 0.0;

    for (int rep = 0; rep < NUM_REPS; ++rep)
    {
        double start = get_time();
        matrix_multiply(A, B, C, N);
        double end = get_time();
        double elapsed = end - start;
        total_time += elapsed;
        printf("Run %d: %.6f seconds\n", rep + 1, elapsed);
    }

    printf("Average time over %d runs: N=%d → %.6f seconds\n", NUM_REPS, N, total_time / NUM_REPS);

    free(A);
    free(B);
    free(C);
    return 0;
}
