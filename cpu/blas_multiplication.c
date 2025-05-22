#include <stdio.h>
#include <stdlib.h>
#include <sys/time.h>
#include <cblas.h>
#include <openblas_config.h> // Needed for openblas_get_num_threads()

#ifndef NUM_REPS
#define NUM_REPS 3
#endif

double get_time()
{
    struct timeval tv;
    gettimeofday(&tv, NULL);
    return tv.tv_sec + tv.tv_usec * 1e-6;
}

int main(int argc, char **argv)
{
    if (argc < 2)
    {
        fprintf(stderr, "Usage: %s <N>\n", argv[0]);
        return 1;
    }

    int N = atoi(argv[1]);
    if (N <= 0)
    {
        fprintf(stderr, "Invalid matrix size: %s\n", argv[1]);
        return 1;
    }

    int size = N * N;
    double *A = malloc(size * sizeof(double));
    double *B = malloc(size * sizeof(double));
    double *A2 = malloc(size * sizeof(double));
    double *C1 = malloc(size * sizeof(double));
    double *C2 = malloc(size * sizeof(double));
    double *C = malloc(size * sizeof(double));

    if (!A || !B || !A2 || !C1 || !C2 || !C)
    {
        fprintf(stderr, "Memory allocation failed!\n");
        return 1;
    }

    for (int i = 0; i < size; ++i)
    {
        A[i] = 1.0;
        B[i] = 2.0;
    }

    double total_time = 0.0;

    for (int rep = 0; rep < NUM_REPS; ++rep)
    {
        double start = get_time();

        cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasTrans,
                    N, N, N, 1.0, B, N, A, N, 0.0, C1, N);

        cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
                    N, N, N, 1.0, A, N, A, N, 0.0, A2, N);

        cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
                    N, N, N, 1.0, A2, N, B, N, 0.0, C2, N);

        for (int i = 0; i < size; ++i)
            C[i] = C1[i] + C2[i];

        double end = get_time();
        double elapsed = end - start;
        total_time += elapsed;
    }

    double avg_time = total_time / NUM_REPS;
    int used_threads = openblas_get_num_threads();

    printf("BLAS: N=%d | Avg time = %.6f seconds | Threads used: %d\n", N, avg_time, used_threads);
    printf("some of the results: C[0] = %f, C[%d] = %f\n", C[0], N * N - 1, C[N * N - 1]);

    free(A);
    free(B);
    free(A2);
    free(C1);
    free(C2);
    free(C);
    return 0;
}
