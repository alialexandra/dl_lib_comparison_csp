#include <stdio.h>
#include <stdlib.h>
#include <sys/time.h>

#ifndef NUM_REPS
#define NUM_REPS 1
#endif

// Naive matrix multiplication
void matmul(const double *A, const double *B, double *C, int n)
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

void matmul_transpose(const double *A, const double *B, double *C, int n)
{
    // Computes C = B * Aᵀ
    for (int i = 0; i < n; ++i)
        for (int j = 0; j < n; ++j)
        {
            double sum = 0.0;
            for (int k = 0; k < n; ++k)
                sum += B[i * n + k] * A[j * n + k]; // Aᵀ[k][j] = A[j][k]
            C[i * n + j] = sum;
        }
}

void matrix_add(const double *A, const double *B, double *C, int n)
{
    for (int i = 0; i < n * n; ++i)
        C[i] = A[i] + B[i];
}

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
    double *C1 = calloc(size, sizeof(double)); // for B × Aᵀ
    double *C2 = calloc(size, sizeof(double)); // for A² × B
    double *C = calloc(size, sizeof(double));  // final result

    if (!A || !B || !C1 || !C2 || !C)
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

        // printf("Run %d: Computing C = B*Aᵀ + A²*B for N = %d\n", rep + 1, N);

        matmul_transpose(A, B, C1, N); // C1 = B × Aᵀ

        double *A_squared = malloc(size * sizeof(double));
        matmul(A, A, A_squared, N); // A² = A × A

        matmul(A_squared, B, C2, N); // C2 = A² × B
        matrix_add(C1, C2, C, N);    // C = C1 + C2

        double end = get_time();
        double elapsed = end - start;
        total_time += elapsed;
        // printf("Run %d completed in %.6f seconds\n", rep + 1, elapsed);

        free(A_squared);
    }

    printf("Average for N = %d over %d runs: %.6f seconds\n", N, NUM_REPS, total_time / NUM_REPS);
    printf("some of the results: C[0] = %f, C[%d] = %f\n", C[0], N * N - 1, C[N * N - 1]);

    free(A);
    free(B);
    free(C1);
    free(C2);
    free(C);
    return 0;
}
