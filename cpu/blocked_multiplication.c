#include <stdio.h>
#include <stdlib.h>
#include <sys/time.h>

#ifndef NUM_REPS
#define NUM_REPS 1
#endif

void transpose(double *A, double *A_T, int n)
{
    for (int i = 0; i < n; ++i)
        for (int j = 0; j < n; ++j)
            A_T[j * n + i] = A[i * n + j];
}

void matrix_add(double *A, double *B, double *C, int n)
{
    for (int i = 0; i < n * n; ++i)
        C[i] = A[i] + B[i];
}

void bmm_blocked(double *A, double *B, double *C, int n, int bs)
{
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

int main(int argc, char **argv)
{
    if (argc < 3)
    {
        fprintf(stderr, "Usage: %s <N> <BLOCK_SIZE>\n", argv[0]);
        return 1;
    }

    int N = atoi(argv[1]);
    int BLOCK_SIZE = atoi(argv[2]);

    if (N <= 0 || BLOCK_SIZE <= 0)
    {
        fprintf(stderr, "Invalid input. N and BLOCK_SIZE must be positive integers.\n");
        return 1;
    }

    int size = N * N;
    double *A = malloc(size * sizeof(double));
    double *B = malloc(size * sizeof(double));
    double *C1 = calloc(size, sizeof(double));
    double *C2 = calloc(size, sizeof(double));
    double *C = calloc(size, sizeof(double));

    if (!A || !B || !C1 || !C2 || !C)
    {
        fprintf(stderr, "Memory allocation failed.\n");
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
        double *A_T = malloc(size * sizeof(double));
        double *A2 = calloc(size, sizeof(double));

        double start = get_time();

        transpose(A, A_T, N);                   // A_T = Aᵀ
        bmm_blocked(B, A_T, C1, N, BLOCK_SIZE); // C1 = B × Aᵀ
        bmm_blocked(A, A, A2, N, BLOCK_SIZE);   // A2 = A × A
        bmm_blocked(A2, B, C2, N, BLOCK_SIZE);  // C2 = A² × B
        matrix_add(C1, C2, C, N);               // C = C1 + C2

        double end = get_time();
        double elapsed = end - start;
        total_time += elapsed;

        // printf("Run %d: %.6f seconds\n", rep + 1, elapsed);

        free(A_T);
        free(A2);
        for (int i = 0; i < size; ++i)
            C1[i] = C2[i] = 0.0;
    }

    // printf("Blocked C = B*Aᵀ + A²*B: N=%d, BLOCK_SIZE=%d → Avg time over %d runs: %.6f seconds\n",
    //        N, BLOCK_SIZE, NUM_REPS, total_time / NUM_REPS);

    double total = total_time / NUM_REPS;
    double gflops = (6.0 * N * N * N) / (total * 1e9);
    printf("%.2f %.2f\n", total, gflops);

    free(A);
    free(B);
    free(C1);
    free(C2);
    free(C);
    return 0;
}
