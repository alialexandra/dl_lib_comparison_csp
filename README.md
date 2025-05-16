# dl_lib_comparison_csp

This is a project for comparing diffferent approaches for matrix multiplication on cpu and gpu and determine if GPU outperforms CPU at some point and how.

cpu/

1. naive_multiplication.c

Simply measuring the running times for different sizes for the matrices.
compilation:

gcc -O3 -o naive_multiplication naive_multiplication.c
./naive_multiplication 1200

2. blocked_multiplication.c

   gcc -O3 -o blocked_multiplication blocked_multiplication.c

   ./blocked_multiplication 1024 32

3. blocked_multiplication_omp.c
   in this implemnetation we have added blocked multiplication and also paralelisation

// no SIMD

gcc -O3 -fopenmp -o blocked_omp_multiplication blocked_omp_multiplication.c

./blocked_omp_multiplication 1024 64 4

we can run normaly with no flags or using also SIMD flags for comparing newly obtained times

// with SIMD

gcc -O3 -march=native -fopenmp -ffast-math -fno-signed-zeros -ffinite-math-only -fno-signaling-nans -fno-trapping-math -fassociative-math -fexcess-precision=fast -mfpmath=sse -o blocked_omp_multiplication_optimised blocked_omp_multiplication.c

./blocked_omp_multiplication_optimised

-march=native enables architecture-specific SIMD (SSE, AVX)

-ffast-math, -funroll-loops boost floating point loop optimization

taskset pins threads to cores, mirroring your first project - dont know if i really want to use this???

what are the other flags do?

In addition to standard compiler optimizations (-O3), we enabled a set of math-specific flags aimed at improving floating-point performance. These include:

-fno-signed-zeros, -ffinite-math-only, -fno-signaling-nans, -fno-trapping-math: These remove unnecessary runtime checks related to IEEE floating-point edge cases (e.g., ±0, NaNs, or exceptions), which are not relevant for our matrix data.

-fassociative-math: Allows the compiler to reorder arithmetic expressions, improving instruction pipelining and vectorization opportunities.

-fexcess-precision=fast: Enables the use of wider internal precision when it results in faster execution.

-mfpmath=sse: Forces the use of SSE instructions for floating-point math on x86 CPUs.

These flags are part of a finer-grained breakdown of the more aggressive -ffast-math flag. By explicitly selecting only those relevant to our workload, we aim to gain performance without compromising correctness

4. using a dedicated library, CBLAS

Libraries like OpenBLAS, MKL, or cuBLAS are already multi-threaded and optimized for your CPU/GPU architecture.

They automatically use:

SIMD/vector instructions (e.g., AVX)

Thread pools (e.g., via OpenMP or pthreads)

Cache-aware algorithms

Metric - Insight

Time vs blocked - CPU How good your hand-tuned CPU really is
Time vs GPU cuBLAS - CPU vs GPU library comparison

Efficiency ceiling - What is achievable without full rewrite

sudo apt-get install libopenblas-dev

// NO SIMD
gcc -O3 -DN=2048 -o blas_multiplication blas_multiplication.c -lopenblas
./blas_multiplication

// with simd

Compiler flags still affect:

Loop wrappers

Preprocessing logic

Memory-related behavior

Timing and logging

Any math done outside BLAS (like adding matrices manually)

gcc -O3 -march=native -ffast-math -fassociative-math \
 -fno-signed-zeros -ffinite-math-only \
 -fno-signaling-nans -fno-trapping-math \
 -fexcess-precision=fast -mfpmath=sse \
 -DN=1024 -DNUM_REPS=3 -o blas_multiplication_flags blas_multiplication.c -lopenblas
./blas_multiplication_flags

======================================================================================================================================

gpu/

# Naive version

nvcc -O3 -DN=1024 -DNUM_REPS=3 -o naive naive.cu

# Shared memory

nvcc -O3 -DN=1024 -DNUM_REPS=3 -DTILE_SIZE=16 -o shared_mem shared_mem.cu

# cuBLAS

nvcc -O3 -DN=1024 -DNUM_REPS=3 -lcublas -o cublas cublas.cu
