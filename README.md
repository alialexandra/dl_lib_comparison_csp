# dl_lib_comparison_csp

This is a project for comparing diffferent approaches for matrix multiplication on cpu and gpu and determine if GPU outperforms CPU at some point and how.

cpu/

1. naive_multiplication.c

Simply measuring the running times for different sizes for the matrices.
compilation:

gcc -O3 -o naive_cpu naive_cpu.c
./naive_cpu

compilation with different sizez for the matrice:

gcc -O3 -DN=2048 -o naive_cpu naive_cpu.c

N should be varied from lets say: 256, 512, 1024, 2048, 4096, 8192, 16 384, etc, will set a limit for when is becoming to slow to follow

each experiment will be run 3 times and the average will be taken in consideration for measurements.

2. blocked_multiplication.c
   gcc -O3 -DN=1024 -DBLOCK_SIZE=64 -DNUM_REPS=3 -o blocked_cpu blocked_multiplication.c

again dn will be the matrix size and also the block size should be varied

3. blocked_multiplication_omp.c
   in this implemnetation we have added blocked multiplication and also paralelisation

// no SIMD

gcc -O3 -march=native -ffast-math -fopenmp
-DN=2048 -DBLOCK_SIZE=64 -DTHREADS=8 -DNUM_REPS=5
-o blocked_omp_multiplication blocked_omp_multiplication.c
./blocked_omp_multiplication

we can run normaly with no flags or using also SIMD flags for comparing newly obtained times

// with SIMD

gcc -O3 -march=native -fopenmp
-fno-signed-zeros -ffinite-math-only
-fno-signaling-nans -fno-trapping-math
-fassociative-math -fexcess-precision=fast -mfpmath=sse -DN=2048 -DBLOCK_SIZE=64 -DTHREADS=8 -DNUM_REPS=5 -o blocked_omp_multiplication_flags blocked_omp_multiplication.c
./blocked_omp_multiplication_flags

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
