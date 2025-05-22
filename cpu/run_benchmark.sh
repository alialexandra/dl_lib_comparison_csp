#!/bin/bash

# Output CSV header
echo "Version,N,BLOCK_SIZE,THREADS,Time_sec,GFLOPs" > results.csv

# Parameters
Ns=(512 1024 2048 4096 8192 16384)
BLOCK_SIZES=(16 32 64 128)
THREADS=(2 4 8 16 32 64)

# Compile all
gcc -O3 -o naive naive_multiplication.c
gcc -O3 -o blocked blocked_multiplication.c
gcc -O3 -fopenmp -o blocked_omp blocked_omp_multiplication.c
gcc -O3 -lopenblas -o blas blas_multiplication.c

for N in "${Ns[@]}"; do
    # -------- Naive version --------
    # echo "Running naive for N=$N"
    # read time flops <<< $(./naive "$N")
    # echo "naive,$N,,,$time,$flops" >> results.csv


    # -------- Blocked version --------
    for BS in "${BLOCK_SIZES[@]}"; do
        echo "Running blocked for N=$N, BS=$BS"
        read time flops <<< $(./blocked "$N" "$BS")
        echo "blocked,$N,$BS,,$time,$flops" >> results.csv
    done

    # -------- Blocked + OMP --------
    for BS in "${BLOCK_SIZES[@]}"; do
        for T in "${THREADS[@]}"; do
            echo "Running blocked_omp for N=$N, BS=$BS, T=$T"
            export OMP_NUM_THREADS=$T
            read time flops <<< $(./blocked_omp "$N" "$BS" "$T")
            echo "blocked_omp,$N,$BS,$T,$time,$flops" >> results.csv

        done
    done

    # -------- BLAS version --------
    echo "Running blas for N=$N"
    read time flops <<< $(./blas "$N")
    echo "blas,$N,,,$time,$flops" >> results.csv

done
