#!/bin/bash

# Output CSV header (append if file exists)
echo "Version,N,BLOCK_SIZE,THREADS,Time_sec,GFLOPs" > results.csv

# Parameters
Ns=(512 1024 2048 4096 8192 16384 32768)
BLOCK_SIZES=(16 32 64 128)

# Compile GPU version
nvcc -O3 -o naive_gpu naive.cu
nvcc -O3 -o shared_gpu shared_mem.cu
nvcc -lcublas -O3 -o cublas_gpu cublas.cu


# Run benchmarks
for N in "${Ns[@]}"; do
    for BS in "${BLOCK_SIZES[@]}"; do
        # -------- Naive version --------
        echo "Running naive_gpu for N=$N, BLOCK_SIZE=$BS"
        read time flops <<< $(./naive_gpu "$N" "$BS")
        echo "naive_gpu,$N,$BS,,$time,$flops" >> results.csv

        # -------- Shared Memory version --------
        echo "Running shared_gpu for N=$N, TILE_SIZE=$BS"
        read time flops <<< $(./shared_gpu "$N" "$BS")
        echo "shared_gpu,$N,$BS,,$time,$flops" >> results.csv

        # -------- cuBLAS version --------
        echo "Running cublas_gpu for N=$N"
        read time flops <<< $(./cublas_gpu "$N")
        echo "cublas_gpu,$N,,,${time},${flops}" >> results.csv

    done
done
