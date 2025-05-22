#!/bin/bash
# This script runs the naive GPU kernel and logs performance metrics.

echo "Runnign the experimetns for CPU implementation"
cd cpu
echo "Running naive CPU kernel..."
./run_naive.sh
echo "Runing Blocked OMP"
./run_blocked_omp.sh
echo "Running BLAS CPU kernel..."
./run_blas.sh
cd ..
cd gpu
echo "running GPU experimetns"
echo "Running naive GPU kernel..."
./run_naive.sh
echo "Running shared GPU kernel..."
./run_shared.sh
echo "Running cublas GPU kernel..."
./run_cublas.sh