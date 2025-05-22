#!/bin/bash

SRC="blas_multiplication.c"
EXE="cpu_blas"
RESULTS_CSV="blas_results.csv"
POWER_LOG="power_blas_tmp.txt"
RESULTS_OUTPUT="blas_output.txt"

echo "Compiling $SRC..."
gcc -O3 -fopenmp -o $EXE $SRC -lopenblas
if [ $? -ne 0 ]; then
  echo "Compilation failed."
  exit 1
fi

if [ ! -f "$RESULTS_CSV" ]; then
  echo "N,threads,avg_time_sec,avg_power_watts,energy_mJ,gflops,gflops_per_watt,energy_per_flop_pj,occupancy_percent" > "$RESULTS_CSV"
fi

for N in 256 512 1024 2048 4096 8192; do
  for THREADS in 2 4 8 16 32; do
    export OPENBLAS_NUM_THREADS=$THREADS
    echo "Running BLAS: N=$N, THREADS=$THREADS"

    sudo powerstat -d 0 -z -R 0.1 2> /dev/null > "$POWER_LOG" &
    POWER_PID=$!

    START=$(date +%s.%N)
    OUTPUT=$(./$EXE $N)
    END=$(date +%s.%N)

    kill $POWER_PID
    wait $POWER_PID 2>/dev/null

    TIME_SEC=$(awk -v s=$START -v e=$END 'BEGIN { print e - s }')

    AVG_PWR=$(grep "Average power" "$POWER_LOG" | awk '{print $3}')
    [[ -z "$AVG_PWR" || "$AVG_PWR" == "nan" ]] && AVG_PWR="25.0"

    ENERGY_MJ=$(awk -v p=$AVG_PWR -v t=$TIME_SEC 'BEGIN { printf("%.4f", p * t * 1000) }')

    FLOPS=$(awk -v n=$N 'BEGIN { print 6 * n * n * n }')
    GFLOPS=$(awk -v f=$FLOPS -v t=$TIME_SEC 'BEGIN { print (t > 0 ? f / (1e9 * t) : 0) }')
    GFLOPS_WATT=$(awk -v g=$GFLOPS -v p=$AVG_PWR 'BEGIN { print (p > 0 ? g / p : 0) }')
    ENERGY_PER_FLOP_PJ=$(awk -v e=$ENERGY_MJ -v f=$FLOPS 'BEGIN { print (f > 0 ? (e * 1e6) / f : 0) }')

    USED_THREADS=$(echo "$OUTPUT" | grep "Threads used" | awk '{print $NF}')
    [[ -z "$USED_THREADS" ]] && USED_THREADS=$THREADS
    TOTAL_THREADS=$(nproc --all)
    OCCUPANCY_CPU=$(awk -v u=$USED_THREADS -v t=$TOTAL_THREADS 'BEGIN { printf("%.2f", 100.0 * u / t) }')

    echo "$N,$USED_THREADS,$TIME_SEC,$AVG_PWR,$ENERGY_MJ,$GFLOPS,$GFLOPS_WATT,$ENERGY_PER_FLOP_PJ,$OCCUPANCY_CPU" >> "$RESULTS_CSV"
    echo "$OUTPUT" >> "$RESULTS_OUTPUT"
    rm -f "$POWER_LOG"

  done
done
rm -f $EXE