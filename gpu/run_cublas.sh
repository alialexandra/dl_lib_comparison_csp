#!/bin/bash

REPS=3
RESULTS_CSV="cublas_gpu_results.csv"
POWER_LOG="power_cublas_tmp.csv"
timestamp=$(date +"%Y%m%d_%H%M%S")

if [ ! -f "$RESULTS_CSV" ]; then
  echo "N,avg_time_sec,avg_power_watts,energy_joules,total_mem_bytes,free_mem_bytes" > "$RESULTS_CSV"
fi

MEM_FREE=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits | head -1)
BYTES_FREE=$((MEM_FREE * 1024 * 1024))
MAX_N=$(awk -v B=$BYTES_FREE 'BEGIN { printf("%d", sqrt(B / (8 * 4))) }')

for N in 256 512 1024 2048 4096 8192; do
  [[ $N -gt $MAX_N ]] && break

  echo "Running cuBLAS kernel: N=$N"
  nvcc -O3 -lcublas -o cublas cublas.cu

  nvidia-smi --query-gpu=timestamp,power.draw --format=csv -l 0.1 > "$POWER_LOG" &
  SMI_PID=$!

  ./cublas $N >/dev/null # Warm-up

  NSYS_QDREP="cublas_${N}_${timestamp}.qdrep"
  NSYS_STATS="cublas_${N}_${timestamp}_stats.txt"
  OUTPUT=$(nsys profile --stats=true -o "$NSYS_QDREP" ./cublas $N)

  kill $SMI_PID

  AVG_TIME=$(echo "$OUTPUT" | grep "Avg time" | awk -F'= ' '{print $2}' | awk '{print $1}')
  [[ ! "$AVG_TIME" =~ ^[0-9]+(\.[0-9]+)?$ ]] && AVG_TIME="0.0"

  MEM_INFO=$(nvidia-smi --query-gpu=memory.total,memory.free --format=csv,noheader,nounits | head -1)
  TOTAL_MEM=$(echo "$MEM_INFO" | cut -d ',' -f1 | xargs)
  FREE_MEM=$(echo "$MEM_INFO" | cut -d ',' -f2 | xargs)

  AVG_PWR=$(awk -F',' '/[0-9]+\.[0-9]+/ {sum+=$2; count++} END {print (count>0 ? sum/count : 0)}' "$POWER_LOG")
  ENERGY=$(awk -v p="$AVG_PWR" -v t="$AVG_TIME" 'BEGIN { printf "%.4f", p * t }')

  echo "$N,$AVG_TIME,$AVG_PWR,$ENERGY,$((TOTAL_MEM*1024*1024)),$((FREE_MEM*1024*1024))" >> "$RESULTS_CSV"

  nsys stats "$NSYS_QDREP" > "$NSYS_STATS"
  rm -f "$POWER_LOG"
done
