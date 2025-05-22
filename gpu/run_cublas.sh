#!/bin/bash

EXE="cublas"
RESULTS_CSV="cublas_gpu_results.csv"
RESULTS_OUTPUT="cublas_output.txt" # Output file name
CLOCK_LOG="clock_cublas_tmp.csv"

# Compile
nvcc -O3 -lcublas -o $EXE cublas.cu

# Header
if [ ! -f "$RESULTS_CSV" ]; then
  echo "N,avg_time_ms,avg_power_watts,energy_mJ,total_mem_bytes,free_mem_bytes,avg_clock_mhz,min_clock_mhz,max_clock_mhz,avg_temp_c,gflops,gflops_per_watt,energy_per_flop_pj" > "$RESULTS_CSV"
fi

# Estimate safe matrix size
MEM_FREE=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits | head -1)
BYTES_FREE=$((MEM_FREE * 1024 * 1024))
MAX_N=$(awk -v B=$BYTES_FREE 'BEGIN { printf("%d", sqrt(B / (8 * 5))) }') # 5 matrices

for N in 256 512 1024 2048 4096 8192; do
  [[ $N -gt $MAX_N ]] && break

  echo "Running cuBLAS kernel: N=$N"

  # Start GPU logging
  nvidia-smi --query-gpu=timestamp,power.draw,clocks.current.graphics,temperature.gpu --format=csv -l 0.1 > "$CLOCK_LOG" &
  SMI_PID=$!

  ./$EXE $N >/dev/null
  OUTPUT=$(./$EXE $N)

  kill $SMI_PID
  wait $SMI_PID 2>/dev/null

  # Extract time
  AVG_TIME_MS=$(echo "$OUTPUT" | grep "Avg time" | awk -F'= ' '{print $2}' | awk '{print $1}')
  [[ ! "$AVG_TIME_MS" =~ ^[0-9]+(\.[0-9]+)?$ ]] && AVG_TIME_MS="0.0"
  TIME_SEC=$(awk -v ms=$AVG_TIME_MS 'BEGIN { print ms / 1000.0 }')

  # Memory info
  MEM_INFO=$(nvidia-smi --query-gpu=memory.total,memory.free --format=csv,noheader,nounits | head -1)
  TOTAL_MEM=$(echo "$MEM_INFO" | cut -d ',' -f1 | xargs)
  FREE_MEM=$(echo "$MEM_INFO" | cut -d ',' -f2 | xargs)

  # Power & energy
  AVG_PWR=$(tail -n +2 "$CLOCK_LOG" | awk -F',' '{gsub(/[^0-9.]/,"",$2); sum+=$2; count++} END {print (count>0 ? sum/count : 0)}')
  [[ -z "$AVG_PWR" || "$AVG_PWR" == "nan" ]] && AVG_PWR="0.0"
  ENERGY_MJ=$(awk -v p=$AVG_PWR -v t=$AVG_TIME_MS 'BEGIN { print (p > 0 ? p * t / 1000 : 0.0) }')

  # Clocks & temp
  AVG_CLOCK=$(tail -n +2 "$CLOCK_LOG" | awk -F',' '{gsub(/[^0-9.]/,"",$3); sum+=$3; count++} END {print (count>0 ? sum/count : 0)}')
  MIN_CLOCK=$(tail -n +2 "$CLOCK_LOG" | awk -F',' '{gsub(/[^0-9.]/,"",$3); if (NR==1 || $3 < min) min=$3} END {print min}')
  MAX_CLOCK=$(tail -n +2 "$CLOCK_LOG" | awk -F',' '{gsub(/[^0-9.]/,"",$3); if ($3 > max) max=$3} END {print max}')
  AVG_TEMP=$(tail -n +2 "$CLOCK_LOG" | awk -F',' '{gsub(/[^0-9.]/,"",$4); sum+=$4; count++} END {print (count>0 ? sum/count : 0)}')

  # GFLOPS metrics
  FLOPS=$(awk -v n=$N 'BEGIN { printf("%.0f", 6 * n * n * n) }')
  GFLOPS=$(awk -v f=$FLOPS -v t=$TIME_SEC 'BEGIN { print (t > 0 ? f / (1e9 * t) : 0) }')
  GFLOPS_PER_WATT=$(awk -v g=$GFLOPS -v p=$AVG_PWR 'BEGIN { print (p > 0 ? g / p : 0) }')
  ENERGY_PER_FLOP_PJ=$(awk -v e=$ENERGY_MJ -v f=$FLOPS 'BEGIN { print (f > 0 ? (e * 1e6) / f : 0) }')

  # Log results
  echo "$N,$AVG_TIME_MS,$AVG_PWR,$ENERGY_MJ,$((TOTAL_MEM*1024*1024)),$((FREE_MEM*1024*1024)),$AVG_CLOCK,$MIN_CLOCK,$MAX_CLOCK,$AVG_TEMP,$GFLOPS,$GFLOPS_PER_WATT,$ENERGY_PER_FLOP_PJ" >> "$RESULTS_CSV"
  echo "$OUTPUT" >> "$RESULTS_OUTPUT"
  rm -f "$CLOCK_LOG"
  
done
rm -f $EXE