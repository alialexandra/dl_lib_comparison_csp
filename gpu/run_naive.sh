#!/bin/bash

REPS=3
RESULTS_CSV="naive_gpu_results.csv"
CLOCK_LOG="clock_naive_tmp.csv"

if [ ! -f "$RESULTS_CSV" ]; then
  echo "N,threads,blocks,avg_time_ms,avg_power_watts,energy_mJ,total_mem_bytes,free_mem_bytes,avg_clock_mhz,min_clock_mhz,max_clock_mhz,avg_temp_c,gflops,gflops_per_watt,energy_per_flop_pj,occupancy_percent" > "$RESULTS_CSV"
fi

MEM_FREE=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits | head -1)
BYTES_FREE=$((MEM_FREE * 1024 * 1024))
MAX_N=$(awk -v B=$BYTES_FREE 'BEGIN { printf("%d", sqrt(B / (8 * 4))) }')

for N in 256 512 1024 2048 4096 8192; do
  [[ $N -gt $MAX_N ]] && break

  for THREADS in 8 16 32; do
    echo "Running naive kernel: N=$N, threads=$THREADS"
    nvcc -O3 -DTHREADS=$THREADS -o naive naive.cu

    # Start power/clock/temp logging
    nvidia-smi --query-gpu=timestamp,power.draw,clocks.gr,temperature.gpu --format=csv -l 0.1 > "$CLOCK_LOG" &
    SMI_PID=$!

    ./naive $N >/dev/null 2>&1  # warm-up
    OUTPUT=$(./naive $N)

    kill $SMI_PID
    wait $SMI_PID 2>/dev/null

    # Extract and validate time
    AVG_TIME_MS=$(echo "$OUTPUT" | grep "Avg time" | awk -F'= ' '{print $2}' | awk '{print $1}')
    [[ ! "$AVG_TIME_MS" =~ ^[0-9]+(\.[0-9]+)?$ ]] && AVG_TIME_MS="0.0"
    TIME_SEC=$(awk -v ms=$AVG_TIME_MS 'BEGIN { print (ms > 0 ? ms / 1000.0 : 0) }')

    # Blocks
    BLOCKS=$(( (N + THREADS - 1) / THREADS ))

    # GPU memory info
    MEM_INFO=$(nvidia-smi --query-gpu=memory.total,memory.free --format=csv,noheader,nounits | head -1)
    TOTAL_MEM=$(echo "$MEM_INFO" | cut -d ',' -f1 | xargs)
    FREE_MEM=$(echo "$MEM_INFO" | cut -d ',' -f2 | xargs)

    # Power
    AVG_PWR=$(awk -F',' '/[0-9]+\.[0-9]+/ {sum+=$2; count++} END {print (count>0 ? sum/count : 0)}' "$CLOCK_LOG")
    [[ -z "$AVG_PWR" || "$AVG_PWR" == "nan" ]] && AVG_PWR="0.0"

    ENERGY_MJ=$(awk -v p=$AVG_PWR -v t=$AVG_TIME_MS 'BEGIN { print (p > 0 ? p * t / 1000 : 0.0) }')

    # Clock + temperature
    AVG_CLOCK=$(tail -n +2 "$CLOCK_LOG" | awk -F',' '{gsub(/[^0-9.]/, "", $3); sum+=$3; count++} END {print (count>0 ? sum/count : 0)}')
    MIN_CLOCK=$(tail -n +2 "$CLOCK_LOG" | awk -F',' '{gsub(/[^0-9.]/, "", $3); if (NR==1 || $3 < min) min=$3} END {print min}')
    MAX_CLOCK=$(tail -n +2 "$CLOCK_LOG" | awk -F',' '{gsub(/[^0-9.]/, "", $3); if ($3 > max) max=$3} END {print max}')
    AVG_TEMP=$(tail -n +2 "$CLOCK_LOG" | awk -F',' '{gsub(/[^0-9.]/, "", $4); sum+=$4; count++} END {print (count>0 ? sum/count : 0)}')


    # FLOPs and efficiency
    FLOPS=$(awk -v n=$N 'BEGIN { printf("%.0f", 6 * n * n * n) }')
    GFLOPS=$(awk -v f=$FLOPS -v t=$TIME_SEC 'BEGIN { print (t > 0 ? f / (1e9 * t) : 0) }')
    GFLOPS_PER_WATT=$(awk -v g=$GFLOPS -v p=$AVG_PWR 'BEGIN { print (p > 0 ? g / p : 0) }')
    ENERGY_PER_FLOP_PJ=$(awk -v e=$ENERGY_MJ -v f=$FLOPS 'BEGIN { print (f > 0 ? (e * 1e6) / f : 0) }')

    # Occupancy
    OCCUPANCY=$(echo "$OUTPUT" | grep "Occupancy" | awk -F'≈ ' '{print $2}' | awk -F'%' '{print $1}')
    [[ ! "$OCCUPANCY" =~ ^[0-9]+(\.[0-9]+)?$ ]] && OCCUPANCY="0.0"

    echo "$N,$THREADS,$BLOCKS,$AVG_TIME_MS,$AVG_PWR,$ENERGY_MJ,$((TOTAL_MEM*1024*1024)),$((FREE_MEM*1024*1024)),$AVG_CLOCK,$MIN_CLOCK,$MAX_CLOCK,$AVG_TEMP,$GFLOPS,$GFLOPS_PER_WATT,$ENERGY_PER_FLOP_PJ,$OCCUPANCY" >> "$RESULTS_CSV"

    rm -f "$CLOCK_LOG"
  done
done
