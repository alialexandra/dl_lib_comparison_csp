#!/bin/bash

REPS=3
RESULTS_CSV="shared_gpu_results.csv"
POWER_LOG="power_shared_tmp.csv"

if [ ! -f "$RESULTS_CSV" ]; then
  echo "N,tile_size,blocks,avg_time_ms,avg_power_watts,energy_mJ,total_mem_bytes,free_mem_bytes" > "$RESULTS_CSV"
fi

MEM_FREE=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits | head -1)
BYTES_FREE=$((MEM_FREE * 1024 * 1024))
MAX_N=$(awk -v B=$BYTES_FREE 'BEGIN { printf("%d", sqrt(B / (8 * 4))) }')

for N in 256 512 1024 2048 4096 8192; do
  [[ $N -gt $MAX_N ]] && break
  for TILE_SIZE in 8 16 32; do
    echo "Running shared memory kernel: N=$N, tile_size=$TILE_SIZE"
    nvcc -O3 -DTILE_SIZE=$TILE_SIZE -o shared shared.cu

    nvidia-smi --query-gpu=timestamp,power.draw --format=csv -l 0.1 > "$POWER_LOG" &
    SMI_PID=$!

    # ./shared $N >/dev/null # Warm-up
    # OUTPUT=$(./shared $N)
    ./shared $N $TILE_SIZE >/dev/null # Warm-up
    OUTPUT=$(./shared $N $TILE_SIZE)


    kill $SMI_PID

    AVG_TIME_MS=$(echo "$OUTPUT" | grep "Avg time" | awk -F'= ' '{print $2}' | awk '{print $1}')
    [[ ! "$AVG_TIME_MS" =~ ^[0-9]+(\.[0-9]+)?$ ]] && AVG_TIME_MS="0.0"

    BLOCKS=$(( (N + TILE_SIZE - 1) / TILE_SIZE ))

    MEM_INFO=$(nvidia-smi --query-gpu=memory.total,memory.free --format=csv,noheader,nounits | head -1)
    TOTAL_MEM=$(echo "$MEM_INFO" | cut -d ',' -f1 | xargs)
    FREE_MEM=$(echo "$MEM_INFO" | cut -d ',' -f2 | xargs)

    AVG_PWR=$(awk -F',' '/[0-9]+\.[0-9]+/ {sum+=$2; count++} END {print (count>0 ? sum/count : 0)}' "$POWER_LOG")
    ENERGY_MJ=$(awk -v p="$AVG_PWR" -v t="$AVG_TIME_MS" 'BEGIN { printf "%.4f", p * t / 1000 }')

    echo "$N,$TILE_SIZE,$BLOCKS,$AVG_TIME_MS,$AVG_PWR,$ENERGY_MJ,$((TOTAL_MEM*1024*1024)),$((FREE_MEM*1024*1024))" >> "$RESULTS_CSV"

    rm -f "$POWER_LOG"
  done
  done
