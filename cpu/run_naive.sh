#!/bin/bash

RESULTS_CSV="cpu_results.csv"
EXE="./naive_multiplication"  # your CPU executable




if [ ! -f "$RESULTS_CSV" ]; then
  echo "N,avg_time_sec,avg_power_watts,energy_mJ,gflops,gflops_per_watt,energy_per_flop_pj,occupancy_percent" > "$RESULTS_CSV"
fi

for N in 256 512 1024 2048; do
  echo "Running CPU kernel: N=$N"

  # Warm-up
  $EXE $N >/dev/null

  # Start timing
  START=$(date +%s.%N)

  # Launch powerstat in background to sample power every 0.1 sec
  sudo powerstat -d 0 -z -R 0.1 2> /dev/null > power_tmp.txt &
  POWER_PID=$!

  # Run the actual experiment
  OUTPUT=$($EXE $N)

  # Stop powerstat
  kill $POWER_PID
  wait $POWER_PID 2>/dev/null

  # End timing
  END=$(date +%s.%N)
  TIME_SEC=$(awk -v s="$START" -v e="$END" 'BEGIN { print e - s }')

  # Extract power in watts (average over samples)
  AVG_PWR=$(grep "Average power" power_tmp.txt | awk '{print $3}')
  [[ -z "$AVG_PWR" ]] && AVG_PWR=25.0

  # Energy = Power × Time
  ENERGY_MJ=$(awk -v p=$AVG_PWR -v t=$TIME_SEC 'BEGIN { printf("%.4f", p * t * 1000) }')

  # GFLOPS = (6 * N^3) / (time * 1e9)
  GFLOPS=$(awk -v n=$N -v t=$TIME_SEC 'BEGIN { printf("%.3f", (6.0 * n * n * n) / (t * 1e9)) }')

  # GFLOPS per watt
  GFLOPS_WATT=$(awk -v g=$GFLOPS -v p=$AVG_PWR 'BEGIN { printf("%.3f", g / p) }')

  # Energy per flop (pJ)
  FLOPS=$(awk -v n=$N 'BEGIN { print 6 * n * n * n }')
  ENERGY_PER_FLOP_PJ=$(awk -v e=$ENERGY_MJ -v f=$FLOPS 'BEGIN { print (f > 0 ? (e * 1e6) / f : 0) }')
  USED_THREADS=1
  TOTAL_THREADS=$(nproc --all)
  OCCUPANCY_CPU=$(awk -v u=$USED_THREADS -v t=$TOTAL_THREADS 'BEGIN { printf("%.2f", 100.0 * u / t) }')

  echo "$N,$TIME_SEC,$AVG_PWR,$ENERGY_MJ,$GFLOPS,$GFLOPS_WATT,$ENERGY_PER_FLOP_PJ,$OCCUPANCY_CPU" >> "$RESULTS_CSV"

  rm -f power_tmp.txt
done
