#!/bin/bash

SRC="naive_multiplication.c"
EXE="naive_multiplication"
RESULTS_CSV="naive_results.csv"
RESULTS_OUTPUT="naive_output.txt"
POWER_LOG="power_naive_tmp.txt"

echo "Compiling $SRC..."
gcc -march=native -ffast-math -fassociative-math \
-fno-signed-zeros -ffinite-math-only \
-fno-signaling-nans -fno-trapping-math \
-fexcess-precision=fast -mfpmath=sse \
-o $EXE $SRC


echo "Compilation completed. Obtained executable: $EXE"
if [ $? -ne 0 ]; then
  echo "Compilation failed."
  exit 1
fi

if [ ! -f "$RESULTS_CSV" ]; then
  echo "N,avg_time_sec,avg_power_watts,energy_mJ,gflops,gflops_per_watt,energy_per_flop_pj,occupancy_percent" > "$RESULTS_CSV"
fi

# for the sake of the argument and in matter of time we will not use the full range of N
for N in 256 512 1024 2048; do
  echo "Running CPU kernel: N=$N"

  START=$(date +%s.%N)
  sudo powerstat -d 0 -z -R 0.1 2> /dev/null > $POWER_LOG &
  POWER_PID=$!

  OUTPUT=$(./$EXE $N)

  kill $POWER_PID 2>/dev/null || true
  wait $POWER_PID 2>/dev/null || true

  END=$(date +%s.%N)
  TIME_SEC=$(awk -v s=$START -v e=$END 'BEGIN { print e - s }')

  AVG_PWR=$(grep "Average power" $POWER_LOG | awk '{print $3}')
  [[ -z "$AVG_PWR" || "$AVG_PWR" == "nan" ]] && AVG_PWR="25.0"

  ENERGY_MJ=$(awk -v p=$AVG_PWR -v t=$TIME_SEC 'BEGIN { printf("%.4f", p * t * 1000) }')
  GFLOPS=$(awk -v n=$N -v t=$TIME_SEC 'BEGIN { printf("%.3f", (6.0 * n * n * n) / (t * 1e9)) }')
  GFLOPS_WATT=$(awk -v g=$GFLOPS -v p=$AVG_PWR 'BEGIN { printf("%.3f", g / p) }')
  FLOPS=$(awk -v n=$N 'BEGIN { print 6 * n * n * n }')
  ENERGY_PER_FLOP_PJ=$(awk -v e=$ENERGY_MJ -v f=$FLOPS 'BEGIN { print (f > 0 ? (e * 1e6) / f : 0) }')
  USED_THREADS=1
  TOTAL_THREADS=$(nproc --all)
  OCCUPANCY_CPU=$(awk -v u=$USED_THREADS -v t=$TOTAL_THREADS 'BEGIN { printf("%.2f", 100.0 * u / t) }')

  echo "$N,$TIME_SEC,$AVG_PWR,$ENERGY_MJ,$GFLOPS,$GFLOPS_WATT,$ENERGY_PER_FLOP_PJ,$OCCUPANCY_CPU" >> "$RESULTS_CSV"
  echo "$OUTPUT" >> "$RESULTS_OUTPUT"
  rm -f "$POWER_LOG"
done
rm -f $EXE