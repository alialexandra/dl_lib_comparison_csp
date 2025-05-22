import pandas as pd
import matplotlib.pyplot as plt

# Load CSVs
cpu_df = pd.read_csv("cpu/results.csv")
gpu_df = pd.read_csv("gpu/results.csv")

# Clean
cpu_df = cpu_df.dropna(subset=["Time_sec"])
gpu_df = gpu_df.dropna(subset=["Time_sec"])
gpu_df = gpu_df[gpu_df["Time_sec"] > 0.001]  # remove 0s

# ---------------------
# Plot 1: cuBLAS vs BLAS
# ---------------------
blas_cpu = cpu_df[cpu_df["Version"] == "blas"]
cublas_gpu = gpu_df[gpu_df["Version"] == "cublas_gpu"]

blas_cpu_best = blas_cpu.groupby("N", as_index=False).agg(cpu_time=("Time_sec", "min"))
cublas_gpu_best = cublas_gpu.groupby("N", as_index=False).agg(gpu_time=("Time_sec", "min"))

merged_blas = pd.merge(blas_cpu_best, cublas_gpu_best, on="N")
merged_blas["Speedup"] = merged_blas["cpu_time"] / merged_blas["gpu_time"]
merged_blas.to_csv("speedup_blas_vs_cublas.csv", index=False)

plt.figure(figsize=(10, 6))
plt.plot(merged_blas["N"], merged_blas["Speedup"], marker='o', label="cuBLAS vs BLAS", color="green")
plt.title("cuBLAS vs BLAS Speedup")
plt.xlabel("Matrix Size (N)")
plt.ylabel("Speedup")
plt.grid(True)
plt.tight_layout()
plt.savefig("speedup_cublas_vs_blas.png", dpi=300)
plt.close()

# -----------------------------
# Plot 2: Best of All Others
# -----------------------------
cpu_other = cpu_df[cpu_df["Version"] != "blas"]
gpu_other = gpu_df[gpu_df["Version"] != "cublas_gpu"]

best_cpu_other = cpu_other.groupby("N", as_index=False).agg(cpu_time=("Time_sec", "min"))
best_gpu_other = gpu_other.groupby("N", as_index=False).agg(gpu_time=("Time_sec", "min"))

merged_other = pd.merge(best_cpu_other, best_gpu_other, on="N")
merged_other["Speedup"] = merged_other["cpu_time"] / merged_other["gpu_time"]
merged_other.to_csv("speedup_best_other.csv", index=False)

plt.figure(figsize=(10, 6))
plt.plot(merged_other["N"], merged_other["Speedup"], marker='s', color="blue", label="Best GPU vs CPU (non-BLAS)")
plt.title("Best GPU vs CPU Speedup (Excl. BLAS)")
plt.xlabel("Matrix Size (N)")
plt.ylabel("Speedup")
plt.grid(True)
plt.tight_layout()
plt.savefig("speedup_best_other.png", dpi=300)
plt.close()

