import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

# Create output folder if it doesn't exist
os.makedirs("plots", exist_ok=True)

# Load data
# Load CSVs
gpu_results_path = "./results.csv"
cpu_results_path = "../cpu/results_v2.csv"
gpu_df = pd.read_csv(gpu_results_path)
cpu_df = pd.read_csv(cpu_results_path)

# Filter and clean
gpu_df_clean = gpu_df.dropna(subset=['Time_sec', 'GFLOPs'])
gpu_df_clean = gpu_df_clean[gpu_df_clean['Version'].isin(['naive_gpu', 'shared_gpu', 'cublas_gpu'])]

cpu_df_clean = cpu_df.dropna(subset=['Time_sec', 'GFLOPs'])
cpu_df_clean = cpu_df_clean[cpu_df_clean['Version'].isin(['blas', 'blocked', 'blocked_omp'])]

# Add group labels
gpu_df_clean["Group"] = "GPU"
cpu_df_clean["Group"] = "CPU"
combined_df = pd.concat([gpu_df_clean, cpu_df_clean], ignore_index=True)

# ----- Plot 1: CPU vs GPU performance (log scale) -----
plt.figure(figsize=(12, 6))
sns.lineplot(
    data=combined_df,
    x="N",
    y="GFLOPs",
    hue="Version",
    style="Group",  # solid for GPU, dashed for CPU
    markers=True,
    linewidth=2,
    palette="tab10",
    errorbar=None  # <- removes shaded areas
)
plt.yscale("log")
plt.title("GFLOPs vs Matrix Size (N): CPU vs GPU")
plt.xlabel("Matrix Size (N)")
plt.ylabel("GFLOPs")
plt.grid(True, axis="y", which="both", linestyle="--", linewidth=0.5)
plt.grid(False, axis="x")  # Disable vertical grid lines
plt.legend(title="Version", fontsize=9)
plt.tight_layout()
plt.savefig("plots/cpu_vs_gpu_gflops_logscale_clean.png", dpi=300)
plt.close()


# ----- Plot 2: Block Size vs GFLOPs (colored by N) -----
plt.figure(figsize=(12, 6))
block_effect = gpu_df_clean.dropna(subset=["BLOCK_SIZE"])
sns.lineplot(
    data=block_effect,
    x="BLOCK_SIZE",
    y="GFLOPs",
    hue="N",       # color-coded by matrix size
    style="Version",
    markers=True,
    dashes=False,
    linewidth=2,
    palette="viridis"
)
plt.title("Effect of Tile/Block Size on GPU Performance")
plt.xlabel("Tile/Block Size")
plt.ylabel("GFLOPs")
plt.grid(True)
plt.tight_layout()
plt.savefig("plots/block_size_vs_gflops_gpu_v2.png", dpi=300)
plt.close()
# ----- Plot 3: Execution Time vs Matrix Size (CPU and GPU) -----
plt.figure(figsize=(12, 6))
sns.lineplot(
    data=combined_df,
    x="N",
    y="Time_sec",
    hue="Version",
    style="Group",
    markers=True,
    linewidth=2,
    palette="tab10"
)

plt.yscale("log")
plt.title("Execution Time vs Matrix Size (N): CPU and GPU (Log Scale)")
plt.xlabel("Matrix Size (N)")
plt.ylabel("Execution Time (seconds, log scale)")
plt.grid(True, which="both", linestyle="--", linewidth=0.5)
plt.tight_layout()
plt.savefig("plots/cpu_vs_gpu_execution_time_logscale_v2.png", dpi=300)
plt.close()

# ----- Plot 4: Execution Time vs N and Block Size (Naive vs Shared GPU) -----

# Filter GPU results for naive and shared implementations only
gpu_naive_shared = gpu_df_clean[
    gpu_df_clean['Version'].isin(['naive_gpu', 'shared_gpu'])
].dropna(subset=['BLOCK_SIZE'])

# Create a label for each bar: "N=..., B=..."
gpu_naive_shared['Label'] = gpu_naive_shared.apply(
    lambda row: f"N={int(row['N'])}\nB={int(row['BLOCK_SIZE'])}", axis=1
)

plt.figure(figsize=(16, 6))
sns.barplot(
    data=gpu_naive_shared,
    x="Label",
    y="Time_sec",
    hue="Version",
    palette="pastel"
)
plt.xticks(rotation=45)
plt.title("Execution Time by N and Block Size (Naive vs Shared GPU)")
plt.xlabel("Matrix Size (N) and Block Size (B)")
plt.ylabel("Execution Time (seconds)")
plt.grid(axis='y', linestyle="--", linewidth=0.5)
plt.tight_layout()
plt.savefig("plots/execution_time_naive_vs_shared_v2.png", dpi=300)
plt.close()
