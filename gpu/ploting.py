import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

# Create output folder if it doesn't exist
os.makedirs("plots", exist_ok=True)

# Load data
gpu_df = pd.read_csv("./results.csv")
cpu_df = pd.read_csv("../cpu/results.csv")

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
    palette="tab10"
)
plt.yscale("log")
plt.title("GFLOPs vs Matrix Size (N): CPU vs GPU (Log Scale)")
plt.xlabel("Matrix Size (N)")
plt.ylabel("GFLOPs (log scale)")
plt.grid(True, which="both", linestyle="--", linewidth=0.5)
plt.tight_layout()
plt.savefig("plots/cpu_vs_gpu_gflops_logscale.png", dpi=300)
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
plt.savefig("plots/block_size_vs_gflops_gpu.png", dpi=300)
plt.close()
