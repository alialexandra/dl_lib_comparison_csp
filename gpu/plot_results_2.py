import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.ticker as ticker
import os

# Load the CSV
df = pd.read_csv("results.csv")

# Keep only naive and shared
df_filtered = df[df['Version'].isin(['naive_gpu', 'shared_gpu'])]

# Filter out values with very low GFLOPs
df_filtered = df_filtered[df_filtered['GFLOPs'] > 100]
df_filtered = df_filtered[df_filtered['N'] > 1024]


# Create readable label
df_filtered['Label'] = df_filtered.apply(lambda row: f"N={row['N']} BLOCK={row['BLOCK_SIZE']}", axis=1)

# Plot
plt.figure(figsize=(18, 8))
#sns.barplot(data=df_filtered, x='Label', y='GFLOPs', hue='Version', ci=None)

sns.barplot(data=df_filtered, x='Label', y='GFLOPs', hue='Version', ci=None)


plt.title("Naive vs Shared GPU GFLOPs by Matrix Size and Block Size")
plt.xlabel("Matrix Size and Block Size")
plt.ylabel("GFLOPs")

# Scientific notation on Y-axis
plt.gca().yaxis.set_major_formatter(ticker.ScalarFormatter(useMathText=True))
plt.ticklabel_format(style='sci', axis='y', scilimits=(0, 0))

# Improve label visibility
plt.xticks(rotation=45, ha='right', fontsize=10)
plt.yticks(fontsize=10)
plt.legend(title='Implementation', fontsize=10, title_fontsize=11)
plt.grid(True)
plt.tight_layout()

# Save
os.makedirs("plots", exist_ok=True)
plt.savefig("plots/gpu_shared_filtered_gflops.png", dpi=300)
plt.close()
