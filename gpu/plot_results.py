import csv
import matplotlib.pyplot as plt
from matplotlib.colors import to_rgba
import os

# Load and parse the CSV
with open("results.csv") as f:
    reader = csv.DictReader(f)
    data = list(reader)

# Utility function to safely convert to float
def safe_float(x):
    try:
        return float(x)
    except:
        return None

# Ensure output directory exists
os.makedirs("plots", exist_ok=True)

# Create faded bar chart
def save_bar_chart(x_labels, values, title, filename, color):
    plt.figure(figsize=(10, 5))
    vmin = min(values)
    vmax = max(values)
    value_range = vmax - vmin if vmax != vmin else 1
    base_rgba = to_rgba(color)
    bar_colors = [
        (base_rgba[0], base_rgba[1], base_rgba[2], 0.3 + 0.7 * (v - vmin) / value_range)
        for v in values
    ]
    plt.bar(x_labels, values, color=bar_colors)
    plt.xticks(rotation=45, ha='right', fontsize=8)
    plt.ylabel("GFLOPs/sec")
    plt.title(title)
    plt.tight_layout()
    plt.savefig(f"plots/{filename}.png")
    plt.close()

# 1. NAIVE GPU
x_labels = []
values = []
for row in data:
    if row["Version"] == "naive_gpu":
        label = f'N={row["N"]}\nB={row["BLOCK_SIZE"]}'
        x_labels.append(label)
        values.append(safe_float(row["GFLOPs"]))

save_bar_chart(x_labels, values, "Naive GPU Matrix Multiplication", "naive_gpu_performance", "mediumseagreen")

# 2. SHARED GPU
x_labels = []
values = []
for row in data:
    if row["Version"] == "shared_gpu":
        label = f'N={row["N"]}\nB={row["BLOCK_SIZE"]}'
        x_labels.append(label)
        values.append(safe_float(row["GFLOPs"]))

save_bar_chart(x_labels, values, "Shared Memory GPU Matrix Multiplication", "shared_gpu_performance", "slateblue")

# 3. cuBLAS GPU
x_labels = []
values = []
for row in data:
    if row["Version"] == "cublas_gpu":
        label = f'N={row["N"]}'
        x_labels.append(label)
        values.append(safe_float(row["GFLOPs"]))

save_bar_chart(x_labels, values, "cuBLAS GPU Matrix Multiplication", "cublas_gpu_performance", "darkorange")
