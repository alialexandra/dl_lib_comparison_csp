import csv
import matplotlib.pyplot as plt
from matplotlib.colors import to_rgba
import os

# Path to CSV file
file_path = "results.csv"

# Utility: safely convert string to float
def safe_float(x):
    try:
        val = float(x)
        return val if val >= 0 else None  # filter out negative values
    except:
        return None

# Ensure output directory exists
os.makedirs("plots", exist_ok=True)

# Load and parse the CSV into a list of dicts
with open(file_path) as f:
    reader = csv.DictReader(f)
    data = list(reader)

# Generic bar chart generator
def save_bar_chart(x_labels, values, title, filename, color):
    # Remove invalid (None) values
    filtered = [(x, v) for x, v in zip(x_labels, values) if v is not None]
    if not filtered:
        print(f"Skipping {title}: no valid data.")
        return
    x_labels, values = zip(*filtered)

    plt.figure(figsize=(12, 6))
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
    plt.savefig(f"plots/{filename}_v2.png")
    plt.close()
    print(f"Saved: plots/{filename}_v2.png")

# === Naive GPU ===
x_labels = []
values = []
for row in data:
    if row["Version"] == "naive_gpu":
        gflops = safe_float(row["GFLOPs"])
        if gflops is not None:
            label = f'N={row["N"]}\nB={row["BLOCK_SIZE"]}'
            x_labels.append(label)
            values.append(gflops)

save_bar_chart(x_labels, values, "Naive GPU Matrix Multiplication", "naive_gpu_performance", "mediumseagreen")

# === Shared Memory GPU ===
x_labels = []
values = []
for row in data:
    if row["Version"] == "shared_gpu":
        gflops = safe_float(row["GFLOPs"])
        if gflops is not None:
            label = f'N={row["N"]}\nB={row["BLOCK_SIZE"]}'
            x_labels.append(label)
            values.append(gflops)

save_bar_chart(x_labels, values, "Shared Memory GPU Matrix Multiplication", "shared_gpu_performance", "slateblue")

# === cuBLAS GPU ===
x_labels = []
values = []
for row in data:
    if row["Version"] == "cublas_gpu":
        gflops = safe_float(row["GFLOPs"])
        if gflops is not None:
            label = f'N={row["N"]}'
            x_labels.append(label)
            values.append(gflops)

save_bar_chart(x_labels, values, "cuBLAS GPU Matrix Multiplication", "cublas_gpu_performance", "darkorange")
