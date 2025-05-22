import csv
import matplotlib.pyplot as plt
from collections import defaultdict
from matplotlib.cm import get_cmap

# Read CSV
with open("results.csv") as f:
    reader = csv.DictReader(f)
    data = list(reader)

def safe_float(x):
    try:
        return float(x)
    except:
        return None

from matplotlib.colors import to_rgba

def save_bar_chart(x_labels, values, title, filename, color):
    plt.figure(figsize=(10, 5))

    # Normalize the values between 0 and 1 for alpha scaling
    vmin = min(values)
    vmax = max(values)
    value_range = vmax - vmin if vmax != vmin else 1  # avoid div by 0

    # Convert base color to RGBA
    base_rgba = to_rgba(color)

    # Create faded colors by varying the alpha
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

# 1. NAIVE
x_labels = []
values = []
for row in data:
    if row["Version"] == "naive":
        x_labels.append(f'N={row["N"]}')
        values.append(safe_float(row["GFLOPs"]))
save_bar_chart(x_labels, values, "Naive Matrix Multiplication", "naive_performance","royalblue")

# 2. BLAS
x_labels = []
values = []
for row in data:
    if row["Version"] == "blas":
        x_labels.append(f'N={row["N"]}')
        values.append(safe_float(row["GFLOPs"]))
save_bar_chart(x_labels, values, "BLAS Matrix Multiplication", "blas_performance","darkorange")

# 3. BLOCKED
x_labels = []
values = []
for row in data:
    if row["Version"] == "blocked":
        label = f'N={row["N"]}\nB={row["BLOCK_SIZE"]}'
        x_labels.append(label)
        values.append(safe_float(row["GFLOPs"]))
save_bar_chart(x_labels, values, "Blocked Matrix Multiplication", "blocked_performance","coral")


def plot_blocked_omp_grouped(data):
    from collections import defaultdict
    import numpy as np

    # Group data by matrix size
    grouped = defaultdict(list)
    for row in data:
        if row["Version"] == "blocked_omp":
            N = int(row["N"])
            bs = int(row["BLOCK_SIZE"])
            t = int(row["THREADS"])
            gflops = float(row["GFLOPs"])
            grouped[N].append(((bs, t), gflops))

    # Color map for each (BLOCK_SIZE, THREADS) combo
    import matplotlib.cm as cm
    import matplotlib.colors as mcolors
    all_keys = sorted({k for lst in grouped.values() for (k, _) in lst})
    cmap = cm.get_cmap("tab20", len(all_keys))
    key_color = {key: cmap(i) for i, key in enumerate(all_keys)}

    fig, ax = plt.subplots(figsize=(14, 6))

    bar_width = 0.8
    spacing = 1  # space between groups
    x_ticks = []
    x_labels = []
    legend_labels = {}
    current_x = 0

    for N in sorted(grouped.keys()):
        items = grouped[N]
        items = sorted(items, key=lambda x: x[0])  # sort by (block, thread)
        local_positions = []

        for (bs, t), gflops in items:
            label = f"B{bs}-T{t}"
            color = key_color[(bs, t)]
            ax.bar(current_x, gflops, width=bar_width, color=color)
            if label not in legend_labels:
                legend_labels[label] = color
            local_positions.append(current_x)
            current_x += 1

        # Add center tick for matrix size group
        center = np.mean(local_positions)
        x_ticks.append(center)
        x_labels.append(f"N={N}")

        current_x += spacing  # space between groups

    ax.set_xticks(x_ticks)
    ax.set_xticklabels(x_labels, fontsize=10)
    ax.set_ylabel("GFLOPs/sec")
    ax.set_title("Blocked + OMP Matrix Multiplication")
    ax.legend([plt.Rectangle((0,0),1,1,color=color) for color in legend_labels.values()],
              list(legend_labels.keys()), fontsize=8, loc="upper left", bbox_to_anchor=(1.01, 1))
    plt.tight_layout()
    plt.savefig("plots/blocked_omp_performance_grouped.png", bbox_inches="tight")
    plt.close()



# 4. BLOCKED + OMP

plot_blocked_omp_grouped(data)
# x_labels = []
# values = []
# for row in data:
#     if row["Version"] == "blocked_omp":
#         label = f'N={row["N"]}\nB={row["BLOCK_SIZE"]}\nT={row["THREADS"]}'
#         x_labels.append(label)
#         values.append(safe_float(row["GFLOPs"]))
# plt.tight_layout()
# plt.subplots_adjust(bottom=0.3)
# save_bar_chart(x_labels, values, "Blocked + OMP Matrix Multiplication", "blocked_omp_performance","orchid")


