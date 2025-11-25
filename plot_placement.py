import matplotlib.pyplot as plt
import matplotlib.patches as patches
import os
import re

def plot_fig(file_path, node_pos):
    fig1 = plt.figure()
    ax1 = plt.gca()
    min_x, min_y = 0, 0
    max_x, max_y = 0, 0
    ax1.axes.xaxis.set_visible(False)
    ax1.axes.yaxis.set_visible(False)
    for node_name in node_pos.keys():
        x, y, size_x, size_y = node_pos[node_name]
        if size_x == 0 or size_y == 0:
            continue
        ax1.add_patch(
            patches.Rectangle(
                (x, y),
                size_x,
                size_y, linewidth=1, edgecolor='k', facecolor='blue', alpha=0.4, fill=True
            )
        )
        min_x, min_y = min(min_x, x), min(min_y, y)
        max_x, max_y = max(max_x, x+size_x), max(max_y, y+size_y)

    plt.xlim(min_x, max_x)
    plt.ylim(min_y, max_y)
    
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.gca().set_aspect('equal')
    fig1.savefig(file_path, dpi=1200, bbox_inches='tight')
    plt.close()

def is_number_regex(s):
    pattern = r'^[-+]?(\d+(\.\d*)?|\.\d+)$'
    return bool(re.match(pattern, s))

def read_node_pos(nodes_path, pl_path):
    node_pos = {}
    with open(nodes_path, "r") as f2:
        for line in f2:
            line = line.strip()
            l = line.split()
            if line and is_number_regex(l[1]):
                macro_name = l[0]
                node_pos[macro_name] = [0, 0, float(l[1]), float(l[2])]

    with open(pl_path, "r") as f2:
        for line in f2:
            line = line.strip()
            l = line.split()
            if line and is_number_regex(l[1]):
                macro_name = l[0]
                if macro_name in node_pos:
                    node_pos[macro_name][0] = float(l[1])
                    node_pos[macro_name][1] = float(l[2])
    
    return node_pos

if __name__ == "__main__":
    benchmark_folder = "ChiPBench"
    dataset = "ariane133"
    seed = 2000

    pl_path = f'./result/{benchmark_folder}/{dataset}/{seed}/{dataset}.pl'
    nodes_path = f'./benchmarks/{benchmark_folder}/{dataset}/{dataset}.nodes'
    node_pos = read_node_pos(nodes_path, pl_path)
    fig_path = pl_path[:-2] + 'png'
    plot_fig(fig_path, node_pos=node_pos)
