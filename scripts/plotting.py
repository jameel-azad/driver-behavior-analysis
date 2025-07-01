import matplotlib.pyplot as plt
import numpy as np
import os
import math

def plot_cluster_profiles(cluster_centers, output_path='figures/cluster_profiles.png'):
    n_clusters = cluster_centers.shape[0]

    # Split into three chunks for brake velocity and acceleration
    partition = math.ceil(cluster_centers.shape[1]/3)
    
    brake = cluster_centers[:, :partition]
    velocity = cluster_centers[:, partition:partition*2]
    acceleration = cluster_centers[:, partition*2:]

    fig, axs = plt.subplots(3, 1, figsize=(20, 12))
    fig.suptitle(f'Cluster Centers ({n_clusters} Clusters)', fontsize=16)
    time = np.arange(80)

    def plot_with_inline_labels(ax, data, title):
        for i in range(n_clusters):
            y = data[i]
            line, = ax.plot(time, y)

            # Midpoint coordinates
            mid_idx = len(time) // 2
            x_base = time[mid_idx]
            y_base = y[mid_idx]

            # Dynamic label offsets based on cluster index
            x_offset = ((i % 5) - 5) * 2          
            y_offset = 0

            ax.text(x_base + x_offset,
                    y_base + y_offset,
                    f'Cluster {i}',
                    fontsize=9,
                    color=line.get_color(),
                    ha='center', va='center',
                    bbox=dict(facecolor='white', edgecolor='none', alpha=0.6))

        ax.set_title(title)
        ax.set_xlim([0, 80])
        ax.set_xlabel("Time")
        ax.set_ylabel(title)

    plot_with_inline_labels(axs[0], brake, "Brake")
    plot_with_inline_labels(axs[1], velocity, "Velocity")
    plot_with_inline_labels(axs[2], acceleration, "Acceleration")

    plt.tight_layout()

    # Save to file
    if not os.path.exists(os.path.dirname(output_path)):
        os.makedirs(os.path.dirname(output_path))
    plt.savefig(output_path, dpi=300)
    plt.show()
