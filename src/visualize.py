import math

import matplotlib.pyplot as plt
import networkx as nx


def visualize_graphs(graphs):
    nb_graphs = len(graphs)
    width = math.ceil(math.sqrt(nb_graphs))
    fig, axs = plt.subplots(width, width, figsize=(8, 8))
    cmap = plt.get_cmap("tab10")
    for i, g in enumerate(graphs):
        row_idx = i // width
        col_idx = i % width
        ax = axs[row_idx, col_idx] if width > 1 else axs[col_idx]
        nx.draw(
            graphs[i],
            ax=ax,
            with_labels=True,
            node_color=cmap(i),
            node_size=350,
            font_size=12,
        )
    plt.show()
