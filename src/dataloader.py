from datasets import load_dataset

import random
import click
from logger import logging
from tqdm import tqdm
import torch
import multiprocessing
from networkx import graph_edit_distance as ged

import networkx as nx

random.seed(42)


def load_data(N_train=200, N_test=100, name='AIDS'):
    dataset = load_dataset(f"graphs-datasets/{name}")["full"]
    dataset = [graph for graph in dataset if graph["num_nodes"] <= 10]
    random.shuffle(dataset)

    train_dataset = dataset[:N_train]
    test_dataset = dataset[N_train : N_test + N_train]
    return train_dataset, test_dataset


def convert_to_networkx_graph(graph):
    # Create an empty NetworkX graph
    G = nx.MultiGraph()

    # Add nodes with features
    for node_id, features in enumerate(graph["node_feat"]):
        G.add_node(node_id, features=features)

    # Add edges with attributes
    for i, (src, dst) in enumerate(zip(graph["edge_index"][0], graph["edge_index"][1])):
        attrs = graph["edge_attr"][i]
        G.add_edge(src, dst, attrs=attrs)

    return G


def node_match(node1, node2):
    if node1["features"] == node2["features"]:
        return True
    return False


def edge_match(edge1, edge2):
    if len(edge1) != len(edge2):
        return False
    if len(edge1) > 1:
        for e1, e2 in zip(edge1, edge2):
            if e1["attrs"] != e2["attrs"]:
                return False
    else:
        if edge1["attrs"] != edge2["attrs"]:
            return False
    return True


def ged_map(i, j, arr):
    dist = ged(arr[i], arr[j], node_match=node_match)
    print(i, j)
    return dist
    # return ged(arr[i], arr[j], node_match=node_match, edge_match=edge_match)


@click.command()
@click.option("--dataset", default="AIDS", type=str)
@click.option("--split", default="train", type=str)
@click.option("--idx-from-start", type=int)
@click.option("--idx-from-end", type=int)
@click.option("--idx-to-start", type=int)
@click.option("--idx-to-end", type=int)
@click.option("--num-workers", type=int)
def main(
    dataset: str,
    split: str,
    idx_from_start: int = None,
    idx_from_end: int = None,
    idx_to_start: int = None,
    idx_to_end: int = None,
    num_workers: int = None,
):
    logging.info(f"Loading dataset {dataset}...")
    train_dataset, test_dataset = load_data(name=dataset)
    logging.info("Converting to networkx graphs...")
    if split == "train":
        data = [convert_to_networkx_graph(graph) for graph in train_dataset]
    elif split == "test":
        data = [convert_to_networkx_graph(graph) for graph in test_dataset]
    else:
        raise ValueError("`split` needs to be either `train` or `test`.")
    if idx_from_start is None:
        idx_from_start = 0
    if idx_from_end is None:
        idx_from_end = len(data)
    idx_from = (idx_from_start, idx_from_end)
    if idx_to_start is None:
        idx_to_start = 0
    if idx_to_end is None:
        idx_to_end = len(data)
    idx_to = (idx_to_start, idx_to_end)
    if num_workers is None:
        num_processes = multiprocessing.cpu_count()
    else:
        num_processes = num_workers
    manager = multiprocessing.Manager()
    shared_data = manager.list(data)
    inputs = [(i, j) for i in range(*idx_from) for j in range(*idx_to)]

    with multiprocessing.Pool(processes=num_processes) as pool:
        results = pool.starmap(
            ged_map, tqdm([(*x, shared_data) for x in inputs], total=len(inputs))
        )

    results = torch.tensor(results).reshape(
        (idx_from_end - idx_from_start, idx_to_end - idx_to_start)
    )
    torch.save(results, f"../Data/{dataset}_{split}_from_{idx_from}_to_{idx_to}")


if __name__ == "__main__":
    main()
