import os
import torch
import networkx as nx
import matplotlib.pyplot as plt
from dataset import HW3Dataset
from torch_geometric.utils import to_networkx

def visualize_graph(G, color):
    plt.figure(figsize=(7,7))
    plt.xticks([])
    plt.yticks([])
    nx.draw_networkx(G, pos=nx.spring_layout(G, seed=42), with_labels=False,
                     node_color=color, cmap="Set2")
    plt.show()

def visualize_embedding(h, color, epoch=None, loss=None):
    plt.figure(figsize=(7,7))
    plt.xticks([])
    plt.yticks([])
    h = h.detach().cpu().numpy()
    plt.scatter(h[:, 0], h[:, 1], s=140, c=color, cmap="Set2")
    if epoch is not None and loss is not None:
        plt.xlabel(f'Epoch: {epoch}, Loss: {loss.item():.4f}', fontsize=16)
    plt.show()

if __name__ == '__main__':
    dataset = HW3Dataset(root='data/hw3/')
    data = dataset[0]
    print(f'Dataset: {dataset}:')
    print('======================')
    print(f'Number of graphs: {len(dataset)}')
    print(f'Number of features: {dataset.num_features}')
    print(f'Number of classes: {dataset.num_classes}')
    print(data)
    print('==============================================================')

    # Gather some statistics about the graph.
    print(f'Number of nodes: {data.num_nodes}')
    print(f'Number of edges: {data.num_edges}')
    print(f'Average node degree: {data.num_edges / data.num_nodes:.2f}')
    print(f'Number of training nodes: {len(data.train_mask)}')
    print(f'Training node label rate: {len(data.train_mask) / data.num_nodes:.2f}')
    print(f'Has isolated nodes: {data.has_isolated_nodes()}')
    print(f'Has self-loops: {data.has_self_loops()}')
    print(f'Is undirected: {data.is_undirected()}')

    # Convert your PyTorch Geometric graph to a NetworkX graph
    networkx_graph = to_networkx(data)

    # Run Dijkstra's algorithm to find the longest path
    longest_path = nx.dag_longest_path(networkx_graph)

    # Print the longest path
    print("Longest path:", longest_path)

    # Get the number of nodes in the graph
    num_nodes = data.num_nodes

    # Initialize variables to store the maximum counts and their corresponding nodes
    max_outgoing_edges = 0
    nodes_with_max_outgoing = []
    max_incoming_edges = 0
    nodes_with_max_incoming = []

    # Iterate over each node in the graph
    for node in range(num_nodes):
        # Count the outgoing edges for the current node
        outgoing_edges = data.out_degree(node)

        # Check if it surpasses the current maximum outgoing count
        if outgoing_edges > max_outgoing_edges:
            max_outgoing_edges = outgoing_edges
            nodes_with_max_outgoing = [node]
        elif outgoing_edges == max_outgoing_edges:
            nodes_with_max_outgoing.append(node)

        # Count the incoming edges for the current node
        incoming_edges = data.in_degree(node)

        # Check if it surpasses the current maximum incoming count
        if incoming_edges > max_incoming_edges:
            max_incoming_edges = incoming_edges
            nodes_with_max_incoming = [node]
        elif incoming_edges == max_incoming_edges:
            nodes_with_max_incoming.append(node)

    # Print the results
    print("Maximum outgoing edges:", max_outgoing_edges)
    print("Nodes with maximum outgoing edges:", nodes_with_max_outgoing)
    print("Maximum incoming edges:", max_incoming_edges)
    print("Nodes with maximum incoming edges:", nodes_with_max_incoming)