import networkx as nx
import matplotlib.pyplot as plt
from dataset import HW3Dataset
from torch_geometric.utils import to_networkx
import numpy as np

if __name__ == '__main__':
    dataset = HW3Dataset(root='data/hw3/')
    data = dataset[0]
    print(f'Dataset: {dataset}:')

    print(f'Number of features: {dataset.num_features}')
    print(f'Number of classes: {dataset.num_classes}')
    print(data)
    print('==============================================================')

    # Graphs statistics.
    print(f'Number of nodes: {data.num_nodes}')
    print(f'Number of edges: {data.num_edges}')
    print(f'Has isolated nodes: {data.has_isolated_nodes()}')
    print(f'Has self-loops: {data.has_self_loops()}')

    networkx_graph = to_networkx(data)
    cycle = nx.find_cycle(networkx_graph, orientation="original")
    print(f'Has cycles: {True if cycle else False}')

    in_degrees = [deg[1] for deg in networkx_graph.in_degree]
    out_degrees = [deg[1] for deg in networkx_graph.out_degree]
    counter = 0
    c=[]
    for i in range(len(in_degrees)):
        if in_degrees[i] == out_degrees[i] == 0:
            c.append(i)
            counter +=1

    in_degree_counts = np.bincount(data.y[c].squeeze(1).numpy())
    plt.bar(range(len(in_degree_counts)), in_degree_counts)
    plt.xlabel("Articles' categories")
    plt.ylabel('Count')
    plt.title('Category Distribution')
    plt.show()
    print("Number of isolated nodes:",counter)
    print("Maximum outgoing edges:", max(out_degrees))
    print("Minimum outgoing edges:", min(out_degrees))
    print("Average out degree:", np.mean(out_degrees))
    print("Median out degree:", np.median(out_degrees))

    print("Maximum incoming edges:", max(in_degrees))
    print("Minimum incoming edges:", min(in_degrees))
    print("Average in degree:", np.mean(in_degrees))
    print("Median in degree:", np.median(in_degrees))

    print(f"Years Range: {min(data.node_year).item()} - {max(data.node_year).item()}")