import torch
import numpy as np
import scipy.sparse as sp
import os
from torch_geometric.loader import DataLoader
from torch.utils.data import random_split


def load_data_splits(batch_size, device = "cuda" if torch.cuda.is_available() else "cpu", node_embedding_type='weighted', seed=42, graph_dir = "./data/modified_graphs"):
    """
    Loads graphs from the modified graphs directory, applies the specified node embedding,
    splits the dataset into train/validation/test, and returns corresponding DataLoaders.

    Parameters:`
        batch_size (int): Batch size for the DataLoader.
        node_embedding_type (str): Type of node embedding to apply. Options:
                                   'weighted', 'unweighted', 'spectral_positional_encoding'.
        seed (int): Random seed for reproducibility of the data splits.
        graph_dir (str): Directory containing the modified graphs.
    Returns:
        train_loader, val_loader, test_loader: DataLoaders for training, validation, and testing.
    """

    # Load graphs from the directory (adjust the directory path as needed)
    graphs = []
    for file in os.listdir(graph_dir):
        if file.endswith(".pt"):
            file_path = os.path.join(graph_dir, file)
            graphs.extend(torch.load(file_path, weights_only=False))

    # Select the appropriate embedding function based on the node_embedding_type argument
    if node_embedding_type == "weighted":
        embedding_func = compute_weighted_node_embeddings
    elif node_embedding_type == "unweighted":
        embedding_func = compute_unweighted_node_embeddings
    elif node_embedding_type == "spectral_positional_encoding":
        embedding_func = compute_spectral_positional_encoding
    else:
        raise ValueError(f"Unknown node embedding type: {node_embedding_type}")

    # Apply the node embedding function to each graph
    for graph in graphs:
        graph.x = embedding_func(graph)
        graph.y = graph.y.float()

    # Split the dataset into train, validation, and test sets
    dataset_size = len(graphs)
    train_size = int(0.7 * dataset_size)
    val_size = int(0.15 * dataset_size)
    test_size = dataset_size - train_size - val_size

    # Set the random seed for reproducibility
    torch.manual_seed(seed)
    generator = torch.Generator().manual_seed(seed)
    train_dataset, val_dataset, test_dataset = random_split(
        graphs, [train_size, val_size, test_size], generator=generator
    )

    # Create DataLoaders for each dataset split
    if device == 'cuda':
        train_loader = DataLoader(
            dataset=train_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=5,          # Spawns 5 subprocesses for data loading
            pin_memory=True,        # Allows async data transfer to GPU (when .to(device, non_blocking=True))
            persistent_workers=True # Keeps workers alive between epochs (newer PyTorch versions)
        )
        val_loader = DataLoader(
            dataset=val_dataset,
            batch_size=batch_size,
            shuffle=False
        )
        test_loader = DataLoader(
            dataset=test_dataset,
            batch_size=batch_size,
            shuffle=False
        )
    else:
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, val_loader, test_loader




def compute_weighted_node_embeddings(graph):
    """
    Computes node embeddings as the WEIGHTED mean of edge features using edge weights.

    Args:
        graph (torch_geometric.data.Data): PyG graph with edge_attr where
            last feature is the edge weight

    Returns:
        torch.Tensor: Weighted node embeddings (num_nodes, num_features-1)
    """
    edge_index = graph.edge_index
    edge_attr = graph.edge_attr
    num_nodes = graph.num_nodes

    assert edge_attr is not None and edge_attr.size(
        1) > 1, "Edge attributes must have at least two features (including weight)."

    edge_weights = edge_attr[:, -1]  # (num_edges,)
    edge_features = edge_attr[:, :-1]  # (num_edges, num_features-1)

    node_embeddings = torch.zeros((num_nodes, edge_features.size(1)), dtype=torch.float32)
    weight_sums = torch.zeros(num_nodes, dtype=torch.float32)


    for i in range(edge_index.size(1)):
        src = edge_index[0, i]
        dst = edge_index[1, i]
        features = edge_features[i]
        weight = edge_weights[i]

        # Weighted accumulation
        weighted_features = features * weight
        node_embeddings[dst] += weighted_features
        weight_sums[dst] += weight

    # Normalize by summed weights
    weight_sums = torch.where(weight_sums == 0,
                             torch.ones_like(weight_sums),
                             weight_sums)

    return node_embeddings / weight_sums.unsqueeze(-1)


def compute_unweighted_node_embeddings(graph):
    """
    Computes node embeddings as the mean of edge features connecting to the node.

    Args:
        graph (torch_geometric.data.Data): PyG graph with edge_attr where
            last feature is the edge weight

    Returns:
        torch.Tensor: Weighted node embeddings (num_nodes, num_features-1)
    """
    # Extract edge indices, edge attributes, and number of nodes
    edge_index = graph.edge_index  # Shape: (2, num_edges)
    edge_attr = graph.edge_attr    # Shape: (num_edges, num_features)
    num_nodes = graph.num_nodes
    # Validate input dimensions
    if edge_attr.dim() != 2 or edge_attr.size(1) < 2:
        raise ValueError("Edge attributes must be 2D with >=2 features (last is weight)")

    # Ensure edge_attr exists and has at least two features
    assert edge_attr is not None and edge_attr.size(1) > 1, "Edge attributes must have at least two features (including weight)."

    # Separate the weights (last feature) from the edge features
    edge_weights = edge_attr[:, -1].unsqueeze(-1)            # Shape: (num_edges,)
    edge_features = edge_attr[:, :-1]          # Shape: (num_edges, num_features - 1)

    # Initialize node embeddings and edge counts
    node_embeddings = torch.zeros((num_nodes, edge_features.size(1)), dtype=torch.float32)  # Shape: (num_nodes, num_features - 1)
    edge_counts = torch.zeros(num_nodes, dtype=torch.float32)                               # Shape: (num_nodes,)

    # Iterate over all edges and accumulate features for each node
    for edge, features in zip(edge_index.T, edge_features):
        src, dst = edge  # Source and destination nodes
        node_embeddings[dst] += features  # Accumulate features
        edge_counts[dst] += 1             # Count the edge

    # Normalize the node embeddings by the number of edges
    edge_counts = edge_counts.clamp_min(1e-10)  # Prevent division by zero
    node_embeddings /= edge_counts.unsqueeze(-1)

    return node_embeddings



def compute_spectral_positional_encoding(graph, num_eigenvectors=64):
    """
    Computes the spectral positional encoding using the eigenvectors of the normalized graph Laplacian.

    Args:
        graph (torch_geometric.data.Data): PyTorch Geometric graph object.
        num_eigenvectors (int): Number of eigenvectors to compute for the encoding.

    Returns:
        torch.Tensor: Node features initialized with spectral positional encoding.
    """
    # Step 1: Convert edge index to adjacency matrix
    num_nodes = graph.num_nodes
    edge_index = graph.edge_index.numpy()
    adj_matrix = sp.coo_matrix((np.ones(edge_index.shape[1]), (edge_index[0], edge_index[1])),
                               shape=(num_nodes, num_nodes))

    # Step 2: Compute degree matrix
    degree_matrix = sp.diags(adj_matrix.sum(axis=1).A1)

    # Steo 3 - compute normalized Laplacian L = D^(-0.5) * (D - A) * D^(-1/2)
    d_inv_sqrt = sp.diags(1.0 / np.sqrt(degree_matrix.diagonal()))
    laplacian = sp.eye(num_nodes) - d_inv_sqrt @ adj_matrix @ d_inv_sqrt

    # Step 4: Compute the first 'num_eigenvectors ' of the Laplacian
    eigenvalues, eigenvectors = np.linalg.eigh(laplacian.toarray())
    eigenvectors = eigenvectors[:, np.argsort(eigenvalues)[:num_eigenvectors]]

    # Convert to pytorch tensor and return
    return torch.tensor(eigenvectors, dtype=torch.float32)
