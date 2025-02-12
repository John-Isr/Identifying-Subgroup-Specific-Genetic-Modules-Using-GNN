import torch
import networkx as nx
from torch_geometric.utils import from_networkx
import os

def run_inference(nx_graph, model_path=os.path.join(".", "experiments", "trained_model.pt"), device='auto'):
    """
    Processes NetworkX graph through trained GNN
    Returns NetworkX graph with node classification feature

    Args:
        nx_graph (nx.Graph): Input graph without edge features
        model_path (str): Path to trained model checkpoint
        device (str): 'cuda', 'cpu', or 'auto'

    Returns:
        nx.Graph: Graph with node 'classification' feature
    """
    # Device detection
    if device == 'auto':
        device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # Convert to PyG format
    pyg_data = from_networkx(nx_graph)

    # Make sure graph has edge features
    if not hasattr(pyg_data, 'edge_attr'):
            raise ValueError("Input graph does not have edge features.")

    # Load model
    model = torch.load(model_path, map_location=device)
    model.eval()

    # Run inference
    with torch.no_grad():
        logits = model(pyg_data.x.float().to(device),
                       pyg_data.edge_index.to(device),
                       pyg_data.edge_attr.float().to(device))
        probs = torch.sigmoid(logits).cpu().numpy()

    # Add predictions to NetworkX graph
    nx.set_node_attributes(nx_graph,
                           {i: float(probs[i]) for i in range(len(probs))},
                           'classification')

    return nx_graph