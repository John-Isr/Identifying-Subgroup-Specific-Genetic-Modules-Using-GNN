import os
import torch
import logging
import argparse
from utils.simulation import run_simulation
from torch_geometric.utils import from_networkx

# Function definitions
def generate_graph(nodes=500, p_cluster_NOT_part_of_process=0.2,
                   p_cluster_part_of_process=0.9, Beta=0.9,
                   lower_correlation_bound=0.7, upper_correlation_bound=0.9):
    try:
        G, m_patient_is_sick, latent_process_cluster, comms, comms_inverted = run_simulation(
            nodes=nodes, p_cluster_NOT_part_of_process=p_cluster_NOT_part_of_process,
            p_cluster_part_of_process=p_cluster_part_of_process, Beta=Beta,
            lower_correlation_bound=lower_correlation_bound,
            upper_correlation_bound=upper_correlation_bound,
            plot_graph_stats=False, simple_feature_vector=False
        )
        data = from_networkx(G, group_edge_attrs='all')
        data.y = torch.tensor(
            [1 if node in comms_inverted[latent_process_cluster] else 0 for node in G.nodes()],
            device="cuda"
        )
        return data
    except Exception as e:
        logging.error(f"Error generating graph: {e}")
        raise

def generate_and_save_graphs(num_graphs, output_dir, condition_name, **kwargs):
    output_file = os.path.join(output_dir, f"{condition_name}_graphs.pt")
    graphs = []
    for i in range(num_graphs):
        graph = generate_graph(**kwargs)
        graphs.append(graph)
        logging.info(f"Generated {i + 1}/{num_graphs} graphs for {condition_name}.")
    torch.save(graphs, output_file)
    logging.info(f"Saved {num_graphs} graphs to {output_file}")

# Condition definitions
conditions = {
    "Optimal": {
        "p_cluster_NOT_part_of_process": 0,
        "p_cluster_part_of_process": 1,
        "Beta": 1,
        "lower_correlation_bound": 0.85,
        "upper_correlation_bound": 0.95
    },
    "Suboptimal": {
        "p_cluster_NOT_part_of_process": 0.1,
        "p_cluster_part_of_process": 0.9,
        "Beta": 0.9,
        "lower_correlation_bound": 0.85,
        "upper_correlation_bound": 0.95
    },
    "Default": {
        "p_cluster_NOT_part_of_process": 0.2,
        "p_cluster_part_of_process": 0.9,
        "Beta": 0.9,
        "lower_correlation_bound": 0.7,
        "upper_correlation_bound": 0.9
    },
    "Hard": {
        "p_cluster_NOT_part_of_process": 0.25,
        "p_cluster_part_of_process": 0.85,
        "Beta": 0.85,
        "lower_correlation_bound": 0.7,
        "upper_correlation_bound": 0.9
    },
}

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate and save graphs.")
    parser.add_argument('--conditions', type=str, nargs='+', choices=['all', 'Optimal', 'Suboptimal', 'Default', 'Hard'], required=True, help="Conditions to simulate. Choices: all, Optimal, Suboptimal, Default, Hard")
    parser.add_argument('--output_dir', type=str, default=os.path.join("..", "data", "graphs"), help="Directory to save the generated graphs")
    parser.add_argument('--num_graphs', type=int, default=1000, help="Number of graphs to generate per condition")
    args = parser.parse_args()

    selected_conditions = conditions.keys() if 'all' in args.conditions else args.conditions

    for condition_name in selected_conditions:
        generate_and_save_graphs(
            num_graphs=args.num_graphs,
            output_dir=args.output_dir,
            condition_name=condition_name,
            **conditions[condition_name]
        )