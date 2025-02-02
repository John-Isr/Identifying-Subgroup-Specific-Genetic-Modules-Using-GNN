import os
import torch
import logging
import argparse

# logging.basicConfig(level=logging.INFO)

def reorder_patients_on_graph(graph):
    edge_attr = graph.edge_attr.float()
    patient_data = edge_attr[:, :-1]
    last_column = edge_attr[:, -1:]
    patient_grades = (patient_data.T @ patient_data).sum(dim=1).tolist()
    sorted_grades = sorted(patient_grades, reverse=True)
    ranks = [sorted_grades.index(grade) for grade in patient_grades]
    reordered_patient_data = patient_data[:, ranks]
    graph.edge_attr = torch.cat([reordered_patient_data, last_column], dim=1)

def save_graph_group(group, name, output_dir):
    output_file = os.path.join(output_dir, f"{name}.pt")
    torch.save(group, output_file)
    logging.info(f"Saved {name} graphs to {output_file}")

def main():
    parser = argparse.ArgumentParser(description="Initialize edge features for graphs.")
    parser.add_argument('--input_dir', type=str, default=os.path.join("data", "graphs"), help="Directory containing the input graph files")
    parser.add_argument('--output_dir', type=str, default=os.path.join("data", "modified_graphs"), help="Directory to save the modified graph files")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    graph_files = {
        "defaultGraphs": os.path.join(args.input_dir, "Default_graphs.pt"),
        "optimalGraphs": os.path.join(args.input_dir, "Optimal_graphs.pt"),
        "suboptimalGraphs": os.path.join(args.input_dir, "Suboptimal_graphs.pt"),
        "hardGraphs": os.path.join(args.input_dir, "Hard_graphs.pt")
    }

    for name, path in graph_files.items():
        graphs = torch.load(path, weights_only=False)
        for graph in graphs:
            reorder_patients_on_graph(graph)
        save_graph_group(graphs, name, args.output_dir)

if __name__ == "__main__":
    main()