import unittest
import torch
import os
import shutil
import argparse
from utils.init_edge_features import reorder_patients_on_graph, save_graph_group, main

class TestInitEdgeFeatures(unittest.TestCase):

    def setUp(self):
        # Create a temporary directory for testing
        self.test_dir = "./test_graphs"
        os.makedirs(self.test_dir, exist_ok=True)

        # Create a sample graph for testing
        self.graph = torch.load(os.path.join("..", "data", "graphs", "Default_graphs.pt"))[0]

    def tearDown(self):
        # Remove the temporary directory after tests
        shutil.rmtree(self.test_dir)

    def test_reorder_patients_on_graph(self):
        original_edge_attr = self.graph.edge_attr.clone()
        reorder_patients_on_graph(self.graph)
        reordered_edge_attr = self.graph.edge_attr

        # Check if the edge attributes have been reordered
        self.assertFalse(torch.equal(original_edge_attr, reordered_edge_attr))

    def test_save_graph_group(self):
        graph_group = [self.graph]
        save_graph_group(graph_group, "test_graph", self.test_dir)

        # Check if the file has been saved
        saved_file_path = os.path.join(self.test_dir, "test_graph.pt")
        self.assertTrue(os.path.exists(saved_file_path))

        # Load the saved file and check its content
        loaded_graph_group = torch.load(saved_file_path)
        self.assertEqual(len(loaded_graph_group), 1)
        self.assertTrue(torch.equal(loaded_graph_group[0].edge_attr, self.graph.edge_attr))

    def test_argument_parsing(self):
        parser = argparse.ArgumentParser(description="Initialize edge features for graphs.")
        parser.add_argument('--input_dir', type=str, default=os.path.join("..", "data", "graphs"), help="Directory containing the input graph files")
        parser.add_argument('--output_dir', type=str, default=os.path.join("..", "data", "modified_graphs"), help="Directory to save the modified graph files")
        args = parser.parse_args(['--input_dir', './input', '--output_dir', './output'])

        self.assertEqual(args.input_dir, './input')
        self.assertEqual(args.output_dir, './output')

    def test_graph_loading(self):
        graph_files = {
            "defaultGraphs": os.path.join("..", "data", "graphs", "Default_graphs.pt"),
        }
        graphs = torch.load(graph_files["defaultGraphs"], weights_only=False)
        self.assertIsInstance(graphs, list)
        self.assertGreater(len(graphs), 0)

if __name__ == "__main__":
    unittest.main()