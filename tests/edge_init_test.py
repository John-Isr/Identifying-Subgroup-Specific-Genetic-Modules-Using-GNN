import unittest
import torch
import os
import shutil
from utils.init_edge_features import reorder_patients_on_graph, save_graph_group


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


if __name__ == "__main__":
    unittest.main()