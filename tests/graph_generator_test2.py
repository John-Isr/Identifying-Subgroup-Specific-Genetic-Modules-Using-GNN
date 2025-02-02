import os
import unittest
import torch
import shutil
from unittest.mock import patch, MagicMock
from utils.graph_generator import generate_and_save_graphs, conditions

class TestGraphGenerator(unittest.TestCase):
    def setUp(self):
        self.test_output_dir = os.path.join("..", "data", "test_graphs")
        os.makedirs(self.test_output_dir, exist_ok=True)
        self.num_graphs = 2  # Small number for testing

    def tearDown(self):
        shutil.rmtree(self.test_output_dir)

    @patch('utils.graph_generator.generate_graph')
    @patch('torch.save')
    def test_generate_and_save_graphs(self, mock_torch_save, mock_generate_graph):
        # Mock the generate_graph function to return a dummy graph
        dummy_graph = MagicMock()
        mock_generate_graph.return_value = dummy_graph

        for condition_name, params in conditions.items():
            generate_and_save_graphs(
                num_graphs=self.num_graphs,
                output_dir=self.test_output_dir,
                condition_name=condition_name,
                **params
            )
            output_file = os.path.join(self.test_output_dir, f"{condition_name}_graphs.pt")
            mock_torch_save.assert_called_with([dummy_graph] * self.num_graphs, output_file)

if __name__ == "__main__":
    unittest.main()