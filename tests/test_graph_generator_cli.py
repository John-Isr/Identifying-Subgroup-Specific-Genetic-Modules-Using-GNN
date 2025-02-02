import os
import unittest
import shutil
import argparse
from unittest.mock import patch, MagicMock
from utils.graph_generator import generate_and_save_graphs, conditions

class TestGraphGeneratorCLI(unittest.TestCase):
    def setUp(self):
        self.test_output_dir = os.path.join("..", "data", "test_graphs")
        os.makedirs(self.test_output_dir, exist_ok=True)
        self.num_graphs = 2  # Small number for testing

    def tearDown(self):
        shutil.rmtree(self.test_output_dir)

    @patch('argparse.ArgumentParser.parse_args')
    @patch('torch.save')
    @patch('utils.graph_generator.generate_graph')
    def test_cli_all_conditions(self, mock_generate_graph, mock_torch_save, mock_parse_args):
        # Mock the command-line arguments
        mock_parse_args.return_value = argparse.Namespace(
            conditions=['all'],
            output_dir=self.test_output_dir,
            num_graphs=self.num_graphs
        )
        # Mock the generate_graph function to return a dummy graph
        dummy_graph = MagicMock()
        mock_generate_graph.return_value = dummy_graph

        # Call the function to generate and save graphs
        for condition_name in conditions.keys():
            generate_and_save_graphs(
                num_graphs=self.num_graphs,
                output_dir=self.test_output_dir,
                condition_name=condition_name,
                **conditions[condition_name]
            )
            # Check if torch.save was called with the correct arguments
            output_file = os.path.join(self.test_output_dir, f"{condition_name}_graphs.pt")
            mock_torch_save.assert_called_with([dummy_graph] * self.num_graphs, output_file)

        # Check if generate_graph was called the expected number of times
        self.assertEqual(mock_generate_graph.call_count, self.num_graphs * len(conditions))

    @patch('argparse.ArgumentParser.parse_args')
    @patch('torch.save')
    @patch('utils.graph_generator.generate_graph')
    def test_cli_single_condition(self, mock_generate_graph, mock_torch_save, mock_parse_args):
        # Mock the command-line arguments
        mock_parse_args.return_value = argparse.Namespace(
            conditions=['Optimal'],
            output_dir=self.test_output_dir,
            num_graphs=self.num_graphs
        )
        # Mock the generate_graph function to return a dummy graph
        dummy_graph = MagicMock()
        mock_generate_graph.return_value = dummy_graph

        # Call the function to generate and save graphs
        generate_and_save_graphs(
            num_graphs=self.num_graphs,
            output_dir=self.test_output_dir,
            condition_name='Optimal',
            **conditions['Optimal']
        )

        # Check if generate_graph was called the expected number of times
        self.assertEqual(mock_generate_graph.call_count, self.num_graphs)
        # Check if torch.save was called with the correct arguments
        output_file = os.path.join(self.test_output_dir, "Optimal_graphs.pt")
        mock_torch_save.assert_called_with([dummy_graph] * self.num_graphs, output_file)

if __name__ == "__main__":
    unittest.main()